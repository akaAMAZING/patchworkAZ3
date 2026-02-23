#!/usr/bin/env python
"""
Patchwork Meta-Analysis: Full-depth analytics for AlphaZero training.

Produces arXiv-style metrics and visualizations:
- First-player advantage (p0 vs p1 win rate over iterations)
- Piece tier list (policy mass when legal, inferred from slot shapes)
- Pass vs buy propensity (action-type distribution)
- Patch placement heatmap (9x9 preferred 1x1 positions)
- Value by game phase (early/mid/late)
- Value calibration (predicted vs target)
- Policy sharpness over game
- Score margin / value distribution
- Ownership heatmap (via model inference, optional)

Usage:
    python tools/meta_analysis.py --config configs/config_best.yaml --run-dir runs/patchwork_production
    python tools/meta_analysis.py --config configs/config_best.yaml --run-dir runs/patchwork_production --iters 40-60 --max-positions 50000 --ownership
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import yaml

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from tqdm import tqdm
    tqdm_write = tqdm.write
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable
    tqdm_write = print

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Action space layout (from ActionEncoder)
PASS_INDEX = 0
PATCH_START = 1
BUY_START = 82
BOARD_SIZE = 9
NUM_SLOTS = 3
NUM_ORIENTS = 8
CELLS = BOARD_SIZE * BOARD_SIZE
MAX_POSITION = 53


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_replay_entries(run_dir: Path) -> List[Tuple[int, Path, int]]:
    """Load replay buffer entries (iter, path, positions) from replay_state.json."""
    state_path = run_dir / "replay_state.json"
    if not state_path.exists():
        return []
    with open(state_path, "r") as f:
        state = json.load(f)
    entries = []
    for e in state:
        p = Path(e["path"])
        if not p.is_absolute():
            p = run_dir / p if (run_dir / p).exists() else REPO_ROOT / p
        if p.exists():
            entries.append((int(e["iteration"]), p, int(e["positions"])))
    return sorted(entries, key=lambda x: x[0])


def load_iteration_summaries(run_dir: Path, iter_start: int, iter_end: int) -> List[dict]:
    """Load iteration_XXX.json for p0/p1 wins and other per-iter metrics."""
    committed = run_dir / "committed"
    results = []
    for it in range(iter_start, iter_end + 1):
        jpath = committed / f"iter_{it:03d}" / f"iteration_{it:03d}.json"
        if not jpath.exists():
            continue
        with open(jpath, "r") as f:
            d = json.load(f)
        sp = d.get("selfplay_stats", {})
        results.append({
            "iteration": it,
            "p0_wins": int(sp.get("p0_wins", 0)),
            "p1_wins": int(sp.get("p1_wins", 0)),
            "num_games": int(sp.get("num_games", 0)),
            "avg_game_length": float(sp.get("avg_game_length", 0)),
            "avg_top1_prob": float(sp.get("avg_top1_prob", 0)),
            "avg_policy_entropy": float(sp.get("avg_policy_entropy", 0)),
        })
    return results


def load_h5_data(paths: List[Path], max_positions: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and merge states, masks, policies, values from multiple HDF5 files. Subsample if needed."""
    all_states, all_masks, all_policies, all_values = [], [], [], []
    total = 0
    for p in paths:
        with h5py.File(p, "r") as f:
            n = f["states"].shape[0]
            if total + n > max_positions:
                take = max_positions - total
                idx = np.sort(rng.choice(n, size=take, replace=False))
            else:
                take = n
                idx = np.arange(n)
            all_states.append(np.asarray(f["states"][idx], dtype=np.float32))
            all_masks.append(np.asarray(f["action_masks"][idx], dtype=np.float32))
            all_policies.append(np.asarray(f["policies"][idx], dtype=np.float32))
            all_values.append(np.asarray(f["values"][idx], dtype=np.float32))
            total += take
        if total >= max_positions:
            break
    return (
        np.concatenate(all_states, axis=0)[:max_positions],
        np.concatenate(all_masks, axis=0)[:max_positions],
        np.concatenate(all_policies, axis=0)[:max_positions],
        np.concatenate(all_values, axis=0)[:max_positions],
    )


def build_slot_shape_to_piece_map() -> Dict[Tuple, int]:
    """Build reverse map: (slot shape as tuple) -> piece_id. Uses StateEncoder slot×orient shapes."""
    from src.network.encoder import StateEncoder
    from src.game.patchwork_engine import PIECE_BY_ID, ORIENT_COUNT
    enc = StateEncoder()
    out = {}
    for pid in PIECE_BY_ID:
        if int(ORIENT_COUNT[pid]) > 0:
            mask = enc._get_slot_orient_shape(int(pid), 0)
            key = tuple(np.round(np.clip(mask, 0, 1)).astype(np.uint8).flatten())
            out[key] = int(pid)
    return out


def infer_piece_ids_from_state(state: np.ndarray, shape_map: Dict) -> List[Optional[int]]:
    """Infer piece_id for slots 0,1,2 from state channels 4-27 (slot×orient shapes)."""
    piece_ids = []
    for slot in range(3):
        base = 4 + slot * 8
        mask = np.zeros((9, 9), dtype=np.float32)
        for o in range(8):
            if state[base + o].sum() > 0:
                mask = np.maximum(mask, state[base + o])
                break
        key = tuple(np.round(np.clip(mask, 0, 1)).astype(np.uint8).flatten())
        piece_ids.append(shape_map.get(key))
    return piece_ids


def compute_first_player_advantage(summaries: List[dict]) -> List[dict]:
    """First-player (p0) win rate by iteration."""
    rows = []
    for s in summaries:
        p0, p1 = s["p0_wins"], s["p1_wins"]
        total = p0 + p1
        wr = p0 / total if total > 0 else 0.5
        rows.append({
            "iteration": s["iteration"],
            "p0_wins": p0,
            "p1_wins": p1,
            "total_games": total,
            "p0_win_rate": wr,
            "elo_advantage": 400 * np.log10(wr / (1 - wr + 1e-9)) if 0 < wr < 1 else 0,
        })
    return rows


def compute_action_type_distribution(masks: np.ndarray, policies: np.ndarray) -> dict:
    """Policy mass on pass (0), patch (1-81), buy (82+). Normalized by legal mass per position."""
    pass_legal = masks[:, PASS_INDEX] > 0
    patch_legal = (masks[:, PATCH_START:BUY_START] > 0).any(axis=1)
    buy_legal = (masks[:, BUY_START:] > 0).any(axis=1)

    pass_mass = np.zeros(len(masks))
    patch_mass = np.zeros(len(masks))
    buy_mass = np.zeros(len(masks))

    for i in range(len(masks)):
        m = masks[i]
        p = policies[i] * m
        s = p.sum()
        if s > 0:
            p = p / s
        pass_mass[i] = p[PASS_INDEX] if pass_legal[i] else np.nan
        patch_mass[i] = p[PATCH_START:BUY_START].sum() if patch_legal[i] else np.nan
        buy_mass[i] = p[BUY_START:].sum() if buy_legal[i] else np.nan

    return {
        "pass_mean": float(np.nanmean(pass_mass)),
        "pass_std": float(np.nanstd(pass_mass)),
        "patch_mean": float(np.nanmean(patch_mass)),
        "patch_std": float(np.nanstd(patch_mass)),
        "buy_mean": float(np.nanmean(buy_mass)),
        "buy_std": float(np.nanstd(buy_mass)),
        "n_pass_legal": int(np.sum(pass_legal)),
        "n_patch_legal": int(np.sum(patch_legal)),
        "n_buy_legal": int(np.sum(buy_legal)),
    }


def compute_patch_placement_heatmap(masks: np.ndarray, policies: np.ndarray) -> np.ndarray:
    """Average policy mass on patch placements (indices 1-81) -> 9x9 heatmap."""
    heat = np.zeros((BOARD_SIZE, BOARD_SIZE))
    count = np.zeros((BOARD_SIZE, BOARD_SIZE))
    for i in range(len(masks)):
        m = masks[i]
        p = policies[i]
        patch_m = m[PATCH_START:BUY_START]
        if patch_m.sum() <= 0:
            continue
        patch_p = p[PATCH_START:BUY_START] * patch_m
        patch_p = patch_p / patch_p.sum()
        for pos in range(81):
            if patch_m[pos] > 0:
                r, c = divmod(pos, BOARD_SIZE)
                heat[r, c] += patch_p[pos]
                count[r, c] += 1
    with np.errstate(divide="ignore", invalid="ignore"):
        heat = np.where(count > 0, heat / count, 0)
    return heat


def compute_piece_ranking(
    states: np.ndarray,
    masks: np.ndarray,
    policies: np.ndarray,
    values: np.ndarray,
    shape_map: Dict,
) -> Tuple[List[dict], List[dict]]:
    """Rank pieces by policy mass and EV when buyable. Returns (ranking, piece_value_overall)."""
    from src.game.patchwork_engine import PIECE_BY_ID

    piece_totals = {
        pid: {
            "mass": 0.0,
            "count": 0,
            "value_sum": 0.0,
            "top_choice_value_sum": 0.0,
            "top_choice_count": 0,
            "by_phase": {"early": {"value_sum": 0.0, "count": 0}, "mid": {"value_sum": 0.0, "count": 0}, "late": {"value_sum": 0.0, "count": 0}},
        }
        for pid in PIECE_BY_ID
    }
    has_phase = states.shape[1] >= 34
    pos_norm = states[:, 34, 0, 0] if has_phase else None  # absolute_time_norm (ch34)

    for i in range(len(states)):
        piece_ids = infer_piece_ids_from_state(states[i], shape_map)
        m = masks[i]
        p = policies[i]
        val = values[i]
        buy_m = m[BUY_START:]
        if buy_m.sum() <= 0:
            continue
        buy_p = p[BUY_START:] * buy_m
        buy_p = buy_p / buy_p.sum()

        # Phase bucket
        if has_phase:
            pn = pos_norm[i]
            if pn < 0.33:
                phase = "early"
            elif pn < 0.66:
                phase = "mid"
            else:
                phase = "late"
        else:
            phase = "mid"

        best_slot = -1
        best_mass = -1.0
        slot_masses = []

        for slot in range(NUM_SLOTS):
            pid = piece_ids[slot]
            if pid is None:
                slot_masses.append(-1.0)
                continue
            slot_orient_size = NUM_ORIENTS * CELLS
            slot_start = BUY_START + slot * slot_orient_size
            slot_end = slot_start + slot_orient_size
            mass = buy_p[slot_start - BUY_START : slot_end - BUY_START].sum()
            slot_masses.append(mass)
            if m[slot_start:slot_end].sum() > 0:
                piece_totals[pid]["mass"] += mass
                piece_totals[pid]["count"] += 1
                piece_totals[pid]["value_sum"] += val
                piece_totals[pid]["by_phase"][phase]["value_sum"] += val
                piece_totals[pid]["by_phase"][phase]["count"] += 1
                if mass > best_mass:
                    best_mass = mass
                    best_slot = slot

        if best_slot >= 0 and piece_ids[best_slot] is not None:
            pid = piece_ids[best_slot]
            piece_totals[pid]["top_choice_value_sum"] += val
            piece_totals[pid]["top_choice_count"] += 1

    rows = []
    for pid in PIECE_BY_ID:
        d = piece_totals[pid]
        if d["count"] > 0:
            rows.append({
                "piece_id": pid,
                "mean_policy_mass": d["mass"] / d["count"],
                "mean_value_when_buyable": d["value_sum"] / d["count"],
                "mean_value_when_top_choice": d["top_choice_value_sum"] / d["top_choice_count"] if d["top_choice_count"] > 0 else None,
                "n_positions": d["count"],
                "n_top_choice": d["top_choice_count"],
                "cost_buttons": PIECE_BY_ID[pid]["cost_buttons"],
                "cost_time": PIECE_BY_ID[pid]["cost_time"],
            })
    rows.sort(key=lambda x: x["mean_policy_mass"], reverse=True)

    # Piece value by phase (for pieces with enough data per phase)
    piece_value_by_phase = []
    for pid in PIECE_BY_ID:
        d = piece_totals[pid]
        if d["count"] < 50:
            continue
        phase_vals = {}
        for ph in ["early", "mid", "late"]:
            c = d["by_phase"][ph]["count"]
            if c >= 10:
                phase_vals[ph] = d["by_phase"][ph]["value_sum"] / c
        if phase_vals:
            piece_value_by_phase.append({
                "piece_id": pid,
                "cost_buttons": PIECE_BY_ID[pid]["cost_buttons"],
                "cost_time": PIECE_BY_ID[pid]["cost_time"],
                "mean_value_overall": d["value_sum"] / d["count"],
                "n_positions": d["count"],
                **{f"mean_value_{ph}": v for ph, v in phase_vals.items()},
            })
    piece_value_by_phase.sort(key=lambda x: x["mean_value_overall"], reverse=True)

    return rows, piece_value_by_phase


def compute_value_by_phase(states: np.ndarray, values: np.ndarray) -> List[dict]:
    """Bucket value by game phase (channel 14 = absolute position [0,1]).
    Includes mean_abs_value (magnitude, doesn't cancel to 0) and value_std.
    """
    if states.shape[1] < 34:
        return []
    pos_norm = states[:, 34, 0, 0]  # absolute_time_norm
    bins = [(0, 0.33), (0.33, 0.66), (0.66, 1.01)]
    phases = ["early", "mid", "late"]
    rows = []
    for (lo, hi), name in zip(bins, phases):
        mask = (pos_norm >= lo) & (pos_norm < hi)
        if mask.sum() > 0:
            v = values[mask]
            rows.append({
                "phase": name,
                "pos_lo": lo,
                "pos_hi": hi,
                "value_mean": float(np.mean(v)),
                "value_std": float(np.std(v)),
                "mean_abs_value": float(np.mean(np.abs(v))),
                "n": int(mask.sum()),
            })
    return rows


def run_value_by_phase_inference(
    config: dict,
    checkpoint_path: Path,
    states: np.ndarray,
    values: np.ndarray,
    device,
) -> List[dict]:
    """Run model inference, bucket predictions by phase. Returns per-phase model stats."""
    import torch
    from src.network.model import create_network, load_model_checkpoint

    if states.shape[1] < 34:
        return []
    pos_norm = states[:, 34, 0, 0]  # absolute_time_norm
    bins = [(0, 0.33), (0.33, 0.66), (0.66, 1.01)]
    phases = ["early", "mid", "late"]

    net = create_network(config)
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    load_model_checkpoint(net, ckpt["model_state_dict"])
    net.to(device).eval()

    preds = []
    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(states), batch_size):
            batch = states[i : i + batch_size]
            x = torch.from_numpy(batch).to(device)
            _, v = net(x)
            preds.append(v.cpu().numpy().flatten())
    preds = np.concatenate(preds)

    rows = []
    for (lo, hi), name in zip(bins, phases):
        mask = (pos_norm >= lo) & (pos_norm < hi)
        if mask.sum() < 20:
            continue
        p = preds[mask]
        t = values[mask]
        rows.append({
            "phase": name,
            "pred_mean": float(np.mean(p)),
            "pred_std": float(np.std(p)),
            "mean_abs_pred": float(np.mean(np.abs(p))),
            "mse_vs_target": float(np.mean((p - t) ** 2)),
            "n": int(mask.sum()),
        })
    return rows


def compute_policy_sharpness_by_phase(states: np.ndarray, masks: np.ndarray, policies: np.ndarray) -> List[dict]:
    """Policy entropy by game phase."""
    if states.shape[1] < 34:
        return []
    pos_norm = states[:, 34, 0, 0]  # absolute_time_norm
    entropies = []
    for i in range(len(masks)):
        m = masks[i]
        p = policies[i] * m
        s = p.sum()
        if s > 0:
            p = p / s
            p = p[p > 0]
            h = -np.sum(p * np.log(p + 1e-12))
            entropies.append((pos_norm[i], h))
        else:
            entropies.append((pos_norm[i], np.nan))
    arr = np.array([(x, h) for x, h in entropies if not np.isnan(h)])
    if len(arr) == 0:
        return []
    pos_norm = arr[:, 0]
    h_arr = arr[:, 1]
    bins = [(0, 0.33), (0.33, 0.66), (0.66, 1.01)]
    phases = ["early", "mid", "late"]
    rows = []
    for (lo, hi), name in zip(bins, phases):
        mask = (pos_norm >= lo) & (pos_norm < hi)
        if mask.sum() > 0:
            rows.append({
                "phase": name,
                "entropy_mean": float(np.mean(h_arr[mask])),
                "entropy_std": float(np.std(h_arr[mask])),
                "n": int(mask.sum()),
            })
    return rows


def run_value_calibration(
    config: dict,
    checkpoint_path: Path,
    states: np.ndarray,
    values: np.ndarray,
    max_samples: int,
    device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run model inference, return predicted vs target for calibration plot."""
    import torch
    from src.network.model import create_network, load_model_checkpoint

    n = min(len(states), max_samples)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(states), size=n, replace=False)
    states_sub = states[idx]
    values_sub = values[idx]

    net = create_network(config)
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    load_model_checkpoint(net, ckpt["model_state_dict"])
    net.to(device).eval()

    preds = []
    batch_size = 256
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = states_sub[i : i + batch_size]
            x = torch.from_numpy(batch).to(device)
            _, v = net(x)
            preds.append(v.cpu().numpy().flatten())
    preds = np.concatenate(preds)
    return preds, values_sub


def run_ownership_inference(
    config: dict,
    checkpoint_path: Path,
    states: np.ndarray,
    n_samples: int,
    device,
) -> Optional[np.ndarray]:
    """Run model with ownership head, return (n, 2, 9, 9) ownership probs."""
    import torch
    import torch.nn.functional as F
    from src.network.model import create_network, load_model_checkpoint

    n = min(len(states), n_samples)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(states), size=n, replace=False)
    states_sub = states[idx]

    net = create_network(config)
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    load_model_checkpoint(net, ckpt["model_state_dict"])
    net.to(device).eval()

    if net.ownership_head is None:
        return None

    with torch.no_grad():
        x = torch.from_numpy(states_sub).to(device)
        trunk = net._trunk_forward(x)
        own_logits = net.ownership_head(trunk)
        own_probs = torch.sigmoid(own_logits).cpu().numpy()
    return own_probs


def save_reports(out_dir: Path, report: dict) -> None:
    """Save JSON summary and CSVs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "meta_analysis_report.json", "w") as f:
        json.dump(report, f, indent=2)

    if "first_player" in report:
        with open(out_dir / "first_player_advantage.csv", "w", newline="") as f:
            import csv
            w = csv.writer(f)
            w.writerow(["iteration", "p0_wins", "p1_wins", "total_games", "p0_win_rate", "elo_advantage"])
            for r in report["first_player"]:
                w.writerow([r["iteration"], r["p0_wins"], r["p1_wins"], r["total_games"], f"{r['p0_win_rate']:.4f}", f"{r['elo_advantage']:.1f}"])

    if "piece_ranking" in report:
        with open(out_dir / "piece_ranking.csv", "w", newline="") as f:
            import csv
            w = csv.writer(f)
            w.writerow(["piece_id", "mean_policy_mass", "mean_value_when_buyable", "mean_value_when_top_choice", "n_positions", "n_top_choice", "cost_buttons", "cost_time"])
            for r in report["piece_ranking"]:
                mvtc = f"{r['mean_value_when_top_choice']:.4f}" if r.get("mean_value_when_top_choice") is not None else ""
                w.writerow([r["piece_id"], f"{r['mean_policy_mass']:.6f}", f"{r.get('mean_value_when_buyable', 0):.4f}", mvtc, r["n_positions"], r.get("n_top_choice", 0), r["cost_buttons"], r["cost_time"]])
    if "piece_value" in report and report["piece_value"]:
        with open(out_dir / "piece_value.csv", "w", newline="") as f:
            import csv
            w = csv.writer(f)
            cols = ["piece_id", "mean_value_overall", "n_positions", "cost_buttons", "cost_time"]
            phase_cols = [k for k in report["piece_value"][0] if k.startswith("mean_value_") and k != "mean_value_overall"]
            cols += sorted(phase_cols)
            w.writerow(cols)
            for r in report["piece_value"]:
                row = [r["piece_id"], f"{r['mean_value_overall']:.4f}", r["n_positions"], r["cost_buttons"], r["cost_time"]]
                row += [f"{r.get(k, 0):.4f}" for k in sorted(phase_cols)]
                w.writerow(row)


def plot_all(report: dict, out_dir: Path) -> None:
    """Generate all matplotlib figures."""
    if not HAS_MATPLOTLIB:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. First-player advantage
    if "first_player" in report and report["first_player"]:
        fp = report["first_player"]
        fig, ax = plt.subplots(figsize=(8, 4))
        iters = [r["iteration"] for r in fp]
        wr = [r["p0_win_rate"] * 100 for r in fp]
        ax.plot(iters, wr, "o-", color="C0")
        ax.axhline(50, color="gray", linestyle="--")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("P0 Win Rate (%)")
        ax.set_title("First-Player Advantage")
        ax.set_ylim(40, 60)
        ax.grid(True, alpha=0.3)
        fig.savefig(out_dir / "first_player_advantage.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 2. Action type distribution (bar)
    if "action_type" in report:
        at = report["action_type"]
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = ["Pass", "Patch (1x1)", "Buy"]
        means = [at["pass_mean"], at["patch_mean"], at["buy_mean"]]
        stds = [at["pass_std"], at["patch_std"], at["buy_std"]]
        x = np.arange(3)
        ax.bar(x, means, yerr=stds, capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Mean policy mass (when legal)")
        ax.set_title("Action type distribution")
        ax.grid(True, alpha=0.3, axis="y")
        fig.savefig(out_dir / "action_type_distribution.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 3. Patch placement heatmap
    if "patch_heatmap" in report:
        heat = np.array(report["patch_heatmap"])
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(heat, cmap="YlOrRd", vmin=0, vmax=heat.max() if heat.size > 0 else 1)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        ax.set_title("Preferred 1×1 patch placement (policy mass)")
        plt.colorbar(im, ax=ax)
        fig.savefig(out_dir / "patch_placement_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 4. Piece ranking (horizontal bar)
    if "piece_ranking" in report and report["piece_ranking"]:
        pr = report["piece_ranking"]
        fig, ax = plt.subplots(figsize=(8, 10))
        labels = [f"Piece {r['piece_id']} (cost {r['cost_buttons']}b/{r['cost_time']}t)" for r in pr]
        vals = [r["mean_policy_mass"] for r in pr]
        y = np.arange(len(labels))
        ax.barh(y, vals)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Mean policy mass when buyable")
        ax.set_title("Piece tier list (policy preference)")
        ax.grid(True, alpha=0.3, axis="x")
        fig.savefig(out_dir / "piece_ranking.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 4b. Piece value (EV when buyable)
    if "piece_value" in report and report["piece_value"]:
        pv = report["piece_value"]
        has_phase = any(k.startswith("mean_value_") and k != "mean_value_overall" for k in pv[0])
        if has_phase:
            fig, axes = plt.subplots(2, 1, figsize=(8, 10))
            ax = axes[0]
        else:
            fig, ax = plt.subplots(figsize=(8, 8))
        labels = [f"Piece {r['piece_id']} ({r['cost_buttons']}b/{r['cost_time']}t)" for r in pv]
        vals = [r["mean_value_overall"] for r in pv]
        y = np.arange(len(labels))
        colors = ["C0" if v >= 0 else "C3" for v in vals]
        ax.barh(y, vals, color=colors)
        ax.axvline(0, color="gray", linestyle="-")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Mean value (EV) when buyable")
        ax.set_title("Piece value — expected outcome when piece is in offer")
        ax.grid(True, alpha=0.3, axis="x")
        if has_phase:
            ax = axes[1]
            phases = ["early", "mid", "late"]
            x = np.arange(len(pv))
            w = 0.25
            for i, ph in enumerate(phases):
                key = f"mean_value_{ph}"
                if key in pv[0]:
                    vals_ph = [r.get(key, 0) for r in pv]
                    offset = (i - 1) * w
                    ax.bar(x + offset, vals_ph, w, label=ph)
            ax.axhline(0, color="gray", linestyle="-")
            ax.set_xticks(x)
            ax.set_xticklabels([r["piece_id"] for r in pv], fontsize=7)
            ax.set_xlabel("Piece ID")
            ax.set_ylabel("Mean value when buyable")
            ax.set_title("Piece value by game phase")
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        fig.savefig(out_dir / "piece_value.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 5. Value by phase (high-quality: mean |value|, model predictions, box plot)
    if "value_by_phase" in report and report["value_by_phase"]:
        vp = report["value_by_phase"]
        phases = [r["phase"] for r in vp]
        has_model = "mean_abs_pred" in vp[0]

        if has_model:
            # Two subplots: (1) magnitude bars, (2) model calibration
            fig, axes = plt.subplots(2, 1, figsize=(7, 8))
            # Top: Position decisiveness (mean |target|) vs model confidence (mean |pred|)
            ax = axes[0]
            x = np.arange(len(phases))
            w = 0.35
            ma_tgt = [r["mean_abs_value"] for r in vp]
            ax.bar(x - w/2, ma_tgt, w, label="Target (mean |value|)", color="C0", alpha=0.8)
            if has_model:
                ma_pred = [r.get("mean_abs_pred", 0) for r in vp]
                ax.bar(x + w/2, ma_pred, w, label="Model (mean |pred|)", color="C1", alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(phases)
            ax.set_ylabel("Magnitude")
            ax.set_title("Value by game phase — position decisiveness & model confidence")
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")
            # Bottom: Calibration error (MSE) by phase
            ax = axes[1]
            mse_vals = [r.get("mse_vs_target", 0) for r in vp]
            ax.bar(x, mse_vals, color="C2", alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(phases)
            ax.set_ylabel("MSE (pred vs target)")
            ax.set_title("Model calibration error by phase")
            ax.grid(True, alpha=0.3, axis="y")
        else:
            # Fallback: just mean |value|
            fig, ax = plt.subplots(figsize=(6, 4))
            x = np.arange(len(phases))
            ma_tgt = [r["mean_abs_value"] for r in vp]
            ax.bar(x, ma_tgt, color="C0", alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(phases)
            ax.set_ylabel("Mean |value| (target)")
            ax.set_title("Value by game phase — position decisiveness")
            ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        fig.savefig(out_dir / "value_by_phase.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 6. Value calibration
    if "calibration_pred" in report and "calibration_target" in report:
        pred = np.array(report["calibration_pred"])
        tgt = np.array(report["calibration_target"])
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(pred, tgt, "o", alpha=0.2, markersize=2)
        deciles = np.percentile(pred, np.linspace(0, 100, 11))
        deciles[0], deciles[-1] = -1.01, 1.01
        centers, avgs = [], []
        for j in range(10):
            mask = (pred >= deciles[j]) & (pred < deciles[j + 1])
            if mask.sum() > 0:
                centers.append((deciles[j] + deciles[j + 1]) / 2)
                avgs.append(float(np.mean(tgt[mask])))
        ax.plot(centers, avgs, "r-o", linewidth=2, markersize=8, label="Decile avg target")
        ax.plot([-1, 1], [-1, 1], "k--", alpha=0.5, label="Perfect")
        ax.set_xlabel("Predicted value")
        ax.set_ylabel("Target value")
        ax.set_title("Value calibration")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(out_dir / "value_calibration.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 7. Ownership heatmap (mean over samples)
    if "ownership_mean" in report:
        om = np.array(report["ownership_mean"])
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for i, (ax, label) in enumerate(zip(axes, ["Current player", "Opponent"])):
            im = ax.imshow(om[i], cmap="RdYlGn", vmin=0, vmax=1)
            ax.set_title(f"P(cell filled) - {label}")
            plt.colorbar(im, ax=ax)
        fig.suptitle("Ownership prediction (game end)")
        fig.savefig(out_dir / "ownership_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close()


def main():
    ap = argparse.ArgumentParser(description="Patchwork meta-analysis: full-depth AlphaZero analytics")
    ap.add_argument("--config", default="configs/config_best.yaml", help="Config path")
    ap.add_argument("--run-dir", type=str, default="runs/patchwork_production", help="Run directory")
    ap.add_argument("--iters", type=str, default="0-100", help="Iteration range for summaries")
    ap.add_argument("--max-positions", type=int, default=100000, help="Max positions to load from HDF5")
    ap.add_argument("--ownership", action="store_true", help="Run ownership inference (slower)")
    ap.add_argument("--ownership-samples", type=int, default=5000, help="Samples for ownership")
    ap.add_argument("--calibration-samples", type=int, default=20000, help="Samples for value calibration")
    ap.add_argument("--out-dir", type=str, default="logs/meta_analysis", help="Output directory")
    ap.add_argument("--no-plots", action="store_true", help="Skip matplotlib plots")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    if not run_dir.exists():
        raise SystemExit(f"Run dir not found: {run_dir}")

    parts = args.iters.split("-")
    iter_start, iter_end = int(parts[0]), int(parts[1]) if len(parts) > 1 else int(parts[0])

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    config = load_config(cfg_path)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir

    rng = np.random.default_rng(42)

    report = {}

    # 1. First-player advantage from iteration summaries
    tqdm_write("=== First-player advantage ===")
    summaries = load_iteration_summaries(run_dir, iter_start, iter_end)
    report["first_player"] = compute_first_player_advantage(summaries)
    if report["first_player"]:
        avg_wr = np.mean([r["p0_win_rate"] for r in report["first_player"]]) * 100
        tqdm_write(f"  Avg P0 win rate: {avg_wr:.1f}% over {len(report['first_player'])} iterations")

    # 2. Load HDF5 data for position-level analysis
    tqdm_write("\n=== Loading replay data ===")
    entries = load_replay_entries(run_dir)
    if not entries:
        committed = run_dir / "committed"
        paths = []
        for it in range(iter_start, min(iter_end + 1, 500)):
            p = committed / f"iter_{it:03d}" / "selfplay.h5"
            if p.exists():
                paths.append(p)
        tqdm_write(f"  No replay_state; using {len(paths)} committed selfplay files")
    else:
        paths = [e[1] for e in entries if iter_start <= e[0] <= iter_end]
        tqdm_write(f"  Loaded {len(paths)} replay entries")

    if not paths:
        tqdm_write("  No HDF5 data found. Skipping position-level analysis.")
    else:
        states, masks, policies, values = load_h5_data(paths, args.max_positions, rng)
        tqdm_write(f"  Loaded {len(states)} positions")

        # Prefer latest committed checkpoint; fallback to best_model.pt
        run_state_path = run_dir / "run_state.json"
        last_comm = iter_end
        if run_state_path.exists():
            try:
                with open(run_state_path) as f:
                    last_comm = json.load(f).get("last_committed_iteration", iter_end)
            except Exception:
                pass
        best_ckpt = run_dir / "committed" / f"iter_{last_comm:03d}" / f"iteration_{last_comm:03d}.pt"
        if not best_ckpt.exists():
            best_ckpt = REPO_ROOT / "checkpoints" / "best_model.pt"

        # 3. Action type distribution
        tqdm_write("\n=== Action type distribution ===")
        report["action_type"] = compute_action_type_distribution(masks, policies)
        at = report["action_type"]
        tqdm_write(f"  Pass: {at['pass_mean']:.3f}±{at['pass_std']:.3f}  Patch: {at['patch_mean']:.3f}  Buy: {at['buy_mean']:.3f}")

        # 4. Patch placement heatmap
        tqdm_write("\n=== Patch placement heatmap ===")
        report["patch_heatmap"] = compute_patch_placement_heatmap(masks, policies).tolist()

        # 5. Piece ranking and piece value (EV when buyable)
        tqdm_write("\n=== Piece ranking & value ===")
        shape_map = build_slot_shape_to_piece_map()
        report["piece_ranking"], report["piece_value"] = compute_piece_ranking(
            states, masks, policies, values, shape_map
        )
        if report["piece_ranking"]:
            top3 = report["piece_ranking"][:3]
            top3_str = [f"pid{r['piece_id']}({r['mean_policy_mass']:.4f})" for r in top3]
            tqdm_write(f"  Top 3 policy: {top3_str}")
        if report.get("piece_value"):
            top3ev = report["piece_value"][:3]
            top3ev_str = [f"pid{r['piece_id']}(EV={r['mean_value_overall']:.4f})" for r in top3ev]
            tqdm_write(f"  Top 3 EV when buyable: {top3ev_str}")

        # 6. Value by phase (target stats + model inference)
        tqdm_write("\n=== Value by game phase ===")
        report["value_by_phase"] = compute_value_by_phase(states, values)
        if best_ckpt.exists():
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            phase_preds = run_value_by_phase_inference(config, best_ckpt, states, values, device)
            by_phase = {r["phase"]: r for r in phase_preds}
            for r in report["value_by_phase"]:
                if r["phase"] in by_phase:
                    r.update(by_phase[r["phase"]])
        for r in report["value_by_phase"]:
            msg = f"  {r['phase']}: mean|val|={r['mean_abs_value']:.3f} n={r['n']}"
            if "mean_abs_pred" in r:
                msg += f"  model mean|pred|={r['mean_abs_pred']:.3f}  mse={r['mse_vs_target']:.4f}"
            tqdm_write(msg)

        # 7. Policy sharpness by phase
        report["policy_sharpness_by_phase"] = compute_policy_sharpness_by_phase(states, masks, policies)

        # 8. Value calibration (need checkpoint)
        tqdm_write("\n=== Value calibration ===")
        if best_ckpt.exists():
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            pred, tgt = run_value_calibration(config, best_ckpt, states, values, args.calibration_samples, device)
            report["calibration_pred"] = pred.tolist()
            report["calibration_target"] = tgt.tolist()
            tqdm_write(f"  Calibration: n={len(pred)}")
        else:
            tqdm_write("  No checkpoint found; skipping calibration")

        # 9. Ownership (optional)
        if args.ownership and best_ckpt.exists():
            tqdm_write("\n=== Ownership inference ===")
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            own = run_ownership_inference(config, best_ckpt, states, args.ownership_samples, device)
            if own is not None:
                report["ownership_mean"] = np.mean(own, axis=0).tolist()
                tqdm_write(f"  Ownership: mean shape (2,9,9)")
            else:
                tqdm_write("  Ownership head not available")

    n_pos = len(states) if paths else 0
    report["meta"] = {
        "iter_start": iter_start,
        "iter_end": iter_end,
        "n_positions_analyzed": n_pos,
        "n_iterations": len(summaries),
    }

    # Save and plot
    save_reports(out_dir, report)
    tqdm_write(f"\nSaved report to {out_dir / 'meta_analysis_report.json'}")

    if not args.no_plots and HAS_MATPLOTLIB:
        plot_all(report, out_dir)
        tqdm_write(f"Saved plots to {out_dir}")
    elif not HAS_MATPLOTLIB:
        tqdm_write("Install matplotlib for plots: pip install matplotlib")


if __name__ == "__main__":
    main()
