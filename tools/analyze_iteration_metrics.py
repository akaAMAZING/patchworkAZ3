#!/usr/bin/env python
"""
Analyze per-iteration self-play and replay metrics.

Prompt 1: Per-iteration distributions (Iter 0-24)
  - (a) Legal move count at root
  - (b) Policy perplexity (exp(entropy)) at root
  - (c) Perplexity / legal_count
  - (d) MCTS root visit entropy
  Plots Iter 0-24.

Prompt 2: Replay value analysis
  - Mean/variance of value targets (score_tanh) per iteration
  - Calibration: bucket predicted value into deciles, show average target per bucket

Run from repo root:
    python tools/analyze_iteration_metrics.py --config configs/config_best.yaml --run-dir runs/patchwork_production
    python tools/analyze_iteration_metrics.py --config configs/config_best.yaml --run-dir runs/patchwork_production --iters 0-24 --calibration
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import h5py
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable
    tqdm.write = print  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _entropy(probs: np.ndarray, eps: float = 1e-12) -> float:
    """Entropy of a probability distribution."""
    p = np.asarray(probs, dtype=np.float64).flatten()
    p = p[p > eps]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))


def compute_per_position_metrics(masks: np.ndarray, policies: np.ndarray) -> dict:
    """Compute legal_count, policy entropy, perplexity per position."""
    legal_count = (masks > 0).sum(axis=1).astype(np.int32)
    entropies = []
    for i in range(policies.shape[0]):
        m = masks[i]
        p = policies[i] * m
        s = p.sum()
        if s > 0:
            p = p / s
        entropies.append(_entropy(p))
    entropy_arr = np.array(entropies, dtype=np.float64)
    perplexity = np.exp(np.clip(entropy_arr, 0, 50))  # avoid overflow
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(legal_count > 0, perplexity / legal_count.astype(np.float64), np.nan)
    return {
        "legal_count": legal_count,
        "policy_entropy": entropy_arr,
        "perplexity": perplexity,
        "perplexity_per_legal": ratio,
    }


def load_selfplay_h5(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load states, action_masks, policies, values from selfplay HDF5."""
    with h5py.File(path, "r") as f:
        masks = np.asarray(f["action_masks"], dtype=np.float32)
        policies = np.asarray(f["policies"], dtype=np.float32)
        values = np.asarray(f["values"], dtype=np.float32)
    return masks, policies, values


def analyze_per_iteration_distributions(
    run_dir: Path,
    iter_start: int,
    iter_end: int,
    out_dir: Path | None,
    no_display: bool = False,
) -> None:
    """Compute and plot (a)-(d) per iteration."""
    if not HAS_MATPLOTLIB:
        print("Install matplotlib for plots: pip install matplotlib")
    committed = run_dir / "committed"
    if not committed.is_dir():
        raise FileNotFoundError(f"No committed dir at {committed}")

    results = []
    it_range = range(iter_start, iter_end + 1)
    for it in tqdm(it_range, desc="Distributions", unit="iter"):
        h5_path = committed / f"iter_{it:03d}" / "selfplay.h5"
        if not h5_path.exists():
            tqdm.write(f"  [skip] iter {it}: {h5_path} not found")
            continue
        masks, policies, values = load_selfplay_h5(h5_path)
        m = compute_per_position_metrics(masks, policies)
        lc = m["legal_count"]
        pe = m["policy_entropy"]
        pp = m["perplexity"]
        r = m["perplexity_per_legal"]
        results.append({
            "iter": it,
            "legal_count_mean": float(np.nanmean(lc)),
            "legal_count_std": float(np.nanstd(lc)),
            "policy_perplexity_mean": float(np.nanmean(pp)),
            "policy_perplexity_std": float(np.nanstd(pp)),
            "perplexity_per_legal_mean": float(np.nanmean(r[~np.isnan(r)])),
            "visit_entropy_mean": float(np.nanmean(pe)),
            "visit_entropy_std": float(np.nanstd(pe)),
            "n_positions": int(masks.shape[0]),
        })
        tqdm.write(
            f"  iter {it:3d}: legal_mean={results[-1]['legal_count_mean']:.1f} "
            f"perplexity={results[-1]['policy_perplexity_mean']:.2f} "
            f"pp/legal={results[-1]['perplexity_per_legal_mean']:.3f} "
            f"entropy={results[-1]['visit_entropy_mean']:.2f} "
            f"n={results[-1]['n_positions']}"
        )

    if not results:
        print("No data to plot.")
        return

    # Write distributions to CSV
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "distributions.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["iter", "legal_count_mean", "legal_count_std", "policy_perplexity_mean",
                        "policy_perplexity_std", "perplexity_per_legal_mean", "visit_entropy_mean",
                        "visit_entropy_std", "n_positions"])
            for r in results:
                w.writerow([r["iter"], f"{r['legal_count_mean']:.4f}", f"{r['legal_count_std']:.4f}",
                            f"{r['policy_perplexity_mean']:.4f}", f"{r['policy_perplexity_std']:.4f}",
                            f"{r['perplexity_per_legal_mean']:.4f}", f"{r['visit_entropy_mean']:.4f}",
                            f"{r['visit_entropy_std']:.4f}", r["n_positions"]])
        tqdm.write(f"Saved {csv_path}")

    iters = [r["iter"] for r in results]
    if not HAS_MATPLOTLIB:
        print("(Install matplotlib for plots)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    # (a) Legal move count
    ax = axes[0, 0]
    ax.errorbar(
        iters,
        [r["legal_count_mean"] for r in results],
        yerr=[r["legal_count_std"] for r in results],
        fmt="o-",
        capsize=3,
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Legal move count (mean ± std)")
    ax.set_title("(a) Legal move count at root")
    ax.grid(True, alpha=0.3)

    # (b) Policy perplexity
    ax = axes[0, 1]
    ax.errorbar(
        iters,
        [r["policy_perplexity_mean"] for r in results],
        yerr=[r["policy_perplexity_std"] for r in results],
        fmt="s-",
        capsize=3,
        color="C1",
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Policy perplexity exp(H)")
    ax.set_title("(b) Policy perplexity at root")
    ax.grid(True, alpha=0.3)

    # (c) Perplexity / legal_count
    ax = axes[1, 0]
    ax.plot(iters, [r["perplexity_per_legal_mean"] for r in results], "^-", color="C2")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Perplexity / legal_count")
    ax.set_title("(c) Perplexity / legal_count")
    ax.grid(True, alpha=0.3)

    # (d) MCTS root visit entropy
    ax = axes[1, 1]
    ax.errorbar(
        iters,
        [r["visit_entropy_mean"] for r in results],
        yerr=[r["visit_entropy_std"] for r in results],
        fmt="d-",
        capsize=3,
        color="C3",
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MCTS root visit entropy")
    ax.set_title("(d) MCTS root visit entropy")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig_path = out_dir / "iter_distributions.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Saved {fig_path}")
    if HAS_MATPLOTLIB and not no_display:
        plt.show()
    else:
        plt.close()


def analyze_value_targets_and_calibration(
    run_dir: Path,
    config: dict,
    iter_start: int,
    iter_end: int,
    max_samples: int,
    out_dir: Path | None,
    plot_calibration: bool = False,
) -> None:
    """Report mean/var of value targets, and calibration (decile buckets)."""
    import torch
    from src.network.model import create_network, load_model_checkpoint

    committed = run_dir / "committed"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tqdm.write(f"Device: {device}")

    last_plot_data = None
    value_stats_rows: list[dict] = []
    calibration_rows: list[dict] = []
    it_range = range(iter_start, iter_end + 1)
    for it in tqdm(it_range, desc="Value/calibration", unit="iter"):
        h5_path = committed / f"iter_{it:03d}" / "selfplay.h5"
        ckpt_path = committed / f"iter_{it:03d}" / f"iteration_{it:03d}.pt"
        if not h5_path.exists():
            tqdm.write(f"  [skip] iter {it}: selfplay.h5 not found")
            continue
        if not ckpt_path.exists():
            # Try checkpoints dir
            ckpt_path = REPO_ROOT / "checkpoints" / f"iteration_{it:03d}.pt"
        if not ckpt_path.exists():
            tqdm.write(f"  [skip] iter {it}: checkpoint not found")
            continue

        with h5py.File(h5_path, "r") as f:
            states = np.asarray(f["states"], dtype=np.float32)
            values = np.asarray(f["values"], dtype=np.float32)

        # Subsample for calibration
        n = min(len(states), max_samples)
        idx = np.random.default_rng(42).choice(len(states), size=n, replace=False)
        states_sub = states[idx]
        values_sub = values[idx]

        # Value target stats
        mean_val = float(np.mean(values_sub))
        var_val = float(np.var(values_sub))
        value_stats_rows.append({"iter": it, "value_mean": mean_val, "value_var": var_val, "n_samples": n})
        tqdm.write(f"  Iter {it}: value_target mean={mean_val:.4f} var={var_val:.4f} n={n}")

        # Load model and get predicted values
        net = create_network(config)
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        load_model_checkpoint(net, ckpt["model_state_dict"])
        net.to(device).eval()

        preds = []
        batch_size = 256
        batch_starts = range(0, n, batch_size)
        with torch.no_grad():
            for i in tqdm(batch_starts, desc=f"Inference iter {it}", unit="batch", leave=False):
                batch = states_sub[i : i + batch_size]
                x = torch.from_numpy(batch).to(device)
                _, v = net(x)
                preds.append(v.cpu().numpy().flatten())
        preds = np.concatenate(preds)

        # Decile buckets: bucket by PREDICTED value, show average TARGET per bucket
        decile_edges = np.percentile(preds, np.linspace(0, 100, 11))
        decile_edges[0] = -1.01
        decile_edges[-1] = 1.01
        tqdm.write("  Calibration (bucket by predicted value decile, avg target):")
        for j in range(10):
            lo, hi = decile_edges[j], decile_edges[j + 1]
            mask = (preds >= lo) & (preds < hi)
            if mask.sum() > 0:
                avg_tgt = float(np.mean(values_sub[mask]))
                calibration_rows.append({"iter": it, "decile": j + 1, "pred_lo": lo, "pred_hi": hi,
                                         "avg_target": avg_tgt, "n": int(mask.sum())})
                tqdm.write(f"    decile {j+1}: pred [{lo:.3f}, {hi:.3f})  avg_target={avg_tgt:.4f}  n={mask.sum()}")
            else:
                tqdm.write(f"    decile {j+1}: empty")

        if plot_calibration and out_dir:
            last_plot_data = (it, preds, values_sub, decile_edges)

    # Write value_stats and calibration to CSV
    if out_dir and (value_stats_rows or calibration_rows):
        out_dir.mkdir(parents=True, exist_ok=True)
        if value_stats_rows:
            csv_path = out_dir / "value_stats.csv"
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["iter", "value_mean", "value_var", "n_samples"])
                for r in value_stats_rows:
                    w.writerow([r["iter"], f"{r['value_mean']:.4f}", f"{r['value_var']:.4f}", r["n_samples"]])
            tqdm.write(f"Saved {csv_path}")
        if calibration_rows:
            csv_path = out_dir / "calibration.csv"
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["iter", "decile", "pred_lo", "pred_hi", "avg_target", "n"])
                for r in calibration_rows:
                    w.writerow([r["iter"], r["decile"], f"{r['pred_lo']:.4f}", f"{r['pred_hi']:.4f}",
                                f"{r['avg_target']:.4f}", r["n"]])
            tqdm.write(f"Saved {csv_path}")

    # Plot calibration for last iter with data
    if plot_calibration and out_dir and last_plot_data is not None and HAS_MATPLOTLIB:
        it, preds, values_sub, decile_edges = last_plot_data
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(preds, values_sub, "o", alpha=0.2, markersize=2)
        centers = []
        avgs = []
        for j in range(10):
            lo, hi = decile_edges[j], decile_edges[j + 1]
            mask = (preds >= lo) & (preds < hi)
            if mask.sum() > 0:
                centers.append((lo + hi) / 2)
                avgs.append(float(np.mean(values_sub[mask])))
        ax.plot(centers, avgs, "r-o", linewidth=2, markersize=8, label="Decile avg target")
        ax.plot([-1, 1], [-1, 1], "k--", alpha=0.5, label="Perfect calibration")
        ax.set_xlabel("Predicted value")
        ax.set_ylabel("Target (score_tanh)")
        ax.set_title(f"Iter {it} value calibration")
        ax.legend()
        ax.grid(True, alpha=0.3)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"calibration_iter{it}.png", dpi=150, bbox_inches="tight")
        print(f"Saved {out_dir / f'calibration_iter{it}.png'}")
        plt.close()


def main():
    ap = argparse.ArgumentParser(description="Analyze per-iteration self-play metrics")
    ap.add_argument("--config", default="configs/config_best.yaml", help="Config file (for calibration model loading)")
    ap.add_argument("--run-dir", type=str, default="runs/patchwork_production", help="Run directory")
    ap.add_argument("--iters", type=str, default="0-24", help="Iteration range, e.g. 0-24")
    ap.add_argument("--distributions", action="store_true", default=True, help="Compute (a)-(d) and plot")
    ap.add_argument("--no-distributions", action="store_false", dest="distributions")
    ap.add_argument("--value-stats", action="store_true", default=False, help="Report value target mean/var per iter")
    ap.add_argument("--calibration", action="store_true", default=False, help="Run value calibration (decile buckets)")
    ap.add_argument("--max-samples", type=int, default=20000, help="Max samples for calibration")
    ap.add_argument("--out-dir", type=str, default="logs/analysis", help="Output directory for plots")
    ap.add_argument("--no-display", action="store_true", help="Skip plt.show() (for batch/headless runs)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    if not run_dir.exists():
        raise SystemExit(f"Run dir not found: {run_dir}")

    parts = args.iters.split("-")
    iter_start = int(parts[0])
    iter_end = int(parts[1]) if len(parts) > 1 else iter_start

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir

    import yaml
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    if args.distributions:
        print("=== Per-iteration distributions (Iter 0-24) ===")
        analyze_per_iteration_distributions(run_dir, iter_start, iter_end, out_dir, no_display=args.no_display)

    if args.value_stats or args.calibration:
        print("\n=== Replay value targets (mean/var) and calibration ===")
        analyze_value_targets_and_calibration(
            run_dir,
            config,
            iter_start,
            iter_end,
            args.max_samples,
            out_dir if args.calibration else None,
            plot_calibration=args.calibration,
        )


if __name__ == "__main__":
    main()
