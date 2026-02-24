#!/usr/bin/env python
"""
Evaluate two model checkpoints head-to-head, including CROSS-ARCHITECTURE (16ch vs 56ch).

Use this to compare same-iteration strength between 16-channel and 56-channel (gold_v2) models.
You need actual 16ch checkpoints (from old training) and 56ch checkpoints (from new gold_v2 training).

Modes:
  1. Same architecture: --config X --model_a A --model_b B  (both use config X)
  2. Cross architecture: --config_a X --config_b Y --model_a A --model_b B
     - Model A: config X (e.g. 16ch) + checkpoint A (must be 16ch)
     - Model B: config Y (e.g. 56ch gold_v2) + checkpoint B (must be 56ch)

Examples:
  # 16ch iter24 vs 56ch iter24 (same iteration, different encodings)
  python tools/eval_latest_vs_oldest_v2.py --config_a configs/config_16ch_legacy.yaml --config_b configs/config_best.yaml --model_a path/to/old_run/iter024.pt --model_b path/to/new_run/iter024.pt --games 50

  # Same-arch (56ch vs 56ch) — works like original
  python tools/eval_latest_vs_oldest_v2.py --config configs/config_best.yaml --model_a 0 --model_b 5 --games 20
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from src.network.model import create_network, load_model_checkpoint
from src.network.encoder import ActionEncoder, get_state_encoder_for_channels
from src.mcts.alphazero_mcts_optimized import create_optimized_mcts
from src.game.patchwork_engine import (
    apply_action_unchecked,
    compute_score_fast,
    current_player_fast,
    get_winner_fast,
    new_game,
    terminal_fast,
)
from elo_system import (
    expected_score,
    load_ratings,
    model_elo_id,
    save_ratings,
    update_ratings_from_match,
    update_ratings_standard_elo_batch,
    wr_to_elo_gap,
)
from cross_arch_eval.run_cross_arch import run_eval_cross_arch as _fixed_run_eval_cross_arch

# Match iteration_*.pt (committed snapshots) and best_model_iter*.pt (gate-promoted copies)
_ITER_PATTERNS = [
    re.compile(r"iteration_(\d+)\.pt$"),
    re.compile(r"best_model_iter(\d+)\.pt$"),
]


def iter_num(p: Path) -> int | None:
    """Extract iteration number if it's an iter checkpoint, else None."""
    for pattern in _ITER_PATTERNS:
        m = pattern.search(p.name)
        if m:
            return int(m.group(1))
    return None


def _label_for(p: Path) -> str:
    """Short readable label for a checkpoint."""
    n = iter_num(p)
    if n is not None:
        return f"iter{n:03d}"
    if p.name == "best_model.pt":
        return "best (current)"
    if p.name == "latest_model.pt":
        return "latest"
    return p.stem


def discover_checkpoints(repo_root: Path) -> list[tuple[Path, str]]:
    """Discover all model checkpoints in checkpoints/ and runs/. Returns list of (path, label)."""
    results: list[tuple[Path, str]] = []
    seen_paths: set[Path] = set()

    def add(p: Path, label: str) -> None:
        p = p.resolve()
        if p in seen_paths:
            return
        seen_paths.add(p)
        results.append((p, label))

    ckpt_dir = repo_root / "checkpoints"
    if ckpt_dir.is_dir():
        for name in ("best_model.pt", "latest_model.pt"):
            f = ckpt_dir / name
            if f.is_file():
                add(f, _label_for(f))
        for f in ckpt_dir.glob("best_model_iter*.pt"):
            if f.is_file():
                add(f, _label_for(f))
        for f in ckpt_dir.glob("iteration_*.pt"):
            if f.is_file():
                add(f, _label_for(f))

    runs_dir = repo_root / "runs"
    if runs_dir.is_dir():
        for run_dir in sorted(runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            committed = run_dir / "committed"
            if committed.is_dir():
                for iter_dir in sorted(committed.iterdir()):
                    if iter_dir.is_dir() and iter_dir.name.startswith("iter_"):
                        for f in iter_dir.glob("iteration_*.pt"):
                            if f.is_file():
                                add(f, f"committed/{_label_for(f)}")
        for run_dir in sorted(runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            staging = run_dir / "staging"
            if staging.is_dir():
                for iter_dir in sorted(staging.iterdir()):
                    if iter_dir.is_dir() and iter_dir.name.startswith("iter_"):
                        for f in iter_dir.glob("iteration_*.pt"):
                            if f.is_file():
                                add(f, f"staging/{_label_for(f)}")

    def sort_key(item: tuple[Path, str]) -> tuple[int, str]:
        path, label = item
        n = iter_num(path)
        return (n if n is not None else 9999, label)

    results.sort(key=sort_key)
    return results


def torch_load(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def run_eval_cross_arch(
    cfg_a: dict,
    cfg_b: dict,
    model_a_path: Path,
    model_b_path: Path,
    games: int,
    sims: int,
    cpuct: float,
    device: torch.device,
    log: logging.Logger,
    elo_file: Path | None = None,
    elo_style: str = "glicko2",
) -> dict | None:
    """Run head-to-head between models with DIFFERENT architectures (16ch vs 56ch)."""
    ch_a = int((cfg_a.get("network") or {}).get("input_channels", 56))
    ch_b = int((cfg_b.get("network") or {}).get("input_channels", 56))

    id_a = model_elo_id(model_a_path)
    id_b = model_elo_id(model_b_path)
    ratings = load_ratings(elo_file) if elo_file else {}
    if elo_file:
        ra = ratings.get(id_a, {}).get("rating", 1500)
        rda = ratings.get(id_a, {}).get("rd", 350)
        rb = ratings.get(id_b, {}).get("rating", 1500)
        rdb = ratings.get(id_b, {}).get("rd", 350)
        exp_a = expected_score(ra, rb, rda, rdb)
        log.info(
            f"CROSS-ARCH: A({ch_a}ch) {id_a} {ra:.0f} ±{rda:.0f}  |  B({ch_b}ch) {id_b} {rb:.0f} ±{rdb:.0f}  |  E[A win]={exp_a*100:.0f}%\n"
        )

    enc_a = get_state_encoder_for_channels(ch_a)
    enc_b = get_state_encoder_for_channels(ch_b)
    action_enc = ActionEncoder()

    model_a = create_network(cfg_a)
    ckpt_a = torch_load(str(model_a_path), device)
    load_model_checkpoint(model_a, ckpt_a["model_state_dict"])
    model_a.to(device).eval()

    model_b = create_network(cfg_b)
    ckpt_b = torch_load(str(model_b_path), device)
    load_model_checkpoint(model_b, ckpt_b["model_state_dict"])
    model_b.to(device).eval()

    mcts_a = create_optimized_mcts(model_a, cfg_a, device, enc_a, action_enc)
    mcts_a.config.simulations = sims
    mcts_a.config.cpuct = cpuct

    mcts_b = create_optimized_mcts(model_b, cfg_b, device, enc_b, action_enc)
    mcts_b.config.simulations = sims
    mcts_b.config.cpuct = cpuct

    max_moves = int(cfg_a["selfplay"]["max_game_length"])
    base_seed = int(cfg_a.get("seed", 42))

    a_wins = 0
    b_wins = 0
    margins: list[float] = []
    move_counts: list[int] = []
    a_wins_as_p0 = 0
    a_games_as_p0 = 0

    label_a = f"{model_a_path.name} ({ch_a}ch)"
    label_b = f"{model_b_path.name} ({ch_b}ch)"

    for game_idx in range(games):
        a_is_p0 = game_idx % 2 == 0
        a_player = 0 if a_is_p0 else 1
        b_player = 1 - a_player

        seed = base_seed + game_idx * 1000
        state = new_game(seed=seed)

        move_number = 0
        while move_number < max_moves and not terminal_fast(state):
            to_move = current_player_fast(state)

            if to_move == a_player:
                vc, *_ = mcts_a.search(state, to_move, move_number, add_noise=False)
                action = mcts_a.select_action(vc, temperature=0.0, deterministic=True)
            else:
                vc, *_ = mcts_b.search(state, to_move, move_number, add_noise=False)
                action = mcts_b.select_action(vc, temperature=0.0, deterministic=True)

            state = apply_action_unchecked(state, action)
            move_number += 1

        winner = get_winner_fast(state)
        p0_score = compute_score_fast(state, 0)
        p1_score = compute_score_fast(state, 1)

        a_score = p0_score if a_is_p0 else p1_score
        b_score = p1_score if a_is_p0 else p0_score
        margin = float(a_score - b_score)
        margins.append(margin)
        move_counts.append(move_number)
        if a_is_p0:
            a_games_as_p0 += 1
            if winner == a_player:
                a_wins_as_p0 += 1

        if winner == a_player:
            a_wins += 1
            winner_str = "A"
        else:
            b_wins += 1
            winner_str = "B"

        log.info(
            f"Game {game_idx:03d} | A={'P0' if a_is_p0 else 'P1'} vs B={'P1' if a_is_p0 else 'P0'} "
            f"| winner={winner_str} | score P0={p0_score:.1f} P1={p1_score:.1f} | moves={move_number}"
        )

    total = a_wins + b_wins
    wr = (a_wins / total) if total else 0.0
    avg_margin = sum(margins) / len(margins) if margins else 0.0
    win_margins = [m for m in margins if m > 0]
    loss_margins = [m for m in margins if m < 0]
    avg_win_margin = sum(win_margins) / len(win_margins) if win_margins else 0.0
    avg_loss_margin = sum(loss_margins) / len(loss_margins) if loss_margins else 0.0
    wr_se = (wr * (1 - wr) / total) ** 0.5 if total else 0.0
    margin_std = (sum((m - avg_margin) ** 2 for m in margins) / len(margins)) ** 0.5 if len(margins) > 1 else 0.0
    max_win = max(margins) if margins else 0.0
    max_loss = min(margins) if margins else 0.0
    avg_moves = sum(move_counts) / len(move_counts) if move_counts else 0.0
    min_moves = min(move_counts) if move_counts else 0
    max_moves_actual = max(move_counts) if move_counts else 0
    a_games_as_p1 = total - a_games_as_p0
    wr_p0 = (a_wins_as_p0 / a_games_as_p0 * 100) if a_games_as_p0 else 0.0
    wr_p1 = ((a_wins - a_wins_as_p0) / a_games_as_p1 * 100) if a_games_as_p1 else 0.0

    log.info(
        f"\nSUMMARY (CROSS-ARCH {ch_a}ch vs {ch_b}ch): A wins={a_wins}, B wins={b_wins}, A winrate={wr*100:.1f}% (SE ±{wr_se*100:.1f}%, n={total})"
    )
    log.info(
        f"  Score margin (A - B): avg={avg_margin:+.1f} ±{margin_std:.1f} pts  |  when A wins: +{avg_win_margin:.1f}  |  when A loses: {avg_loss_margin:.1f}"
    )
    log.info(f"  Largest margin: A +{max_win:.0f}  |  A {max_loss:+.0f}")
    log.info(f"  Game length: avg={avg_moves:.0f} moves  (min={min_moves}, max={max_moves_actual})")
    log.info(
        f"  A as P0: {a_wins_as_p0}/{a_games_as_p0} ({wr_p0:.0f}%)  |  A as P1: {a_wins - a_wins_as_p0}/{a_games_as_p1} ({wr_p1:.0f}%)"
    )
    log.info(f"  A: {label_a}")
    log.info(f"  B: {label_b}")
    implied_elo = wr_to_elo_gap(wr)
    log.info(f"  [Calibration] Observed WR {wr*100:.1f}% implies ~{implied_elo:.0f} Elo gap")

    if elo_file and margins:
        outcomes = [1.0 if m > 0 else 0.0 for m in margins]
        if elo_style == "standard":
            ratings, (new_ra, new_rda), (new_rb, new_rdb) = update_ratings_standard_elo_batch(
                ratings, id_a, id_b, outcomes
            )
        else:
            ratings, (new_ra, new_rda), (new_rb, new_rdb) = update_ratings_from_match(
                ratings, id_a, id_b, outcomes, margins
            )
        log.info(
            f"\nELO (post): A {id_a} {new_ra:.0f} ±{new_rda:.0f}  |  B {id_b} {new_rb:.0f} ±{new_rdb:.0f}  |  gap={new_ra - new_rb:.0f}"
        )
        save_ratings(elo_file, ratings)

    del model_a, model_b, mcts_a, mcts_b
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "a_wins": a_wins,
        "b_wins": b_wins,
        "total": total,
        "wr": wr,
        "wr_se": wr_se,
        "implied_elo": implied_elo,
    }


def run_eval(
    cfg: dict,
    model_a_path: Path,
    model_b_path: Path,
    games: int,
    sims: int,
    cpuct: float,
    device: torch.device,
    log: logging.Logger,
    elo_file: Path | None = None,
    elo_style: str = "glicko2",
) -> dict | None:
    """Run head-to-head evaluation (same architecture). Delegates to original logic."""
    from eval_latest_vs_oldest import run_eval as _run_eval

    return _run_eval(cfg, model_a_path, model_b_path, games, sims, cpuct, device, log, elo_file, elo_style)


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate checkpoints head-to-head (including 16ch vs 56ch cross-arch)."
    )
    ap.add_argument("--config", default="configs/config_best.yaml", help="Default config (used for both when same-arch)")
    ap.add_argument(
        "--config_a",
        help="Config for model A (e.g. configs/config_16ch_legacy.yaml for 16ch)",
    )
    ap.add_argument(
        "--config_b",
        help="Config for model B (e.g. configs/config_best.yaml for 56ch gold_v2)",
    )
    ap.add_argument("--games", type=int, default=50, help="Number of games")
    ap.add_argument("--override_sims", type=int, default=0, help="0 = use config eval sims")
    ap.add_argument("--override_cpuct", type=float, default=0.0, help="0 = use config eval cpuct")

    ap.add_argument("--model_a", required=True, help="Model A: index (0,1,2...) or path to .pt")
    ap.add_argument("--model_b", required=True, help="Model B: index or path to .pt")

    ap.add_argument("--elo", action="store_true", default=True, help="Update ELO ratings (default)")
    ap.add_argument("--no-elo", action="store_false", dest="elo", help="Disable ELO tracking")
    ap.add_argument("--elo-file", type=str, default="elo_ratings_v2.json", help="Path to ELO ratings JSON (v2 default)")
    ap.add_argument(
        "--elo-style",
        choices=["glicko2", "standard"],
        default="standard",
        help="glicko2 or standard Elo",
    )

    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("eval_v2")

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path

    def load_cfg(p: Path) -> dict:
        with open(p, "r") as f:
            return yaml.safe_load(f)

    cfg = load_cfg(cfg_path)
    cfg_a_path = Path(args.config_a) if args.config_a else cfg_path
    cfg_b_path = Path(args.config_b) if args.config_b else cfg_path
    if not cfg_a_path.is_absolute():
        cfg_a_path = REPO_ROOT / cfg_a_path
    if not cfg_b_path.is_absolute():
        cfg_b_path = REPO_ROOT / cfg_b_path

    if not cfg_a_path.exists():
        raise SystemExit(f"Config A not found: {cfg_a_path}")
    if not cfg_b_path.exists():
        raise SystemExit(f"Config B not found: {cfg_b_path}")

    cfg_a = load_cfg(cfg_a_path)
    cfg_b = load_cfg(cfg_b_path)

    eval_cfg = cfg_a.get("evaluation", {}).get("eval_mcts", {}) or cfg.get("evaluation", {}).get("eval_mcts", {})
    sims = int(args.override_sims) if args.override_sims > 0 else int(eval_cfg.get("simulations", 192))
    cpuct = float(args.override_cpuct) if args.override_cpuct > 0 else float(eval_cfg.get("cpuct", 1.4))
    device = torch.device(
        "cuda" if (cfg_a.get("hardware", {}).get("device") == "cuda" and torch.cuda.is_available()) else "cpu"
    )
    elo_file = (REPO_ROOT / args.elo_file) if args.elo else None

    choices = discover_checkpoints(REPO_ROOT)

    def resolve(s: str) -> Path:
        if s.isdigit():
            idx = int(s)
            if 0 <= idx < len(choices):
                return choices[idx][0]
            raise SystemExit(f"Index {idx} out of range (0-{len(choices)-1})")
        p = Path(s)
        if not p.is_absolute():
            p = REPO_ROOT / p
        if not p.exists():
            raise SystemExit(f"File not found: {p}")
        return p

    path_a = resolve(args.model_a)
    path_b = resolve(args.model_b)
    if path_a == path_b:
        raise SystemExit("Model A and B must be different.")

    ch_a = int((cfg_a.get("network") or {}).get("input_channels", 56))
    ch_b = int((cfg_b.get("network") or {}).get("input_channels", 56))

    cross_arch = ch_a != ch_b
    if cross_arch:
        log.info(f"CROSS-ARCHITECTURE: Model A ({ch_a}ch) vs Model B ({ch_b}ch)")
        log.info(f"  A: {path_a.name}  config={cfg_a_path.name}")
        log.info(f"  B: {path_b.name}  config={cfg_b_path.name}")
        log.info(f"  Games: {args.games}  Sims: {sims}\n")
        _fixed_run_eval_cross_arch(
            cfg_a, cfg_b, path_a, path_b, args.games, sims, cpuct, device, log, elo_file=elo_file, elo_style=args.elo_style
        )
    else:
        log.info(f"Same architecture ({ch_a}ch): {path_a.name} vs {path_b.name}")
        log.info(f"  Games: {args.games}  Sims: {sims}\n")
        run_eval(
            cfg_a, path_a, path_b, args.games, sims, cpuct, device, log, elo_file=elo_file, elo_style=args.elo_style
        )


if __name__ == "__main__":
    main()
