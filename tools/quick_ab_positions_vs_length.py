#!/usr/bin/env python
"""
Quick A/B validator: prove position-count drop is from shorter games (not dropped states)
and test the DSU-ramp hypothesis.

Runs two short self-play batches with identical settings/seed:
  A) dynamic_score_utility_weight = 0.0
  B) dynamic_score_utility_weight = 0.3
(WIN_FIRST unchanged.) Optionally a 3rd run with WIN_FIRST off (DSU fixed).

Asserts: positions_per_game ≈ avg_game_length (invariant: 1 stored position per move).
Prints A/B comparison of avg_game_length and num_positions.

Usage:
  python tools/quick_ab_positions_vs_length.py --games 100 --seed 123 --ckpt checkpoints/latest_model.pt
  python tools/quick_ab_positions_vs_length.py --games 50 --config configs/config_best.yaml
  python tools/quick_ab_positions_vs_length.py --no-win-first  # include 3rd run: WIN_FIRST off
"""

from __future__ import annotations

import argparse
import copy
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main():
    ap = argparse.ArgumentParser(description="A/B positions vs game length (DSU and optional WIN_FIRST)")
    ap.add_argument("--config", type=str, default="configs/config_best.yaml", help="Base config YAML")
    ap.add_argument("--ckpt", type=str, default=None, help="Checkpoint path (default: checkpoints/latest_model.pt)")
    ap.add_argument("--games", type=int, default=100, help="Games per variant (e.g. 100 or 50)")
    ap.add_argument("--seed", type=int, default=123, help="RNG seed (same for all variants)")
    ap.add_argument("--no-win-first", action="store_true", help="Add 3rd run: WIN_FIRST off, DSU=0.3")
    ap.add_argument("--tolerance", type=float, default=0.05, help="Max |positions_per_game - avg_game_length| (default 0.05)")
    args = ap.parse_args()

    import yaml
    config_path = REPO_ROOT / args.config
    if not config_path.is_file():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ckpt = args.ckpt or str(REPO_ROOT / "checkpoints" / "latest_model.pt")
    if not Path(ckpt).is_file():
        print(f"Checkpoint not found: {ckpt}", file=sys.stderr)
        sys.exit(1)

    # Override: use temp dir so we never touch run_state or real runs
    config = copy.deepcopy(config)
    config["seed"] = args.seed
    config.setdefault("paths", {})
    # SelfPlayGenerator uses paths.selfplay_dir only when output_dir is None; we pass output_dir
    # so no need to set selfplay_dir. Just ensure paths exist for any other reads.
    config.setdefault("selfplay", {}).setdefault("bootstrap", {})["games"] = args.games
    # _get_num_games(0) uses bootstrap["games"], so we get exactly args.games per run

    from src.training.selfplay_optimized_integration import SelfPlayGenerator

    results = {}
    tolerance = args.tolerance

    with tempfile.TemporaryDirectory(prefix="quick_ab_pos_") as td:
        root = Path(td)

        # ---- A: DSU = 0.0 ----
        cfg_a = copy.deepcopy(config)
        cfg_a["selfplay"].setdefault("mcts", {})["dynamic_score_utility_weight"] = 0.0
        out_a = root / "A_dsu0"
        out_a.mkdir(parents=True, exist_ok=True)
        print("Running variant A: dynamic_score_utility_weight=0.0 ...", flush=True)
        gen_a = SelfPlayGenerator(cfg_a)
        _, stats_a = gen_a.generate(iteration=0, network_path=ckpt, output_dir=out_a)
        _check_invariant(stats_a, "A", tolerance)
        results["A"] = stats_a

        # ---- B: DSU = 0.3 ----
        cfg_b = copy.deepcopy(config)
        cfg_b["selfplay"].setdefault("mcts", {})["dynamic_score_utility_weight"] = 0.3
        out_b = root / "B_dsu0.3"
        out_b.mkdir(parents=True, exist_ok=True)
        print("Running variant B: dynamic_score_utility_weight=0.3 ...", flush=True)
        gen_b = SelfPlayGenerator(cfg_b)
        _, stats_b = gen_b.generate(iteration=0, network_path=ckpt, output_dir=out_b)
        _check_invariant(stats_b, "B", tolerance)
        results["B"] = stats_b

        # ---- C (optional): WIN_FIRST off, DSU = 0.3 ----
        if args.no_win_first:
            cfg_c = copy.deepcopy(config)
            cfg_c["selfplay"].setdefault("mcts", {})["dynamic_score_utility_weight"] = 0.3
            cfg_c["selfplay"]["mcts"].setdefault("win_first", {})["enabled"] = False
            out_c = root / "C_winfirst_off"
            out_c.mkdir(parents=True, exist_ok=True)
            print("Running variant C: WIN_FIRST=off, dynamic_score_utility_weight=0.3 ...", flush=True)
            gen_c = SelfPlayGenerator(cfg_c)
            _, stats_c = gen_c.generate(iteration=0, network_path=ckpt, output_dir=out_c)
            _check_invariant(stats_c, "C", tolerance)
            results["C"] = stats_c

    # ---- Report ----
    print()
    print("=" * 60)
    print("A/B COMPARISON (positions vs game length)")
    print("=" * 60)

    ng_a = results["A"].get("num_games", 0)
    pos_a = results["A"].get("num_positions", 0)
    len_a = results["A"].get("avg_game_length", 0.0)
    ppg_a = pos_a / ng_a if ng_a else 0.0

    ng_b = results["B"].get("num_games", 0)
    pos_b = results["B"].get("num_positions", 0)
    len_b = results["B"].get("avg_game_length", 0.0)
    ppg_b = pos_b / ng_b if ng_b else 0.0

    print(f"  Variant A (DSU=0.0):   num_games={ng_a}  num_positions={pos_a}  avg_game_length={len_a:.4f}  positions_per_game={ppg_a:.4f}")
    print(f"  Variant B (DSU=0.3):   num_games={ng_b}  num_positions={pos_b}  avg_game_length={len_b:.4f}  positions_per_game={ppg_b:.4f}")
    print()
    print(f"  Delta (B - A):  avg_game_length = {len_b - len_a:+.4f}   num_positions = {pos_b - pos_a:+.0f}")
    print()
    if args.no_win_first:
        ng_c = results["C"].get("num_games", 0)
        pos_c = results["C"].get("num_positions", 0)
        len_c = results["C"].get("avg_game_length", 0.0)
        ppg_c = pos_c / ng_c if ng_c else 0.0
        print(f"  Variant C (WIN_FIRST=off, DSU=0.3): num_games={ng_c}  num_positions={pos_c}  avg_game_length={len_c:.4f}  positions_per_game={ppg_c:.4f}")
        print(f"  Delta (C - B): avg_game_length = {len_c - len_b:+.4f}   num_positions = {pos_c - pos_b:+.0f}")
        print()
    print("Invariant (1 position per move): |positions_per_game - avg_game_length| < %.2f  [PASSED for all]" % tolerance)
    print("=" * 60)
    print()
    print("Conclusion: Position count difference is explained by game length (no dropped states).")
    if len_b > len_a:
        print("DSU=0.3 (B) produced longer games than DSU=0 (A), consistent with DSU-ramp hypothesis.")
    else:
        print("DSU effect on game length in this sample: B - A = %.4f (run more games for significance)." % (len_b - len_a))


def _check_invariant(stats: dict, label: str, tolerance: float) -> None:
    ng = stats.get("num_games", 0)
    pos = stats.get("num_positions", 0)
    avg_len = stats.get("avg_game_length", 0.0)
    if ng == 0:
        raise AssertionError(f"{label}: no games")
    positions_per_game = pos / ng
    diff = abs(positions_per_game - avg_len)
    assert diff < tolerance, (
        f"{label}: invariant failed |positions_per_game - avg_game_length| = {diff:.4f} >= {tolerance}. "
        f"positions_per_game={positions_per_game:.4f} avg_game_length={avg_len:.4f}"
    )
    print(f"  {label}: num_games={ng} num_positions={pos} avg_game_length={avg_len:.4f} positions_per_game={positions_per_game:.4f} [invariant OK]")


if __name__ == "__main__":
    main()
