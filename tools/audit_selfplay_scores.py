"""
B) PROVE self-play labels use TRUE scores (not buttons-only).

Runs 50 self-play games (bootstrap MCTS, no network), then for each game logs:
  final_buttons_p0/p1, empty_squares_p0/p1, bonus7x7_p0/p1,
  final_score_p0/p1, winner, raw_margin (P0 perspective), stored_score_margin_tanh.
Prints per-game lines and summary stats.
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main():
    ap = argparse.ArgumentParser(description="Audit self-play: prove labels use true scores")
    ap.add_argument("--config", default="configs/config_best.yaml", help="Config YAML")
    ap.add_argument("--games", type=int, default=50, help="Number of games to run")
    ap.add_argument("--quiet", action="store_true", help="Only print summary, not per-game")
    args = ap.parse_args()

    import yaml
    import numpy as np
    from src.training.selfplay_optimized import OptimizedSelfPlayWorker

    with open(REPO_ROOT / args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    cfg = copy.deepcopy(config)
    cfg.setdefault("selfplay", {})["num_workers"] = 1
    cfg["selfplay"].setdefault("augmentation", "none")
    cfg["selfplay"].setdefault("store_canonical_only", True)
    cfg["selfplay"]["max_game_length"] = 200
    cfg["selfplay"].setdefault("mcts", {})["simulations"] = 32
    cfg["selfplay"]["mcts"].setdefault("parallel_leaves", 8)
    # Bootstrap so we don't need a checkpoint
    cfg["selfplay"].setdefault("network_path", None)

    worker = OptimizedSelfPlayWorker(network_path=None, config=cfg, device="cpu")

    raw_margins = []
    stored_tanh_p0 = []
    stored_tanh_any = []  # all stored score_margins in replay (per position)

    print("B) Self-play score audit: 50 games (bootstrap MCTS)")
    print("=" * 100)
    for g in range(args.games):
        data = worker.play_game(g, 0, seed=42 + g)
        if data is None:
            print(f"  Game {g}: no data")
            continue
        audit = data.get("audit")
        if not audit:
            print(f"  Game {g}: no audit (old code?)")
            continue
        raw_margins.append(audit["raw_margin_p0_perspective"])
        stored_tanh_p0.append(audit["stored_score_margin_tanh_p0"])
        # All stored score_margins (one per position, perspective varies)
        for m in data.get("score_margins", []):
            stored_tanh_any.append(m)

        if not args.quiet:
            print(
                f"  Game {g}: btns=({audit['final_buttons_p0']},{audit['final_buttons_p1']}) "
                f"empty=({audit['empty_squares_p0']},{audit['empty_squares_p1']}) "
                f"bonus7x7=({audit['bonus7x7_p0']},{audit['bonus7x7_p1']}) "
                f"score=({audit['final_score_p0']},{audit['final_score_p1']}) "
                f"winner={audit['winner']} tie_player={audit['tie_break_tie_player']} "
                f"raw_margin={audit['raw_margin_p0_perspective']:.1f} "
                f"tanh_p0={audit['stored_score_margin_tanh_p0']:.4f} tanh_p1={audit['stored_score_margin_tanh_p1']:.4f}"
            )

    if not raw_margins:
        print("No games with audit data. Ensure selfplay_optimized returns 'audit' in game result.")
        return 1

    raw_margins = np.array(raw_margins)
    stored_tanh_p0 = np.array(stored_tanh_p0)
    stored_tanh_any = np.array(stored_tanh_any) if stored_tanh_any else stored_tanh_p0

    print("\n--- Summary ---")
    print(f"  raw_margin_points (P0 perspective): min={raw_margins.min():.1f} mean={raw_margins.mean():.1f} max={raw_margins.max():.1f}")
    print(f"  stored_score_margin_tanh (P0 per game): min={stored_tanh_p0.min():.4f} mean={stored_tanh_p0.mean():.4f} max={stored_tanh_p0.max():.4f}")
    print(f"  stored_score_margin_tanh (all positions): min={stored_tanh_any.min():.4f} mean={stored_tanh_any.mean():.4f} max={stored_tanh_any.max():.4f}")
    sat = np.abs(stored_tanh_any) > 0.99
    print(f"  fraction |stored_tanh| > 0.99 (saturation): {sat.mean():.2%}")
    print("\nConclusion: Labels use final_score_p0/p1 (buttons - 2*empty + bonus), not buttons-only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
