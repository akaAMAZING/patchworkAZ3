#!/usr/bin/env python3
"""
Sanity-check PATCH tie-break behavior (root-only) in sure-winning positions.

Reports:
- how often patch_tiebreak would change the chosen PATCH placement
- average improvements in empty_components / isolated_1x1 / empty_squares on changed cases

This does NOT aim to prove strength; it’s a diagnostic for win% saturation cases.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.game.patchwork_engine import (
    apply_action_unchecked,
    current_player_fast,
    empty_count_from_occ,
    legal_actions_fast,
    new_game,
    terminal_fast,
    AT_PATCH,
    P0_OCC0,
    P0_OCC1,
    P0_OCC2,
    P1_OCC0,
    P1_OCC1,
    P1_OCC2,
)
from src.mcts.alphazero_mcts_optimized import create_optimized_mcts
from src.network.encoder import GoldV2StateEncoder, ActionEncoder
from src.network.model import create_network, load_model_checkpoint, get_state_dict_for_inference
from src.training.run_layout import committed_dir, get_run_root
from src.utils.packing_metrics import fragmentation_from_occ_words


def _packing_metrics_for_player(state: np.ndarray, player: int) -> Tuple[int, int, int]:
    if int(player) == 0:
        occ = (int(state[P0_OCC0]), int(state[P0_OCC1]), int(state[P0_OCC2]))
    else:
        occ = (int(state[P1_OCC0]), int(state[P1_OCC1]), int(state[P1_OCC2]))
    empty = int(empty_count_from_occ(*occ))
    comp, iso = fragmentation_from_occ_words(*occ)
    return empty, int(comp), int(iso)


def _infer_checkpoint_path(config: dict, iter_n: int) -> Path:
    run_root = get_run_root(config)
    return committed_dir(run_root, int(iter_n)) / f"iteration_{iter_n:03d}.pt"


def main() -> int:
    ap = argparse.ArgumentParser(description="PATCH tie-break saturation diagnostics.")
    ap.add_argument("--config", type=str, default="configs/config_continue_from_iter70.yaml")
    ap.add_argument("--checkpoint", type=str, default=None, help="Path to .pt checkpoint (overrides --iter)")
    ap.add_argument("--iter", type=int, default=90, help="Committed iter number when --checkpoint is omitted")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--sims", type=int, default=256, help="MCTS simulations per diagnostic search")
    ap.add_argument("--max-positions", type=int, default=120, help="Max positions to evaluate")
    ap.add_argument("--max-steps", type=int, default=140, help="Max steps to roll forward per game")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--win-prob-floor", type=float, default=0.98, help="Only count positions where best_Qv implies p>=floor")
    args = ap.parse_args()

    import yaml

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ckpt = Path(args.checkpoint) if args.checkpoint else _infer_checkpoint_path(config, int(args.iter))
    if not ckpt.is_absolute():
        ckpt = (REPO_ROOT / ckpt).resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    device = torch.device(args.device)
    # create_network expects the full config dict (with top-level "network" key)
    network = create_network(config).to(device)
    ck = torch.load(str(ckpt), map_location=device)
    state_dict = get_state_dict_for_inference(ck, config, for_selfplay=True)
    load_model_checkpoint(network, state_dict)
    network.eval()

    state_encoder = GoldV2StateEncoder()
    action_encoder = ActionEncoder()

    # Force deterministic root selection; we want consistent comparisons.
    cfg_base = dict(config)
    cfg_base.setdefault("selfplay", {}).setdefault("mcts", {})
    cfg_base["selfplay"]["mcts"]["simulations"] = int(args.sims)
    cfg_base["selfplay"]["mcts"]["enable_tree_reuse"] = False

    mcts = create_optimized_mcts(network, cfg_base, device, state_encoder, action_encoder, enable_tree_reuse=False)

    # Ensure tie-break is configured (but we’ll apply it via override to avoid mutating config files).
    mcts.config.patch_tiebreak.enabled = True
    mcts.config.patch_tiebreak.mode = "packing"
    mcts.config.patch_tiebreak.win_prob_floor = float(args.win_prob_floor)

    positions_checked = 0
    positions_triggered = 0
    changed = 0
    sum_d_empty = 0.0
    sum_d_comp = 0.0
    sum_d_iso = 0.0

    rng = np.random.default_rng(int(args.seed))
    game_seed = int(args.seed)

    while positions_checked < int(args.max_positions):
        st = new_game(seed=game_seed)
        game_seed += 1
        steps = 0

        while (not terminal_fast(st)) and steps < int(args.max_steps) and positions_checked < int(args.max_positions):
            to_move = int(current_player_fast(st))
            legal = legal_actions_fast(st)
            if not legal:
                break

            # Only inspect positions where a PATCH move is available.
            has_patch = any(int(a[0]) == int(AT_PATCH) for a in legal)
            if has_patch:
                visit_counts, _, _root_q_utility = mcts.search(st, to_move, add_noise=False, root_legal_actions=legal)
                root = getattr(mcts, "_root", None)
                if root is not None and root._visit_count is not None and root._value_sum is not None:
                    visits = root._visit_count.astype(np.float64)
                    has_visits = visits > 0
                    safe = np.where(has_visits, visits, 1.0)
                    qv = np.where(has_visits, root._value_sum / safe, -np.inf)
                    best_qv = float(np.max(qv))
                    best_p = 0.5 * (best_qv + 1.0)
                else:
                    best_p = 0.0

                positions_checked += 1

                if best_p >= float(args.win_prob_floor):
                    positions_triggered += 1

                    # Compare selection with and without tie-break on the same root stats.
                    a_no = mcts.select_action(visit_counts, temperature=0.0, deterministic=True, patch_tiebreak_override=False)
                    a_tb = mcts.select_action(visit_counts, temperature=0.0, deterministic=True, patch_tiebreak_override=True)

                    if int(a_no[0]) == int(AT_PATCH) and int(a_tb[0]) == int(AT_PATCH) and a_no != a_tb:
                        changed += 1

                        b0 = _packing_metrics_for_player(st, to_move)
                        s_no = apply_action_unchecked(st, a_no)
                        s_tb = apply_action_unchecked(st, a_tb)
                        b1 = _packing_metrics_for_player(s_no, to_move)
                        b2 = _packing_metrics_for_player(s_tb, to_move)

                        # Improvement = (no_tiebreak - tiebreak) on AFTER metrics (lower is better).
                        sum_d_empty += float(b1[0] - b2[0])
                        sum_d_comp += float(b1[1] - b2[1])
                        sum_d_iso += float(b1[2] - b2[2])

            # Roll forward with a cheap policy to find more patch positions.
            # (Random among legal; bias toward BUY by skipping PATCH sometimes.)
            if has_patch and rng.random() < 0.70:
                # Prefer non-patch to keep game moving; if only patch exists, we’ll take patch.
                non_patch = [a for a in legal if int(a[0]) != int(AT_PATCH)]
                action = non_patch[int(rng.integers(0, len(non_patch)))] if non_patch else legal[int(rng.integers(0, len(legal)))]
            else:
                action = legal[int(rng.integers(0, len(legal)))]

            st = apply_action_unchecked(st, action)
            steps += 1

    print("PATCH tie-break diagnostics")
    print(f"  positions_checked={positions_checked}")
    print(f"  positions_triggered(p>={args.win_prob_floor:.2f})={positions_triggered}")
    print(f"  changed_patch_choice={changed}")
    if changed > 0:
        print("  avg_improvement_on_changed_cases (no_tiebreak - tiebreak) [positive = better]:")
        print(f"    empty_squares: {sum_d_empty / changed:.3f}")
        print(f"    empty_components: {sum_d_comp / changed:.3f}")
        print(f"    isolated_1x1: {sum_d_iso / changed:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

