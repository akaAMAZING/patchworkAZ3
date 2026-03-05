"""
Measure how often DSU (dynamic score utility) changes the chosen move.

Procedure:
1) Sample N random states from selfplay (mid/late game).
2) For each state, run MCTS twice with identical settings except:
   A) dynamic_score_utility_weight = 0.0
   B) dynamic_score_utility_weight = 0.3
   (win_first and gate_dsu disabled so DSU is not gated.)
3) Use sims=512 (or 256) for meaningful comparison.
4) Record chosen action by visits, and top 5 actions with visits, Q, score_est, combined_utility.
5) Print: % states where chosen action differs, average |score_est difference|, 3 flip examples.

If flip-rate is ~0% with gating off, DSU may not be applied or score head is constant.
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np


def _decode_action(action: Tuple) -> str:
    """Human-readable decoded action."""
    atype = int(action[0])
    if atype == 0:
        return "PASS"
    if atype == 1:
        return f"PATCH(pos={action[1]})"
    if atype == 2:
        return f"BUY(off={action[1]}, pid={action[2]}, orient={action[3]}, top={action[4]}, left={action[5]})"
    return str(action)


def _get_chosen_and_top5(mcts) -> Tuple[Optional[Tuple], List[Dict[str, Any]]]:
    """From MCTS root after search: chosen action (argmax visits) and top 5 with visits, Q, score_est, combined."""
    root = mcts._root
    if root is None or not root.legal_actions or root._visit_count is None:
        return None, []
    n = len(root.legal_actions)
    visits = root._visit_count
    total_value = root._total_value
    value_sum = root._value_sum
    score_sum = root._score_sum
    actions = root.legal_actions

    order = np.argsort(-visits)
    chosen_action = actions[int(order[0])] if n else None

    top5: List[Dict[str, Any]] = []
    for idx in range(min(5, n)):
        i = int(order[idx])
        v = int(visits[i])
        if v == 0:
            q_val = score_val = combined = 0.0
        else:
            q_val = float(value_sum[i] / v)
            score_val = float(score_sum[i] / v)
            combined = float(total_value[i] / v)
        top5.append({
            "action": actions[i],
            "visits": v,
            "Q": q_val,
            "score_est": score_val,
            "combined_utility": combined,
        })
    return chosen_action, top5


def main() -> int:
    import numpy as np
    import yaml
    import torch
    from src.game.patchwork_engine import (
        new_game,
        apply_action_unchecked,
        terminal_fast,
        current_player_fast,
        legal_actions_fast,
    )
    from src.mcts.alphazero_mcts_optimized import create_optimized_mcts
    from src.network.encoder import StateEncoder, ActionEncoder
    from src.network.model import create_network, load_model_checkpoint

    ap = argparse.ArgumentParser(description="DSU effect rate: how often does DSU change the chosen move?")
    ap.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    ap.add_argument("--config", default="configs/config_best.yaml")
    ap.add_argument("--n-states", type=int, default=200, help="Number of states to sample")
    ap.add_argument("--sims", type=int, default=512, help="MCTS simulations per run (256 or 512)")
    ap.add_argument("--move-min", type=int, default=12, help="Min moves before saving state (mid-game)")
    ap.add_argument("--move-max", type=int, default=50, help="Max moves (late-game cap)")
    ap.add_argument("--seed", type=int, default=999, help="RNG seed for state sampling")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    with open(REPO_ROOT / args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = create_network(config)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    load_model_checkpoint(network, ckpt.get("model_state_dict", ckpt))
    network = network.to(device)
    network.eval()
    state_encoder = StateEncoder()
    action_encoder = ActionEncoder()

    # Build two MCTS configs: DSU=0 and DSU=0.3, win_first and gate_dsu OFF
    def make_mcts_cfg(dynamic_w: float):
        cfg = copy.deepcopy(config)
        sp = cfg.setdefault("selfplay", {}).setdefault("mcts", {})
        sp["simulations"] = args.sims
        sp["parallel_leaves"] = min(64, max(8, args.sims // 4))
        sp["dynamic_score_utility_weight"] = dynamic_w
        sp["win_first"] = {"enabled": False, "gate_dsu_enabled": False}
        return cfg

    cfg_a = make_mcts_cfg(0.0)
    cfg_b = make_mcts_cfg(0.3)
    mcts_a = create_optimized_mcts(network, cfg_a, device, state_encoder, action_encoder)
    mcts_b = create_optimized_mcts(network, cfg_b, device, state_encoder, action_encoder)

    # ----- 1) Sample N mid/late game states -----
    states: List[Tuple[np.ndarray, int, int]] = []  # (state, seed, move_count)
    games = 0
    while len(states) < args.n_states and games < 800:
        seed = rng.randint(0, 999999)
        st = new_game(seed=seed)
        move_count = 0
        while not terminal_fast(st) and move_count < args.move_max:
            legal = legal_actions_fast(st)
            if not legal:
                break
            if args.move_min <= move_count:
                states.append((st.copy(), seed, move_count))
                if len(states) >= args.n_states:
                    break
            st = apply_action_unchecked(st, rng.choice(legal))
            move_count += 1
        games += 1

    states = states[: args.n_states]
    print(f"DSU effect rate audit: {len(states)} states, sims={args.sims}, DSU=0 vs DSU=0.3 (win_first OFF)")
    print()

    flips: List[Tuple[int, Tuple, Tuple, List, List]] = []  # state_idx, action_a, action_b, top5_a, top5_b
    score_est_diffs: List[float] = []
    n_differ = 0
    n_compared = 0

    for state_idx, (s, _seed, move_count) in enumerate(states):
        to_move = current_player_fast(s)
        legal = legal_actions_fast(s)
        if not legal:
            continue

        # Run A: DSU=0
        mcts_a.clear_tree()
        vc_a, _time_a, _q_a = mcts_a.search(s, to_move, move_number=move_count, add_noise=False, root_legal_actions=legal)
        chosen_a, top5_a = _get_chosen_and_top5(mcts_a)

        # Run B: DSU=0.3
        mcts_b.clear_tree()
        vc_b, _time_b, _q_b = mcts_b.search(s, to_move, move_number=move_count, add_noise=False, root_legal_actions=legal)
        chosen_b, top5_b = _get_chosen_and_top5(mcts_b)

        if chosen_a is None or chosen_b is None:
            continue
        n_compared += 1
        if chosen_a != chosen_b:
            n_differ += 1
            score_a = next((x["score_est"] for x in top5_a if x["action"] == chosen_a), 0.0)
            score_b = next((x["score_est"] for x in top5_b if x["action"] == chosen_b), 0.0)
            score_est_diffs.append(abs(score_b - score_a))
            if len(flips) < 3:
                flips.append((state_idx, chosen_a, chosen_b, top5_a, top5_b))

    pct = 100.0 * (n_differ / n_compared) if n_compared else 0.0
    avg_diff = (sum(score_est_diffs) / len(score_est_diffs)) if score_est_diffs else 0.0

    print("Results")
    print("  states compared: {}".format(n_compared))
    print("  % states where chosen action differs (DSU=0 vs DSU=0.3): {:.1f}%".format(pct))
    print("  average |score_est difference| between chosen actions (when differ): {:.4f}".format(avg_diff))
    print()

    if flips:
        print("3 examples where DSU flips the move:")
        for k, (state_idx, chosen_a, chosen_b, top5_a, top5_b) in enumerate(flips, 1):
            print(f"  --- Example {k} (state_index={state_idx}) ---")
            print(f"  DSU=0   chosen: {_decode_action(chosen_a)}")
            print(f"  DSU=0.3 chosen: {_decode_action(chosen_b)}")
            print("  Top 5 (DSU=0):   ", end="")
            for x in top5_a[:5]:
                print(f" {_decode_action(x['action'])[:20]} N={x['visits']} Q={x['Q']:.3f} score={x['score_est']:.3f} comb={x['combined_utility']:.3f}  ", end="")
            print()
            print("  Top 5 (DSU=0.3): ", end="")
            for x in top5_b[:5]:
                print(f" {_decode_action(x['action'])[:20]} N={x['visits']} Q={x['Q']:.3f} score={x['score_est']:.3f} comb={x['combined_utility']:.3f}  ", end="")
            print()
    else:
        print("No flip examples (chosen action never differed in sampled states).")
        if pct < 1.0 and args.sims >= 256:
            print("If flip-rate is ~0% with sims>=256 and gating off, DSU may not be applied or score head is constant.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
