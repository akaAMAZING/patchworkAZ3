"""
F) Search behavior sanity check (win_first + DSU gating).

Run MCTS from a late-game position with:
  - win_first ON / OFF
  - gate_dsu ON / OFF
  - dynamic_score_utility_weight 0.0 vs 0.3
Show which move MCTS prefers and the evaluated values/scores.

Uses a fixed seed to play to a late-game state, then runs 4 MCTS configs
with few sims and reports best action and root Q.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main():
    import yaml
    import numpy as np
    from src.game.patchwork_engine import (
        new_game,
        apply_action_unchecked,
        terminal_fast,
        current_player_fast,
        legal_actions_fast,
    )
    from src.mcts.alphazero_mcts_optimized import create_optimized_mcts
    from src.network.encoder import StateEncoder, ActionEncoder, encode_state_multimodal, get_slot_piece_id
    from src.network.model import create_network, load_model_checkpoint

    ap = __import__("argparse").ArgumentParser(description="F) MCTS win_first + DSU audit")
    ap.add_argument("--config", default="configs/config_best.yaml")
    ap.add_argument("--checkpoint", type=str, default=None, help="Path to .pt (if None, pure MCTS)")
    ap.add_argument("--moves", type=int, default=28, help="Play this many moves to get late-game state")
    ap.add_argument("--sims", type=int, default=64, help="MCTS simulations per config")
    args = ap.parse_args()

    with open(REPO_ROOT / args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Build late-game state (deterministic)
    st = new_game(seed=123, starting_buttons=5, starting_player=0)
    move_count = 0
    while not terminal_fast(st) and move_count < args.moves:
        legal = legal_actions_fast(st)
        if not legal:
            break
        # Deterministic: first legal action
        st = apply_action_unchecked(st, legal[0])
        move_count += 1

    to_move = current_player_fast(st)
    legal = legal_actions_fast(st)
    if not legal:
        print("No legal actions at late-game state (terminal or no moves). Reduce --moves.")
        return 1
    print(f"F) MCTS audit from late-game state (after {move_count} moves), to_move={to_move}, num_legal={len(legal)}")
    print()

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    network = None
    if args.checkpoint and Path(args.checkpoint).is_file():
        net_config = (config.get("network") or {}).copy()
        network = create_network(net_config)
        load_model_checkpoint(network, args.checkpoint, device=device, strict=False)
        network = network.to(device)
        network.eval()
    else:
        print("No checkpoint provided. MCTS requires a network or eval_client.")
        print("Run with --checkpoint path/to/iteration_001.pt (e.g. from runs/.../committed/iter_001/).")
        return 0
    state_encoder = StateEncoder()
    action_encoder = ActionEncoder()

    configs = [
        ("win_first=ON gate_dsu=ON  dynamic_w=0.3", {"win_first": {"enabled": True, "gate_dsu_enabled": True}, "dynamic_score_utility_weight": 0.3}),
        ("win_first=ON gate_dsu=ON  dynamic_w=0.0", {"win_first": {"enabled": True, "gate_dsu_enabled": True}, "dynamic_score_utility_weight": 0.0}),
        ("win_first=OFF gate_dsu=OFF dynamic_w=0.3", {"win_first": {"enabled": False, "gate_dsu_enabled": False}, "dynamic_score_utility_weight": 0.3}),
        ("win_first=OFF gate_dsu=OFF dynamic_w=0.0", {"win_first": {"enabled": False, "gate_dsu_enabled": False}, "dynamic_score_utility_weight": 0.0}),
    ]

    for label, mcts_overrides in configs:
        cfg = copy.deepcopy(config)
        sp = cfg.setdefault("selfplay", {}).setdefault("mcts", {})
        for k, v in mcts_overrides.items():
            if k == "win_first":
                sp["win_first"] = {**(sp.get("win_first") or {}), **v}
            else:
                sp[k] = v
        sp["simulations"] = args.sims
        sp["parallel_leaves"] = min(8, args.sims // 2)

        mcts = create_optimized_mcts(
            network, cfg, __import__("torch").device(device),
            state_encoder, action_encoder,
        )
        mcts.clear_tree()
        visit_counts, search_time, root_q = mcts.search(
            st, to_move, move_number=move_count, add_noise=False, root_legal_actions=legal
        )
        best_action = max(visit_counts, key=visit_counts.get)
        best_visits = visit_counts[best_action]
        root_score = getattr(mcts, "_search_root_score", None)
        print(f"  {label}")
        print(f"    best_action={best_action}  visits={best_visits}  root_q={root_q:.4f}  root_score_est={root_score}")
    print()
    print("Conclusion: Compare best_action and root_q across configs; dynamic_w=0.3 should favor margin when win is likely.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
