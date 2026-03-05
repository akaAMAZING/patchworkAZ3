"""
PROVE value/score sign handling is correct in Patchwork's non-alternating turn order.

When to_move stays the same across a move -> backed-up value/score must NOT be negated.
When to_move changes -> must BE negated (zero-sum).

Uses a dummy constant network and instruments MCTS backup to assert sign rules.
No checkpoint required.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from src.mcts.alphazero_mcts_optimized import (
    OptimizedAlphaZeroMCTS,
    MCTSConfig,
    WinFirstConfig,
    MCTSNode,
)


def _decode_action(action: Tuple) -> str:
    atype = int(action[0])
    if atype == 0:
        return "PASS"
    if atype == 1:
        return f"PATCH(pos={action[1]})"
    if atype == 2:
        return f"BUY(off={action[1]}, pid={action[2]}, orient={action[3]}, top={action[4]}, left={action[5]})"
    return str(action)


def find_state_with_both_turn_behaviors(
    max_seeds: int = 500,
    move_min: int = 5,
    move_max: int = 45,
    seed_start: int = 0,
) -> Optional[Tuple[Any, int, int, Tuple, Tuple]]:
    """Find (state, game_seed, move_count, action_same_turn, action_turn_passes)."""
    from src.game.patchwork_engine import (
        new_game,
        apply_action_unchecked,
        terminal_fast,
        current_player_fast,
        legal_actions_fast,
    )
    rng = random.Random(42)
    for _ in range(max_seeds):
        seed = seed_start + rng.randint(0, 999999)
        st = new_game(seed=seed)
        move_count = 0
        while not terminal_fast(st) and move_count < move_max:
            legal = legal_actions_fast(st)
            if not legal:
                break
            if move_count >= move_min:
                to_move = current_player_fast(st)
                action_same = None
                action_flip = None
                for a in legal:
                    st1 = apply_action_unchecked(st.copy(), a)
                    to1 = current_player_fast(st1)
                    if to1 == to_move:
                        action_same = a
                    else:
                        action_flip = a
                if action_same is not None and action_flip is not None:
                    return (st.copy(), seed, move_count, action_same, action_flip)
            st = apply_action_unchecked(st, rng.choice(legal))
            move_count += 1
    return None


class DummyConstantNetwork(torch.nn.Module):
    """Returns constant value and score for any input (for sign audit)."""

    def __init__(self, value_const: float = 0.4, score_points_const: float = 10.0):
        super().__init__()
        self.value_const = value_const
        self.score_points_const = score_points_const
        # So MCTS uses legacy encoder (not multimodal)
        self.conv_input = SimpleNamespace(in_channels=16)

    def forward(
        self,
        state: "torch.Tensor",
        action_mask: "torch.Tensor",
        x_global=None,
        x_track=None,
        shop_ids=None,
        shop_feats=None,
    ):
        B = state.shape[0]
        device = state.device
        dtype = state.dtype
        policy = torch.zeros(B, 2026, device=device, dtype=dtype)
        value = torch.full((B, 1), self.value_const, device=device, dtype=dtype)
        # Bins -100..100 -> index for +10 is 110
        score_logits = torch.zeros(B, 201, device=device, dtype=dtype)
        idx_10 = int(self.score_points_const + 100)
        idx_10 = max(0, min(200, idx_10))
        score_logits[:, idx_10] = 10.0
        return policy, value, score_logits


class InstrumentedMCTS(OptimizedAlphaZeroMCTS):
    """MCTS that prints and asserts on every backup step."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._backup_log: List[Dict[str, Any]] = []
        self._first_fail: Optional[Dict[str, Any]] = None

    def _backup_path(
        self,
        search_path: List[Tuple[MCTSNode, tuple]],
        leaf_v: float,
        leaf_s: float,
        leaf_u: float,
        leaf_to_move: int,
    ) -> None:
        raw_leaf = (leaf_v, leaf_s, leaf_u)
        for parent_node, action_taken in reversed(search_path):
            parent_node.remove_virtual_loss(action_taken, self.config.virtual_loss)
            if parent_node.to_move == leaf_to_move:
                vp, sp, up = leaf_v, leaf_s, leaf_u
                negated = False
            else:
                vp, sp, up = -leaf_v, -leaf_s, -leaf_u
                negated = True
            # Assert
            expect_negated = (parent_node.to_move != leaf_to_move)
            if negated != expect_negated:
                if self._first_fail is None:
                    self._first_fail = {
                        "parent_to_move": int(parent_node.to_move),
                        "leaf_to_move": int(leaf_to_move),
                        "action": action_taken,
                        "raw_leaf": raw_leaf,
                        "vp_sp_up": (vp, sp, up),
                        "negated": negated,
                        "expect_negated": expect_negated,
                    }
            self._backup_log.append({
                "parent_to_move": int(parent_node.to_move),
                "child_to_move": int(leaf_to_move),
                "raw_leaf_v": leaf_v,
                "raw_leaf_s": leaf_s,
                "raw_leaf_u": leaf_u,
                "after_v": vp,
                "after_s": sp,
                "after_u": up,
                "negated": negated,
            })
            parent_node.update(action_taken, vp, sp, up)


def main() -> int:
    import yaml
    from src.game.patchwork_engine import (
        apply_action_unchecked,
        current_player_fast,
        legal_actions_fast,
    )
    from src.network.encoder import StateEncoder, ActionEncoder

    ap = argparse.ArgumentParser(description="MCTS turn-order sign audit (value/score backup)")
    ap.add_argument("--config", default="configs/config_best.yaml")
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--max-seeds", type=int, default=500)
    ap.add_argument("--verbose", action="store_true", help="Print every backup step")
    args = ap.parse_args()

    # 1) Find state with both same-turn and turn-passes actions
    result = find_state_with_both_turn_behaviors(max_seeds=args.max_seeds)
    if result is None:
        print("FAIL: Could not find a state with both same-turn and turn-passes actions.")
        return 1
    state, game_seed, move_count, action_same, action_flip = result
    to_move = current_player_fast(state)
    legal = legal_actions_fast(state)
    print(f"Found state: seed={game_seed}, move_count={move_count}, to_move={to_move}")
    print(f"  action_same_turn:   {_decode_action(action_same)}")
    print(f"  action_turn_passes: {_decode_action(action_flip)}")
    print()

    # 2) Dummy network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_net = DummyConstantNetwork(value_const=0.4, score_points_const=10.0)
    dummy_net.to(device)
    dummy_net.eval()

    # 3) Config: small sims, no win_first, no noise
    with open(REPO_ROOT / args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    cfg = config.get("selfplay", {}).get("mcts", {}) or {}
    mcts_config = MCTSConfig(
        simulations=args.sims,
        parallel_leaves=min(8, args.sims // 2),
        cpuct=float(cfg.get("cpuct", 1.5)),
        temperature=0.0,
        temperature_threshold=0,
        root_dirichlet_alpha=0.0,
        root_noise_weight=0.0,
        virtual_loss=float(cfg.get("virtual_loss", 1.0)),
        fpu_reduction=float(cfg.get("fpu_reduction", 0.25)),
        static_score_utility_weight=0.0,
        dynamic_score_utility_weight=0.0,
        score_utility_scale=30.0,
        score_min=-100,
        score_max=100,
        enable_tree_reuse=False,
        win_first=WinFirstConfig(enabled=False, gate_dsu_enabled=False),
    )
    state_encoder = StateEncoder()
    action_encoder = ActionEncoder()

    # Use instrumented MCTS: create and replace _backup_path by using our subclass
    mcts = InstrumentedMCTS(
        dummy_net,
        mcts_config,
        device,
        state_encoder,
        action_encoder,
        eval_client=None,
        inference_settings={"use_amp": False},
        full_config=config,
    )
    mcts.clear_tree()
    mcts.search(state, to_move, move_number=move_count, add_noise=False, root_legal_actions=legal)

    # 4) Assertions and report
    if args.verbose:
        for i, step in enumerate(mcts._backup_log):
            print(
                f"  backup[{i}] parent.to_move={step['parent_to_move']} child.to_move={step['child_to_move']} "
                f"raw_leaf=(v={step['raw_leaf_v']:.3f}, s={step['raw_leaf_s']:.3f}, u={step['raw_leaf_u']:.3f}) "
                f"after=(v={step['after_v']:.3f}, s={step['after_s']:.3f}, u={step['after_u']:.3f}) "
                f"negated={step['negated']}"
            )

    failed = False
    for step in mcts._backup_log:
        parent = step["parent_to_move"]
        child = step["child_to_move"]
        if parent == child:
            if step["negated"]:
                failed = True
                if mcts._first_fail is None:
                    mcts._first_fail = {
                        "parent_to_move": parent,
                        "leaf_to_move": child,
                        "action": None,
                        "raw_leaf": (step["raw_leaf_v"], step["raw_leaf_s"], step["raw_leaf_u"]),
                        "vp_sp_up": (step["after_v"], step["after_s"], step["after_u"]),
                        "negated": True,
                        "expect_negated": False,
                    }
                break
        else:
            if not step["negated"]:
                failed = True
                if mcts._first_fail is None:
                    mcts._first_fail = {
                        "parent_to_move": parent,
                        "leaf_to_move": child,
                        "action": None,
                        "raw_leaf": (step["raw_leaf_v"], step["raw_leaf_s"], step["raw_leaf_u"]),
                        "vp_sp_up": (step["after_v"], step["after_s"], step["after_u"]),
                        "negated": False,
                        "expect_negated": True,
                    }
                break

    if mcts._first_fail is not None:
        failed = True

    if failed and mcts._first_fail:
        print()
        print("FAIL: Turn-order sign violation")
        print(f"  state seed: {game_seed}")
        print(f"  action: {_decode_action(mcts._first_fail['action']) if mcts._first_fail.get('action') else 'N/A'}")
        print(f"  parent.to_move: {mcts._first_fail['parent_to_move']}")
        print(f"  child.to_move:  {mcts._first_fail['leaf_to_move']}")
        print(f"  raw_leaf (v,s,u): {mcts._first_fail['raw_leaf']}")
        print(f"  after conversion (vp,sp,up): {mcts._first_fail['vp_sp_up']}")
        print(f"  expected negated: {mcts._first_fail['expect_negated']}  actual negated: {mcts._first_fail['negated']}")
        return 1

    print()
    print("PASS: Value/score sign handling is correct.")
    print("  On edges where child.to_move == parent.to_move, sign was NOT flipped.")
    print("  On edges where child.to_move != parent.to_move, sign was flipped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
