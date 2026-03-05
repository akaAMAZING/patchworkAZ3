"""
Quantify how often win_first's DSU gate is near zero during actual self-play.

Procedure:
1) Sample 5,000 root states encountered during self-play (or replay + 1 root eval).
2) For each state compute root_q (value used by win_first) and gate_dsu_value
   using the same gate function and thresholds as MCTS.
3) Print distribution: mean/median root_q, fraction gate_dsu < 0.05, > 0.50, |root_q| > 0.65.
4) If gate_dsu < 0.05 for most states, DSU is effectively disabled in training.

Config switch comparison:
- Run once with win_first.gate_dsu_enabled=true, once with false.
- Compare: chosen move differs %, and average final empty squares over 200 games.
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


def compute_gate_dsu(
    root_q: float,
    gate_dsu_win_start: float = 0.65,
    gate_dsu_win_full: float = 0.90,
    gate_dsu_power: float = 2.0,
    gate_dsu_loss_start: float = -0.90,
    gate_dsu_loss_full: float = -0.98,
) -> float:
    """Same formula as MCTS _compute_dsu_gate(). Returns scalar in [0, 1]."""
    v = root_q
    # Win gate: ramp from win_start to win_full
    denom_win = max(1e-9, gate_dsu_win_full - gate_dsu_win_start)
    t_win = (v - gate_dsu_win_start) / denom_win
    t_win = max(0.0, min(1.0, t_win))
    g_win = t_win ** gate_dsu_power
    # Loss gate: ramp when v very negative
    denom_loss = max(1e-9, gate_dsu_loss_start - gate_dsu_loss_full)
    t_loss = (gate_dsu_loss_start - v) / denom_loss
    t_loss = max(0.0, min(1.0, t_loss))
    g_loss = t_loss ** gate_dsu_power
    return max(g_win, g_loss)


def total_empty_squares(state: np.ndarray) -> int:
    """Total empty squares (both players) from occupancy words."""
    from src.game.patchwork_engine import (
        empty_count_from_occ,
        P0_OCC0,
        P0_OCC1,
        P0_OCC2,
        P1_OCC0,
        P1_OCC1,
        P1_OCC2,
    )
    e0 = empty_count_from_occ(
        int(state[P0_OCC0]), int(state[P0_OCC1]), int(state[P0_OCC2])
    )
    e1 = empty_count_from_occ(
        int(state[P1_OCC0]), int(state[P1_OCC1]), int(state[P1_OCC2])
    )
    return e0 + e1


def main() -> int:
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

    ap = argparse.ArgumentParser(description="Win-first DSU gate activity and gate-on vs gate-off comparison")
    ap.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    ap.add_argument("--config", default="configs/config_best.yaml")
    ap.add_argument("--n-roots", type=int, default=5000, help="Root states to sample for gate distribution")
    ap.add_argument("--n-games-compare", type=int, default=200, help="Games for gate-on vs gate-off comparison")
    ap.add_argument("--sims-collect", type=int, default=32, help="MCTS sims when collecting roots")
    ap.add_argument("--sims-compare", type=int, default=64, help="MCTS sims for move-diff comparison")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    with open(REPO_ROOT / args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = REPO_ROOT / ckpt_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = create_network(config)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    load_model_checkpoint(network, ckpt.get("model_state_dict", ckpt))
    network = network.to(device)
    network.eval()

    # Gate params from config (same as MCTS)
    mcts_cfg = config.get("selfplay", {}).get("mcts", {}) or {}
    wf_cfg = mcts_cfg.get("win_first") or {}
    gate_win_start = float(wf_cfg.get("gate_dsu_win_start", 0.65))
    gate_win_full = float(wf_cfg.get("gate_dsu_win_full", 0.90))
    gate_power = float(wf_cfg.get("gate_dsu_power", 2.0))
    gate_loss_start = float(wf_cfg.get("gate_dsu_loss_start", -0.90))
    gate_loss_full = float(wf_cfg.get("gate_dsu_loss_full", -0.98))

    state_encoder = StateEncoder()
    action_encoder = ActionEncoder()

    def make_mcts_cfg(gate_dsu_enabled: bool, sims: int) -> dict:
        cfg = copy.deepcopy(config)
        sp = cfg.setdefault("selfplay", {}).setdefault("mcts", {})
        sp["simulations"] = sims
        sp["parallel_leaves"] = min(32, max(8, sims // 4))
        wf = sp.setdefault("win_first", {})
        wf["enabled"] = True
        wf["gate_dsu_enabled"] = gate_dsu_enabled
        return cfg

    # -------------------------------------------------------------------------
    # Part 1: Collect 5,000 root states and gate distribution (with gate enabled)
    # -------------------------------------------------------------------------
    cfg_collect = make_mcts_cfg(gate_dsu_enabled=True, sims=args.sims_collect)
    mcts_collect = create_optimized_mcts(network, cfg_collect, device, state_encoder, action_encoder)

    root_qs: List[float] = []
    gate_values: List[float] = []
    collected_states: List[Tuple[np.ndarray, int, int]] = []  # (state, to_move, move_number)

    games = 0
    while len(root_qs) < args.n_roots and games < 500:
        seed = rng.randint(0, 999999)
        st = new_game(seed=seed)
        move_count = 0
        while not terminal_fast(st) and len(root_qs) < args.n_roots:
            legal = legal_actions_fast(st)
            if not legal:
                break
            to_move = current_player_fast(st)
            mcts_collect.clear_tree()
            _vc, _t, root_q = mcts_collect.search(st, to_move, move_number=move_count, add_noise=False, root_legal_actions=legal)
            # Gate uses _search_root_value (set at root expand), not post-search root_q
            v_for_gate = getattr(mcts_collect, "_search_root_value", root_q)
            g = compute_gate_dsu(
                v_for_gate,
                gate_dsu_win_start=gate_win_start,
                gate_dsu_win_full=gate_win_full,
                gate_dsu_power=gate_power,
                gate_dsu_loss_start=gate_loss_start,
                gate_dsu_loss_full=gate_loss_full,
            )
            root_qs.append(v_for_gate)
            gate_values.append(g)
            collected_states.append((st.copy(), to_move, move_count))
            action = mcts_collect.select_action(_vc, temperature=0.0, deterministic=True)
            st = apply_action_unchecked(st, action)
            move_count += 1
        games += 1

    root_qs = root_qs[: args.n_roots]
    gate_values = gate_values[: args.n_roots]
    collected_states = collected_states[: args.n_roots]

    # Distribution
    arr_q = np.array(root_qs, dtype=np.float64)
    arr_g = np.array(gate_values, dtype=np.float64)
    n = len(arr_q)
    frac_gate_lt_005 = np.sum(arr_g < 0.05) / n if n else 0.0
    frac_gate_gt_050 = np.sum(arr_g > 0.50) / n if n else 0.0
    frac_abs_q_gt_065 = np.sum(np.abs(arr_q) > 0.65) / n if n else 0.0

    print("=== Win-first DSU gate activity (self-play roots) ===")
    print(f"  Root states: {n}")
    print(f"  root_q: mean = {float(np.mean(arr_q)):.4f}, median = {float(np.median(arr_q)):.4f}")
    print(f"  gate_dsu: mean = {float(np.mean(arr_g)):.4f}, median = {float(np.median(arr_g)):.4f}")
    print(f"  fraction gate_dsu < 0.05: {frac_gate_lt_005:.2%}")
    print(f"  fraction gate_dsu > 0.50: {frac_gate_gt_050:.2%}")
    print(f"  fraction |root_q| > 0.65:  {frac_abs_q_gt_065:.2%}")
    if frac_gate_lt_005 > 0.5:
        print("  Conclusion: DSU is effectively DISABLED in training (gate < 0.05 for most states).")
    else:
        print("  Conclusion: DSU gate is active for a substantial fraction of states.")
    print()

    # -------------------------------------------------------------------------
    # Part 2: Config switch — chosen move differs %, avg final empty (200 games each)
    # -------------------------------------------------------------------------
    mcts_gate_on = create_optimized_mcts(
        network, make_mcts_cfg(gate_dsu_enabled=True, sims=args.sims_compare),
        device, state_encoder, action_encoder,
    )
    mcts_gate_off = create_optimized_mcts(
        network, make_mcts_cfg(gate_dsu_enabled=False, sims=args.sims_compare),
        device, state_encoder, action_encoder,
    )

    # 2a) Chosen move differs: run 200 games with gate ON; at each step also run MCTS with gate OFF on same state
    move_diff_count = 0
    move_diff_total = 0
    final_empties_gate_on: List[int] = []

    for g in range(args.n_games_compare):
        seed = rng.randint(0, 999999)
        st = new_game(seed=seed)
        move_count = 0
        while not terminal_fast(st):
            legal = legal_actions_fast(st)
            if not legal:
                break
            to_move = current_player_fast(st)
            # Gate ON search and action
            mcts_gate_on.clear_tree()
            vc_on, _, _ = mcts_gate_on.search(st, to_move, move_number=move_count, add_noise=False, root_legal_actions=legal)
            action_on = mcts_gate_on.select_action(vc_on, temperature=0.0, deterministic=True)
            # Gate OFF on same state
            mcts_gate_off.clear_tree()
            vc_off, _, _ = mcts_gate_off.search(st, to_move, move_number=move_count, add_noise=False, root_legal_actions=legal)
            action_off = mcts_gate_off.select_action(vc_off, temperature=0.0, deterministic=True)
            # Compare (tuple equality)
            move_diff_total += 1
            if action_on != action_off:
                move_diff_count += 1
            st = apply_action_unchecked(st, action_on)
            move_count += 1
        final_empties_gate_on.append(total_empty_squares(st))

    # 2b) 200 games with gate OFF for final empty
    final_empties_gate_off: List[int] = []
    for g in range(args.n_games_compare):
        seed = rng.randint(0, 999999)
        st = new_game(seed=seed)
        move_count = 0
        while not terminal_fast(st):
            legal = legal_actions_fast(st)
            if not legal:
                break
            to_move = current_player_fast(st)
            mcts_gate_off.clear_tree()
            vc, _, _ = mcts_gate_off.search(st, to_move, move_number=move_count, add_noise=False, root_legal_actions=legal)
            action = mcts_gate_off.select_action(vc, temperature=0.0, deterministic=True)
            st = apply_action_unchecked(st, action)
            move_count += 1
        final_empties_gate_off.append(total_empty_squares(st))

    move_diff_pct = (100.0 * move_diff_count / move_diff_total) if move_diff_total else 0.0
    avg_empty_on = float(np.mean(final_empties_gate_on)) if final_empties_gate_on else 0.0
    avg_empty_off = float(np.mean(final_empties_gate_off)) if final_empties_gate_off else 0.0

    print("=== Config switch: gate_dsu_enabled true vs false ===")
    print(f"  Games (gate-on trajectory for move diff): {args.n_games_compare}")
    print(f"  Chosen move differs: {move_diff_count} / {move_diff_total} = {move_diff_pct:.1f}%")
    print(f"  Average final empty squares (gate ON):  {avg_empty_on:.2f}")
    print(f"  Average final empty squares (gate OFF): {avg_empty_off:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
