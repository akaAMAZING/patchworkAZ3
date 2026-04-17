#!/usr/bin/env python3
"""
Head-to-head search A/B test: SAME checkpoint on both sides, different search configurations.

Agents (both use the SAME weights, e.g. iter_069):
  A) baseline search:
       - PW OFF
       - DSU/gate baseline: gate_dsu_enabled TRUE, dynamic_score_utility_weight 0.30
  B) PW search:
       - PW ON (k_root=64, k0=32, k_sqrt_coef=8)
       - DSU/gate identical to A (gate_dsu_enabled TRUE, dynamic 0.30)

Protocol:
  - Play A vs B directly.
  - Alternate which agent is player 0 each game (to cancel first-move advantage).
  - Same seeds/RNG schedule for both.

Reports:
  - B win rate vs A
  - Avg terminal empty squares and fragmentation metrics (empty components, isolated 1x1 holes) for BOTH agents
  - Avg root legal count and avg root expanded count for BOTH agents
"""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch

# Repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.game.patchwork_engine import (
    apply_action_unchecked,
    current_player_fast,
    empty_count_from_occ,
    get_winner_fast,
    legal_actions_fast,
    new_game,
    terminal_fast,
    BOARD_SIZE,
    BOARD_CELLS,
    P0_OCC0,
    P0_OCC1,
    P0_OCC2,
    P1_OCC0,
    P1_OCC1,
    P1_OCC2,
)
from src.mcts.alphazero_mcts_optimized import create_optimized_mcts
from src.network.encoder import GoldV2StateEncoder
from src.network.model import create_network, load_model_checkpoint, get_state_dict_for_inference
from src.training.evaluation import build_eval_schedule
from src.training.run_layout import get_run_root, committed_dir


def _empty_squares(state) -> tuple:
    """Return (empty_p0, empty_p1) from state occupancy."""
    e0 = empty_count_from_occ(int(state[P0_OCC0]), int(state[P0_OCC1]), int(state[P0_OCC2]))
    e1 = empty_count_from_occ(int(state[P1_OCC0]), int(state[P1_OCC1]), int(state[P1_OCC2]))
    return e0, e1


def _fragmentation_from_occ(occ0: int, occ1: int, occ2: int) -> tuple:
    """
    Fragmentation metrics for a single board:
      - empty_components: number of connected components of empty cells (4-neighbour)
      - isolated_1x1_holes: empty cells with zero empty neighbours (4-neighbour)
    """
    occ0_i, occ1_i, occ2_i = int(occ0), int(occ1), int(occ2)
    empty = np.zeros(BOARD_CELLS, dtype=np.bool_)
    for idx in range(BOARD_CELLS):
        word = idx >> 5
        bit = idx & 31
        mask = 1 << bit
        if word == 0:
            occupied = (occ0_i & mask) != 0
        elif word == 1:
            occupied = (occ1_i & mask) != 0
        else:
            occupied = (occ2_i & mask) != 0
        empty[idx] = not occupied

    visited = np.zeros(BOARD_CELLS, dtype=np.bool_)
    components = 0
    isolated = 0

    for start in range(BOARD_CELLS):
        if not empty[start] or visited[start]:
            continue
        components += 1
        stack = [start]
        visited[start] = True
        size = 0
        has_neighbour = False
        while stack:
            i = stack.pop()
            size += 1
            r, c = divmod(i, BOARD_SIZE)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    ni = nr * BOARD_SIZE + nc
                    if empty[ni]:
                        has_neighbour = True
                        if not visited[ni]:
                            visited[ni] = True
                            stack.append(ni)
        if size == 1 and not has_neighbour:
            isolated += 1

    return components, isolated


def _fragmentation(state) -> tuple:
    """Return fragmentation metrics for both players: (comp0, iso0, comp1, iso1)."""
    c0, i0 = _fragmentation_from_occ(int(state[P0_OCC0]), int(state[P0_OCC1]), int(state[P0_OCC2]))
    c1, i1 = _fragmentation_from_occ(int(state[P1_OCC0]), int(state[P1_OCC1]), int(state[P1_OCC2]))
    return c0, i0, c1, i1


def make_config_baseline(config: dict) -> dict:
    """Baseline search config: PW OFF; DSU/gate baseline (gate ON, dynamic 0.30)."""
    cfg = copy.deepcopy(config)
    sp = cfg.setdefault("selfplay", {}).setdefault("mcts", {})
    sp["dynamic_score_utility_weight"] = 0.30
    pw = sp.setdefault("progressive_widening", {})
    pw["enabled"] = False
    wf = sp.setdefault("win_first", {})
    wf["gate_dsu_enabled"] = True
    return cfg


# Packing variant presets: A2=PW packing off; B/C/D=training-style; E=aggressive "max pack" (play/stress test)
PACKING_VARIANTS = {
    "B": {  # current
        "alpha": 0.10,
        "weights": {"adj_edges": 0.5, "corner_bonus": 0.0, "iso_hole_penalty": 4.0, "frontier_penalty": 0.0},
    },
    "C": {
        "alpha": 0.25,
        "weights": {"adj_edges": 0.25, "corner_bonus": 0.0, "iso_hole_penalty": 6.0, "frontier_penalty": 0.0},
    },
    "D": {
        "alpha": 0.50,
        "weights": {"adj_edges": 0.10, "corner_bonus": 0.0, "iso_hole_penalty": 8.0, "frontier_penalty": 0.0},
    },
    "E": {  # max pack: avoid cavities, prefer larger pieces; for play/stress test, not training
        "alpha": 0.90,
        "weights": {
            "adj_edges": 0.05,
            "corner_bonus": 0.0,
            "iso_hole_penalty": 10.0,
            "frontier_penalty": 0.0,
            "area_bonus": 0.40,
        },
    },
}


def make_config_pw(config: dict, packing_variant: str | None = None) -> dict:
    """PW search config: PW ON (k_root=64,k0=32,k_sqrt_coef=8); DSU/gate same as baseline.
    If packing_variant is 'B', 'C', 'D', or 'E', enable packing_ordering with that variant's weights."""
    cfg = copy.deepcopy(config)
    sp = cfg.setdefault("selfplay", {}).setdefault("mcts", {})
    sp["dynamic_score_utility_weight"] = 0.30
    pw = sp.setdefault("progressive_widening", {})
    pw["enabled"] = True
    pw["k_root"] = 64
    pw["k0"] = 32
    pw["k_sqrt_coef"] = 8
    pw["always_include_pass"] = True
    pw["always_include_patch"] = True
    if packing_variant and packing_variant in PACKING_VARIANTS:
        preset = PACKING_VARIANTS[packing_variant]
        po = sp.setdefault("packing_ordering", {})
        po["enabled"] = True
        po["alpha"] = preset["alpha"]
        po["use_log_prior"] = True
        po["weights"] = dict(preset["weights"])
        po["local_check_radius"] = 2
    wf = sp.setdefault("win_first", {})
    wf["gate_dsu_enabled"] = True
    return cfg


def main():
    ap = argparse.ArgumentParser(description="Head-to-head search A/B: baseline vs PW (same checkpoint).")
    ap.add_argument("--config", type=str, default="configs/config_continue_from_iter70.yaml", help="Config path (main config)")
    ap.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to .pt checkpoint (overrides --iter if set).",
    )
    ap.add_argument(
        "--iter",
        type=int,
        default=69,
        help="Iteration number for default checkpoint path (default: 69). Use with config's run_id.",
    )
    ap.add_argument("--games", type=int, default=200, help="Total games (A vs B, alternating colours).")
    ap.add_argument("--sims", type=int, default=208, help="MCTS simulations per move.")
    ap.add_argument("--seed", type=int, default=42, help="Base seed for game schedule.")
    ap.add_argument(
        "--progress-interval",
        type=int,
        default=10,
        help="Print progress every N games (0 = no per-game progress).",
    )
    ap.add_argument(
        "--packing",
        action="store_true",
        help="Run A2 vs packing variants (B,C,D,E). --games is per matchup.",
    )
    ap.add_argument(
        "--only-variant",
        type=str,
        default=None,
        choices=["B", "C", "D", "E"],
        help="If set with --packing, run only A2 vs this variant (e.g. --packing --only-variant E).",
    )
    args = ap.parse_args()

    import yaml

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_root = get_run_root(config)
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = committed_dir(run_root, args.iter) / f"iteration_{args.iter:03d}.pt"
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    device = torch.device(config.get("hardware", {}).get("device", "cuda"))
    state_encoder = GoldV2StateEncoder()
    from src.network.encoder import ActionEncoder

    action_encoder = ActionEncoder()

    network = create_network(config)
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    state_dict = get_state_dict_for_inference(ckpt, config, for_selfplay=False)
    load_model_checkpoint(network, state_dict)
    network.to(device)
    network.eval()

    cfg_a = make_config_baseline(config)
    use_packing_variants = args.packing

    if use_packing_variants:
        cfg_a2 = make_config_pw(config, packing_variant=None)  # PW on, packing off
        only_v = getattr(args, "only_variant", None)
        variant_names = [only_v] if only_v else ["B", "C", "D", "E"]
        mcts_a2 = create_optimized_mcts(network, cfg_a2, device, state_encoder, action_encoder)
        variants = []
        for v_name in variant_names:
            cfg_v = make_config_pw(config, packing_variant=v_name)
            mcts_v = create_optimized_mcts(network, cfg_v, device, state_encoder, action_encoder)
            variants.append((v_name, mcts_v, PACKING_VARIANTS[v_name]))
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Games: {args.games} per matchup (A2 vs {', '.join(variant_names)})  Sims: {args.sims}  Seed: {args.seed}")
        print("A2: PW search, packing OFF (k_root=64, k0=32, gate_dsu on, dynamic 0.30)")
        for v_name in variant_names:
            preset = PACKING_VARIANTS[v_name]
            w = preset["weights"]
            area = w.get("area_bonus", 0.0)
            area_s = f", area={area}" if area else ""
            print(f"{v_name}:  PW + packing  (alpha={preset['alpha']}, adj={w.get('adj_edges')}, iso={w.get('iso_hole_penalty')}, corner=0, front=0{area_s})")
    else:
        cfg_b = make_config_pw(config, packing_variant=None)
        mcts_a = create_optimized_mcts(network, cfg_a, device, state_encoder, action_encoder)
        mcts_b = create_optimized_mcts(network, cfg_b, device, state_encoder, action_encoder)
        variants = [("B", mcts_b, None)]
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Games: {args.games}  Sims: {args.sims}  Seed: {args.seed}")
        print("A:  baseline search (PW off, gate_dsu on, dynamic 0.30)")
        print("B:  PW search      (PW on, k_root=64,k0=32, gate_dsu on, dynamic 0.30)")
    print()

    # When --packing: baseline is A2 (PW on, packing off). Otherwise A (PW off).
    mcts_baseline = mcts_a2 if use_packing_variants else mcts_a

    def _avg(xs):
        return float(np.mean(xs)) if xs else 0.0

    max_moves = int(config.get("selfplay", {}).get("max_game_length", 200))
    sims = int(args.sims)
    n_games = args.games

    # Results per variant: { "B": {wins, empty_a, empty_opp, ...}, ... } (a/b = baseline vs variant)
    results = {}
    for v_name, mcts_opp, _preset in variants:
        results[v_name] = {
            "wins": 0,
            "wins_when_opp_p0": 0,   # variant (B/C/D) was P0 and won
            "wins_when_baseline_p0": 0,  # baseline was P0 but variant won (variant was P1)
            "games_baseline_p0": 0,
            "empty_a": [],
            "empty_opp": [],
            "comp_a": [],
            "comp_opp": [],
            "iso_a": [],
            "iso_opp": [],
            "root_legal_a": [],
            "root_legal_opp": [],
            "root_expanded_a": [],
            "root_expanded_opp": [],
        }

    for v_name, mcts_opp, _preset in variants:
        seed_offset = {"B": 0, "C": 1, "D": 2, "E": 3}.get(v_name, 0)
        schedule = build_eval_schedule(n_games, args.seed + seed_offset * 10000, paired_eval=True)
        res = results[v_name]

        for game_idx, (seed, _) in enumerate(schedule):
            baseline_is_p0 = (game_idx % 2 == 0)
            state = new_game(seed=seed)
            move_number = 0
            mcts_baseline.clear_tree()
            mcts_opp.clear_tree()

            while move_number < max_moves:
                if terminal_fast(state):
                    break
                to_move = current_player_fast(state)
                if (to_move == 0 and baseline_is_p0) or (to_move == 1 and not baseline_is_p0):
                    mcts_baseline.config.simulations = sims
                    visit_counts, _, _ = mcts_baseline.search(state, to_move, move_number, add_noise=False)
                    action = mcts_baseline.select_action(visit_counts, temperature=0.0, deterministic=True)
                    res["root_legal_a"].append(mcts_baseline.get_root_legal_count())
                    res["root_expanded_a"].append(mcts_baseline.get_root_expanded_count())
                else:
                    mcts_opp.config.simulations = sims
                    visit_counts, _, _ = mcts_opp.search(state, to_move, move_number, add_noise=False)
                    action = mcts_opp.select_action(visit_counts, temperature=0.0, deterministic=True)
                    res["root_legal_opp"].append(mcts_opp.get_root_legal_count())
                    res["root_expanded_opp"].append(mcts_opp.get_root_expanded_count())
                state = apply_action_unchecked(state, action)
                move_number += 1

            winner = get_winner_fast(state)
            e0, e1 = _empty_squares(state)
            c0, i0, c1, i1 = _fragmentation(state)  # (comp_p0, iso_p0, comp_p1, iso_p1)
            if baseline_is_p0:
                opp_won = winner == 1
                e_a, e_opp = e0, e1
                c_a, i_a, c_opp, i_opp = c0, i0, c1, i1
            else:
                opp_won = winner == 0
                e_a, e_opp = e1, e0
                c_a, i_a, c_opp, i_opp = c1, i1, c0, i0
            if opp_won:
                res["wins"] += 1
                if baseline_is_p0:
                    res["wins_when_baseline_p0"] += 1  # variant was P1 and won
                else:
                    res["wins_when_opp_p0"] += 1  # variant was P0 and won
            if baseline_is_p0:
                res["games_baseline_p0"] += 1
            res["empty_a"].append(e_a)
            res["empty_opp"].append(e_opp)
            res["comp_a"].append(c_a)
            res["comp_opp"].append(c_opp)
            res["iso_a"].append(i_a)
            res["iso_opp"].append(i_opp)

            if args.progress_interval > 0:
                n_done = game_idx + 1
                if n_done % args.progress_interval == 0 or n_done == n_games:
                    wr = res["wins"] / n_done if n_done else 0.0
                    label = "A2" if use_packing_variants else "A"
                    print(f"[{label}_vs_{v_name}] games={n_done}/{n_games}  {v_name}_wins={res['wins']}  {v_name}_WR={wr:.1%}", flush=True)

    # Report
    if use_packing_variants:
        run_variant_names = list(results.keys())
        print(f"=== Results ({', '.join(run_variant_names)} vs A2) ===")
        for v_name in run_variant_names:
            res = results[v_name]
            total = n_games
            wr = res["wins"] / total if total else 0.0
            n_baseline_p0 = res["games_baseline_p0"]
            n_opp_p0 = total - n_baseline_p0
            print(f"  {v_name} win rate vs A2: {wr:.2%}  ({res['wins']}/{total})")
            print(f"    -> {v_name} wins when {v_name} is P0: {res['wins_when_opp_p0']}/{n_opp_p0}  |  when A2 is P0 ({v_name} is P1): {res['wins_when_baseline_p0']}/{n_baseline_p0}")
        print("  (If one column is 100% and the other 0%, outcomes are first-player dominated; agents are ~equal.)")
        print()
        print("  A2 (PW, packing off) — averaged across all matchups:")
        empty_a = [x for r in results.values() for x in r["empty_a"]]
        comp_a = [x for r in results.values() for x in r["comp_a"]]
        iso_a = [x for r in results.values() for x in r["iso_a"]]
        rl_a = [x for r in results.values() for x in r["root_legal_a"]]
        re_a = [x for r in results.values() for x in r["root_expanded_a"]]
        print(f"    Avg final empty squares: {_avg(empty_a):.1f}")
        print(f"    Avg empty components:    {_avg(comp_a):.1f}")
        print(f"    Avg isolated 1x1 holes:  {_avg(iso_a):.1f}")
        print(f"    Avg root legal count:    {_avg(rl_a):.1f}")
        print(f"    Avg root expanded count: {_avg(re_a):.1f}")
        print()
        for v_name in run_variant_names:
            res = results[v_name]
            print(f"  {v_name} (PW + packing)")
            print(f"    Avg final empty squares: {_avg(res['empty_opp']):.1f}")
            print(f"    Avg empty components:    {_avg(res['comp_opp']):.1f}")
            print(f"    Avg isolated 1x1 holes:  {_avg(res['iso_opp']):.1f}")
            print(f"    Avg root legal count:    {_avg(res['root_legal_opp']):.1f}")
            print(f"    Avg root expanded count: {_avg(res['root_expanded_opp']):.1f}")
            print()
    else:
        res = results["B"]
        total = n_games
        wr_b = res["wins"] / total if total else 0.0
        print("=== Results (B vs A) ===")
        print(f"  B win rate vs A: {wr_b:.2%}  ({res['wins']}/{total})")
        print()
        print("  A (baseline search)")
        print(f"    Avg final empty squares: {_avg(res['empty_a']):.1f}")
        print(f"    Avg empty components:    {_avg(res['comp_a']):.1f}")
        print(f"    Avg isolated 1x1 holes:  {_avg(res['iso_a']):.1f}")
        print(f"    Avg root legal count:    {_avg(res['root_legal_a']):.1f}")
        print(f"    Avg root expanded count: {_avg(res['root_expanded_a']):.1f}")
        print()
        print("  B (PW search)")
        print(f"    Avg final empty squares: {_avg(res['empty_opp']):.1f}")
        print(f"    Avg empty components:    {_avg(res['comp_opp']):.1f}")
        print(f"    Avg isolated 1x1 holes:  {_avg(res['iso_opp']):.1f}")
        print(f"    Avg root legal count:    {_avg(res['root_legal_opp']):.1f}")
        print(f"    Avg root expanded count: {_avg(res['root_expanded_opp']):.1f}")


if __name__ == "__main__":
    main()

