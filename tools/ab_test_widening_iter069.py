#!/usr/bin/env python3
"""
A/B test: baseline MCTS vs progressive-widening MCTS using the SAME checkpoint (e.g. iter_069).

Compares:
  A)  Baseline:  PW OFF; gate_dsu_enabled TRUE; dynamic_score_utility_weight 0.30
  B)  New:       PW ON;  gate_dsu_enabled FALSE; dynamic 0.12  (for reference)
  B') PW-only:   PW ON;  gate_dsu_enabled TRUE; dynamic 0.30    (isolate PW effect)

Opponent: --opponent previous_best | packer | mixed
  previous_best = iter_068 (or pure-MCTS 64); packer = greedy human proxy; mixed = 50%% each.

--sweep-pw: PW parameter sweep only (no A/B/Bp). Runs 4 configs (k_root, k0, k_sqrt_coef):
  A=(48,32,8), B=(64,32,8), C=(80,32,8), D=(64,32,6); DSU baseline; --sweep-games per config (default 150).

Reports per arm/config: win rate, avg final empty (P0, P1), root legal/expanded/K_buy.
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
    compute_score_fast,
    empty_count_from_occ,
    get_winner_fast,
    legal_actions_fast,
    new_game,
    terminal_fast,
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
from src.training.evaluation import build_eval_schedule, PureMCTSEvaluator
from src.training.run_layout import get_run_root, committed_dir
from src.training.packer_opponent import PackerOpponent


def _empty_squares_both_players(state) -> tuple:
    """Return (empty_p0, empty_p1) from state occupancy."""
    e0 = empty_count_from_occ(
        int(state[P0_OCC0]), int(state[P0_OCC1]), int(state[P0_OCC2])
    )
    e1 = empty_count_from_occ(
        int(state[P1_OCC0]), int(state[P1_OCC1]), int(state[P1_OCC2])
    )
    return (e0, e1)


def _fragmentation_from_occ(occ0: int, occ1: int, occ2: int) -> tuple:
    """
    Fragmentation metrics for a single board:
      - empty_components: number of connected components of empty cells (4-neighbour)
      - isolated_1x1_holes: empty cells with zero empty neighbours (4-neighbour)
    """
    occ0_i, occ1_i, occ2_i = int(occ0), int(occ1), int(occ2)
    # Build empty mask
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
        # New component
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
        # An isolated 1x1 hole is a component of size 1 with no empty neighbours.
        if size == 1 and not has_neighbour:
            isolated += 1

    return components, isolated


def _fragmentation_both_players(state) -> tuple:
    """Return fragmentation metrics for both players: (comp0, iso0, comp1, iso1)."""
    c0, i0 = _fragmentation_from_occ(int(state[P0_OCC0]), int(state[P0_OCC1]), int(state[P0_OCC2]))
    c1, i1 = _fragmentation_from_occ(int(state[P1_OCC0]), int(state[P1_OCC1]), int(state[P1_OCC2]))
    return c0, i0, c1, i1


def make_config_baseline(config: dict) -> dict:
    """Config for A: progressive widening OFF; DSU gating on; dynamic 0.30."""
    cfg = copy.deepcopy(config)
    sp = cfg.setdefault("selfplay", {}).setdefault("mcts", {})
    sp["dynamic_score_utility_weight"] = 0.30
    pw = sp.setdefault("progressive_widening", {})
    pw["enabled"] = False
    wf = sp.setdefault("win_first", {})
    wf["gate_dsu_enabled"] = True
    return cfg


def make_config_new(config: dict) -> dict:
    """Config for B: progressive widening ON; DSU gating OFF; dynamic 0.12."""
    cfg = copy.deepcopy(config)
    sp = cfg.setdefault("selfplay", {}).setdefault("mcts", {})
    sp["dynamic_score_utility_weight"] = 0.12
    pw = sp.setdefault("progressive_widening", {})
    pw["enabled"] = True
    pw["k_root"] = 64
    pw["k0"] = 32
    pw["k_sqrt_coef"] = 16
    pw["always_include_pass"] = True
    pw["always_include_patch"] = True
    wf = sp.setdefault("win_first", {})
    wf["gate_dsu_enabled"] = False
    return cfg


def make_config_b_prime(config: dict) -> dict:
    """Config for B': PW ON; DSU/gate EXACTLY baseline (gate_dsu_enabled True, dynamic 0.30). Isolates PW effect."""
    cfg = copy.deepcopy(config)
    sp = cfg.setdefault("selfplay", {}).setdefault("mcts", {})
    sp["dynamic_score_utility_weight"] = 0.30
    pw = sp.setdefault("progressive_widening", {})
    pw["enabled"] = True
    pw["k_root"] = 64
    pw["k0"] = 32
    pw["k_sqrt_coef"] = 16
    pw["always_include_pass"] = True
    pw["always_include_patch"] = True
    wf = sp.setdefault("win_first", {})
    wf["gate_dsu_enabled"] = True
    return cfg


def make_config_pw_sweep(config: dict, k_root: int, k0: int, k_sqrt_coef: int) -> dict:
    """Config for PW sweep: only PW params (k_root, k0, k_sqrt_coef); DSU baseline (gate on, dynamic 0.30)."""
    cfg = copy.deepcopy(config)
    sp = cfg.setdefault("selfplay", {}).setdefault("mcts", {})
    sp["dynamic_score_utility_weight"] = 0.30
    pw = sp.setdefault("progressive_widening", {})
    pw["enabled"] = True
    pw["k_root"] = k_root
    pw["k0"] = k0
    pw["k_sqrt_coef"] = k_sqrt_coef
    pw["always_include_pass"] = True
    pw["always_include_patch"] = True
    wf = sp.setdefault("win_first", {})
    wf["gate_dsu_enabled"] = True
    return cfg


def _run_pw_sweep(
    network,
    config: dict,
    device: torch.device,
    state_encoder,
    action_encoder,
    opponent_mcts,
    pure_mcts_opponent,
    packer_opponent,
    opponent_mode: str,
    sweep_games: int,
    sims: int,
    seed: int,
    progress_interval: int,
    opponent_label: str,
) -> None:
    """Run PW parameter sweep: configs A,B,C,D; DSU baseline; report per config."""
    # (k_root, k0, k_sqrt_coef): A=(48,32,8), B=(64,32,8), C=(80,32,8), D=(64,32,6)
    sweep_specs = [
        ("A", 48, 32, 8),
        ("B", 64, 32, 8),
        ("C", 80, 32, 8),
        ("D", 64, 32, 6),
    ]
    print("PW parameter sweep (DSU baseline: gate on, dynamic 0.30)")
    print(f"Opponent: {opponent_label}  Games per config: {sweep_games}  Sims: {sims}  Seed: {seed}")
    print()

    all_stats = {}
    for name, k_root, k0, k_sqrt in sweep_specs:
        cfg = make_config_pw_sweep(config, k_root, k0, k_sqrt)
        mcts = create_optimized_mcts(network, cfg, device, state_encoder, action_encoder)
        label = f"PW_sweep_{name}(k_root={k_root},k0={k0},k_sqrt={k_sqrt})"
        print(f"Running {label}...")
        all_stats[name] = run_games(
            mcts, opponent_mcts, pure_mcts_opponent,
            sweep_games, seed, sims, device, cfg, label,
            progress_interval=progress_interval,
            packer_opponent=packer_opponent,
            opponent_mode=opponent_mode,
        )
        print()

    def _k_str(s):
        return f"{s:.1f}" if s is not None else "N/A"

    print("=== PW Sweep Results ===")
    for name in ["A", "B", "C", "D"]:
        s = all_stats[name]
        kr, k0, ks = next((x[1], x[2], x[3]) for x in sweep_specs if x[0] == name)
        print(f"  {name} (k_root={kr}, k0={k0}, k_sqrt_coef={ks})")
        print(f"    Win rate vs {opponent_label}: {s['win_rate']:.2%}  ({s['wins']}/{s['total_games']})")
        print(f"    Avg final empty squares: P0={s['avg_empty_p0']:.1f}  P1={s['avg_empty_p1']:.1f}")
        print(f"    Avg root legal: {s['avg_root_legal_count']:.1f}  expanded: {s['avg_root_expanded_count']:.1f}  K_buy: {_k_str(s['avg_root_K_buy'])}")
        print()
    print("Pick the config that maximizes win rate vs packer with fewer empties; then lock and continue training from iter_069.")


def run_games(
    model_mcts,
    opponent_mcts,
    pure_mcts_opponent,
    num_games: int,
    base_seed: int,
    sims: int,
    device: torch.device,
    config: dict,
    label: str,
    progress_interval: int = 20,
    packer_opponent=None,
    opponent_mode: str = "previous_best",
):
    """Play num_games: model vs opponent. Return stats and root widening stats. Print progress every progress_interval games.

    opponent_mode: "previous_best" | "packer" | "mixed"
      previous_best: use opponent_mcts (or pure_mcts_opponent)
      packer: use packer_opponent
      mixed: 50% previous_best, 50% packer (by game index)
    """
    model_mcts.config.simulations = int(sims)
    paired = True
    schedule = build_eval_schedule(num_games, base_seed, paired_eval=paired)

    results = []
    root_legal_counts = []
    root_expanded_counts = []
    root_n_total_list = []
    root_K_buy_list = []
    game_lengths = []
    empty_p0_list = []
    empty_p1_list = []
    empty_comp_p0_list = []
    empty_comp_p1_list = []
    iso_holes_p0_list = []
    iso_holes_p1_list = []

    max_moves = int(config.get("selfplay", {}).get("max_game_length", 200))

    for game_idx, (seed, model_plays_first) in enumerate(schedule):
        state = new_game(seed=seed)
        move_number = 0
        model_mcts.clear_tree()

        while move_number < max_moves:
            if terminal_fast(state):
                break
            to_move = current_player_fast(state)
            is_model_turn = (to_move == 0 and model_plays_first) or (to_move == 1 and not model_plays_first)

            if is_model_turn:
                visit_counts, _, _ = model_mcts.search(state, to_move, move_number, add_noise=False)
                action = model_mcts.select_action(visit_counts, temperature=0.0, deterministic=True)
                root_legal_counts.append(model_mcts.get_root_legal_count())
                root_expanded_counts.append(model_mcts.get_root_expanded_count())
                root_n_total_list.append(model_mcts.get_root_n_total())
                k_buy = model_mcts.get_root_K_buy()
                root_K_buy_list.append(k_buy if k_buy is not None else -1)
            else:
                use_packer = (
                    opponent_mode == "packer"
                    or (opponent_mode == "mixed" and (game_idx % 2) != 0)
                )
                if use_packer and packer_opponent is not None:
                    action = packer_opponent.get_move(state, seed_offset=move_number)
                elif opponent_mcts is not None:
                    visit_counts, _, _ = opponent_mcts.search(state, to_move, move_number, add_noise=False)
                    action = opponent_mcts.select_action(visit_counts, temperature=0.0, deterministic=True)
                else:
                    action = pure_mcts_opponent.get_move(state, seed_offset=move_number)

            state = apply_action_unchecked(state, action)
            move_number += 1

        winner = get_winner_fast(state)
        model_won = (winner == 0 and model_plays_first) or (winner == 1 and not model_plays_first)
        e0, e1 = _empty_squares_both_players(state)
        c0, i0, c1, i1 = _fragmentation_both_players(state)
        results.append(model_won)
        game_lengths.append(move_number)
        empty_p0_list.append(e0)
        empty_p1_list.append(e1)
        empty_comp_p0_list.append(c0)
        empty_comp_p1_list.append(c1)
        iso_holes_p0_list.append(i0)
        iso_holes_p1_list.append(i1)

        # Progress: every progress_interval games or on last game
        n_done = game_idx + 1
        if progress_interval > 0 and (n_done % progress_interval == 0 or n_done == num_games):
            wins_so_far = sum(results)
            wr = wins_so_far / n_done if n_done else 0.0
            avg_len = float(np.mean(game_lengths)) if game_lengths else 0.0
            avg_legal = float(np.mean(root_legal_counts)) if root_legal_counts else 0.0
            avg_exp = float(np.mean(root_expanded_counts)) if root_expanded_counts else 0.0
            avg_n = float(np.mean(root_n_total_list)) if root_n_total_list else 0.0
            k_vals = [x for x in root_K_buy_list if x >= 0]
            avg_k = float(np.mean(k_vals)) if k_vals else None
            k_str = f"{avg_k:.1f}" if avg_k is not None else "N/A"
            print(
                f"  [{label}] games={n_done}/{num_games}  wins={wins_so_far}  WR={wr:.1%}  "
                f"avg_len={avg_len:.1f}  root_legal={avg_legal:.1f}  root_expanded={avg_exp:.1f}  N={avg_n:.1f}  K_buy={k_str}",
                flush=True,
            )

    wins = sum(results)
    k_buy_vals = [x for x in root_K_buy_list if x >= 0]
    return {
        "label": label,
        "win_rate": wins / len(results) if results else 0.0,
        "wins": wins,
        "total_games": len(results),
        "avg_game_length": float(np.mean(game_lengths)) if game_lengths else 0.0,
        "avg_empty_p0": float(np.mean(empty_p0_list)) if empty_p0_list else 0.0,
        "avg_empty_p1": float(np.mean(empty_p1_list)) if empty_p1_list else 0.0,
        "avg_empty_components_p0": float(np.mean(empty_comp_p0_list)) if empty_comp_p0_list else 0.0,
        "avg_empty_components_p1": float(np.mean(empty_comp_p1_list)) if empty_comp_p1_list else 0.0,
        "avg_isolated_1x1_holes_p0": float(np.mean(iso_holes_p0_list)) if iso_holes_p0_list else 0.0,
        "avg_isolated_1x1_holes_p1": float(np.mean(iso_holes_p1_list)) if iso_holes_p1_list else 0.0,
        "avg_root_legal_count": float(np.mean(root_legal_counts)) if root_legal_counts else 0.0,
        "avg_root_expanded_count": float(np.mean(root_expanded_counts)) if root_expanded_counts else 0.0,
        "avg_root_total_visits": float(np.mean(root_n_total_list)) if root_n_total_list else 0.0,
        "avg_root_K_buy": float(np.mean(k_buy_vals)) if k_buy_vals else None,
    }


def main():
    ap = argparse.ArgumentParser(description="A/B test: baseline vs progressive-widening MCTS (same checkpoint)")
    ap.add_argument("--config", type=str, default="configs/config_continue_from_iter70.yaml", help="Config path (main config)")
    ap.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to .pt checkpoint (default: runs/<run_id>/committed/iter_069/iteration_069.pt)",
    )
    ap.add_argument("--games", type=int, default=200, help="Games per arm (default 200)")
    ap.add_argument("--sims", type=int, default=208, help="MCTS simulations (default 208)")
    ap.add_argument("--seed", type=int, default=42, help="Base seed for game schedule")
    ap.add_argument("--progress-interval", type=int, default=20, help="Print progress every N games (0=no progress)")
    ap.add_argument(
        "--arms",
        type=str,
        default="A,B,Bp",
        help="Comma-separated arms to run: A (baseline), B (PW+DSU change), Bp (PW-only). Default: A,B,Bp",
    )
    ap.add_argument(
        "--opponent",
        type=str,
        choices=["previous_best", "packer", "mixed"],
        default="previous_best",
        help="Opponent: previous_best (iter_068), packer (greedy human proxy), or mixed (50%% each). Default: previous_best",
    )
    ap.add_argument(
        "--sweep-pw",
        action="store_true",
        help="PW parameter sweep only: run configs A,B,C,D (k_root,k0,k_sqrt_coef); DSU baseline; report best vs opponent.",
    )
    ap.add_argument(
        "--sweep-games",
        type=int,
        default=150,
        help="Games per config in --sweep-pw mode (default 150).",
    )
    args = ap.parse_args()
    arms_requested = [s.strip() for s in args.arms.split(",") if s.strip()]

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
        checkpoint_path = committed_dir(run_root, 69) / "iteration_069.pt"
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    # Opponent: previous-best (iter_068) or pure-MCTS 64
    opponent_path = committed_dir(run_root, 68) / "iteration_068.pt"
    opponent_mcts = None
    pure_mcts_opponent = None
    if opponent_path.exists():
        opponent_network = create_network(config)
        opp_ckpt = torch.load(str(opponent_path), map_location="cpu", weights_only=False)
        opp_sd = get_state_dict_for_inference(opp_ckpt, config, for_selfplay=False)
        load_model_checkpoint(opponent_network, opp_sd)
        opponent_network.to(config.get("hardware", {}).get("device", "cuda"))
        opponent_network.eval()
        enc_ver = str((config.get("data", {}) or {}).get("encoding_version", ""))
        in_ch = int((config.get("network", {}) or {}).get("input_channels", 56))
        if enc_ver.lower() in ("gold_v2_32ch", "gold_v2_multimodal") or in_ch == 56:
            state_encoder = GoldV2StateEncoder()
        else:
            from src.network.encoder import StateEncoder
            state_encoder = StateEncoder()
        from src.network.encoder import ActionEncoder
        action_encoder = ActionEncoder()
        opp_cfg = copy.deepcopy(config)
        opp_cfg["selfplay"]["mcts"]["simulations"] = 64
        dev = torch.device(config.get("hardware", {}).get("device", "cuda"))
        opponent_mcts = create_optimized_mcts(
            opponent_network, opp_cfg, dev, state_encoder, action_encoder,
        )
        opponent_mcts.config.simulations = 64
        _opponent_src = "previous_best(iter_068)"
    else:
        pure_mcts_opponent = PureMCTSEvaluator(simulations=64, exploration=1.4, seed=config.get("seed", 42))
        _opponent_src = "pure_mcts_64"
    if args.opponent == "packer":
        opponent_label = "packer"
    elif args.opponent == "mixed":
        opponent_label = f"mixed (50% {_opponent_src}, 50% packer)"
    else:
        opponent_label = _opponent_src

    packer_opponent = PackerOpponent() if args.opponent in ("packer", "mixed") else None

    device = torch.device(config.get("hardware", {}).get("device", "cuda"))
    state_encoder = GoldV2StateEncoder()
    from src.network.encoder import ActionEncoder
    action_encoder = ActionEncoder()

    # Load single checkpoint for both A and B (and sweep)
    network = create_network(config)
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    state_dict = get_state_dict_for_inference(ckpt, config, for_selfplay=False)
    load_model_checkpoint(network, state_dict)
    network.to(device)
    network.eval()

    # PW sweep mode: only PW params change; DSU baseline (gate on, dynamic 0.30)
    if args.sweep_pw:
        _run_pw_sweep(
            network, config, device, state_encoder, action_encoder,
            opponent_mcts, pure_mcts_opponent, packer_opponent, args.opponent,
            args.sweep_games, args.sims, args.seed, args.progress_interval,
            opponent_label,
        )
        return

    config_a = make_config_baseline(config)
    config_b = make_config_new(config)
    config_b_prime = make_config_b_prime(config)

    mcts_a = create_optimized_mcts(network, config_a, device, state_encoder, action_encoder)
    mcts_b = create_optimized_mcts(network, config_b, device, state_encoder, action_encoder)
    mcts_b_prime = create_optimized_mcts(network, config_b_prime, device, state_encoder, action_encoder)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Opponent: {opponent_label}")
    print(f"Games per arm: {args.games}  Sims: {args.sims}  Seed: {args.seed}  Arms: {arms_requested}")
    print(f"Progress every: {args.progress_interval} games")
    print()

    all_stats = {}

    if "A" in arms_requested:
        print("Running A (baseline: PW off, DSU gate on, dynamic 0.30)...")
        all_stats["A"] = run_games(
            mcts_a, opponent_mcts, pure_mcts_opponent,
            args.games, args.seed, args.sims, device, config_a, "A_baseline",
            progress_interval=args.progress_interval,
            packer_opponent=packer_opponent,
            opponent_mode=args.opponent,
        )
        print()

    if "B" in arms_requested:
        print("Running B (new: PW on, DSU gate off, dynamic 0.12)...")
        all_stats["B"] = run_games(
            mcts_b, opponent_mcts, pure_mcts_opponent,
            args.games, args.seed, args.sims, device, config_b, "B_new_PW",
            progress_interval=args.progress_interval,
            packer_opponent=packer_opponent,
            opponent_mode=args.opponent,
        )
        print()

    if "Bp" in arms_requested:
        print("Running B' (PW-only: PW on, DSU gate on, dynamic 0.30)...")
        all_stats["Bp"] = run_games(
            mcts_b_prime, opponent_mcts, pure_mcts_opponent,
            args.games, args.seed, args.sims, device, config_b_prime, "Bp_PW_only",
            progress_interval=args.progress_interval,
            packer_opponent=packer_opponent,
            opponent_mode=args.opponent,
        )
        print()

    def _k_buy_str(s):
        return f"{s:.1f}" if s is not None else "N/A"

    print("=== Results ===")
    arm_descriptions = {
        "A": "A (baseline: PW off, gate_dsu on, dynamic 0.30)",
        "B": "B (new: PW on, gate_dsu off, dynamic 0.12)",
        "Bp": "B' (PW-only: PW on, gate_dsu on, dynamic 0.30)",
    }
    for arm in ["A", "B", "Bp"]:
        if arm not in all_stats:
            continue
        s = all_stats[arm]
        print(f"  {arm_descriptions[arm]}")
        print(f"    Win rate vs {opponent_label}: {s['win_rate']:.2%}  ({s['wins']}/{s['total_games']})")
        print(f"    Avg game length: {s['avg_game_length']:.1f}")
        print(f"    Avg final empty squares: P0={s['avg_empty_p0']:.1f}  P1={s['avg_empty_p1']:.1f}")
        print(f"    Avg empty components:    P0={s['avg_empty_components_p0']:.1f}  P1={s['avg_empty_components_p1']:.1f}")
        print(f"    Avg isolated 1x1 holes:  P0={s['avg_isolated_1x1_holes_p0']:.1f}  P1={s['avg_isolated_1x1_holes_p1']:.1f}")
        print(f"    Avg root legal count: {s['avg_root_legal_count']:.1f}  Avg root expanded count: {s['avg_root_expanded_count']:.1f}")
        print(f"    Avg root total visits (N): {s['avg_root_total_visits']:.1f}  Avg root K_buy: {_k_buy_str(s['avg_root_K_buy'])}")
        print()
    print("Decision: If B' (PW-only) >= A winrate and fewer empties, keep PW and revert DSU/gate to baseline. Only if B beats both, keep DSU/gate change.")


if __name__ == "__main__":
    main()
