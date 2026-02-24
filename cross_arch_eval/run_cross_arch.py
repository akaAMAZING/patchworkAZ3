"""
Fixed cross-architecture head-to-head evaluation (e.g. 16ch vs 56ch).

Root cause of the bug this fixes:
    AlphaZeroMCTS sets `_use_multimodal = use_film or (in_ch == 56)`.
    The legacy 16ch config has use_film=True (plane-based FiLM, no separate
    multimodal tensors), so `_use_multimodal` incorrectly becomes True.
    MCTS then calls encode_state_multimodal() which returns 32 spatial
    channels, crashing the 16ch conv_input that expects 16.

Fix: after creating MCTS for each model, force `_use_multimodal = False`
for any model whose conv_input expects fewer than 32 channels.
The 16ch model derives FiLM conditioning from plane means of the 16ch
spatial tensor itself (film_input_plane_indices path in model.py), so it
works correctly with x_global=None.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TOOLS_DIR = REPO_ROOT / "tools"
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


def _get_conv_in_channels(network: torch.nn.Module) -> int:
    """Return the number of spatial input channels the network expects."""
    conv = getattr(network, "conv_input", None)
    if conv is not None:
        return int(getattr(conv, "in_channels", 56))
    return 56


def _torch_load(path: str, device: torch.device):
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
    elo_file: Optional[Path] = None,
    elo_style: str = "glicko2",
) -> Optional[dict]:
    """
    Run head-to-head between two models that may have different architectures.

    Works for any combination of 16ch, 32ch, 56ch networks.
    Automatically patches MCTS._use_multimodal based on the actual
    conv_input channel count rather than the (unreliable) use_film flag.
    """
    # Resolve channel counts from network configs
    ch_a = int((cfg_a.get("network") or {}).get("input_channels", 56))
    ch_b = int((cfg_b.get("network") or {}).get("input_channels", 56))

    # Optional Elo book-keeping (re-uses elo_system from tools/ if available)
    id_a = id_b = None
    ratings = {}
    try:
        from elo_system import (
            expected_score,
            load_ratings,
            model_elo_id,
            save_ratings,
            update_ratings_from_match,
            update_ratings_standard_elo_batch,
            wr_to_elo_gap,
        )
        id_a = model_elo_id(model_a_path)
        id_b = model_elo_id(model_b_path)
        if elo_file:
            ratings = load_ratings(elo_file)
            ra  = ratings.get(id_a, {}).get("rating", 1500)
            rda = ratings.get(id_a, {}).get("rd", 350)
            rb  = ratings.get(id_b, {}).get("rating", 1500)
            rdb = ratings.get(id_b, {}).get("rd", 350)
            exp_a = expected_score(ra, rb, rda, rdb)
            log.info(
                f"CROSS-ARCH: A({ch_a}ch) {id_a} {ra:.0f} ±{rda:.0f}  |"
                f"  B({ch_b}ch) {id_b} {rb:.0f} ±{rdb:.0f}  |  E[A win]={exp_a*100:.0f}%\n"
            )
        _elo_available = True
    except ImportError:
        _elo_available = False
        elo_file = None

    # ------------------------------------------------------------------ #
    # Build model A
    # ------------------------------------------------------------------ #
    enc_a = get_state_encoder_for_channels(ch_a)
    action_enc = ActionEncoder()

    model_a = create_network(cfg_a)
    ckpt_a = _torch_load(str(model_a_path), device)
    load_model_checkpoint(model_a, ckpt_a["model_state_dict"])
    model_a.to(device).eval()

    mcts_a = create_optimized_mcts(model_a, cfg_a, device, enc_a, action_enc)
    mcts_a.config.simulations = sims
    mcts_a.config.cpuct = cpuct

    # *** THE FIX ***
    # If the network's conv_input expects fewer than 32 channels (i.e. it is a
    # legacy 16ch model), force _use_multimodal=False so MCTS uses the
    # LegacyStateEncoder (16ch) instead of encode_state_multimodal (32ch).
    # The 16ch model's FiLM conditioning is plane-based (film_input_plane_indices)
    # and works correctly with x_global=None.
    conv_in_a = _get_conv_in_channels(model_a)
    if conv_in_a < 32:
        mcts_a._use_multimodal = False
        log.info(f"[cross_arch_eval] Model A: conv_input={conv_in_a}ch → forced _use_multimodal=False (legacy encoder)")

    # ------------------------------------------------------------------ #
    # Build model B
    # ------------------------------------------------------------------ #
    enc_b = get_state_encoder_for_channels(ch_b)

    model_b = create_network(cfg_b)
    ckpt_b = _torch_load(str(model_b_path), device)
    load_model_checkpoint(model_b, ckpt_b["model_state_dict"])
    model_b.to(device).eval()

    mcts_b = create_optimized_mcts(model_b, cfg_b, device, enc_b, action_enc)
    mcts_b.config.simulations = sims
    mcts_b.config.cpuct = cpuct

    conv_in_b = _get_conv_in_channels(model_b)
    if conv_in_b < 32:
        mcts_b._use_multimodal = False
        log.info(f"[cross_arch_eval] Model B: conv_input={conv_in_b}ch → forced _use_multimodal=False (legacy encoder)")

    # ------------------------------------------------------------------ #
    # Game loop
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # Results
    # ------------------------------------------------------------------ #
    total = a_wins + b_wins
    wr = (a_wins / total) if total else 0.0
    wr_se = (wr * (1 - wr) / total) ** 0.5 if total else 0.0
    avg_margin = sum(margins) / len(margins) if margins else 0.0
    margin_std = (
        (sum((m - avg_margin) ** 2 for m in margins) / len(margins)) ** 0.5
        if len(margins) > 1
        else 0.0
    )
    win_margins  = [m for m in margins if m > 0]
    loss_margins = [m for m in margins if m < 0]
    avg_win_margin  = sum(win_margins)  / len(win_margins)  if win_margins  else 0.0
    avg_loss_margin = sum(loss_margins) / len(loss_margins) if loss_margins else 0.0
    max_win  = max(margins) if margins else 0.0
    max_loss = min(margins) if margins else 0.0
    avg_moves     = sum(move_counts) / len(move_counts) if move_counts else 0.0
    min_moves_val = min(move_counts) if move_counts else 0
    max_moves_val = max(move_counts) if move_counts else 0
    a_games_as_p1 = total - a_games_as_p0
    wr_p0 = (a_wins_as_p0 / a_games_as_p0 * 100) if a_games_as_p0 else 0.0
    wr_p1 = ((a_wins - a_wins_as_p0) / a_games_as_p1 * 100) if a_games_as_p1 else 0.0

    # Elo gap helper (may not be available)
    try:
        implied_elo = wr_to_elo_gap(wr)
    except Exception:
        implied_elo = 400 * (np.log10(wr / (1 - wr)) if 0 < wr < 1 else float("nan"))

    log.info(
        f"\nSUMMARY (CROSS-ARCH {ch_a}ch vs {ch_b}ch): A wins={a_wins}, B wins={b_wins},"
        f" A winrate={wr*100:.1f}% (SE ±{wr_se*100:.1f}%, n={total})"
    )
    log.info(
        f"  Score margin (A - B): avg={avg_margin:+.1f} ±{margin_std:.1f} pts"
        f"  |  when A wins: +{avg_win_margin:.1f}  |  when A loses: {avg_loss_margin:.1f}"
    )
    log.info(f"  Largest margin: A +{max_win:.0f}  |  A {max_loss:+.0f}")
    log.info(f"  Game length: avg={avg_moves:.0f} moves  (min={min_moves_val}, max={max_moves_val})")
    log.info(
        f"  A as P0: {a_wins_as_p0}/{a_games_as_p0} ({wr_p0:.0f}%)"
        f"  |  A as P1: {a_wins - a_wins_as_p0}/{a_games_as_p1} ({wr_p1:.0f}%)"
    )
    log.info(f"  A: {label_a}")
    log.info(f"  B: {label_b}")
    log.info(f"  [Calibration] Observed WR {wr*100:.1f}% implies ~{implied_elo:.0f} Elo gap")

    if _elo_available and elo_file and margins and id_a and id_b:
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
            f"\nELO (post): A {id_a} {new_ra:.0f} ±{new_rda:.0f}"
            f"  |  B {id_b} {new_rb:.0f} ±{new_rdb:.0f}  |  gap={new_ra - new_rb:.0f}"
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
