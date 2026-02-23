#!/usr/bin/env python
"""
Fast sanity checks for overnight safety.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import List

import numpy as np

# Make `python tools/*.py` runnable from repo root without manual PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.game.patchwork_engine import (
    apply_action_unchecked,
    compute_score_fast,
    current_player_fast,
    get_winner_fast,
    legal_actions_fast,
    new_game,
    terminal_fast,
)
from src.mcts.alphazero_mcts_optimized import engine_action_to_flat_index
from src.network.encoder import ActionEncoder, GoldV2StateEncoder
from src.training.evaluation import PureMCTSEvaluator


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def check_action_roundtrip(steps: int, seed: int) -> None:
    rng = random.Random(seed)
    encoder = ActionEncoder()
    state = new_game(seed=seed)

    for _ in range(steps):
        legal = legal_actions_fast(state)
        _assert(len(legal) > 0, "No legal actions in non-terminal state")

        for action in legal[: min(64, len(legal))]:
            idx_fast = int(engine_action_to_flat_index(action))
            idx_enc = int(encoder.encode_action(action))
            _assert(idx_fast == idx_enc, "MCTS/encoder index mismatch")

            decoded = encoder.decode_action(idx_enc)
            idx_re = int(encoder.encode_action(decoded))
            _assert(idx_re == idx_enc, "encode/decode round-trip mismatch")

        state = apply_action_unchecked(state, rng.choice(legal))
        if terminal_fast(state):
            state = new_game(seed=rng.randint(1, 10_000_000))


def check_legality_masks(steps: int, seed: int) -> None:
    rng = random.Random(seed + 1)
    encoder = ActionEncoder()
    state = new_game(seed=seed + 1)

    for _ in range(steps):
        legal = legal_actions_fast(state)
        _assert(len(legal) > 0, "No legal actions available")
        _, mask = encoder.encode_legal_actions(legal)
        _assert(float(mask.sum()) >= 1.0, "Mask has no legal action")
        for action in legal:
            idx = encoder.encode_action(action)
            _assert(mask[idx] == 1.0, "Legal action missing from mask")

        state = apply_action_unchecked(state, rng.choice(legal))
        if terminal_fast(state):
            state = new_game(seed=rng.randint(1, 10_000_000))


def check_terminal_consistency(num_games: int, seed: int, max_moves: int = 250) -> None:
    rng = random.Random(seed + 2)
    for g in range(num_games):
        state = new_game(seed=seed + g)
        for _ in range(max_moves):
            if terminal_fast(state):
                break
            legal = legal_actions_fast(state)
            _assert(len(legal) > 0, "Non-terminal state has no legal actions")
            state = apply_action_unchecked(state, rng.choice(legal))

        _assert(terminal_fast(state), "Game did not reach terminal within max_moves")
        s0 = int(compute_score_fast(state, 0))
        s1 = int(compute_score_fast(state, 1))
        winner = int(get_winner_fast(state))
        _assert(winner in (0, 1), "Winner must be 0 or 1")
        if s0 > s1:
            _assert(winner == 0, "Winner-score inconsistency (p0)")
        elif s1 > s0:
            _assert(winner == 1, "Winner-score inconsistency (p1)")


def _play_pure_game(seed: int, sims: int, max_moves: int = 200) -> int:
    p0 = PureMCTSEvaluator(simulations=sims, seed=seed)
    p1 = PureMCTSEvaluator(simulations=sims, seed=seed)
    state = new_game(seed=seed)
    for move in range(max_moves):
        if terminal_fast(state):
            break
        to_move = current_player_fast(state)
        if to_move == 0:
            action = p0.get_move(state, seed_offset=move)
        else:
            action = p1.get_move(state, seed_offset=move)
        state = apply_action_unchecked(state, action)
    _assert(terminal_fast(state), "Pure-vs-pure game did not terminate")
    return int(get_winner_fast(state))


def check_avs_a_fairness(num_pairs: int, seed: int, sims: int) -> None:
    # Candidate A and baseline B are identical pure-MCTS agents.
    # Paired seeds + side swap should give ~50%.
    a_points = 0.0
    total_games = 0
    for i in range(num_pairs):
        s = seed + i * 17
        winner_g1 = _play_pure_game(s, sims=sims)  # A as P0
        a_points += 1.0 if winner_g1 == 0 else 0.0
        total_games += 1

        winner_g2 = _play_pure_game(s, sims=sims)  # A as P1 (interpretation swap)
        a_points += 1.0 if winner_g2 == 1 else 0.0
        total_games += 1

    rate = a_points / float(total_games)
    print(f"A-vs-A paired rate: {rate:.3f} over {total_games} games")
    _assert(0.40 <= rate <= 0.60, "A-vs-A fairness drift outside [0.40, 0.60]")


def check_state_encoder_shape(seed: int) -> None:
    encoder = GoldV2StateEncoder()
    state = new_game(seed=seed)
    to_move = current_player_fast(state)
    x, _, _, _, _ = encoder.encode_state_multimodal(state, to_move)
    _assert(x.shape == (56, 9, 9), "Encoded state shape mismatch")
    _assert(str(x.dtype) == "float32", "Encoded state dtype mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description="Patchwork AlphaZero sanity checks")
    parser.add_argument("--quick", action="store_true", help="Run lightweight quick checks")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.quick:
        action_steps = 12
        mask_steps = 12
        terminal_games = 6
        pairs = 4
        sims = 8
    else:
        action_steps = 60
        mask_steps = 60
        terminal_games = 20
        pairs = 20
        sims = 16

    print("Running sanity checks...")
    check_state_encoder_shape(args.seed)
    check_action_roundtrip(steps=action_steps, seed=args.seed)
    check_legality_masks(steps=mask_steps, seed=args.seed)
    check_terminal_consistency(num_games=terminal_games, seed=args.seed)
    check_avs_a_fairness(num_pairs=pairs, seed=args.seed, sims=sims)
    print("All sanity checks passed.")


if __name__ == "__main__":
    main()
