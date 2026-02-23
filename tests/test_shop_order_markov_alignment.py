"""
Shop order Markov alignment — verify encoder/engine semantics match exactly.

CRITICAL INVARIANT: remaining_after_pawn[0:3] == [slot0_id, slot1_id, slot2_id]
and after simulating buy slot s, the new state satisfies the same property.
"""

from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np
import pytest

from src.game.patchwork_engine import (
    new_game,
    legal_actions_fast,
    apply_action_unchecked,
    terminal_fast,
    current_player_fast,
    AT_BUY,
)
from src.network.shop_debug import (
    get_remaining_after_pawn,
    get_slot_piece_ids_from_engine,
    assert_shop_order_alignment,
    debug_dump_shop_state,
)


def _play_random_moves(state: np.ndarray, num_moves: int, seed: int) -> List[np.ndarray]:
    """Generate list of reachable states by playing random legal moves."""
    rng = random.Random(seed)
    states = [state.copy()]
    for _ in range(num_moves):
        s = states[-1]
        if terminal_fast(s):
            break
        legal = legal_actions_fast(s)
        if not legal:
            break
        action = rng.choice(legal)
        next_s = apply_action_unchecked(s, action)
        states.append(next_s)
    return states


def _collect_reachable_states(num_games: int, moves_per_game: int, base_seed: int) -> List[np.ndarray]:
    """Collect reachable states from many random games."""
    all_states: List[np.ndarray] = []
    for g in range(num_games):
        st = new_game(seed=base_seed + g)
        path = _play_random_moves(st, moves_per_game, base_seed + 1000 + g)
        all_states.extend(path)
    return all_states


def test_shop_order_alignment_initial_state():
    """Initial state: assert alignment."""
    state = new_game(seed=42)
    assert_shop_order_alignment(state)


def test_shop_order_alignment_after_buy_slot0():
    """After buying slot 0, alignment holds."""
    state = new_game(seed=123)
    legal = legal_actions_fast(state)
    buy_actions = [a for a in legal if a[0] == AT_BUY and a[1] == 1]
    if not buy_actions:
        pytest.skip("No buy slot0 available in initial state")
    action = buy_actions[0]
    next_state = apply_action_unchecked(state, action)
    assert_shop_order_alignment(next_state)


def test_shop_order_alignment_after_buy_slot1():
    """After buying slot 1, alignment holds."""
    state = new_game(seed=456)
    legal = legal_actions_fast(state)
    buy_actions = [a for a in legal if a[0] == AT_BUY and a[1] == 2]
    if not buy_actions:
        pytest.skip("No buy slot1 available")
    action = buy_actions[0]
    next_state = apply_action_unchecked(state, action)
    assert_shop_order_alignment(next_state)


def test_shop_order_alignment_after_buy_slot2():
    """After buying slot 2, alignment holds."""
    state = new_game(seed=789)
    legal = legal_actions_fast(state)
    buy_actions = [a for a in legal if a[0] == AT_BUY and a[1] == 3]
    if not buy_actions:
        pytest.skip("No buy slot2 available")
    action = buy_actions[0]
    next_state = apply_action_unchecked(state, action)
    assert_shop_order_alignment(next_state)


def test_shop_order_markov_alignment():
    """
    For at least 200 reachable states (multiple seeds, random games):
    assert remaining_after_pawn[0:3] == [slot0_id, slot1_id, slot2_id].
    """
    # Generate ~200+ reachable states: 10 games × ~25 moves each = 250+ states
    states = _collect_reachable_states(num_games=10, moves_per_game=30, base_seed=9000)
    # Filter to states with non-empty circle (shop exists)
    states_with_shop = [s for s in states if int(s[12]) > 0]  # CIRCLE_LEN = 12
    assert len(states_with_shop) >= 200, (
        f"Need at least 200 states with shop, got {len(states_with_shop)}"
    )

    for i, state in enumerate(states_with_shop):
        try:
            assert_shop_order_alignment(state)
        except AssertionError as e:
            print(debug_dump_shop_state(state))
            raise AssertionError(f"State index {i} failed: {e}") from e


def test_shop_order_after_sequential_buys():
    """Simulate buy slot0, then buy slot0 again, etc. Alignment holds after each."""
    state = new_game(seed=111)
    for _ in range(5):
        if terminal_fast(state) or int(state[12]) == 0:
            break
        legal = legal_actions_fast(state)
        buy_actions = [a for a in legal if a[0] == AT_BUY]
        if not buy_actions:
            break
        action = buy_actions[0]
        state = apply_action_unchecked(state, action)
        assert_shop_order_alignment(state)


def test_debug_dump_shop_state_smoke():
    """debug_dump_shop_state runs without error and contains expected keys."""
    state = new_game(seed=42)
    out = debug_dump_shop_state(state)
    assert "NEUTRAL" in out
    assert "slot0" in out
    assert "remaining_after_pawn" in out
