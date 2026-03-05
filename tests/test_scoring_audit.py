"""
A) PROVE the game scoring function is correct.

Authoritative formula (patchwork_engine.compute_score / compute_score_fast):
  score = buttons - 2 * empty_squares + bonus_7x7
  bonus_7x7 = 7 if player has the 7x7 tile, else 0
  Tie-break: get_winner returns 1 - TIE_PLAYER when scores are equal.
"""
from __future__ import annotations

import numpy as np

from src.game.patchwork_engine import (
    state_from_dict,
    new_game,
    compute_score,
    compute_score_fast,
    get_winner,
    get_winner_fast,
    empty_count_from_occ,
    P0_OCC0, P0_OCC1, P0_OCC2,
    P1_OCC0, P1_OCC1, P1_OCC2,
    BOARD_SIZE,
)


def _board_rows(filled_count: int) -> list:
    """Create 9x9 board with first `filled_count` cells filled (row-major). '.' = empty, '1' = filled."""
    rows = []
    for r in range(BOARD_SIZE):
        row = []
        for c in range(BOARD_SIZE):
            idx = r * BOARD_SIZE + c
            row.append("1" if idx < filled_count else ".")
        rows.append("".join(row))
    return rows


def _make_terminal_state(
    p0_buttons: int,
    p1_buttons: int,
    p0_filled: int,
    p1_filled: int,
    bonus_owner: int = -1,
    tie_player: int = 0,
    circle_seed: int = 0,
) -> np.ndarray:
    """Build a terminal-like state with given buttons and filled counts. Uses state_from_dict."""
    from src.game.patchwork_engine import make_shuffled_circle
    circle, neutral = make_shuffled_circle(seed=circle_seed)
    p0_board = _board_rows(min(81, max(0, p0_filled)))
    p1_board = _board_rows(min(81, max(0, p1_filled)))
    d = {
        "players": [
            {"position": 53, "buttons": p0_buttons, "board": p0_board, "income": 0},
            {"position": 53, "buttons": p1_buttons, "board": p1_board, "income": 0},
        ],
        "circle": circle,
        "neutral": neutral,
        "randomize_circle": False,
        "bonus_owner": bonus_owner,
        "pending_patches": 0,
        "pending_owner": -1,
        "tie_player": tie_player,
        "edition": "revised",
    }
    return state_from_dict(d)


def test_score_formula_buttons_minus_2_per_empty():
    """Same buttons, different empty squares: score differs by exactly 2 per empty."""
    # P0: 80 filled (1 empty), P1: 81 filled (0 empty). Same buttons.
    st1 = _make_terminal_state(p0_buttons=50, p1_buttons=50, p0_filled=80, p1_filled=81, bonus_owner=-1)
    sc0_1 = compute_score_fast(st1, 0)
    sc1_1 = compute_score_fast(st1, 1)
    # P0: 50 - 2*1 + 0 = 48.  P1: 50 - 0 + 0 = 50.
    assert sc0_1 == 48, f"expected 48 got {sc0_1}"
    assert sc1_1 == 50, f"expected 50 got {sc1_1}"

    # P0: 79 filled (2 empty), same buttons
    st2 = _make_terminal_state(p0_buttons=50, p1_buttons=50, p0_filled=79, p1_filled=81, bonus_owner=-1)
    sc0_2 = compute_score_fast(st2, 0)
    assert sc0_2 == 46, f"expected 46 (50 - 2*2) got {sc0_2}"
    assert sc0_1 - sc0_2 == 2, "one extra empty should reduce score by exactly 2"
    assert sc0_2 - 46 == 0


def test_score_difference_exactly_2_per_empty():
    """Two states: same player buttons, differ by 1 empty -> scores differ by exactly 2."""
    # State A: P0 has 70 filled (11 empty)
    st_a = _make_terminal_state(p0_buttons=40, p1_buttons=40, p0_filled=70, p1_filled=81)
    # State B: P0 has 71 filled (10 empty)
    st_b = _make_terminal_state(p0_buttons=40, p1_buttons=40, p0_filled=71, p1_filled=81)
    score_a = compute_score_fast(st_a, 0)
    score_b = compute_score_fast(st_b, 0)
    assert score_b - score_a == 2, f"one fewer empty should add 2 points: {score_a} -> {score_b}"


def test_7x7_bonus():
    """7x7 bonus: +7 for the player who has it."""
    # Full board = has 7x7 (any 7x7 region is filled). Give P0 the bonus.
    st_no_bonus = _make_terminal_state(30, 30, 81, 81, bonus_owner=-1)
    st_p0_bonus = _make_terminal_state(30, 30, 81, 81, bonus_owner=0)
    st_p1_bonus = _make_terminal_state(30, 30, 81, 81, bonus_owner=1)

    assert compute_score_fast(st_no_bonus, 0) == 30, "full board, no bonus"
    assert compute_score_fast(st_no_bonus, 1) == 30

    assert compute_score_fast(st_p0_bonus, 0) == 37, "P0 gets +7"
    assert compute_score_fast(st_p0_bonus, 1) == 30
    assert compute_score_fast(st_p1_bonus, 0) == 30
    assert compute_score_fast(st_p1_bonus, 1) == 37


def test_tie_break_rule():
    """When scores are equal, get_winner returns 1 - TIE_PLAYER."""
    st = _make_terminal_state(40, 40, 81, 81, bonus_owner=-1, tie_player=0)
    assert compute_score_fast(st, 0) == compute_score_fast(st, 1)
    winner = get_winner_fast(st)
    assert winner == 1, "TIE_PLAYER=0 -> winner = 1-0 = 1"

    st1 = _make_terminal_state(40, 40, 81, 81, bonus_owner=-1, tie_player=1)
    winner1 = get_winner_fast(st1)
    assert winner1 == 0, "TIE_PLAYER=1 -> winner = 1-1 = 0"


def test_empty_count_from_occ():
    """empty_count_from_occ = 81 - popcount(occ0|occ1|occ2)."""
    st = _make_terminal_state(0, 0, 80, 0)
    e0 = empty_count_from_occ(
        int(st[P0_OCC0]), int(st[P0_OCC1]), int(st[P0_OCC2])
    )
    e1 = empty_count_from_occ(
        int(st[P1_OCC0]), int(st[P1_OCC1]), int(st[P1_OCC2])
    )
    assert e0 == 1, f"P0 has 80 filled -> 1 empty, got {e0}"
    assert e1 == 81, f"P1 has 0 filled -> 81 empty, got {e1}"
