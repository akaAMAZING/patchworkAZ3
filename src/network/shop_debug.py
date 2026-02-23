"""
Shop order debug and verification utilities.

Authoritative reference for pawn/shop indexing convention.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from src.game.patchwork_engine import (
    CIRCLE_LEN,
    CIRCLE_START,
    NEUTRAL,
    MAX_CIRCLE,
)


# =============================================================================
# Pawn/Shop Indexing Convention (AUTHORITATIVE)
# =============================================================================
#
# state[NEUTRAL] = index in the circle array of the 1x1 neutral pawn piece (piece_id=32).
# The circle is stored at state[CIRCLE_START : CIRCLE_START + n] where n = state[CIRCLE_LEN].
#
# Convention: NEUTRAL points to the PIECE that is the pawn (the 1x1 marker).
# The next 3 pieces clockwise are buyable:
#   slot0 = circle[(neutral + 1) % n]
#   slot1 = circle[(neutral + 2) % n]
#   slot2 = circle[(neutral + 3) % n]
#
# "remaining_after_pawn" = piece IDs in clockwise order starting immediately AFTER the pawn:
#   remaining[k] = circle[(neutral + 1 + k) % n]  for k = 0, 1, ..., n-1
#
# INVARIANT: remaining_after_pawn[0:3] == [slot0_id, slot1_id, slot2_id]
#
# After buying slot s (offset = s+1):
#   idx = (neutral + s + 1) % n  (index of bought piece)
#   Piece is removed; circle shifts.
#   new_neutral = (idx - 1) % (n-1)  (points to piece that was immediately before bought piece)
# =============================================================================


def get_remaining_after_pawn(state: np.ndarray) -> List[int]:
    """
    Return piece IDs in clockwise order starting immediately after the neutral pawn.

    remaining[0] = slot0, remaining[1] = slot1, remaining[2] = slot2.
    """
    n = int(state[CIRCLE_LEN])
    if n <= 0:
        return []
    neutral = int(state[NEUTRAL])
    return [int(state[CIRCLE_START + (neutral + 1 + k) % n]) for k in range(n)]


def get_slot_piece_ids_from_engine(state: np.ndarray) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Get slot0, slot1, slot2 piece IDs using engine semantics (offset 1,2,3)."""
    from src.network.encoder import get_slot_piece_id
    s0 = get_slot_piece_id(state, 0)
    s1 = get_slot_piece_id(state, 1)
    s2 = get_slot_piece_id(state, 2)
    return (s0, s1, s2)


def debug_dump_shop_state(state: np.ndarray) -> str:
    """
    Print human-readable shop state for debugging.

    Returns a string with:
    - pawn representation (raw NEUTRAL index)
    - full ring order (piece ids in clockwise order as stored)
    - visible slots (slot0/1/2 piece ids)
    - computed remaining_after_pawn (first 10)
    """
    n = int(state[CIRCLE_LEN])
    neutral = int(state[NEUTRAL])
    circle = [int(state[CIRCLE_START + i]) for i in range(n)]
    slot0, slot1, slot2 = get_slot_piece_ids_from_engine(state)
    remaining = get_remaining_after_pawn(state)

    lines = [
        "=== Shop State Debug ===",
        f"  CIRCLE_LEN (n): {n}",
        f"  NEUTRAL (pawn index): {neutral}",
        f"  Full ring (piece ids, indices 0..n-1): {circle}",
        f"  Visible slots: slot0={slot0} slot1={slot1} slot2={slot2}",
        f"  remaining_after_pawn (first 10): {remaining[:10]}",
    ]
    return "\n".join(lines)


def assert_shop_order_alignment(state: np.ndarray) -> None:
    """
    Assert remaining_after_pawn[0:3] == [slot0_id, slot1_id, slot2_id].

    Raises AssertionError with details if mismatch.
    """
    remaining = get_remaining_after_pawn(state)
    slot0, slot1, slot2 = get_slot_piece_ids_from_engine(state)
    expected = [slot0, slot1, slot2]
    actual = remaining[:3] if len(remaining) >= 3 else remaining + [None] * (3 - len(remaining))
    assert actual[0] == expected[0], (
        f"slot0 mismatch: remaining[0]={actual[0]} vs slot0={expected[0]}"
    )
    assert actual[1] == expected[1], (
        f"slot1 mismatch: remaining[1]={actual[1]} vs slot1={expected[1]}"
    )
    assert actual[2] == expected[2], (
        f"slot2 mismatch: remaining[2]={actual[2]} vs slot2={expected[2]}"
    )
