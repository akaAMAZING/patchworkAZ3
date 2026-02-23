"""
D4 action equivariance — verify spatial planes, orientation channels, and
policy/action indices transform consistently with engine semantics.

CRITICAL: A legal in S <=> A' legal in S'; and sym(step(S,A)) == step(S',A').
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

import numpy as np
import pytest

from src.game.patchwork_engine import (
    new_game,
    legal_actions_fast,
    apply_action_unchecked,
    terminal_fast,
    current_player_fast,
    AT_PASS,
    AT_PATCH,
    AT_BUY,
    BOARD_SIZE,
    ORIENT_SHAPE_H,
    ORIENT_SHAPE_W,
)
from src.network.encoder import StateEncoder, get_slot_piece_id
from src.network.d4_augmentation import (
    transform_position,
    get_orient_after_transform,
    transform_buy_top_left,
    transform_legalTL_planes,
    _transform_board_plane,
    D4_COUNT,
    D4_NAMES,
)


# =============================================================================
# Engine state spatial transform (boards only; circle/positions unchanged)
# =============================================================================

def _occ_words_from_mask(mask: np.ndarray) -> Tuple[int, int, int]:
    """Encode 9x9 binary mask to (occ0, occ1, occ2) words. Returns values for np.int32 storage."""
    w0, w1, w2 = np.uint32(0), np.uint32(0), np.uint32(0)
    for idx in range(81):
        if mask.flat[idx] <= 0:
            continue
        word = idx >> 5
        bit = idx & 31
        b = np.uint32(1) << np.uint32(bit)
        if word == 0:
            w0 |= b
        elif word == 1:
            w1 |= b
        else:
            w2 |= b
    return (int(w0) & 0xFFFFFFFF, int(w1) & 0xFFFFFFFF, int(w2) & 0xFFFFFFFF)


def transform_engine_state_boards(state: np.ndarray, transform_idx: int) -> np.ndarray:
    """
    Return new state with P0 and P1 board occupancies spatially transformed.
    Circle, positions, buttons, etc. unchanged.
    """
    enc = StateEncoder()
    out = state.copy()

    for player in (0, 1):
        base = 0 if player == 0 else 6
        o0 = int(state[base + 3])
        o1 = int(state[base + 4])
        o2 = int(state[base + 5])
        mask = enc._decode_occ_words(o0, o1, o2)
        transformed = _transform_board_plane(mask, transform_idx)
        no0, no1, no2 = _occ_words_from_mask(transformed)
        out[base + 3] = np.uint32(no0).astype(np.int32)
        out[base + 4] = np.uint32(no1).astype(np.int32)
        out[base + 5] = np.uint32(no2).astype(np.int32)

    return out


def _transform_buy_position(
    top: int, left: int, piece_id: int, orient: int, orient_new: int, transform_idx: int
) -> Tuple[int, int]:
    """
    Transform (top, left) for a buy action. The piece's bounding box changes with
    orientation; the new top-left is the min of the transformed corners.
    """
    BS = BOARD_SIZE
    old_h = int(ORIENT_SHAPE_H[piece_id, orient])
    old_w = int(ORIENT_SHAPE_W[piece_id, orient])

    br = top + old_h - 1
    bc = left + old_w - 1

    def corners(r0, c0, r1, c1):
        return [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]

    if transform_idx == 0:
        return (top, left)
    if transform_idx == 1:  # r90: (r,c)->(c, BS-1-r)
        pts = [(c, BS - 1 - r) for r, c in corners(top, left, br, bc)]
    elif transform_idx == 2:  # r180
        pts = [(BS - 1 - r, BS - 1 - c) for r, c in corners(top, left, br, bc)]
    elif transform_idx == 3:  # r270: (r,c)->(BS-1-c, r)
        pts = [(BS - 1 - c, r) for r, c in corners(top, left, br, bc)]
    elif transform_idx == 4:  # m: (r,c)->(r, BS-1-c)
        pts = [(r, BS - 1 - c) for r, c in corners(top, left, br, bc)]
    elif transform_idx == 5:  # m_r90
        pts = [(r, BS - 1 - c) for r, c in corners(top, left, br, bc)]
        pts = [(c, BS - 1 - r) for r, c in pts]
    elif transform_idx == 6:  # m_r180
        pts = [(r, BS - 1 - c) for r, c in corners(top, left, br, bc)]
        pts = [(BS - 1 - r, BS - 1 - c) for r, c in pts]
    elif transform_idx == 7:  # m_r270
        pts = [(r, BS - 1 - c) for r, c in corners(top, left, br, bc)]
        pts = [(BS - 1 - c, r) for r, c in pts]
    else:
        return (top, left)

    min_r = min(p[0] for p in pts)
    min_c = min(p[1] for p in pts)
    return (min_r, min_c)


def transform_engine_action(
    action: Tuple,
    state: np.ndarray,
    transform_idx: int,
) -> Tuple:
    """Transform engine action for application in transformed state."""
    atype = int(action[0])
    if atype == AT_PASS:
        return action

    if atype == AT_PATCH:
        idx = int(action[1])
        r, c = idx // BOARD_SIZE, idx % BOARD_SIZE
        rn, cn = transform_position(r, c, transform_idx)
        new_idx = rn * BOARD_SIZE + cn
        return (AT_PATCH, new_idx, 0, 0, 0, 0)

    if atype == AT_BUY:
        offset = int(action[1])
        piece_id = int(action[2])
        orient = int(action[3])
        top = int(action[4])
        left = int(action[5])
        orient_new = get_orient_after_transform(piece_id, transform_idx, orient)
        top_new, left_new = _transform_buy_position(
            top, left, piece_id, orient, orient_new, transform_idx
        )
        return (AT_BUY, offset, piece_id, orient_new, top_new, left_new)

    return action


def _states_canonically_equal(s1: np.ndarray, s2: np.ndarray) -> bool:
    """Compare states: same circle, positions, occupancies (order may differ)."""
    if s1.shape != s2.shape:
        return False
    return np.array_equal(s1, s2)


# =============================================================================
# legalTL from engine (for gold_v2-style validation)
# =============================================================================

def compute_legalTL_simple(state: np.ndarray, to_move: int) -> np.ndarray:
    """Compute legalTL by iterating legal buy actions from engine."""
    legal = legal_actions_fast(state)
    legalTL = np.zeros((24, 9, 9), dtype=np.float32)
    n = int(state[12])
    neutral = int(state[13])

    for a in legal:
        if a[0] != AT_BUY:
            continue
        offset, piece_id, orient, top, left = a[1], a[2], a[3], a[4], a[5]
        slot = offset - 1
        piece_idx = (neutral + offset) % n
        if int(state[18 + piece_idx]) != piece_id:
            continue
        ch = slot * 8 + orient
        legalTL[ch, top, left] = 1.0

    return legalTL


# =============================================================================
# Tests
# =============================================================================

def test_d4_action_equivariance_patch():
    """Patch placement: A legal in S <=> A' legal in S'; step(S,A) transform == step(S',A')."""
    state = new_game(seed=42)
    for _ in range(15):
        legal = legal_actions_fast(state)
        patch_actions = [a for a in legal if a[0] == AT_PATCH]
        if not patch_actions:
            state = apply_action_unchecked(state, legal[0])
            continue
        action = patch_actions[0]
        for ti in range(D4_COUNT):
            S_prime = transform_engine_state_boards(state, ti)
            A_prime = transform_engine_action(action, state, ti)
            in_S = action in legal
            legal_prime = legal_actions_fast(S_prime)
            in_S_prime = A_prime in legal_prime
            assert in_S == in_S_prime, (
                f"ti={D4_NAMES[ti]}: A legal in S={in_S} but A' legal in S'={in_S_prime}"
            )
            if in_S:
                S1 = apply_action_unchecked(state, action)
                S2 = apply_action_unchecked(S_prime, A_prime)
                S1_transformed = transform_engine_state_boards(S1, ti)
                assert _states_canonically_equal(S1_transformed, S2), (
                    f"ti={D4_NAMES[ti]}: sym(step(S,A)) != step(S',A')"
                )
        state = apply_action_unchecked(state, action)
        if terminal_fast(state):
            break


def test_d4_action_equivariance_buy_legal_in_transformed():
    """
    Buy action: A legal in S => A' legal in S'.
    Uses dimension-aware position transform for engine-level correctness.
    Known to fail due to orient remap producing wrong shape for some piece/transform combos.
    """
    state = new_game(seed=99)
    for _ in range(20):
        legal = legal_actions_fast(state)
        buy_actions = [a for a in legal if a[0] == AT_BUY]
        if not buy_actions:
            if legal:
                state = apply_action_unchecked(state, legal[0])
            continue
        action = buy_actions[0]
        for ti in range(D4_COUNT):
            S_prime = transform_engine_state_boards(state, ti)
            A_prime = transform_engine_action(action, state, ti)
            legal_prime = legal_actions_fast(S_prime)
            assert A_prime in legal_prime, (
                f"ti={D4_NAMES[ti]}: A' must be legal in S'"
            )
        state = apply_action_unchecked(state, action)
        if terminal_fast(state) or int(state[12]) == 0:
            break


def test_d4_action_equivariance_many_states():
    """Run legal equivariance (A legal <=> A' legal) on many random states. Skip full step equivariance for buy (see docs)."""
    rng = random.Random(7777)
    states = []
    for seed in range(50):
        st = new_game(seed=seed)
        for _ in range(25):
            if terminal_fast(st):
                break
            legal = legal_actions_fast(st)
            if not legal:
                break
            states.append(st.copy())
            st = apply_action_unchecked(st, rng.choice(legal))

    tested = 0
    for state in states:
        legal = legal_actions_fast(state)
        if not legal:
            continue
        action = rng.choice(legal)
        for ti in range(D4_COUNT):
            S_prime = transform_engine_state_boards(state, ti)
            A_prime = transform_engine_action(action, state, ti)
            in_S = action in legal
            legal_prime = legal_actions_fast(S_prime)
            in_S_prime = A_prime in legal_prime
            assert in_S == in_S_prime, f"ti={ti} legal mismatch"
            tested += 1
        if tested >= 100:
            break


@pytest.mark.slow
def test_legalTL_transform_consistency():
    """
    For each state S and symmetry sym:
    L = legalTL from engine for S
    L' = legalTL from engine for S' (transformed boards)
    assert L' == sym_transform(L) with dimension-aware buy transform.
    """
    state = new_game(seed=123)
    to_move = current_player_fast(state)
    slot_piece_ids = [get_slot_piece_id(state, s) for s in range(3)]

    for _ in range(5):
        legal = legal_actions_fast(state)
        if not legal or terminal_fast(state) or int(state[12]) == 0:
            break

        L = compute_legalTL_simple(state, to_move)

        for ti in range(D4_COUNT):
            S_prime = transform_engine_state_boards(state, ti)
            to_move_prime = current_player_fast(S_prime)
            L_prime_from_engine = compute_legalTL_simple(S_prime, to_move_prime)

            L_transformed = transform_legalTL_planes(L, ti, slot_piece_ids)

            assert L_transformed.shape == L_prime_from_engine.shape
            diff = np.abs(L_transformed - L_prime_from_engine)
            max_diff = diff.max()
            assert max_diff < 0.5, (
                f"ti={D4_NAMES[ti]}: legalTL transform mismatch, max_diff={max_diff}"
            )

        state = apply_action_unchecked(state, legal[0])
        to_move = current_player_fast(state)


def test_transform_action_vector_buy_matches_engine_dimension_aware():
    """
    For random reachable states and random legal buy actions, verify that the
    remapped policy index from transform_action_vector corresponds to the
    engine-equivalent transformed action.
    """
    from src.network.d4_augmentation import (
        transform_action_vector,
        BUY_START,
        NUM_ORIENTS,
    )

    rng = random.Random(4242)
    for _ in range(50):
        state = new_game(seed=rng.randint(0, 9999))
        for step in range(30):
            if terminal_fast(state) or int(state[12]) == 0:
                break
            legal = legal_actions_fast(state)
            buy_actions = [a for a in legal if a[0] == AT_BUY]
            if not buy_actions:
                if legal:
                    state = apply_action_unchecked(state, legal[0])
                continue
            action = rng.choice(buy_actions)
            offset, piece_id, orient, top, left = action[1], action[2], action[3], action[4], action[5]
            slot = offset - 1
            slot_piece_ids = [get_slot_piece_id(state, s) for s in range(3)]
            policy_idx = BUY_START + (slot * NUM_ORIENTS + orient) * 81 + top * 9 + left

            mask = np.zeros(2026, dtype=np.float32)
            mask[0] = 1.0
            mask[policy_idx] = 1.0

            for ti in range(D4_COUNT):
                mask_t = transform_action_vector(mask, ti, slot_piece_ids)
                A_prime = transform_engine_action(action, state, ti)
                offset_prime, orient_new, top_new, left_new = (
                    A_prime[1], A_prime[3], A_prime[4], A_prime[5]
                )
                slot_prime = offset_prime - 1
                expected_idx = BUY_START + (slot_prime * NUM_ORIENTS + orient_new) * 81 + top_new * 9 + left_new
                assert mask_t[expected_idx] > 0, (
                    f"ti={D4_NAMES[ti]}: transformed policy should have 1 at {expected_idx}, got {mask_t[expected_idx]}"
                )
            state = apply_action_unchecked(state, action)
