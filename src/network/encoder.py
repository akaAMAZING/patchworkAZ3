"""
Patchwork State + Action Encoders (Slot-Based Action Space)

StateEncoder: Legacy encoder (deprecated). Use GoldV2StateEncoder for 56ch gold_v2.
ActionEncoder: Maps between engine actions (numeric tuples) and network action indices.
              Includes D4 augmentation (vertical/horizontal board flips) for training.

Total action space size: 82 + 3*8*81 = 2026
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    import numba as _numba
    _HAS_NUMBA = True
    _njit = _numba.njit(cache=True, fastmath=True, nogil=True)
except ImportError:
    _HAS_NUMBA = False
    _njit = lambda f: f  # pure-Python fallback — correct but slower

# Import from the game engine
from src.game.patchwork_engine import (
    PIECE_BY_ID,
    PIECE_AUGMENTATION,
    BOARD_SIZE,
    BUTTONS_AFTER,
    PATCHES_AFTER,
    CIRCLE_LEN,
    NEUTRAL,
    BONUS_OWNER,
    PENDING_PATCHES,
    P0_POS,
    P0_BUTTONS,
    P0_INCOME,
    P0_OCC0,
    P0_OCC1,
    P0_OCC2,
    P1_POS,
    P1_BUTTONS,
    P1_INCOME,
    P1_OCC0,
    P1_OCC1,
    P1_OCC2,
    CIRCLE_START,
    AT_PASS,
    AT_PATCH,
    AT_BUY,
    ORIENT_COUNT,
    ORIENT_INCOME,
    ORIENT_HEIGHT,
    ORIENT_WIDTH,
    ORIENT_SHAPE_H,
    ORIENT_SHAPE_W,
    MASK_W0,
    MASK_W1,
    MASK_W2,
    PIECE_COST_BUTTONS,
    PIECE_COST_TIME,
    EDITION_CODE,
    TIE_PLAYER,
    SEVEN_MASK_W0,
    SEVEN_MASK_W1,
    SEVEN_MASK_W2,
)
from src.network.gold_v2_constants import (
    C_SPATIAL,
    C_SPATIAL_ENC,
    C_TRACK,
    F_GLOBAL,
    F_SHOP,
    NMAX,
    TRACK_LEN,
)


# Source-of-truth constants (must match engine)
TRACK_LENGTH = 53
MAX_PIECE_COST = 10
MAX_PIECE_TIME = 6
MAX_PIECE_INCOME = 2
MAX_BUTTONS = 40
MAX_INCOME = 12
BOARD_SIZE_INT = 9
_LOG1P_200 = math.log1p(200.0)   # precomputed: log1p(200) for button normalisation
_LOG1P_20  = math.log1p(20.0)    # precomputed: log1p(20)  for cost normalisation


def _button_marks_crossed(prev_pos: int, new_pos: int) -> int:
    """Steps to next income trigger strictly > prev_pos; if at/after last, return 0."""
    return int(sum(1 for m in BUTTONS_AFTER.tolist() if int(prev_pos) < int(m) <= int(new_pos)))


def _patches_crossed(prev_pos: int, new_pos: int, opponent_pos: int, edition_code: int) -> int:
    """Same using PATCHES_AFTER, opponent_pos for eligibility."""
    return int(sum(1 for m in PATCHES_AFTER.tolist() if int(prev_pos) < int(m) <= int(new_pos) and int(opponent_pos) < int(m)))


def _clamp_pos(pos: int) -> int:
    return TRACK_LENGTH if int(pos) > TRACK_LENGTH else int(pos)


# ---------------------------------------------------------------------------
# Numba JIT helpers — used by GoldV2StateEncoder.encode_into
# ---------------------------------------------------------------------------

@_njit
def _popcount32(x):
    """Bit population count (Brian Kernighan) — Numba replacement for int.bit_count()."""
    count = 0
    x = x & 0xFFFFFFFF
    while x:
        x = x & (x - 1)
        count += 1
    return count


@_njit
def _jit_clamp_pos(pos, track_length):
    return track_length if pos > track_length else pos


@_njit
def _jit_buttons_crossed(prev_pos, new_pos, buttons_after):
    count = 0
    for i in range(len(buttons_after)):
        m = buttons_after[i]
        if prev_pos < m and m <= new_pos:
            count += 1
    return count


@_njit
def _jit_patches_crossed(prev_pos, new_pos, opp_pos, patches_after):
    count = 0
    for i in range(len(patches_after)):
        m = patches_after[i]
        if prev_pos < m and m <= new_pos and opp_pos < m:
            count += 1
    return count


@_njit
def _encode_scalars_shop_jit(
    slot_pids,       # (3,)    int32 — piece_id per visible slot, -1 if empty
    shop_pids,       # (NMAX,) int32 — circle piece_ids in shop order
    n_shop,          # int — actual number of items in shop_pids
    n_circle,        # int — total circle length
    c_pos, c_buttons, c_income, c_occ0, c_occ1, c_occ2,
    o_pos, o_buttons, o_income, o_occ0, o_occ1, o_occ2,
    bonus_owner, pending, tie_player, to_move,
    piece_cost_lut,    # (MAX_PID,) int32
    piece_time_lut,    # (MAX_PID,) int32
    piece_income_lut,  # (MAX_PID,) int32
    piece_area_lut,    # (MAX_PID,) int32
    orient_count_lut,  # (MAX_PID,) int32
    buttons_after,     # (M,) int64
    patches_after,     # (K,) int64
    seven_mask_w0,     # (9,) int64
    seven_mask_w1,     # (9,) int64
    seven_mask_w2,     # (9,) int64
    x_global,          # (F_GLOBAL,)      float32 — output
    x_track,           # (C_TRACK, TRACK_LEN) float32 — output
    shop_ids,          # (NMAX,)          int16   — output
    shop_feats,        # (NMAX, F_SHOP)   float32 — output
    track_length,      # int = 53
    track_len,         # int = 54
    log1p_200,         # float
    log1p_20,          # float
):
    """JIT-compiled scalar/shop encoder: fills x_global, x_track, shop_ids, shop_feats."""

    # Mask occupancy to 32-bit unsigned range for correct bitwise arithmetic
    c_o0 = c_occ0 & 0xFFFFFFFF
    c_o1 = c_occ1 & 0xFFFFFFFF
    c_o2 = c_occ2 & 0xFFFFFFFF
    o_o0 = o_occ0 & 0xFFFFFFFF
    o_o1 = o_occ1 & 0xFFFFFFFF
    o_o2 = o_occ2 & 0xFFFFFFFF

    denom_pos = float(track_len)

    # ---- x_global[0-8]: position & resource features ----
    x_global[0] = c_pos / denom_pos
    x_global[1] = o_pos / denom_pos
    x_global[2] = (c_pos - o_pos) / denom_pos
    cb_c = c_buttons if c_buttons < 200 else 200
    ob_c = o_buttons if o_buttons < 200 else 200
    x_global[3] = math.log1p(cb_c) / log1p_200
    x_global[4] = math.log1p(ob_c) / log1p_200
    x_global[5] = math.tanh((c_buttons - o_buttons) / 40.0)
    ci_c = c_income if c_income < 15 else 15
    oi_c = o_income if o_income < 15 else 15
    x_global[6] = math.tanh(ci_c / 15.0)
    x_global[7] = math.tanh(oi_c / 15.0)
    x_global[8] = math.tanh((c_income - o_income) / 10.0)

    # ---- x_global[9-10]: board fill fractions ----
    cur_filled = _popcount32(c_o0) + _popcount32(c_o1) + _popcount32(c_o2)
    opp_filled = _popcount32(o_o0) + _popcount32(o_o1) + _popcount32(o_o2)
    x_global[9]  = cur_filled / 81.0
    x_global[10] = opp_filled / 81.0

    # ---- x_global[11]: bonus tile owner ----
    if bonus_owner == -1:
        x_global[11] = 0.0
    elif bonus_owner == to_move:
        x_global[11] = 1.0
    else:
        x_global[11] = 0.5

    # ---- x_global[12-13]: 7x7 completion distance (cur / opp) ----
    best7 = 49
    for k in range(9):
        m0 = seven_mask_w0[k]
        m1 = seven_mask_w1[k]
        m2 = seven_mask_w2[k]
        if (c_o0 & m0) == m0 and (c_o1 & m1) == m1 and (c_o2 & m2) == m2:
            best7 = 0
            break
        fi = _popcount32(c_o0 & m0) + _popcount32(c_o1 & m1) + _popcount32(c_o2 & m2)
        miss = 49 - fi
        if miss < best7:
            best7 = miss
    x_global[12] = best7 / 49.0

    best7o = 49
    for k in range(9):
        m0 = seven_mask_w0[k]
        m1 = seven_mask_w1[k]
        m2 = seven_mask_w2[k]
        if (o_o0 & m0) == m0 and (o_o1 & m1) == m1 and (o_o2 & m2) == m2:
            best7o = 0
            break
        fi = _popcount32(o_o0 & m0) + _popcount32(o_o1 & m1) + _popcount32(o_o2 & m2)
        miss = 49 - fi
        if miss < best7o:
            best7o = miss
    x_global[13] = best7o / 49.0

    # ---- x_global[14-21]: track state ----
    x_global[14] = (pending if pending < 5 else 5) / 5.0
    remaining = 0
    if n_circle > 0:
        for i in range(len(patches_after)):
            if o_pos < patches_after[i]:
                remaining += 1
    x_global[15] = (remaining if remaining < 5 else 5) / 5.0
    for i in range(len(patches_after)):
        x_global[16 + i] = 1.0 if o_pos < patches_after[i] else 0.0
    x_global[21] = 1.0 if tie_player == to_move else 0.0

    # ---- x_global[22-27]: pass action features ----
    pass_new = _jit_clamp_pos(o_pos + 1, track_length)
    pass_steps = pass_new - c_pos
    pass_inc = _jit_buttons_crossed(c_pos, pass_new, buttons_after)
    pass_pat = _jit_patches_crossed(c_pos, pass_new, o_pos, patches_after)
    x_global[22] = pass_steps / denom_pos
    x_global[23] = (pass_inc if pass_inc < 9 else 9) / 9.0
    x_global[24] = (pass_pat if pass_pat < 2 else 2) / 2.0
    pass_btn_delta = pass_inc * c_income
    x_global[25] = math.tanh(pass_btn_delta / 30.0)
    x_global[26] = pass_new / denom_pos
    x_global[27] = 1.0 if pass_new == o_pos else 0.0

    # ---- x_global[28-60]: per-slot (3 x 11) buy action features ----
    for s in range(3):
        pid = slot_pids[s]
        base = 28 + s * 11
        if pid < 0:
            continue
        cost    = piece_cost_lut[pid]
        tv      = piece_time_lut[pid]
        inc_g   = piece_income_lut[pid]
        area    = piece_area_lut[pid]
        buy_new = _jit_clamp_pos(c_pos + tv, track_length)
        buy_inc = _jit_buttons_crossed(c_pos, buy_new, buttons_after)
        buy_pat = _jit_patches_crossed(c_pos, buy_new, o_pos, patches_after)
        ibg     = buy_inc * c_income
        x_global[base]      = 1.0 if c_buttons >= cost else 0.0
        x_global[base + 1]  = math.log1p(cost if cost < 20 else 20) / log1p_20
        x_global[base + 2]  = (tv if tv < 6 else 6) / 6.0
        x_global[base + 3]  = (inc_g if inc_g < 3 else 3) / 3.0
        x_global[base + 4]  = (area if area < 9 else 9) / 9.0
        x_global[base + 5]  = (buy_inc if buy_inc < 9 else 9) / 9.0
        x_global[base + 6]  = math.tanh(ibg / 80.0)
        x_global[base + 7]  = (buy_pat if buy_pat < 2 else 2) / 2.0
        x_global[base + 8]  = math.tanh((-cost + ibg) / 40.0)
        x_global[base + 9]  = buy_new / denom_pos
        x_global[base + 10] = 1.0 if buy_new == o_pos else 0.0

    # ---- x_track (C_TRACK x TRACK_LEN): one-hot positions & event markers ----
    tc = c_pos if c_pos < track_len else track_len - 1
    to_ = o_pos if o_pos < track_len else track_len - 1
    x_track[0, tc]  = 1.0
    x_track[1, to_] = 1.0
    for i in range(len(buttons_after)):
        m = buttons_after[i]
        if m < track_len:
            x_track[2, m] = 1.0
    for i in range(len(patches_after)):
        m = patches_after[i]
        if o_pos < m and m < track_len:
            x_track[3, m] = 1.0
    pl = _jit_clamp_pos(o_pos + 1, track_length)
    if pl < track_len:
        x_track[4, pl] = 1.0
    for s in range(3):
        pid = slot_pids[s]
        if pid >= 0:
            land = _jit_clamp_pos(c_pos + piece_time_lut[pid], track_length)
            if land < track_len:
                x_track[5 + s, land] = 1.0

    # ---- shop_ids + shop_feats: 33-item circle loop ----
    if n_shop > 0:
        denom_sh = float(n_circle - 1) if n_circle > 1 else 1.0
        for i in range(n_shop):
            pid = shop_pids[i]
            shop_ids[i] = pid
            cost  = piece_cost_lut[pid]
            tv    = piece_time_lut[pid]
            inc   = piece_income_lut[pid]
            area  = piece_area_lut[pid]
            n_or  = orient_count_lut[pid]
            shop_feats[i, 0] = i / denom_sh
            shop_feats[i, 1] = math.log1p(cost if cost < 20 else 20) / log1p_20
            shop_feats[i, 2] = (tv if tv < 6 else 6) / 6.0
            shop_feats[i, 3] = (inc if inc < 3 else 3) / 3.0
            shop_feats[i, 4] = (area if area < 9 else 9) / 9.0
            shop_feats[i, 5] = n_or / 8.0
            shop_feats[i, 6] = 1.0 if c_buttons >= cost else 0.0
            shop_feats[i, 7] = 1.0 if o_buttons >= cost else 0.0
            # shop_feats[i, 8] and [i, 9]: num_legal placements, computed on-GPU


def _dist_to_next_income(pos: int) -> int:
    """Distance in steps to next income trigger strictly > pos; if at/after last, return 0."""
    for m in BUTTONS_AFTER.tolist():
        if int(pos) < int(m):
            return int(m) - int(pos)
    return 0


def _dist_to_next_patch(pos: int) -> int:
    """Distance in steps to next patch trigger strictly > pos; if at/after last, return 0."""
    for m in PATCHES_AFTER.tolist():
        if int(pos) < int(m):
            return int(m) - int(pos)
    return 0


# =========================================================================
# Shared slot lookup (unified — used by both encoder and MCTS)
# =========================================================================

def get_slot_piece_id(state: np.ndarray, slot_idx: int) -> Optional[int]:
    """
    Get piece ID for a slot in the circle.

    Slots 0-2 correspond to the next 1-3 pieces clockwise of the neutral pawn.
    Returns None if the slot doesn't exist (circle too small).
    """
    n = int(state[CIRCLE_LEN])
    if n <= 0:
        return None
    max_pick = min(3, n)
    if slot_idx < 0 or slot_idx >= max_pick:
        return None
    neutral = int(state[NEUTRAL])
    piece_idx = (neutral + slot_idx + 1) % n
    return int(state[CIRCLE_START + piece_idx])


# =========================================================================
# State Encoder (Legacy — deprecated; 56ch gold_v2 is production)
# =========================================================================

class StateEncoder:
    """Legacy encoder (deprecated). Use GoldV2StateEncoder for 56ch gold_v2."""

    BOARD_SIZE = 9
    NUM_CHANNELS = 61
    MAX_BUTTONS = MAX_BUTTONS
    MAX_INCOME = MAX_INCOME
    MAX_POSITION = TRACK_LENGTH
    MAX_PIECE_COST = MAX_PIECE_COST
    MAX_PIECE_TIME = MAX_PIECE_TIME
    MAX_PIECE_INCOME = MAX_PIECE_INCOME

    def __init__(self) -> None:
        self.piece_costs = {pid: int(piece["cost_buttons"]) for pid, piece in PIECE_BY_ID.items()}
        self._slot_orient_shape_masks = self._precompute_slot_orient_shape_masks()
        self._coord_row_norm, self._coord_col_norm = self._precompute_coord_planes()

    def _precompute_coord_planes(self) -> Tuple[np.ndarray, np.ndarray]:
        """Precompute coord_row_norm and coord_col_norm (row/(BOARD_SIZE-1), col/(BOARD_SIZE-1))."""
        denom = float(BOARD_SIZE_INT - 1)
        row = np.zeros((BOARD_SIZE_INT, BOARD_SIZE_INT), dtype=np.float32)
        col = np.zeros((BOARD_SIZE_INT, BOARD_SIZE_INT), dtype=np.float32)
        for r in range(BOARD_SIZE_INT):
            for c in range(BOARD_SIZE_INT):
                row[r, c] = r / denom
                col[r, c] = c / denom
        return row, col

    def _precompute_slot_orient_shape_masks(self) -> Dict[Tuple[int, int], np.ndarray]:
        """Precompute (piece_id, orient) -> 9x9 top-left anchored shape mask from engine MASK_W*."""
        out: Dict[Tuple[int, int], np.ndarray] = {}
        for pid in PIECE_BY_ID:
            n_orient = int(ORIENT_COUNT[pid])
            for o in range(min(8, n_orient)):
                # Engine stores mask for placement at (top, left); pos=0 => top=0, left=0
                for pos in range(BOARD_SIZE * BOARD_SIZE):
                    m0 = int(MASK_W0[pid, o, pos])
                    m1 = int(MASK_W1[pid, o, pos])
                    m2 = int(MASK_W2[pid, o, pos])
                    if (m0 | m1 | m2) != 0:
                        mask = self._decode_occ_words(m0, m1, m2)
                        out[(pid, o)] = mask
                        break
        return out

    def _get_slot_orient_shape(self, piece_id: int, orient: int) -> np.ndarray:
        """Return 9x9 shape mask for (piece_id, orient). Uses engine orientation semantics."""
        key = (piece_id, orient)
        if key in self._slot_orient_shape_masks:
            return self._slot_orient_shape_masks[key].copy()
        mask = np.zeros((BOARD_SIZE_INT, BOARD_SIZE_INT), dtype=np.float32)
        n_orient = int(ORIENT_COUNT[piece_id])
        if orient < n_orient:
            for pos in range(BOARD_SIZE * BOARD_SIZE):
                m0 = MASK_W0[piece_id, orient, pos]
                m1 = MASK_W1[piece_id, orient, pos]
                m2 = MASK_W2[piece_id, orient, pos]
                if (m0 | m1 | m2) != 0:
                    mask = self._decode_occ_words(int(m0), int(m1), int(m2))
                    break
        return mask

    @staticmethod
    def _decode_occ_words(occ0_i: int, occ1_i: int, occ2_i: int) -> np.ndarray:
        """Decode 3×32-bit occupancy words into a (9,9) float32 mask."""
        words = np.array([
            int(occ0_i) & 0xFFFFFFFF,
            int(occ1_i) & 0xFFFFFFFF,
            int(occ2_i) & 0xFFFFFFFF,
        ], dtype=np.uint32)
        byte_arr = words.view(np.uint8)
        all_bits = np.unpackbits(byte_arr, bitorder='little')
        return all_bits[:81].astype(np.float32).reshape(BOARD_SIZE, BOARD_SIZE)

    def encode_state(self, state: np.ndarray, to_move: int) -> np.ndarray:
        """Encode game state from current player's perspective (legacy encoder)."""
        channels = np.zeros((self.NUM_CHANNELS, self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.float32)

        if int(to_move) == 0:
            c_pos, c_buttons, c_income = int(state[P0_POS]), int(state[P0_BUTTONS]), int(state[P0_INCOME])
            o_pos, o_buttons, o_income = int(state[P1_POS]), int(state[P1_BUTTONS]), int(state[P1_INCOME])
            c_occ0, c_occ1, c_occ2 = int(state[P0_OCC0]), int(state[P0_OCC1]), int(state[P0_OCC2])
            o_occ0, o_occ1, o_occ2 = int(state[P1_OCC0]), int(state[P1_OCC1]), int(state[P1_OCC2])
        else:
            c_pos, c_buttons, c_income = int(state[P1_POS]), int(state[P1_BUTTONS]), int(state[P1_INCOME])
            o_pos, o_buttons, o_income = int(state[P0_POS]), int(state[P0_BUTTONS]), int(state[P0_INCOME])
            c_occ0, c_occ1, c_occ2 = int(state[P1_OCC0]), int(state[P1_OCC1]), int(state[P1_OCC2])
            o_occ0, o_occ1, o_occ2 = int(state[P0_OCC0]), int(state[P0_OCC1]), int(state[P0_OCC2])

        ed = int(state[EDITION_CODE]) if state.shape[0] > EDITION_CODE else 0

        # Spatial (0-27)
        channels[0] = self._decode_occ_words(c_occ0, c_occ1, c_occ2)
        channels[1] = self._decode_occ_words(o_occ0, o_occ1, o_occ2)
        channels[2] = self._coord_row_norm
        channels[3] = self._coord_col_norm

        # Slots 0-2, orients 0-7 (4-27)
        for slot_idx in range(3):
            pid = get_slot_piece_id(state, slot_idx)
            for orient in range(8):
                ch = 4 + slot_idx * 8 + orient
                if pid is not None and orient < int(ORIENT_COUNT[pid]):
                    channels[ch] = self._get_slot_orient_shape(pid, orient)

        # Scalar channels (28-60) — full-plane, never spatially transformed
        channels[28].fill(min(c_income, self.MAX_INCOME) / float(self.MAX_INCOME))
        channels[29].fill(min(o_income, self.MAX_INCOME) / float(self.MAX_INCOME))
        channels[30].fill((c_pos - o_pos) / float(TRACK_LENGTH))
        cb = min(c_buttons, self.MAX_BUTTONS) / float(self.MAX_BUTTONS)
        ob = min(o_buttons, self.MAX_BUTTONS) / float(self.MAX_BUTTONS)
        channels[31].fill(cb - ob)

        bonus_owner = int(state[BONUS_OWNER])
        if bonus_owner == -1:
            channels[32].fill(0.0)
        elif bonus_owner == int(to_move):
            channels[32].fill(1.0)
        else:
            channels[32].fill(0.5)

        pending = int(state[PENDING_PATCHES])
        channels[33].fill(min(pending, 5) / 5.0)
        channels[34].fill(c_pos / float(TRACK_LENGTH))
        channels[35].fill(min(c_buttons, self.MAX_BUTTONS) / float(self.MAX_BUTTONS))

        channels[36].fill(_dist_to_next_income(c_pos) / float(TRACK_LENGTH))
        channels[37].fill(_dist_to_next_income(o_pos) / float(TRACK_LENGTH))
        channels[38].fill(_dist_to_next_patch(c_pos) / float(TRACK_LENGTH))
        channels[39].fill(_dist_to_next_patch(o_pos) / float(TRACK_LENGTH))

        # pass_* (engine formula: new_pos = clamp(o_pos+1), prev=c_pos)
        pass_new_pos = _clamp_pos(o_pos + 1)
        pass_steps = pass_new_pos - c_pos
        pass_income_marks = _button_marks_crossed(c_pos, pass_new_pos)
        pass_patches = _patches_crossed(c_pos, pass_new_pos, o_pos, ed)
        channels[40].fill(pass_steps / float(TRACK_LENGTH))
        channels[41].fill(min(pass_income_marks, 6) / 6.0)
        channels[42].fill(min(pass_patches, 5) / 5.0)

        # Per-slot cost/time/income (43-51)
        for slot_idx in range(3):
            pid = get_slot_piece_id(state, slot_idx)
            base = 43 + slot_idx * 3
            if pid is not None:
                cost = int(PIECE_COST_BUTTONS[pid])
                time_val = int(PIECE_COST_TIME[pid])
                income = 0
                for o in range(int(ORIENT_COUNT[pid])):
                    income = max(income, int(ORIENT_INCOME[pid, o]))
                channels[base].fill(max(0, min(cost, self.MAX_PIECE_COST)) / float(self.MAX_PIECE_COST))
                channels[base + 1].fill(max(0, min(time_val, self.MAX_PIECE_TIME)) / float(self.MAX_PIECE_TIME))
                channels[base + 2].fill(max(0, min(income, self.MAX_PIECE_INCOME)) / float(self.MAX_PIECE_INCOME))

        # Per-slot can_afford + buy_*crossed (52-60)
        for slot_idx in range(3):
            pid = get_slot_piece_id(state, slot_idx)
            cost = int(PIECE_COST_BUTTONS[pid]) if pid is not None else 999
            time_val = int(PIECE_COST_TIME[pid]) if pid is not None else 0
            can_afford = 1.0 if (pid is not None and c_buttons >= cost) else 0.0
            buy_new_pos = _clamp_pos(c_pos + time_val) if pid is not None else c_pos
            buy_income = _button_marks_crossed(c_pos, buy_new_pos) if pid is not None else 0
            buy_patches = _patches_crossed(c_pos, buy_new_pos, o_pos, ed) if pid is not None else 0
            channels[52 + slot_idx * 3].fill(can_afford)
            channels[52 + slot_idx * 3 + 1].fill(min(buy_income, 6) / 6.0)
            channels[52 + slot_idx * 3 + 2].fill(min(buy_patches, 5) / 5.0)

        return channels

    def encode_batch(self, states: List[np.ndarray], to_move_list: List[int]) -> torch.Tensor:
        """Encode batch of states."""
        batch_size = len(states)
        encoded = np.zeros((batch_size, self.NUM_CHANNELS, self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.float32)
        for i, (state, to_move) in enumerate(zip(states, to_move_list)):
            encoded[i] = self.encode_state(state, to_move)
        return torch.from_numpy(encoded)


# =========================================================================
# Gold v2 Multimodal State Encoder
# =========================================================================

def _count_legal_placements_clip20(
    occ0: int, occ1: int, occ2: int,
    piece_id: int,
    max_count: int = 20,
) -> int:
    """Count legal placements for a piece (all orients). Early-exit at max_count."""
    n_orient = int(ORIENT_COUNT[piece_id])
    count = 0
    o0, o1, o2 = np.uint32(occ0), np.uint32(occ1), np.uint32(occ2)
    for orient in range(n_orient):
        for pos_idx in range(81):
            m0 = int(MASK_W0[piece_id, orient, pos_idx])
            m1 = int(MASK_W1[piece_id, orient, pos_idx])
            m2 = int(MASK_W2[piece_id, orient, pos_idx])
            if (m0 | m1 | m2) == 0:
                continue
            if (o0 & m0) != 0 or (o1 & m1) != 0 or (o2 & m2) != 0:
                continue
            count += 1
            if count >= max_count:
                return count
    return count


def _valid_7x7_empty(occ0: int, occ1: int, occ2: int) -> np.ndarray:
    """Returns 9x9 plane: 1 at (r,c) if 7x7 with TL at (r,c) is fully empty. Only (0,0)..(2,2) can be 1."""
    out = np.zeros((9, 9), dtype=np.float32)
    o0, o1, o2 = np.uint32(int(occ0) & 0xFFFFFFFF), np.uint32(int(occ1) & 0xFFFFFFFF), np.uint32(int(occ2) & 0xFFFFFFFF)
    for k in range(9):
        top, left = k // 3, k % 3
        m0 = int(SEVEN_MASK_W0[k])
        m1 = int(SEVEN_MASK_W1[k])
        m2 = int(SEVEN_MASK_W2[k])
        if (o0 & m0) == 0 and (o1 & m1) == 0 and (o2 & m2) == 0:
            out[top, left] = 1.0
    return out


def _frontier_plane(occ0: int, occ1: int, occ2: int) -> np.ndarray:
    """Empty cells adjacent (4-neighbor) to filled cells (vectorised — no Python loop)."""
    occ = StateEncoder._decode_occ_words(int(occ0), int(occ1), int(occ2)).astype(bool)
    padded = np.zeros((11, 11), dtype=bool)
    padded[1:-1, 1:-1] = occ
    adj = padded[:-2, 1:-1] | padded[2:, 1:-1] | padded[1:-1, :-2] | padded[1:-1, 2:]
    return (adj & ~occ).astype(np.float32)


class GoldV2StateEncoder:
    """Gold v2 multimodal encoder.

    encoding_version="gold_v2_36ch" (default): x_spatial (36,9,9), x_global (65,)
        Includes packing quality planes ch 32-35 and global[61-64].
    encoding_version="gold_v2_32ch" (legacy): x_spatial (32,9,9), x_global (61,)
        Omits packing quality planes. Required for iter192 and earlier checkpoints.

    In both cases channels 8-31 are slot×orient shape planes and legalTL (ch 32-55 or 36-59)
    is computed on-GPU by DeterministicLegalityModule in PatchworkNetwork.forward().
    """

    def __init__(self, encoding_version: str | None = None) -> None:
        from src.network.gold_v2_constants import ENCODING_VERSION as _CURRENT
        _enc = (encoding_version or _CURRENT).lower()
        self._legacy_32ch: bool = _enc in ("gold_v2_32ch", "gold_v2_multimodal")
        _n_sp: int = 32 if self._legacy_32ch else C_SPATIAL_ENC  # spatial channels (32 or 36)
        _n_g: int  = 61 if self._legacy_32ch else F_GLOBAL        # global dims (61 or 65)
        self._coord_row = np.array(
            [[r / 8.0 for c in range(9)] for r in range(9)],
            dtype=np.float32,
        )
        self._coord_col = np.array(
            [[c / 8.0 for c in range(9)] for r in range(9)],
            dtype=np.float32,
        )
        _se = StateEncoder()
        self._slot_orient_shape_masks = _se._precompute_slot_orient_shape_masks()
        self._get_shape = _se._get_slot_orient_shape

        # Pre-compute per-piece area (orient-0 cell count) and max income for fast dict lookup.
        # Replaces per-call: sum(1 for r ... for c ... if mask[r,c] > 0)
        self._piece_area: Dict[int, int] = {}
        self._piece_income: Dict[int, int] = {}
        for pid in PIECE_BY_ID:
            pid_int = int(pid)
            mask0 = self._slot_orient_shape_masks.get((pid_int, 0))
            self._piece_area[pid_int] = int(mask0.sum()) if mask0 is not None else 0
            n_o = int(ORIENT_COUNT[pid_int])
            self._piece_income[pid_int] = max(int(ORIENT_INCOME[pid_int, o]) for o in range(n_o))

        # ---- Numba JIT LUTs: 1-D numpy arrays indexed by piece_id ----
        # Replaces per-call dict lookups with O(1) array indexing inside the JIT function.
        _all_pids = sorted(int(pid) for pid in PIECE_BY_ID)
        _max_pid = _all_pids[-1] + 1 if _all_pids else 0
        self._piece_cost_lut   = np.zeros(_max_pid, dtype=np.int32)
        self._piece_time_lut   = np.zeros(_max_pid, dtype=np.int32)
        self._piece_income_lut = np.zeros(_max_pid, dtype=np.int32)
        self._piece_area_lut   = np.zeros(_max_pid, dtype=np.int32)
        self._orient_count_lut = np.zeros(_max_pid, dtype=np.int32)
        for pid_int in _all_pids:
            self._piece_cost_lut[pid_int]   = int(PIECE_COST_BUTTONS[pid_int])
            self._piece_time_lut[pid_int]   = int(PIECE_COST_TIME[pid_int])
            self._piece_income_lut[pid_int] = self._piece_income.get(pid_int, 0)
            self._piece_area_lut[pid_int]   = self._piece_area.get(pid_int, 0)
            self._orient_count_lut[pid_int] = int(ORIENT_COUNT[pid_int])

        # int64 copies of track marker / bitmask arrays for Numba bitwise ops
        self._buttons_after_i64 = BUTTONS_AFTER.astype(np.int64)
        self._patches_after_i64 = PATCHES_AFTER.astype(np.int64)
        self._seven_mask_w0_i64 = SEVEN_MASK_W0.astype(np.int64)
        self._seven_mask_w1_i64 = SEVEN_MASK_W1.astype(np.int64)
        self._seven_mask_w2_i64 = SEVEN_MASK_W2.astype(np.int64)

        # Pre-allocated working buffers for JIT pre-extraction (avoids hot-path malloc)
        self._buf_slot_pids = np.zeros(3, dtype=np.int32)
        self._buf_shop_pids = np.zeros(NMAX, dtype=np.int32)

        # Pre-allocated instance buffers — reused across encode calls (no per-call malloc).
        # encode_state_multimodal writes here then returns copies; encode_into writes in-place.
        self._buf_spatial    = np.zeros((_n_sp, 9, 9), dtype=np.float32)
        self._buf_global     = np.zeros(_n_g, dtype=np.float32)
        self._buf_track      = np.zeros((C_TRACK, TRACK_LEN), dtype=np.float32)
        self._buf_shop_ids   = np.full(NMAX, -1, dtype=np.int16)
        self._buf_shop_feats = np.zeros((NMAX, F_SHOP), dtype=np.float32)

    def _get_slot_orient_shape(self, piece_id: int, orient: int) -> np.ndarray:
        key = (piece_id, orient)
        if key in self._slot_orient_shape_masks:
            return self._slot_orient_shape_masks[key].copy()
        return self._get_shape(piece_id, orient)

    @staticmethod
    def _decode_occ_words(occ0_i: int, occ1_i: int, occ2_i: int) -> np.ndarray:
        """Decode 3×32-bit occupancy words into a (9,9) float32 mask."""
        words = np.array([
            int(occ0_i) & 0xFFFFFFFF,
            int(occ1_i) & 0xFFFFFFFF,
            int(occ2_i) & 0xFFFFFFFF,
        ], dtype=np.uint32)
        byte_arr = words.view(np.uint8)
        all_bits = np.unpackbits(byte_arr, bitorder="little")
        return all_bits[:81].astype(np.float32).reshape(BOARD_SIZE_INT, BOARD_SIZE_INT)

    def encode_state_multimodal(
        self, state: np.ndarray, to_move: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Encode game state into multimodal inputs.

        Returns:
            x_spatial: (36, 9, 9) float32  — channels 0-35 only; legalTL (36-59) computed on GPU
            x_global: (65,) float32
            x_track: (8, 54) float32
            shop_ids: (33,) int16, pad=-1
            shop_feats: (33, 10) float32
        """
        self.encode_into(
            state, to_move,
            self._buf_spatial, self._buf_global, self._buf_track,
            self._buf_shop_ids, self._buf_shop_feats,
        )
        return (
            self._buf_spatial.copy(),
            self._buf_global.copy(),
            self._buf_track.copy(),
            self._buf_shop_ids.copy(),
            self._buf_shop_feats.copy(),
        )

    def encode_into(
        self,
        state: np.ndarray,
        to_move: int,
        x_spatial: np.ndarray,
        x_global: np.ndarray,
        x_track: np.ndarray,
        shop_ids: np.ndarray,
        shop_feats: np.ndarray,
    ) -> None:
        """
        Zero-copy encode: write state directly into shared-memory buffer views.
        Uses gold_v2_36ch logic (36 spatial channels, 65 global, 8x54 track, shop).
        Caller must provide pre-allocated views (e.g. from WorkerSharedBuffer).
        """
        if int(to_move) == 0:
            c_pos     = int(state[P0_POS])
            c_buttons = int(state[P0_BUTTONS])
            c_income  = int(state[P0_INCOME])
            c_occ0, c_occ1, c_occ2 = int(state[P0_OCC0]), int(state[P0_OCC1]), int(state[P0_OCC2])
            o_pos     = int(state[P1_POS])
            o_buttons = int(state[P1_BUTTONS])
            o_income  = int(state[P1_INCOME])
            o_occ0, o_occ1, o_occ2 = int(state[P1_OCC0]), int(state[P1_OCC1]), int(state[P1_OCC2])
        else:
            c_pos     = int(state[P1_POS])
            c_buttons = int(state[P1_BUTTONS])
            c_income  = int(state[P1_INCOME])
            c_occ0, c_occ1, c_occ2 = int(state[P1_OCC0]), int(state[P1_OCC1]), int(state[P1_OCC2])
            o_pos     = int(state[P0_POS])
            o_buttons = int(state[P0_BUTTONS])
            o_income  = int(state[P0_INCOME])
            o_occ0, o_occ1, o_occ2 = int(state[P0_OCC0]), int(state[P0_OCC1]), int(state[P0_OCC2])

        n_circle   = int(state[CIRCLE_LEN])
        neutral    = int(state[NEUTRAL])
        bonus_own  = int(state[BONUS_OWNER])
        pending    = int(state[PENDING_PATCHES])
        tie_player = int(state[TIE_PLAYER])

        # Zero buffers (SHM may contain stale data)
        x_spatial.fill(0.0)
        x_global.fill(0.0)
        x_track.fill(0.0)
        shop_ids.fill(-1)
        shop_feats.fill(0.0)

        # ---- ch 0-7: numpy spatial planes (vectorised — keep as numpy) ----
        x_spatial[0] = StateEncoder._decode_occ_words(c_occ0, c_occ1, c_occ2)
        x_spatial[1] = StateEncoder._decode_occ_words(o_occ0, o_occ1, o_occ2)
        x_spatial[2] = self._coord_row
        x_spatial[3] = self._coord_col
        x_spatial[4] = _frontier_plane(c_occ0, c_occ1, c_occ2)
        x_spatial[5] = _frontier_plane(o_occ0, o_occ1, o_occ2)
        x_spatial[6] = _valid_7x7_empty(c_occ0, c_occ1, c_occ2)
        x_spatial[7] = _valid_7x7_empty(o_occ0, o_occ1, o_occ2)

        # ---- ch 8-31: slot×orient shape masks (dict lookup + numpy copy) ----
        # legalTL is NOT stored here — DeterministicLegalityModule computes it on-GPU.
        for slot in range(3):
            pid = get_slot_piece_id(state, slot)
            n_orient = int(ORIENT_COUNT[pid]) if pid is not None else 8
            for o in range(8):
                ch_shape = 8 + slot * 8 + o
                if pid is not None and o < n_orient:
                    mask = self._slot_orient_shape_masks.get((pid, o))
                    if mask is not None:
                        x_spatial[ch_shape] = mask  # numpy copy — no intermediate alloc

        # ---- pre-extract slot / shop piece IDs for JIT (avoids dict inside JIT) ----
        slot_pids = self._buf_slot_pids
        for s in range(3):
            p = get_slot_piece_id(state, s)
            slot_pids[s] = p if p is not None else -1

        n_shop = min(n_circle, NMAX) if n_circle > 0 else 0
        shop_pids = self._buf_shop_pids
        for i in range(n_shop):
            piece_idx = (neutral + 1 + i) % n_circle
            shop_pids[i] = int(state[CIRCLE_START + piece_idx])

        # ---- x_global + x_track + shop_ids/feats: Numba JIT ----
        _encode_scalars_shop_jit(
            slot_pids, shop_pids, n_shop, n_circle,
            c_pos, c_buttons, c_income, c_occ0, c_occ1, c_occ2,
            o_pos, o_buttons, o_income, o_occ0, o_occ1, o_occ2,
            bonus_own, pending, tie_player, int(to_move),
            self._piece_cost_lut, self._piece_time_lut,
            self._piece_income_lut, self._piece_area_lut, self._orient_count_lut,
            self._buttons_after_i64, self._patches_after_i64,
            self._seven_mask_w0_i64, self._seven_mask_w1_i64, self._seven_mask_w2_i64,
            x_global, x_track, shop_ids, shop_feats,
            TRACK_LENGTH, TRACK_LEN, _LOG1P_200, _LOG1P_20,
        )

        # ---- ch 32-35 and global[61-64]: packing quality planes (36ch encoding only) ----
        if not self._legacy_32ch:
            # ch 32-33: isolated holes (empty cells with all 4 cardinal neighbors filled/OOB)
            # ch 32 = current player board, ch 33 = opponent board
            for ch_occ, ch_dst in ((0, 32), (1, 33)):
                occ = x_spatial[ch_occ].astype(bool)
                empty = ~occ
                # Pad with True (filled): OOB is treated as a wall
                padded = np.pad(occ, 1, mode='constant', constant_values=True)
                n_nbr = padded[:-2, 1:-1]
                s_nbr = padded[2:,  1:-1]
                w_nbr = padded[1:-1, :-2]
                e_nbr = padded[1:-1, 2:]
                x_spatial[ch_dst] = (empty & n_nbr & s_nbr & w_nbr & e_nbr).astype(np.float32)

            # ch 34-35: constrained empty (at most 1 empty 4-neighbor; isolated + dead-end cells)
            # ch 34 = current player board, ch 35 = opponent board
            for ch_occ, ch_dst in ((0, 34), (1, 35)):
                occ = x_spatial[ch_occ].astype(bool)
                empty = ~occ
                padded_e = np.pad(empty, 1, mode='constant', constant_values=False)
                n_e = padded_e[:-2, 1:-1]
                s_e = padded_e[2:,  1:-1]
                w_e = padded_e[1:-1, :-2]
                e_e = padded_e[1:-1, 2:]
                empty_nbr = (n_e.view(np.uint8) + s_e.view(np.uint8)
                             + w_e.view(np.uint8) + e_e.view(np.uint8)).astype(np.int32)
                x_spatial[ch_dst] = (empty & (empty_nbr <= 1)).astype(np.float32)

            # x_global[61-64]: normalized counts (divisor 27 = rough max isolated holes in 9×9)
            x_global[61] = float(x_spatial[32].sum()) / 27.0  # cur isolated holes
            x_global[62] = float(x_spatial[33].sum()) / 27.0  # opp isolated holes
            x_global[63] = float(x_spatial[34].sum()) / 27.0  # cur constrained empty
            x_global[64] = float(x_spatial[35].sum()) / 27.0  # opp constrained empty


# Module-level singleton encoder — avoids recreating the encoder (and
# re-running _precompute_slot_orient_shape_masks for all 264 piece×orient
# combos) on every single MCTS leaf call.
_GOLD_V2_ENCODER: Optional[GoldV2StateEncoder] = None


def encode_state_multimodal(
    state: np.ndarray, to_move: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convenience: encode state using a module-level singleton GoldV2StateEncoder."""
    global _GOLD_V2_ENCODER
    if _GOLD_V2_ENCODER is None:
        _GOLD_V2_ENCODER = GoldV2StateEncoder()
    return _GOLD_V2_ENCODER.encode_state_multimodal(state, to_move)


# =========================================================================
# Legacy State Encoder (16 channels — for 16ch cross-arch eval)
# =========================================================================

class LegacyStateEncoder:
    """16-channel encoding (legacy layout). Use for 16ch checkpoints in cross-arch eval."""

    BOARD_SIZE = 9
    NUM_CHANNELS = 16
    MAX_BUTTONS = MAX_BUTTONS
    MAX_INCOME = MAX_INCOME
    MAX_POSITION = TRACK_LENGTH
    MAX_PIECE_COST = MAX_PIECE_COST

    def __init__(self) -> None:
        self.piece_costs = {pid: int(piece["cost_buttons"]) for pid, piece in PIECE_BY_ID.items()}
        self.piece_shape_masks = self._precompute_piece_shape_masks()

    def _precompute_piece_shape_masks(self) -> Dict[int, np.ndarray]:
        out: Dict[int, np.ndarray] = {}
        for pid, piece in PIECE_BY_ID.items():
            shape = piece["shape"]
            h, w = len(shape), len(shape[0])
            mask = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.float32)
            for r in range(h):
                for c in range(w):
                    if int(shape[r][c]) != 0:
                        mask[r, c] = 1.0
            out[int(pid)] = mask
        return out

    @staticmethod
    def _decode_occ_words(occ0_i: int, occ1_i: int, occ2_i: int) -> np.ndarray:
        words = np.array([
            int(occ0_i) & 0xFFFFFFFF,
            int(occ1_i) & 0xFFFFFFFF,
            int(occ2_i) & 0xFFFFFFFF,
        ], dtype=np.uint32)
        byte_arr = words.view(np.uint8)
        all_bits = np.unpackbits(byte_arr, bitorder="little")
        return all_bits[:81].astype(np.float32).reshape(BOARD_SIZE_INT, BOARD_SIZE_INT)

    def encode_state(self, state: np.ndarray, to_move: int) -> np.ndarray:
        """Encode game state from current player's perspective (16 channels)."""
        channels = np.zeros((self.NUM_CHANNELS, self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.float32)

        if int(to_move) == 0:
            c_pos, c_buttons, c_income = int(state[P0_POS]), int(state[P0_BUTTONS]), int(state[P0_INCOME])
            o_pos, o_buttons, o_income = int(state[P1_POS]), int(state[P1_BUTTONS]), int(state[P1_INCOME])
            c_occ0, c_occ1, c_occ2 = int(state[P0_OCC0]), int(state[P0_OCC1]), int(state[P0_OCC2])
            o_occ0, o_occ1, o_occ2 = int(state[P1_OCC0]), int(state[P1_OCC1]), int(state[P1_OCC2])
        else:
            c_pos, c_buttons, c_income = int(state[P1_POS]), int(state[P1_BUTTONS]), int(state[P1_INCOME])
            o_pos, o_buttons, o_income = int(state[P0_POS]), int(state[P0_BUTTONS]), int(state[P0_INCOME])
            c_occ0, c_occ1, c_occ2 = int(state[P1_OCC0]), int(state[P1_OCC1]), int(state[P1_OCC2])
            o_occ0, o_occ1, o_occ2 = int(state[P0_OCC0]), int(state[P0_OCC1]), int(state[P0_OCC2])

        channels[0] = self._decode_occ_words(c_occ0, c_occ1, c_occ2)
        channels[1].fill(min(c_income, self.MAX_INCOME) / float(self.MAX_INCOME))
        channels[2] = self._decode_occ_words(o_occ0, o_occ1, o_occ2)
        channels[3].fill(min(o_income, self.MAX_INCOME) / float(self.MAX_INCOME))

        n = int(state[CIRCLE_LEN])
        if n > 0:
            max_pick = min(3, n)
            affordable_count = 0
            total_norm_cost = 0.0
            slot_count = 0
            for slot_idx in range(max_pick):
                pid = get_slot_piece_id(state, slot_idx)
                if pid is None:
                    continue
                cost = int(self.piece_costs.get(pid, 0))
                total_norm_cost += cost / float(self.MAX_PIECE_COST)
                slot_count += 1
                if c_buttons >= cost:
                    affordable_count += 1
            if slot_count > 0:
                channels[4].fill(affordable_count / float(slot_count))
                channels[5].fill(total_norm_cost / float(slot_count))

        channels[6].fill((c_pos / float(self.MAX_POSITION)) - (o_pos / float(self.MAX_POSITION)))
        cb = min(c_buttons, self.MAX_BUTTONS) / float(self.MAX_BUTTONS)
        ob = min(o_buttons, self.MAX_BUTTONS) / float(self.MAX_BUTTONS)
        channels[7].fill(cb - ob)
        channels[8].fill((float(int(state[NEUTRAL])) / float(n)) if n > 0 else 0.0)

        bonus_owner = int(state[BONUS_OWNER])
        if bonus_owner == -1:
            channels[9].fill(0.0)
        elif bonus_owner == int(to_move):
            channels[9].fill(1.0)
        else:
            channels[9].fill(0.5)

        pending = int(state[PENDING_PATCHES])
        if pending > 0:
            channels[10].fill(min(pending, 5) / 5.0)

        for slot_idx in range(3):
            pid = get_slot_piece_id(state, slot_idx)
            if pid is not None:
                mask = self.piece_shape_masks.get(pid)
                if mask is not None:
                    channels[11 + slot_idx] = mask.copy()

        channels[14].fill(min(c_pos, self.MAX_POSITION) / float(self.MAX_POSITION))
        channels[15].fill(min(c_buttons, self.MAX_BUTTONS) / float(self.MAX_BUTTONS))

        return channels


def get_state_encoder_for_channels(num_channels: int):
    """Return encoder matching the given channel count (16, 32, 36, 56, or 60).

    32 / 56 → GoldV2StateEncoder in legacy 32ch mode (for iter192 and earlier).
    36 / 60 → GoldV2StateEncoder in new 36ch mode (gold_v2_36ch encoding).
    """
    n = int(num_channels)
    if n == 16:
        return LegacyStateEncoder()
    if n in (32, 56):
        return GoldV2StateEncoder("gold_v2_32ch")
    if n in (36, 60):
        return GoldV2StateEncoder()
    raise ValueError(f"No encoder for {num_channels} channels; expected 16, 32, 36, 56, or 60")


# =========================================================================
# Action Encoder
# =========================================================================

class ActionEncoder:
    """
    Slot-based action encoding.

    Index 0: Pass
    Indices 1-81: Patch placements (board_pos)
    Indices 82+: Buy actions (slot * 648 + orient * 81 + pos)

    Provides D4 augmentation methods for vertical and horizontal board flips.
    """

    BOARD_SIZE = 9
    NUM_SLOTS = 3
    NUM_ORIENTS = 8
    PASS_INDEX = 0
    PATCH_START = 1
    BUY_START = PATCH_START + BOARD_SIZE * BOARD_SIZE  # 82
    ACTION_SPACE_SIZE = BUY_START + NUM_SLOTS * NUM_ORIENTS * (BOARD_SIZE * BOARD_SIZE)  # 2026

    def __init__(self) -> None:
        self._vflip_patch_map = self._build_patch_flip_map("vertical")
        self._hflip_patch_map = self._build_patch_flip_map("horizontal")
        self._vflip_buy_map: Dict[int, Dict[int, int]] = {}
        self._hflip_buy_map: Dict[int, Dict[int, int]] = {}

    # ------------------------------------------------------------------
    # Core encode / decode (unchanged)
    # ------------------------------------------------------------------

    def encode_action(self, action: Tuple) -> int:
        """Encode action tuple to flat index."""
        if not action:
            raise ValueError("Empty action")

        if isinstance(action[0], (int, np.integer)):
            at = int(action[0])
            if at == AT_PASS:
                return self.PASS_INDEX
            if at == AT_PATCH:
                return self.PATCH_START + int(action[1])
            if at == AT_BUY:
                offset = int(action[1])
                orient = int(action[3])
                top = int(action[4])
                left = int(action[5])
                slot_index = offset - 1
                pos = top * self.BOARD_SIZE + left
                return self.BUY_START + (slot_index * self.NUM_ORIENTS + orient) * (self.BOARD_SIZE * self.BOARD_SIZE) + pos
            raise ValueError(f"Unknown action type {at}")

        action_type = action[0]
        if action_type == "pass":
            return self.PASS_INDEX
        if action_type == "patch":
            return self.PATCH_START + int(action[1])
        if action_type == "buy_slot":
            slot_index = int(action[1])
            orient = int(action[2])
            top = int(action[3])
            left = int(action[4])
            pos = top * self.BOARD_SIZE + left
            return self.BUY_START + (slot_index * self.NUM_ORIENTS + orient) * (self.BOARD_SIZE * self.BOARD_SIZE) + pos
        if action_type == "buy":
            offset = int(action[1])
            orient = int(action[3])
            top = int(action[4])
            left = int(action[5])
            slot_index = offset - 1
            pos = top * self.BOARD_SIZE + left
            return self.BUY_START + (slot_index * self.NUM_ORIENTS + orient) * (self.BOARD_SIZE * self.BOARD_SIZE) + pos
        raise ValueError(f"Unknown action type {action_type}")

    def decode_action(self, action_index: int) -> Tuple:
        """Decode action index to canonical tuple."""
        if action_index == self.PASS_INDEX:
            return ("pass",)
        if self.PATCH_START <= action_index < self.BUY_START:
            return ("patch", action_index - self.PATCH_START)
        if action_index >= self.BUY_START:
            rel = action_index - self.BUY_START
            slot_orient, pos = divmod(rel, self.BOARD_SIZE * self.BOARD_SIZE)
            slot_index, orient = divmod(slot_orient, self.NUM_ORIENTS)
            top, left = divmod(pos, self.BOARD_SIZE)
            return ("buy_slot", slot_index, orient, top, left)
        raise ValueError(f"Invalid action index: {action_index}")

    def encode_legal_actions(self, actions: List[Tuple]) -> Tuple[np.ndarray, np.ndarray]:
        """Encode list of legal actions to indices and mask."""
        indices = np.zeros((len(actions),), dtype=np.int32)
        mask = np.zeros((self.ACTION_SPACE_SIZE,), dtype=np.float32)
        for i, a in enumerate(actions):
            idx = self.encode_action(a)
            indices[i] = idx
            mask[idx] = 1.0
        return indices, mask

    def create_target_policy(
        self,
        visit_counts: Dict[Tuple, int],
        temperature: float = 1.0,
        mode: str = "visits",
    ) -> np.ndarray:
        """Create target policy from MCTS visit counts.

        Args:
            visit_counts: Map from action tuple to visit count.
            temperature: Used only when mode="visits_temperature_shaped". Must be > 0.
            mode: "visits" (default) = normalized counts pi(a)=N(a)/sum(N);
                  "visits_temperature_shaped" (legacy) = pi ∝ count^(1/T).

        Returns:
            Policy array (sum=1) of shape (ACTION_SPACE_SIZE,).
        """
        policy = np.zeros((self.ACTION_SPACE_SIZE,), dtype=np.float32)

        if mode == "visits":
            # Raw normalized visit counts — no temperature shaping
            total = sum(max(0, c) for c in visit_counts.values())
            if total <= 0:
                return policy
            for action, count in visit_counts.items():
                if count <= 0:
                    continue
                idx = self.encode_action(action)
                policy[idx] = float(count) / total
            return policy

        if mode == "visits_temperature_shaped":
            # Legacy: log(count)/T, softmax
            if temperature <= 0:
                raise ValueError(f"temperature must be positive for shaped mode, got {temperature}")
            logits = np.full((self.ACTION_SPACE_SIZE,), -np.inf, dtype=np.float64)
            for action, count in visit_counts.items():
                if count <= 0:
                    continue
                idx = self.encode_action(action)
                logits[idx] = math.log(float(count)) / temperature
            valid = np.isfinite(logits)
            if not np.any(valid):
                return policy
            logits_valid = logits[valid]
            logits_valid -= np.max(logits_valid)
            exp_logits = np.exp(logits_valid)
            policy[valid] = exp_logits.astype(np.float32)
            total = float(policy.sum())
            if total > 0:
                policy /= total
            return policy

        raise ValueError(f"policy_target_mode must be 'visits' or 'visits_temperature_shaped', got {mode!r}")

    # ------------------------------------------------------------------
    # D4 Augmentation: board flips
    # ------------------------------------------------------------------

    @staticmethod
    def _build_patch_flip_map(flip_type: str) -> np.ndarray:
        BS = 9
        mapping = np.zeros(BS * BS, dtype=np.int32)
        for pos in range(BS * BS):
            row, col = divmod(pos, BS)
            if flip_type == "vertical":
                new_row, new_col = BS - 1 - row, col
            else:
                new_row, new_col = row, BS - 1 - col
            mapping[pos] = new_row * BS + new_col
        return mapping

    def _get_buy_flip_index(
        self,
        slot_index: int,
        orient: int,
        top: int,
        left: int,
        piece_id: int,
        flip_type: str,
    ) -> int:
        aug = PIECE_AUGMENTATION.get(piece_id)
        if aug is None:
            return -1

        n_orient = aug["n_orient"]
        if orient >= n_orient:
            return -1

        if flip_type == "vertical":
            new_orient = aug["vflip_orient"][orient]
        else:
            new_orient = aug["hflip_orient"][orient]

        old_h = int(ORIENT_SHAPE_H[piece_id, orient])
        old_w = int(ORIENT_SHAPE_W[piece_id, orient])
        new_h = int(ORIENT_SHAPE_H[piece_id, new_orient])
        new_w = int(ORIENT_SHAPE_W[piece_id, new_orient])

        BS = self.BOARD_SIZE

        if flip_type == "vertical":
            new_top = BS - top - old_h
            new_left = left
        else:
            new_top = top
            new_left = BS - left - old_w

        if new_top < 0 or new_top + new_h > BS:
            return -1
        if new_left < 0 or new_left + new_w > BS:
            return -1

        new_pos = new_top * BS + new_left
        return (
            self.BUY_START
            + (slot_index * self.NUM_ORIENTS + new_orient) * (BS * BS)
            + new_pos
        )

    def flip_action_mask_v(self, mask: np.ndarray, slot_piece_ids: List[Optional[int]]) -> np.ndarray:
        return self._flip_action_vector(mask, slot_piece_ids, "vertical")

    def flip_policy_v(self, policy: np.ndarray, slot_piece_ids: List[Optional[int]]) -> np.ndarray:
        return self._flip_action_vector(policy, slot_piece_ids, "vertical")

    def flip_action_mask_h(self, mask: np.ndarray, slot_piece_ids: List[Optional[int]]) -> np.ndarray:
        return self._flip_action_vector(mask, slot_piece_ids, "horizontal")

    def flip_policy_h(self, policy: np.ndarray, slot_piece_ids: List[Optional[int]]) -> np.ndarray:
        return self._flip_action_vector(policy, slot_piece_ids, "horizontal")

    def _flip_action_vector(self, arr: np.ndarray, slot_piece_ids: List[Optional[int]], flip_type: str) -> np.ndarray:
        new_arr = np.zeros_like(arr)
        patch_map = self._vflip_patch_map if flip_type == "vertical" else self._hflip_patch_map

        new_arr[self.PASS_INDEX] = arr[self.PASS_INDEX]

        patch_vals = arr[self.PATCH_START:self.BUY_START]
        if np.any(patch_vals):
            for old_pos in range(len(patch_map)):
                val = patch_vals[old_pos]
                if val != 0:
                    new_pos = patch_map[old_pos]
                    new_arr[self.PATCH_START + new_pos] += val

        for old_idx in range(self.BUY_START, self.ACTION_SPACE_SIZE):
            val = arr[old_idx]
            if val == 0:
                continue

            rel = old_idx - self.BUY_START
            slot_orient, pos = divmod(rel, self.BOARD_SIZE * self.BOARD_SIZE)
            slot_index, orient = divmod(slot_orient, self.NUM_ORIENTS)
            top, left = divmod(pos, self.BOARD_SIZE)

            if slot_index >= len(slot_piece_ids):
                continue
            piece_id = slot_piece_ids[slot_index]
            if piece_id is None:
                continue

            new_idx = self._get_buy_flip_index(
                slot_index, orient, top, left, piece_id, flip_type
            )
            if new_idx != -1:
                new_arr[new_idx] += val

        return new_arr

    def _augment_flip(
        self,
        encoded_state: np.ndarray,
        policy: np.ndarray,
        action_mask: np.ndarray,
        slot_piece_ids: List[Optional[int]],
        flip_type: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply board flip. Flips all C_SPATIAL_ENC spatial channels (boards, coords, shapes, packing planes)."""
        SPATIAL = list(range(C_SPATIAL_ENC))  # all encoded spatial channels
        new_state = encoded_state.copy()
        ch_axis = 0 if flip_type == "vertical" else 1
        for ch in SPATIAL:
            new_state[ch] = np.flip(encoded_state[ch], axis=ch_axis).copy()

        new_policy = self._flip_action_vector(policy, slot_piece_ids, flip_type)
        new_mask = self._flip_action_vector(action_mask, slot_piece_ids, flip_type)

        total = new_policy.sum()
        if total > 0:
            new_policy /= total

        new_mask = (new_mask > 0).astype(np.float32)

        return new_state, new_policy, new_mask

    def augment_vertical_flip(
        self,
        encoded_state: np.ndarray,
        policy: np.ndarray,
        action_mask: np.ndarray,
        slot_piece_ids: List[Optional[int]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._augment_flip(
            encoded_state, policy, action_mask, slot_piece_ids, "vertical"
        )

    def augment_horizontal_flip(
        self,
        encoded_state: np.ndarray,
        policy: np.ndarray,
        action_mask: np.ndarray,
        slot_piece_ids: List[Optional[int]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._augment_flip(
            encoded_state, policy, action_mask, slot_piece_ids, "horizontal"
        )

    def augment_vh_flip(
        self,
        encoded_state: np.ndarray,
        policy: np.ndarray,
        action_mask: np.ndarray,
        slot_piece_ids: List[Optional[int]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        vs, vp, vm = self._augment_flip(
            encoded_state, policy, action_mask, slot_piece_ids, "vertical"
        )
        if vp.sum() == 0:
            return vs, vp, vm
        return self._augment_flip(vs, vp, vm, slot_piece_ids, "horizontal")
