#!/usr/bin/env python3
"""
Patchwork Game Engine (Flat Int32 State; Numba-optional)

This file is intended to be a *rules-faithful* in-process engine matching the
game-rule semantics implemented in `patchwork_api.py` (the HTTP oracle).

Key fidelity points (vs the prior engine version):
- Special-patch marker positions (revised edition): [26, 32, 38, 44, 50]
- `state_from_dict` mirrors API parsing behavior:
    * If `circle` omitted or empty => initialize shuffled circle (seeded if provided)
    * Back-compat heuristic: if a "fresh" game is sent with the default ordered circle,
      auto-randomize unless `randomize_circle` is explicitly false.
    * If `income` is missing for a player, it is inferred from board '2' cells.
- `new_game` supports `starting_player` and `edition`, consistent with API /new.
- Robust import: if Numba cannot be imported, the engine falls back to pure Python
  implementations (slower, but correct) so importing this module never hard-fails.

State layout (np.ndarray, dtype=int32, shape=(STATE_SIZE,)):
  Player 0: P0_POS, P0_BUTTONS, P0_INCOME, P0_OCC0, P0_OCC1, P0_OCC2
  Player 1: P1_POS, P1_BUTTONS, P1_INCOME, P1_OCC0, P1_OCC1, P1_OCC2
  Global:   CIRCLE_LEN, NEUTRAL, BONUS_OWNER, PENDING_PATCHES, PENDING_OWNER, TIE_PLAYER
  Circle array: CIRCLE_START .. CIRCLE_START+MAX_CIRCLE-1
  Edition:  EDITION_CODE (0=revised)

Action representation (numeric tuples, len=6):
  (AT_PASS, 0, 0, 0, 0, 0)
  (AT_PATCH, idx, 0, 0, 0, 0)
  (AT_BUY, offset, piece_id, orient, top, left)
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np

# =============================================================================
# Optional Numba acceleration
# =============================================================================

try:
    from numba import njit, types  # type: ignore
    from numba.typed import List as NList  # type: ignore
    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    _NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):  # type: ignore
        # no-op decorator fallback
        if args and callable(args[0]):
            return args[0]

        def _wrap(fn):
            return fn

        return _wrap

    types = None  # type: ignore
    NList = None  # type: ignore


# =============================================================================
# Constants
# =============================================================================

BOARD_SIZE: int = 9
BOARD_CELLS: int = BOARD_SIZE * BOARD_SIZE
SCOREBOARD_LENGTH: int = 53

BUTTONS_AFTER = np.asarray([5, 11, 17, 23, 29, 35, 41, 47, 53], dtype=np.int32)

# Special patch marker positions (revised edition only; original is obsolete)
EDITION_PATCHES: Dict[str, List[int]] = {
    "revised":  [26, 32, 38, 44, 50],
}
DEFAULT_EDITION: str = "revised"
EDITION_TO_CODE: Dict[str, int] = {"revised": 0}
CODE_TO_EDITION: Dict[int, str] = {0: "revised"}

PATCHES_AFTER = np.asarray(EDITION_PATCHES["revised"], dtype=np.int32)

AT_PASS: int = 0
AT_PATCH: int = 1
AT_BUY: int = 2

# State indices
P0_POS = 0
P0_BUTTONS = 1
P0_INCOME = 2
P0_OCC0 = 3
P0_OCC1 = 4
P0_OCC2 = 5

P1_POS = 6
P1_BUTTONS = 7
P1_INCOME = 8
P1_OCC0 = 9
P1_OCC1 = 10
P1_OCC2 = 11

CIRCLE_LEN = 12
NEUTRAL = 13
BONUS_OWNER = 14
PENDING_PATCHES = 15
PENDING_OWNER = 16
TIE_PLAYER = 17

CIRCLE_START = 18
MAX_CIRCLE = 33

# Prior versions used STATE_SIZE_BASE = CIRCLE_START + MAX_CIRCLE (51).
# We keep the same circle layout and append EDITION_CODE at the end for minimal disruption.
STATE_SIZE_BASE = CIRCLE_START + MAX_CIRCLE
EDITION_CODE = STATE_SIZE_BASE
STATE_SIZE = STATE_SIZE_BASE + 1

# Type alias
GameState = np.ndarray


# =============================================================================
# Piece definitions
# =============================================================================

PIECES: List[dict] = [
    {"id": 0, "cost_buttons": 5, "cost_time": 5, "shape": [[2, 2, 1], [0, 1, 0], [0, 1, 0]]},
    {"id": 1, "cost_buttons": 7, "cost_time": 2, "shape": [[2, 0, 0, 0], [2, 1, 1, 1], [1, 0, 0, 0]]},
    {"id": 2, "cost_buttons": 1, "cost_time": 2, "shape": [[0, 0, 0, 1], [1, 1, 1, 1], [1, 0, 0, 0]]},
    {"id": 3, "cost_buttons": 2, "cost_time": 1, "shape": [[0, 1, 0, 0], [1, 1, 1, 1], [0, 0, 1, 0]]},
    {"id": 4, "cost_buttons": 5, "cost_time": 4, "shape": [[0, 2, 0], [1, 2, 1], [0, 1, 0]]},
    {"id": 5, "cost_buttons": 0, "cost_time": 3, "shape": [[0, 1, 0, 0], [1, 2, 1, 1], [0, 1, 0, 0]]},
    {"id": 6, "cost_buttons": 1, "cost_time": 4, "shape": [[0, 0, 1, 0, 0], [1, 1, 2, 1, 1], [0, 0, 1, 0, 0]]},
    {"id": 7, "cost_buttons": 8, "cost_time": 6, "shape": [[2, 2, 0], [2, 1, 1], [0, 0, 1]]},
    {"id": 8, "cost_buttons": 2, "cost_time": 3, "shape": [[1, 0, 1], [1, 1, 1], [1, 0, 1]]},
    {"id": 9, "cost_buttons": 10, "cost_time": 4, "shape": [[2, 2, 0], [0, 2, 1], [0, 0, 1]]},
    {"id": 10, "cost_buttons": 3, "cost_time": 6, "shape": [[0, 2, 0], [2, 1, 1], [1, 0, 1]]},
    {"id": 11, "cost_buttons": 6, "cost_time": 5, "shape": [[2, 2], [1, 1]]},
    {"id": 12, "cost_buttons": 7, "cost_time": 1, "shape": [[2, 1, 1, 1, 1]]},
    {"id": 13, "cost_buttons": 3, "cost_time": 3, "shape": [[2, 1, 1, 1]]},
    {"id": 14, "cost_buttons": 2, "cost_time": 2, "shape": [[1, 1, 1]]},
    {"id": 15, "cost_buttons": 1, "cost_time": 5, "shape": [[2, 1, 1, 1], [1, 0, 0, 1]]},
    {"id": 16, "cost_buttons": 3, "cost_time": 4, "shape": [[0, 0, 2, 0], [1, 1, 1, 1]]},
    {"id": 17, "cost_buttons": 7, "cost_time": 4, "shape": [[0, 2, 2, 0], [1, 1, 1, 1]]},
    {"id": 18, "cost_buttons": 2, "cost_time": 2, "shape": [[0, 1, 0], [1, 1, 1]]},
    {"id": 19, "cost_buttons": 2, "cost_time": 2, "shape": [[1, 1, 0], [1, 1, 1]]},
    {"id": 20, "cost_buttons": 10, "cost_time": 3, "shape": [[2, 0, 0, 0], [2, 1, 1, 1]]},
    {"id": 21, "cost_buttons": 10, "cost_time": 5, "shape": [[2, 2, 0, 0], [2, 1, 1, 1]]},
    {"id": 22, "cost_buttons": 3, "cost_time": 2, "shape": [[2, 1, 0], [0, 1, 1]]},
    {"id": 23, "cost_buttons": 7, "cost_time": 6, "shape": [[2, 2, 0], [0, 2, 1]]},
    {"id": 24, "cost_buttons": 4, "cost_time": 2, "shape": [[1, 1, 1, 0], [0, 1, 1, 1]]},
    {"id": 25, "cost_buttons": 2, "cost_time": 3, "shape": [[2, 1, 1, 0], [0, 0, 1, 1]]},
    {"id": 26, "cost_buttons": 1, "cost_time": 2, "shape": [[1, 1, 1], [1, 0, 1]]},
    {"id": 27, "cost_buttons": 4, "cost_time": 6, "shape": [[2, 0, 0], [2, 1, 1]]},
    {"id": 28, "cost_buttons": 4, "cost_time": 2, "shape": [[2, 0, 0], [1, 1, 1]]},
    {"id": 29, "cost_buttons": 1, "cost_time": 3, "shape": [[1, 0], [1, 1]]},
    {"id": 30, "cost_buttons": 3, "cost_time": 1, "shape": [[1, 0], [1, 1]]},
    {"id": 31, "cost_buttons": 5, "cost_time": 3, "shape": [[0, 2, 1, 0], [1, 1, 1, 1], [0, 1, 1, 0]]},
    {"id": 32, "cost_buttons": 2, "cost_time": 1, "shape": [[1, 1]]},
]
PIECE_BY_ID: Dict[int, dict] = {int(p["id"]): p for p in PIECES}

N_PIECES: int = len(PIECES)
MAX_ORIENTS: int = 8

PIECE_COST_BUTTONS = np.zeros(N_PIECES, dtype=np.int32)
PIECE_COST_TIME = np.zeros(N_PIECES, dtype=np.int32)

ORIENT_COUNT = np.zeros(N_PIECES, dtype=np.int32)
ORIENT_INCOME = np.zeros((N_PIECES, MAX_ORIENTS), dtype=np.int32)

MASK_W0 = np.zeros((N_PIECES, MAX_ORIENTS, BOARD_CELLS), dtype=np.uint32)
MASK_W1 = np.zeros((N_PIECES, MAX_ORIENTS, BOARD_CELLS), dtype=np.uint32)
MASK_W2 = np.zeros((N_PIECES, MAX_ORIENTS, BOARD_CELLS), dtype=np.uint32)

SEVEN_MASK_W0 = np.zeros(9, dtype=np.uint32)
SEVEN_MASK_W1 = np.zeros(9, dtype=np.uint32)
SEVEN_MASK_W2 = np.zeros(9, dtype=np.uint32)

# Per-orient bounding box: ORIENT_HEIGHT[pid, orient], ORIENT_WIDTH[pid, orient]
ORIENT_HEIGHT = np.zeros((N_PIECES, MAX_ORIENTS), dtype=np.int32)
ORIENT_WIDTH = np.zeros((N_PIECES, MAX_ORIENTS), dtype=np.int32)

# Per-orient FULL shape-array dimensions (len(shape), len(shape[0])).
ORIENT_SHAPE_H = np.zeros((N_PIECES, MAX_ORIENTS), dtype=np.int32)
ORIENT_SHAPE_W = np.zeros((N_PIECES, MAX_ORIENTS), dtype=np.int32)


def _rotate_ccw(shape):
    rows, cols = len(shape), len(shape[0])
    return [[shape[r][c] for r in range(rows)] for c in range(cols - 1, -1, -1)]


def _flip_vertical(shape):
    return list(reversed([row[:] for row in shape]))


def _flip_horizontal(shape):
    return [list(reversed(row)) for row in shape]


def _canonical_shape(shape):
    return tuple(tuple(int(v) for v in row) for row in shape)


def _canonical_cells(shape):
    """Canonical form as frozenset of (r,c) filled-cell offsets, origin-normalised."""
    cells = set()
    for r, row in enumerate(shape):
        for c, v in enumerate(row):
            if int(v) != 0:
                cells.add((r, c))
    if not cells:
        return frozenset()
    min_r = min(r for r, c in cells)
    min_c = min(c for r, c in cells)
    return frozenset((r - min_r, c - min_c) for r, c in cells)


def _shape_dims(shape):
    """Bounding box (h, w) of filled cells."""
    cells = [(r, c) for r, row in enumerate(shape) for c, v in enumerate(row) if int(v) != 0]
    if not cells:
        return 0, 0
    return (
        max(r for r, c in cells) - min(r for r, c in cells) + 1,
        max(c for r, c in cells) - min(c for r, c in cells) + 1,
    )


def _mask_words_for_cells(cells, top, left):
    w0 = np.uint32(0)
    w1 = np.uint32(0)
    w2 = np.uint32(0)
    for (r, c) in cells:
        idx = (top + r) * BOARD_SIZE + (left + c)
        word = idx >> 5
        bit = idx & 31
        b = np.uint32(1) << np.uint32(bit)
        if word == 0:
            w0 |= b
        elif word == 1:
            w1 |= b
        else:
            w2 |= b
    return w0, w1, w2


def _get_orient_shapes(base_shape):
    """Get list of unique oriented shapes in canonical engine/API order."""
    seen = set()
    shapes = []
    cur = [row[:] for row in base_shape]
    for _ in range(4):
        for f in (False, True):
            s = _flip_vertical(cur) if f else [row[:] for row in cur]
            key = _canonical_shape(s)
            if key not in seen:
                seen.add(key)
                shapes.append(s)
        cur = _rotate_ccw(cur)
    return shapes


def _build_tables():
    for p in PIECES:
        pid = int(p["id"])
        PIECE_COST_BUTTONS[pid] = int(p["cost_buttons"])
        PIECE_COST_TIME[pid] = int(p["cost_time"])

    k = 0
    for top in range(3):
        for left in range(3):
            cells = [(r, c) for r in range(7) for c in range(7)]
            w0, w1, w2 = _mask_words_for_cells(cells, top, left)
            SEVEN_MASK_W0[k] = w0
            SEVEN_MASK_W1[k] = w1
            SEVEN_MASK_W2[k] = w2
            k += 1

    for p in PIECES:
        pid = int(p["id"])
        shapes = _get_orient_shapes(p["shape"])
        ORIENT_COUNT[pid] = len(shapes)
        for o_idx, s in enumerate(shapes):
            h, w = len(s), len(s[0])
            if h > BOARD_SIZE or w > BOARD_SIZE:
                continue
            cells = []
            income = 0
            for r in range(h):
                for c in range(w):
                    v = int(s[r][c])
                    if v != 0:
                        cells.append((r, c))
                        if v == 2:
                            income += 1
            ORIENT_INCOME[pid, o_idx] = income
            bh, bw = _shape_dims(s)
            ORIENT_HEIGHT[pid, o_idx] = bh
            ORIENT_WIDTH[pid, o_idx] = bw
            ORIENT_SHAPE_H[pid, o_idx] = h
            ORIENT_SHAPE_W[pid, o_idx] = w
            for top in range(BOARD_SIZE - h + 1):
                for left in range(BOARD_SIZE - w + 1):
                    pos = top * BOARD_SIZE + left
                    w0, w1, w2 = _mask_words_for_cells(cells, top, left)
                    MASK_W0[pid, o_idx, pos] = w0
                    MASK_W1[pid, o_idx, pos] = w1
                    MASK_W2[pid, o_idx, pos] = w2


_build_tables()


# =============================================================================
# Piece augmentation metadata (used externally; preserved)
# =============================================================================

def _build_augmentation_tables() -> Dict[int, dict]:
    tables: Dict[int, dict] = {}
    for p in PIECES:
        pid = int(p["id"])
        shapes = _get_orient_shapes(p["shape"])
        orient_keys = [_canonical_cells(s) for s in shapes]
        n = len(shapes)
        dims = [_shape_dims(s) for s in shapes]

        vflip_map = []
        hflip_map = []
        for s in shapes:
            vf = _flip_vertical(s)
            vk = _canonical_cells(vf)
            vflip_map.append(orient_keys.index(vk))

            hf = _flip_horizontal(s)
            hk = _canonical_cells(hf)
            hflip_map.append(orient_keys.index(hk))

        tables[pid] = {
            "n_orient": n,
            "dims": dims,
            "vflip_orient": vflip_map,
            "hflip_orient": hflip_map,
        }
    return tables


PIECE_AUGMENTATION: Dict[int, dict] = _build_augmentation_tables()


# =============================================================================
# State upgrade (backwards compatibility)
# =============================================================================

def upgrade_state(state: np.ndarray) -> np.ndarray:
    """Upgrade older (length=STATE_SIZE_BASE) states to the new layout with EDITION_CODE."""
    if not isinstance(state, np.ndarray) or state.dtype != np.int32 or state.ndim != 1:
        raise ValueError("state must be a 1D np.ndarray(dtype=int32)")
    if state.shape[0] == STATE_SIZE:
        return state
    if state.shape[0] == STATE_SIZE_BASE:
        s2 = np.full((STATE_SIZE,), -1, dtype=np.int32)
        s2[:STATE_SIZE_BASE] = state
        s2[EDITION_CODE] = np.int32(EDITION_TO_CODE[DEFAULT_EDITION])
        return s2
    raise ValueError(f"Unexpected state size {state.shape[0]} (expected {STATE_SIZE_BASE} or {STATE_SIZE})")


def _edition_code(state: np.ndarray) -> int:
    if state.shape[0] == STATE_SIZE_BASE:
        return EDITION_TO_CODE[DEFAULT_EDITION]
    code = int(state[EDITION_CODE])
    return code if code in CODE_TO_EDITION else EDITION_TO_CODE[DEFAULT_EDITION]


# =============================================================================
# Pure-Python helpers (also used by validators)
# =============================================================================

def _player_base_py(player_idx: int) -> int:
    return 0 if int(player_idx) == 0 else 6


def _get_occ_words_py(state: np.ndarray, player_idx: int) -> Tuple[np.uint32, np.uint32, np.uint32]:
    base = _player_base_py(player_idx)
    return (np.uint32(state[base + 3]), np.uint32(state[base + 4]), np.uint32(state[base + 5]))


def _is_bit_set_py(occ0: np.uint32, occ1: np.uint32, occ2: np.uint32, idx: int) -> bool:
    i = int(idx)
    word = i >> 5
    bit = i & 31
    mask = np.uint32(1) << np.uint32(bit)
    if word == 0:
        return (occ0 & mask) != np.uint32(0)
    if word == 1:
        return (occ1 & mask) != np.uint32(0)
    return (occ2 & mask) != np.uint32(0)


def _empty_count_words_py(occ0: np.uint32, occ1: np.uint32, occ2: np.uint32) -> int:
    filled = int(int(occ0).bit_count() + int(occ1).bit_count() + int(occ2).bit_count())
    return BOARD_CELLS - filled


def _has_seven_by_seven_py(occ0: np.uint32, occ1: np.uint32, occ2: np.uint32) -> bool:
    for i in range(9):
        m0 = SEVEN_MASK_W0[i]
        m1 = SEVEN_MASK_W1[i]
        m2 = SEVEN_MASK_W2[i]
        if (occ0 & m0) == m0 and (occ1 & m1) == m1 and (occ2 & m2) == m2:
            return True
    return False


def _button_marks_crossed_py(prev_pos: int, new_pos: int) -> int:
    return int(sum(1 for m in BUTTONS_AFTER.tolist() if int(prev_pos) < int(m) <= int(new_pos)))


def _patches_crossed_py(prev_pos: int, new_pos: int, opponent_pos: int, edition_code: int) -> int:
    return int(sum(1 for m in PATCHES_AFTER.tolist() if int(prev_pos) < int(m) <= int(new_pos) and int(opponent_pos) < int(m)))


def _clamp_pos_py(pos: int) -> int:
    # Match patchwork_api.py: only clamp overrun beyond SCOREBOARD_LENGTH
    return SCOREBOARD_LENGTH if int(pos) > SCOREBOARD_LENGTH else int(pos)


# =============================================================================
# Numba (or Python fallback) implementations
# =============================================================================

if _NUMBA_AVAILABLE:
    @njit(cache=True, fastmath=True)
    def _popcount32_nb(x):
        v = x
        v = v - ((v >> np.uint32(1)) & np.uint32(0x55555555))
        v = (v & np.uint32(0x33333333)) + ((v >> np.uint32(2)) & np.uint32(0x33333333))
        v = (v + (v >> np.uint32(4))) & np.uint32(0x0F0F0F0F)
        v = v + (v >> np.uint32(8))
        v = v + (v >> np.uint32(16))
        return np.int32(v & np.uint32(0x3F))


    @njit(cache=True, fastmath=True)
    def _empty_count_words_nb(occ0, occ1, occ2):
        filled = _popcount32_nb(occ0) + _popcount32_nb(occ1) + _popcount32_nb(occ2)
        return np.int32(BOARD_CELLS - int(filled))


    @njit(cache=True, fastmath=True)
    def _is_bit_set_nb(occ0, occ1, occ2, idx):
        i = int(idx)
        word = i >> 5
        bit = i & 31
        mask = np.uint32(1) << np.uint32(bit)
        if word == 0:
            return (occ0 & mask) != np.uint32(0)
        if word == 1:
            return (occ1 & mask) != np.uint32(0)
        return (occ2 & mask) != np.uint32(0)


    @njit(cache=True, fastmath=True)
    def _set_bit_nb(occ0, occ1, occ2, idx):
        i = int(idx)
        word = i >> 5
        bit = i & 31
        mask = np.uint32(1) << np.uint32(bit)
        if word == 0:
            occ0 |= mask
        elif word == 1:
            occ1 |= mask
        else:
            occ2 |= mask
        return occ0, occ1, occ2


    @njit(cache=True, fastmath=True)
    def _has_seven_by_seven_nb(occ0, occ1, occ2):
        for i in range(9):
            m0 = SEVEN_MASK_W0[i]
            m1 = SEVEN_MASK_W1[i]
            m2 = SEVEN_MASK_W2[i]
            if (occ0 & m0) == m0 and (occ1 & m1) == m1 and (occ2 & m2) == m2:
                return True
        return False


    @njit(cache=True, fastmath=True)
    def _clamp_pos_nb(pos):
        return np.int32(SCOREBOARD_LENGTH if pos > SCOREBOARD_LENGTH else pos)


    @njit(cache=True, fastmath=True)
    def _button_marks_crossed_nb(prev_pos, new_pos):
        cnt = 0
        for i in range(BUTTONS_AFTER.shape[0]):
            m = BUTTONS_AFTER[i]
            if prev_pos < m <= new_pos:
                cnt += 1
        return np.int32(cnt)


    @njit(cache=True, fastmath=True)
    def _patches_crossed_nb(prev_pos, new_pos, opponent_pos, edition_code):
        cnt = 0
        for i in range(PATCHES_AFTER.shape[0]):
            m = PATCHES_AFTER[i]
            if prev_pos < m <= new_pos and opponent_pos < m:
                cnt += 1
        return np.int32(cnt)


    @njit(cache=True, fastmath=True)
    def _player_base_nb(player_idx):
        return np.int32(0) if player_idx == 0 else np.int32(6)


    @njit(cache=True, fastmath=True)
    def _get_occ_words_nb(state, player_idx):
        base = _player_base_nb(player_idx)
        return (np.uint32(state[base + 3]), np.uint32(state[base + 4]), np.uint32(state[base + 5]))


    @njit(cache=True, fastmath=True)
    def _set_occ_words_nb(state, player_idx, occ0, occ1, occ2):
        base = _player_base_nb(player_idx)
        state[base + 3] = np.int32(occ0)
        state[base + 4] = np.int32(occ1)
        state[base + 5] = np.int32(occ2)


    @njit(cache=True, fastmath=True)
    def _current_player_nb(state):
        if state[PENDING_PATCHES] > 0:
            return np.int32(state[PENDING_OWNER])
        p0_pos = state[P0_POS]
        p1_pos = state[P1_POS]
        if p0_pos < p1_pos:
            return np.int32(0)
        if p1_pos < p0_pos:
            return np.int32(1)
        return np.int32(state[TIE_PLAYER])


    @njit(cache=True, fastmath=True)
    def _terminal_nb(state):
        return (
            state[P0_POS] >= SCOREBOARD_LENGTH
            and state[P1_POS] >= SCOREBOARD_LENGTH
            and state[PENDING_PATCHES] == 0
        )


    _ACTION_TUPLE = types.UniTuple(types.int64, 6)


    @njit(cache=True, fastmath=True)
    def _legal_actions_raw_nb(state):
        actions = NList.empty_list(_ACTION_TUPLE)
        if _terminal_nb(state):
            return actions

        pl = _current_player_nb(state)

        if state[PENDING_PATCHES] > 0:
            occ0, occ1, occ2 = _get_occ_words_nb(state, pl)
            for idx in range(BOARD_CELLS):
                if not _is_bit_set_nb(occ0, occ1, occ2, np.int32(idx)):
                    actions.append((AT_PATCH, idx, 0, 0, 0, 0))
            # CRITICAL FIX: If board is completely full, allow discarding patch at idx 0
            if len(actions) == 0:
                actions.append((AT_PATCH, 0, 0, 0, 0, 0))
            return actions

        base = _player_base_nb(pl)
        pos = state[base + 0]
        buttons = state[base + 1]

        if pos < SCOREBOARD_LENGTH:
            actions.append((AT_PASS, 0, 0, 0, 0, 0))

        n = int(state[CIRCLE_LEN])
        if n == 0:
            return actions

        neutral = int(state[NEUTRAL])
        max_pick = 3 if n >= 3 else n
        occ0, occ1, occ2 = _get_occ_words_nb(state, pl)

        for offset in range(1, max_pick + 1):
            piece_idx = (neutral + offset) % n
            piece_id = int(state[CIRCLE_START + piece_idx])
            if buttons < PIECE_COST_BUTTONS[piece_id]:
                continue
            ocount = int(ORIENT_COUNT[piece_id])
            for orient in range(ocount):
                for pos_idx in range(BOARD_CELLS):
                    m0 = MASK_W0[piece_id, orient, pos_idx]
                    m1 = MASK_W1[piece_id, orient, pos_idx]
                    m2 = MASK_W2[piece_id, orient, pos_idx]
                    if (m0 | m1 | m2) == np.uint32(0):
                        continue
                    if (occ0 & m0) != np.uint32(0) or (occ1 & m1) != np.uint32(0) or (occ2 & m2) != np.uint32(0):
                        continue
                    top = pos_idx // BOARD_SIZE
                    left = pos_idx - top * BOARD_SIZE
                    actions.append((AT_BUY, offset, piece_id, orient, top, left))
        return actions


    @njit(cache=True, fastmath=True)
    def _maybe_claim_bonus_nb(occ0, occ1, occ2, bonus_owner, pl):
        if bonus_owner != -1:
            return bonus_owner
        return pl if _has_seven_by_seven_nb(occ0, occ1, occ2) else bonus_owner


    @njit(cache=True, fastmath=True)
    def _apply_action_nb(state, action):
        if _terminal_nb(state):
            return state.copy()

        a0 = int(action[0])
        pl = _current_player_nb(state)
        pb = _player_base_nb(pl)
        ob = _player_base_nb(np.int32(1 - int(pl)))
        new_state = state.copy()

        p_pos = new_state[pb + 0]
        p_buttons = new_state[pb + 1]
        p_income = new_state[pb + 2]
        o_pos = new_state[ob + 0]

        p_occ0 = np.uint32(new_state[pb + 3])
        p_occ1 = np.uint32(new_state[pb + 4])
        p_occ2 = np.uint32(new_state[pb + 5])

        ed = np.int32(new_state[EDITION_CODE])

        if a0 == AT_PATCH:
            idx = np.int32(action[1])
            p_occ0, p_occ1, p_occ2 = _set_bit_nb(p_occ0, p_occ1, p_occ2, idx)
            _set_occ_words_nb(new_state, pl, p_occ0, p_occ1, p_occ2)
            new_state[BONUS_OWNER] = _maybe_claim_bonus_nb(p_occ0, p_occ1, p_occ2, np.int32(new_state[BONUS_OWNER]), pl)
            new_pending = np.int32(new_state[PENDING_PATCHES] - 1)
            new_state[PENDING_PATCHES] = new_pending
            new_state[PENDING_OWNER] = np.int32(new_state[PENDING_OWNER]) if new_pending > 0 else np.int32(-1)
            return new_state

        if a0 == AT_PASS:
            prev = np.int32(p_pos)
            new_pos = _clamp_pos_nb(np.int32(o_pos + 1))
            steps = np.int32(new_pos - prev)
            new_buttons = np.int32(p_buttons + steps)
            income_marks = _button_marks_crossed_nb(prev, new_pos)
            if income_marks > 0:
                new_buttons = np.int32(new_buttons + income_marks * np.int32(p_income))
            gained_patches = _patches_crossed_nb(prev, new_pos, np.int32(o_pos), ed)
            tie_player = np.int32(new_state[TIE_PLAYER])
            if new_pos == np.int32(o_pos):
                tie_player = pl
            new_state[pb + 0] = np.int32(new_pos)
            new_state[pb + 1] = np.int32(new_buttons)
            new_state[pb + 2] = np.int32(p_income)
            new_state[PENDING_PATCHES] = np.int32(gained_patches)
            new_state[PENDING_OWNER] = pl if gained_patches > 0 else np.int32(-1)
            new_state[TIE_PLAYER] = tie_player
            return new_state

        if a0 == AT_BUY:
            offset = int(action[1])
            piece_id = int(action[2])
            orient = int(action[3])
            top = int(action[4])
            left = int(action[5])

            n = int(new_state[CIRCLE_LEN])
            idx = (int(new_state[NEUTRAL]) + offset) % n
            cost_buttons = PIECE_COST_BUTTONS[piece_id]
            cost_time = PIECE_COST_TIME[piece_id]

            pos_idx = top * BOARD_SIZE + left
            m0 = MASK_W0[piece_id, orient, pos_idx]
            m1 = MASK_W1[piece_id, orient, pos_idx]
            m2 = MASK_W2[piece_id, orient, pos_idx]

            p_occ0_2 = p_occ0 | m0
            p_occ1_2 = p_occ1 | m1
            p_occ2_2 = p_occ2 | m2
            income2 = np.int32(p_income + ORIENT_INCOME[piece_id, orient])

            for j in range(idx, n - 1):
                new_state[CIRCLE_START + j] = new_state[CIRCLE_START + j + 1]
            new_state[CIRCLE_START + (n - 1)] = np.int32(-1)
            new_state[CIRCLE_LEN] = np.int32(n - 1)

            new_len = n - 1
            if new_len > 0:
                new_state[NEUTRAL] = np.int32((idx - 1) % new_len)
            else:
                new_state[NEUTRAL] = np.int32(0)

            prev = np.int32(p_pos)
            new_pos = _clamp_pos_nb(np.int32(p_pos + cost_time))
            new_buttons = np.int32(p_buttons - cost_buttons)
            income_marks = _button_marks_crossed_nb(prev, new_pos)
            if income_marks > 0:
                new_buttons = np.int32(new_buttons + income_marks * income2)
            gained_patches = _patches_crossed_nb(prev, new_pos, np.int32(o_pos), ed)
            new_state[BONUS_OWNER] = _maybe_claim_bonus_nb(p_occ0_2, p_occ1_2, p_occ2_2, np.int32(new_state[BONUS_OWNER]), pl)

            tie_player = np.int32(new_state[TIE_PLAYER])
            if new_pos == np.int32(o_pos):
                tie_player = pl
            new_state[pb + 0] = np.int32(new_pos)
            new_state[pb + 1] = np.int32(new_buttons)
            new_state[pb + 2] = income2
            _set_occ_words_nb(new_state, pl, p_occ0_2, p_occ1_2, p_occ2_2)
            new_state[PENDING_PATCHES] = np.int32(gained_patches)
            new_state[PENDING_OWNER] = pl if gained_patches > 0 else np.int32(-1)
            new_state[TIE_PLAYER] = tie_player
            return new_state

        return new_state

else:
    # -------------------------
    # Pure Python fallbacks
    # -------------------------

    def _current_player_nb(state: np.ndarray) -> np.int32:
        if int(state[PENDING_PATCHES]) > 0:
            return np.int32(int(state[PENDING_OWNER]))
        p0_pos = int(state[P0_POS])
        p1_pos = int(state[P1_POS])
        if p0_pos < p1_pos:
            return np.int32(0)
        if p1_pos < p0_pos:
            return np.int32(1)
        return np.int32(int(state[TIE_PLAYER]))


    def _terminal_nb(state: np.ndarray) -> bool:
        return bool(
            int(state[P0_POS]) >= SCOREBOARD_LENGTH
            and int(state[P1_POS]) >= SCOREBOARD_LENGTH
            and int(state[PENDING_PATCHES]) == 0
        )


    def _legal_actions_raw_nb(state: np.ndarray):
        actions: List[Tuple[int, int, int, int, int, int]] = []
        if _terminal_nb(state):
            return actions

        pl = int(_current_player_nb(state))

        if int(state[PENDING_PATCHES]) > 0:
            occ0, occ1, occ2 = _get_occ_words_py(state, pl)
            for idx in range(BOARD_CELLS):
                if not _is_bit_set_py(occ0, occ1, occ2, idx):
                    actions.append((AT_PATCH, idx, 0, 0, 0, 0))
            # CRITICAL FIX: If board is completely full, allow discarding patch at idx 0
            if len(actions) == 0:
                actions.append((AT_PATCH, 0, 0, 0, 0, 0))  # BUGFIX: Use 0 not idx
            return actions

        base = _player_base_py(pl)
        pos = int(state[base + 0])
        buttons = int(state[base + 1])

        if pos < SCOREBOARD_LENGTH:
            actions.append((AT_PASS, 0, 0, 0, 0, 0))

        n = int(state[CIRCLE_LEN])
        if n == 0:
            return actions

        neutral = int(state[NEUTRAL])
        max_pick = 3 if n >= 3 else n
        occ0, occ1, occ2 = _get_occ_words_py(state, pl)

        for offset in range(1, max_pick + 1):
            piece_idx = (neutral + offset) % n
            piece_id = int(state[CIRCLE_START + piece_idx])
            if buttons < int(PIECE_COST_BUTTONS[piece_id]):
                continue
            ocount = int(ORIENT_COUNT[piece_id])
            for orient in range(ocount):
                for pos_idx in range(BOARD_CELLS):
                    m0 = MASK_W0[piece_id, orient, pos_idx]
                    m1 = MASK_W1[piece_id, orient, pos_idx]
                    m2 = MASK_W2[piece_id, orient, pos_idx]
                    if int(m0 | m1 | m2) == 0:
                        continue
                    if (occ0 & m0) != np.uint32(0) or (occ1 & m1) != np.uint32(0) or (occ2 & m2) != np.uint32(0):
                        continue
                    top = pos_idx // BOARD_SIZE
                    left = pos_idx - top * BOARD_SIZE
                    actions.append((AT_BUY, offset, piece_id, orient, top, left))
        return actions


    def _maybe_claim_bonus_py(occ0, occ1, occ2, bonus_owner, pl):
        if int(bonus_owner) != -1:
            return np.int32(int(bonus_owner))
        return np.int32(pl) if _has_seven_by_seven_py(occ0, occ1, occ2) else np.int32(int(bonus_owner))


    def _apply_action_nb(state: np.ndarray, action):
        if _terminal_nb(state):
            return state.copy()

        a0 = int(action[0])
        pl = int(_current_player_nb(state))
        pb = _player_base_py(pl)
        ob = _player_base_py(1 - pl)
        new_state = state.copy()

        p_pos = int(new_state[pb + 0])
        p_buttons = int(new_state[pb + 1])
        p_income = int(new_state[pb + 2])
        o_pos = int(new_state[ob + 0])

        p_occ0 = np.uint32(new_state[pb + 3])
        p_occ1 = np.uint32(new_state[pb + 4])
        p_occ2 = np.uint32(new_state[pb + 5])

        ed = int(new_state[EDITION_CODE])

        if a0 == AT_PATCH:
            idx = int(action[1])
            # set bit
            word = idx >> 5
            bit = idx & 31
            mask = np.uint32(1) << np.uint32(bit)
            if word == 0:
                p_occ0 |= mask
            elif word == 1:
                p_occ1 |= mask
            else:
                p_occ2 |= mask

            new_state[pb + 3] = np.int32(p_occ0)
            new_state[pb + 4] = np.int32(p_occ1)
            new_state[pb + 5] = np.int32(p_occ2)

            new_state[BONUS_OWNER] = _maybe_claim_bonus_py(p_occ0, p_occ1, p_occ2, new_state[BONUS_OWNER], pl)
            new_pending = int(new_state[PENDING_PATCHES]) - 1
            new_state[PENDING_PATCHES] = np.int32(new_pending)
            new_state[PENDING_OWNER] = np.int32(int(new_state[PENDING_OWNER]) if new_pending > 0 else -1)
            return new_state

        if a0 == AT_PASS:
            prev = p_pos
            new_pos = _clamp_pos_py(o_pos + 1)
            steps = new_pos - prev
            new_buttons = p_buttons + steps
            income_marks = _button_marks_crossed_py(prev, new_pos)
            if income_marks > 0:
                new_buttons += income_marks * p_income
            gained_patches = _patches_crossed_py(prev, new_pos, o_pos, ed)
            tie_player = int(new_state[TIE_PLAYER])
            if new_pos == o_pos:
                tie_player = pl
            new_state[pb + 0] = np.int32(new_pos)
            new_state[pb + 1] = np.int32(new_buttons)
            new_state[pb + 2] = np.int32(p_income)
            new_state[PENDING_PATCHES] = np.int32(gained_patches)
            new_state[PENDING_OWNER] = np.int32(pl if gained_patches > 0 else -1)
            new_state[TIE_PLAYER] = np.int32(tie_player)
            return new_state

        if a0 == AT_BUY:
            offset = int(action[1])
            piece_id = int(action[2])
            orient = int(action[3])
            top = int(action[4])
            left = int(action[5])

            n = int(new_state[CIRCLE_LEN])
            idx = (int(new_state[NEUTRAL]) + offset) % n
            cost_buttons = int(PIECE_COST_BUTTONS[piece_id])
            cost_time = int(PIECE_COST_TIME[piece_id])

            pos_idx = top * BOARD_SIZE + left
            m0 = MASK_W0[piece_id, orient, pos_idx]
            m1 = MASK_W1[piece_id, orient, pos_idx]
            m2 = MASK_W2[piece_id, orient, pos_idx]

            p_occ0_2 = p_occ0 | m0
            p_occ1_2 = p_occ1 | m1
            p_occ2_2 = p_occ2 | m2
            income2 = p_income + int(ORIENT_INCOME[piece_id, orient])

            # remove from circle
            for j in range(idx, n - 1):
                new_state[CIRCLE_START + j] = new_state[CIRCLE_START + j + 1]
            new_state[CIRCLE_START + (n - 1)] = np.int32(-1)
            new_state[CIRCLE_LEN] = np.int32(n - 1)

            new_len = n - 1
            new_state[NEUTRAL] = np.int32((idx - 1) % new_len) if new_len > 0 else np.int32(0)

            prev = p_pos
            new_pos = _clamp_pos_py(p_pos + cost_time)
            new_buttons = p_buttons - cost_buttons
            income_marks = _button_marks_crossed_py(prev, new_pos)
            if income_marks > 0:
                new_buttons += income_marks * income2
            gained_patches = _patches_crossed_py(prev, new_pos, o_pos, ed)

            new_state[BONUS_OWNER] = _maybe_claim_bonus_py(p_occ0_2, p_occ1_2, p_occ2_2, new_state[BONUS_OWNER], pl)

            tie_player = int(new_state[TIE_PLAYER])
            if new_pos == o_pos:
                tie_player = pl

            new_state[pb + 0] = np.int32(new_pos)
            new_state[pb + 1] = np.int32(new_buttons)
            new_state[pb + 2] = np.int32(income2)
            new_state[pb + 3] = np.int32(p_occ0_2)
            new_state[pb + 4] = np.int32(p_occ1_2)
            new_state[pb + 5] = np.int32(p_occ2_2)
            new_state[PENDING_PATCHES] = np.int32(gained_patches)
            new_state[PENDING_OWNER] = np.int32(pl if gained_patches > 0 else -1)
            new_state[TIE_PLAYER] = np.int32(tie_player)
            return new_state

        return new_state


# =============================================================================
# Public API (upgrade + strict validation)
# =============================================================================

def current_player(state: np.ndarray) -> int:
    s = upgrade_state(state)
    return int(_current_player_nb(s))


def terminal(state: np.ndarray) -> bool:
    s = upgrade_state(state)
    return bool(_terminal_nb(s))


def legal_actions_raw(state: np.ndarray):
    s = upgrade_state(state)
    return _legal_actions_raw_nb(s)


def legal_actions_list(state: np.ndarray) -> List[Tuple[int, ...]]:
    return list(legal_actions_raw(state))


def _validate_action_py(state: np.ndarray, action) -> None:
    """Paranoid validator (mirrors patchwork_api.py apply_action checks)."""
    if terminal(state):
        return

    if action is None or len(action) < 1:
        raise ValueError("Invalid action")

    a0 = int(action[0])
    pl = current_player(state)

    # Pending patch placement has priority
    if int(state[PENDING_PATCHES]) > 0:
        if a0 != AT_PATCH:
            raise ValueError("Must place patches first")
        if int(state[PENDING_OWNER]) != pl:
            raise ValueError("Not the pending patch owner")

    if a0 == AT_PATCH:
        if int(state[PENDING_PATCHES]) <= 0:
            raise ValueError("No pending patches to place")
        idx = int(action[1])
        if not (0 <= idx < BOARD_CELLS):
            raise ValueError(f"Patch idx={idx} out of range [0, {BOARD_CELLS})")
        occ0, occ1, occ2 = _get_occ_words_py(state, pl)
        # CRITICAL FIX: Allow occupied square only if board is completely full (discard patch)
        if _is_bit_set_py(occ0, occ1, occ2, idx):
            empty = _empty_count_words_py(occ0, occ1, occ2)
            if empty > 0:
                raise ValueError("Patch on occupied square")
            # else: board full, allow discard by placing on occupied square
        return

    if int(state[PENDING_PATCHES]) > 0:
        raise ValueError("Must place patches first")

    if a0 == AT_PASS:
        base = _player_base_py(pl)
        if int(state[base + 0]) >= SCOREBOARD_LENGTH:
            raise ValueError("Cannot pass from final space")
        return

    if a0 == AT_BUY:
        offset = int(action[1])
        piece_id = int(action[2])
        orient = int(action[3])
        top = int(action[4])
        left = int(action[5])

        n = int(state[CIRCLE_LEN])
        if n == 0:
            raise ValueError("No pieces left to buy")
        max_pick = 3 if n >= 3 else n
        if not (1 <= offset <= max_pick):
            raise ValueError(f"Buy offset={offset} out of range [1, {max_pick}]")

        idx = (int(state[NEUTRAL]) + offset) % n
        if int(state[CIRCLE_START + idx]) != piece_id:
            raise ValueError("Buy mismatch")

        base = _player_base_py(pl)
        buttons = int(state[base + 1])
        if buttons < int(PIECE_COST_BUTTONS[piece_id]):
            raise ValueError("Not enough buttons")

        ocount = int(ORIENT_COUNT[piece_id])
        if not (0 <= orient < ocount):
            raise ValueError(f"orient={orient} out of range")

        h = int(ORIENT_SHAPE_H[piece_id, orient])
        w = int(ORIENT_SHAPE_W[piece_id, orient])
        if not (0 <= top <= BOARD_SIZE - h):
            raise ValueError("top out of range for orientation")
        if not (0 <= left <= BOARD_SIZE - w):
            raise ValueError("left out of range for orientation")

        pos_idx = top * BOARD_SIZE + left
        m0 = MASK_W0[piece_id, orient, pos_idx]
        m1 = MASK_W1[piece_id, orient, pos_idx]
        m2 = MASK_W2[piece_id, orient, pos_idx]
        if int(m0 | m1 | m2) == 0:
            raise ValueError("Invalid placement for orientation")

        occ0, occ1, occ2 = _get_occ_words_py(state, pl)
        if (occ0 & m0) != np.uint32(0) or (occ1 & m1) != np.uint32(0) or (occ2 & m2) != np.uint32(0):
            raise ValueError("Overlap")

        return

    raise ValueError(f"Unknown action type {a0}")


def apply_action(state: np.ndarray, action) -> np.ndarray:
    s = upgrade_state(state)
    _validate_action_py(s, action)
    out = _apply_action_nb(s, action)
    # Ensure edition code stays valid
    out = upgrade_state(out)
    out[EDITION_CODE] = np.int32(_edition_code(out))
    return out


# =============================================================================
# Scoring helpers
# =============================================================================

def empty_count_from_occ(occ0: int, occ1: int, occ2: int) -> int:
    """Count empty squares from occupancy words."""
    return _empty_count_words_py(np.uint32(np.int32(occ0)), np.uint32(np.int32(occ1)), np.uint32(np.int32(occ2)))


def compute_score(state: np.ndarray, player_idx: int) -> int:
    """Compute final score for a player (matches patchwork_api final_score)."""
    s = upgrade_state(state)
    if int(player_idx) == 0:
        buttons = int(s[P0_BUTTONS])
        occ0, occ1, occ2 = int(s[P0_OCC0]), int(s[P0_OCC1]), int(s[P0_OCC2])
    else:
        buttons = int(s[P1_BUTTONS])
        occ0, occ1, occ2 = int(s[P1_OCC0]), int(s[P1_OCC1]), int(s[P1_OCC2])
    empty = empty_count_from_occ(occ0, occ1, occ2)
    bonus = 7 if int(s[BONUS_OWNER]) == int(player_idx) else 0
    return buttons - 2 * empty + bonus


def get_winner(state: np.ndarray) -> int:
    """Get winner. Returns 0 or 1 (uses tie-breaker on draw)."""
    s = upgrade_state(state)
    p0_score = compute_score(s, 0)
    p1_score = compute_score(s, 1)
    if p0_score > p1_score:
        return 0
    if p1_score > p0_score:
        return 1
    return 1 - int(s[TIE_PLAYER])


# =============================================================================
# Fast-path functions for MCTS / selfplay inner loops
# =============================================================================
# These skip upgrade_state and validation for maximum speed.
# ONLY use when the state is known to be valid (size=STATE_SIZE)
# and actions are known to be legal (from legal_actions_list).

def apply_action_unchecked(state: np.ndarray, action) -> np.ndarray:
    """Apply action WITHOUT validation or upgrade_state.

    Use ONLY when:
    - state is already valid (created by new_game or previous apply_action)
    - action is from legal_actions_list (guaranteed legal)

    ~10-50x faster than apply_action() for MCTS inner loops.
    """
    out = _apply_action_nb(state, action)
    return out


def terminal_fast(state: np.ndarray) -> bool:
    """Fast terminal check — skips upgrade_state."""
    return bool(_terminal_nb(state))


def current_player_fast(state: np.ndarray) -> int:
    """Fast current_player — skips upgrade_state."""
    return int(_current_player_nb(state))


def legal_actions_fast(state: np.ndarray) -> list:
    """Fast legal_actions — skips upgrade_state wrapper."""
    return list(_legal_actions_raw_nb(state))


def compute_score_fast(state: np.ndarray, player_idx: int) -> int:
    """Fast compute_score — skips upgrade_state.

    Use ONLY on states produced by apply_action_unchecked / new_game.
    """
    if int(player_idx) == 0:
        buttons = int(state[P0_BUTTONS])
        occ0, occ1, occ2 = int(state[P0_OCC0]), int(state[P0_OCC1]), int(state[P0_OCC2])
    else:
        buttons = int(state[P1_BUTTONS])
        occ0, occ1, occ2 = int(state[P1_OCC0]), int(state[P1_OCC1]), int(state[P1_OCC2])
    empty = _empty_count_words_py(np.uint32(np.int32(occ0)), np.uint32(np.int32(occ1)), np.uint32(np.int32(occ2)))
    bonus = 7 if int(state[BONUS_OWNER]) == int(player_idx) else 0
    return buttons - 2 * empty + bonus


def get_winner_fast(state: np.ndarray) -> int:
    """Fast get_winner — skips upgrade_state.

    Use ONLY on states produced by apply_action_unchecked / new_game.
    """
    p0_score = compute_score_fast(state, 0)
    p1_score = compute_score_fast(state, 1)
    if p0_score > p1_score:
        return 0
    if p1_score > p0_score:
        return 1
    return 1 - int(state[TIE_PLAYER])


# =============================================================================
# Circle + JSON parsing (API-faithful)
# =============================================================================

def make_shuffled_circle(seed: Optional[int] = None) -> Tuple[List[int], int]:
    ids = [int(p["id"]) for p in PIECES]
    rng: random.Random = random.Random(seed) if seed is not None else random.SystemRandom()
    rng.shuffle(ids)
    neutral = ids.index(32) if 32 in ids else 0
    return ids, int(neutral)


def _is_fresh_player_words(pos: int, occ0: int, occ1: int, occ2: int, income: int) -> bool:
    return int(pos) == 0 and int(income) == 0 and int(occ0) == 0 and int(occ1) == 0 and int(occ2) == 0


def _looks_like_fresh_game(p0: dict, p1: dict, d: dict, circle_in) -> bool:
    try:
        ordered = list(map(int, circle_in)) == list(range(len(PIECES)))
    except Exception:
        ordered = False

    def _player_fresh(p: dict) -> bool:
        return (
            int(p.get("position", 0)) == 0
            and int(p.get("income", 0)) == 0
            and all(ch == "." for ch in "".join(p.get("board", ["........."] * 9)))
        )

    # Mirror patchwork_api.py heuristic fields
    return (
        ordered
        and _player_fresh(p0)
        and _player_fresh(p1)
        and int(d.get("pending_patches", 0)) == 0
        and int(d.get("bonus_owner", -1)) == -1
    )


def _parse_board_rows(rows: List[str]) -> Tuple[np.int32, np.int32, np.int32, int]:
    """Parse list[str] board rows into occ words and inferred income (from '2' cells)."""
    if not isinstance(rows, list) or len(rows) != BOARD_SIZE or not all(isinstance(r, str) for r in rows):
        raise ValueError("Board must be list[str] with 9 rows.")
    w0, w1, w2 = np.uint32(0), np.uint32(0), np.uint32(0)
    income = 0
    for r, row in enumerate(rows):
        if len(row) != BOARD_SIZE:
            raise ValueError("Each board row must be length 9")
        for c, ch in enumerate(row):
            if ch in (".", "0", " "):
                continue
            if ch in ("1", "#", "X"):
                pass
            elif ch == "2":
                income += 1
            else:
                raise ValueError(f"Unknown board cell {ch!r}")
            idx = r * BOARD_SIZE + c
            word = idx >> 5
            bit = idx & 31
            b = np.uint32(1) << np.uint32(bit)
            if word == 0:
                w0 |= b
            elif word == 1:
                w1 |= b
            else:
                w2 |= b
    return np.int32(w0), np.int32(w1), np.int32(w2), int(income)


def new_game(
    seed: Optional[int] = None,
    starting_buttons: int = 5,
    starting_player: int = 0,
    edition: str = DEFAULT_EDITION,
) -> np.ndarray:
    """Create fresh game state (matches patchwork_api /new behavior)."""
    if edition not in EDITION_PATCHES:
        raise ValueError(f"Unknown edition {edition!r}. Valid: {sorted(EDITION_PATCHES.keys())}")
    if starting_player not in (0, 1):
        starting_player = 0

    circle, neutral = make_shuffled_circle(seed=seed)
    s = np.full((STATE_SIZE,), -1, dtype=np.int32)

    s[P0_POS] = 0
    s[P0_BUTTONS] = np.int32(starting_buttons)
    s[P0_INCOME] = 0
    s[P0_OCC0] = 0
    s[P0_OCC1] = 0
    s[P0_OCC2] = 0

    s[P1_POS] = 0
    s[P1_BUTTONS] = np.int32(starting_buttons)
    s[P1_INCOME] = 0
    s[P1_OCC0] = 0
    s[P1_OCC1] = 0
    s[P1_OCC2] = 0

    s[CIRCLE_LEN] = np.int32(len(circle))
    s[NEUTRAL] = np.int32(neutral)
    s[BONUS_OWNER] = np.int32(-1)
    s[PENDING_PATCHES] = np.int32(0)
    s[PENDING_OWNER] = np.int32(-1)
    s[TIE_PLAYER] = np.int32(starting_player)

    for i, pid in enumerate(circle):
        s[CIRCLE_START + i] = np.int32(pid)

    s[EDITION_CODE] = np.int32(EDITION_TO_CODE[edition])
    return s


def state_to_dict(state: np.ndarray) -> dict:
    """Convert flat state to dict (compatible with patchwork_api state_to_dict)."""
    s = upgrade_state(state)

    def _occ_words_to_rows(w0_i, w1_i, w2_i):
        w0, w1, w2 = np.uint32(np.int32(w0_i)), np.uint32(np.int32(w1_i)), np.uint32(np.int32(w2_i))
        rows = []
        for r in range(BOARD_SIZE):
            chars = []
            for c in range(BOARD_SIZE):
                idx = r * BOARD_SIZE + c
                word = idx >> 5
                bit = idx & 31
                mask = np.uint32(1) << np.uint32(bit)
                if word == 0:
                    occ = (w0 & mask) != np.uint32(0)
                elif word == 1:
                    occ = (w1 & mask) != np.uint32(0)
                else:
                    occ = (w2 & mask) != np.uint32(0)
                chars.append("1" if occ else ".")
            rows.append("".join(chars))
        return rows

    p0_board = _occ_words_to_rows(int(s[P0_OCC0]), int(s[P0_OCC1]), int(s[P0_OCC2]))
    p1_board = _occ_words_to_rows(int(s[P1_OCC0]), int(s[P1_OCC1]), int(s[P1_OCC2]))
    n = int(s[CIRCLE_LEN])
    circle = [int(s[CIRCLE_START + i]) for i in range(n)]
    ed = CODE_TO_EDITION.get(_edition_code(s), DEFAULT_EDITION)

    return {
        "edition": ed,
        "players": [
            {"position": int(s[P0_POS]), "buttons": int(s[P0_BUTTONS]), "board": p0_board, "income": int(s[P0_INCOME])},
            {"position": int(s[P1_POS]), "buttons": int(s[P1_BUTTONS]), "board": p1_board, "income": int(s[P1_INCOME])},
        ],
        "circle": circle,
        "neutral": int(s[NEUTRAL]),
        "bonus_owner": int(s[BONUS_OWNER]),
        "pending_patches": int(s[PENDING_PATCHES]),
        "pending_owner": int(s[PENDING_OWNER]),
        "tie_player": int(s[TIE_PLAYER]),
    }


def state_from_dict(d: dict) -> np.ndarray:
    """Convert dict to flat state (mirrors patchwork_api state_from_dict parsing)."""
    if not isinstance(d, dict):
        raise ValueError("state must be a dict")
    players = d.get("players", [])
    if not isinstance(players, list) or len(players) != 2:
        raise ValueError("state.players must have length 2")

    p0_in = players[0]
    p1_in = players[1]
    if not isinstance(p0_in, dict) or not isinstance(p1_in, dict):
        raise ValueError("players entries must be dicts")

    # Parse boards; infer income if omitted
    p0_occ0, p0_occ1, p0_occ2, p0_board_income = _parse_board_rows(p0_in.get("board"))
    p1_occ0, p1_occ1, p1_occ2, p1_board_income = _parse_board_rows(p1_in.get("board"))

    p0_income = int(p0_in.get("income", p0_board_income))
    p1_income = int(p1_in.get("income", p1_board_income))

    # Edition
    edition = str(d.get("edition", DEFAULT_EDITION))
    if edition not in EDITION_PATCHES:
        raise ValueError(f"Unknown edition {edition!r}. Valid: {sorted(EDITION_PATCHES.keys())}")
    ed_code = EDITION_TO_CODE[edition]

    # Circle initialization / heuristic randomization
    circle_in = d.get("circle", None)
    randomize_circle = d.get("randomize_circle", None)  # default True when omitted
    seed = d.get("seed", None)

    if circle_in is None or (isinstance(circle_in, (list, tuple)) and len(circle_in) == 0):
        circle, neutral = make_shuffled_circle(seed=seed)
    else:
        if (randomize_circle is None or bool(randomize_circle)) and _looks_like_fresh_game(p0_in, p1_in, d, circle_in):
            circle, neutral = make_shuffled_circle(seed=seed)
        else:
            circle = [int(x) for x in circle_in]
            neutral = int(d.get("neutral", 0))
            neutral = 0 if len(circle) == 0 else max(0, min(neutral, len(circle) - 1))

    # Validate circle contents
    if len(set(circle)) != len(circle):
        raise ValueError("circle contains duplicate piece IDs")
    for pid in circle:
        if pid not in PIECE_BY_ID:
            raise ValueError(f"circle contains unknown piece ID {pid}")

    bonus_owner = int(d.get("bonus_owner", -1))
    if bonus_owner not in (-1, 0, 1):
        raise ValueError("bonus_owner must be -1, 0, or 1")

    pending_patches = int(d.get("pending_patches", 0))
    pending_owner = int(d.get("pending_owner", -1))
    if pending_patches < 0:
        raise ValueError("pending_patches cannot be negative")
    if pending_patches > 0 and pending_owner not in (0, 1):
        raise ValueError("pending_patches > 0 but pending_owner is not 0 or 1")
    if pending_patches == 0 and pending_owner != -1:
        raise ValueError("pending_patches == 0 but pending_owner != -1")

    tie_player = int(d.get("tie_player", 0))
    if tie_player not in (0, 1):
        raise ValueError("tie_player must be 0 or 1")

    s = np.full((STATE_SIZE,), -1, dtype=np.int32)

    s[P0_POS] = np.int32(int(p0_in.get("position", 0)))
    s[P0_BUTTONS] = np.int32(int(p0_in.get("buttons", 0)))
    s[P0_INCOME] = np.int32(p0_income)
    s[P0_OCC0], s[P0_OCC1], s[P0_OCC2] = p0_occ0, p0_occ1, p0_occ2

    s[P1_POS] = np.int32(int(p1_in.get("position", 0)))
    s[P1_BUTTONS] = np.int32(int(p1_in.get("buttons", 0)))
    s[P1_INCOME] = np.int32(p1_income)
    s[P1_OCC0], s[P1_OCC1], s[P1_OCC2] = p1_occ0, p1_occ1, p1_occ2

    s[CIRCLE_LEN] = np.int32(len(circle))
    for i, pid in enumerate(circle):
        s[CIRCLE_START + i] = np.int32(pid)
    # Fill remaining slots with -1 (already)
    s[NEUTRAL] = np.int32(neutral)
    s[BONUS_OWNER] = np.int32(bonus_owner)
    s[PENDING_PATCHES] = np.int32(pending_patches)
    s[PENDING_OWNER] = np.int32(pending_owner)
    s[TIE_PLAYER] = np.int32(tie_player)
    s[EDITION_CODE] = np.int32(ed_code)

    return s


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    # Basic sanity: generate a game, take first legal action repeatedly.
    s = new_game(seed=42, starting_buttons=5, starting_player=0, edition="revised")
    for _ in range(200):
        if terminal(s):
            break
        acts = legal_actions_list(s)
        if not acts:
            break
        s = apply_action(s, acts[0])
    print("Smoke test OK. terminal=", terminal(s), "winner=", get_winner(s), "edition=", CODE_TO_EDITION.get(_edition_code(s), DEFAULT_EDITION))
