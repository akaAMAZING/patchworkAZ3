"""
Packing placement-ordering heuristic for BUY actions (Progressive Widening).

Fully optimized: bitboard-only (no Python loops), flat array caches, precomputed window masks.
Used only to RANK BUY actions for PW.
"""

from __future__ import annotations

import numpy as np

from src.game.patchwork_engine import (
    BOARD_SIZE,
    BOARD_CELLS,
    P0_OCC0,
    P0_OCC1,
    P0_OCC2,
    P1_OCC0,
    P1_OCC1,
    P1_OCC2,
    MASK_W0,
    MASK_W1,
    MASK_W2,
    ORIENT_SHAPE_H,
    ORIENT_SHAPE_W,
    N_PIECES,
    MAX_ORIENTS,
)
from src.game.patchwork_engine import current_player_fast

# ---------------------------------------------------------------------------
# Bitboard constants (81 bits; cell (r,c) = bit r*9+c)
# ---------------------------------------------------------------------------
_FULL_MASK = (1 << BOARD_CELLS) - 1

# Column masks for boundary-respecting shifts
_COL0_MASK = sum(1 << (i * BOARD_SIZE) for i in range(BOARD_SIZE))
_COL8_MASK = sum(1 << (i * BOARD_SIZE + (BOARD_SIZE - 1)) for i in range(BOARD_SIZE))

# Cells on board edge (row/col 0 or 8)
_EDGE_MASK = 0
for r in range(BOARD_SIZE):
    for c in range(BOARD_SIZE):
        if r == 0 or r == BOARD_SIZE - 1 or c == 0 or c == BOARD_SIZE - 1:
            _EDGE_MASK |= 1 << (r * BOARD_SIZE + c)

# True corners only: (0,0), (0,8), (8,0), (8,8) — corner_bonus rewards only these, not any edge
_CORNER_MASK = (1 << 0) | (1 << 8) | (1 << 72) | (1 << 80)


def _neigh(bb: int) -> int:
    """4-neighbor expansion of bitboard (respecting 9x9 boundaries). Pure bit-ops."""
    up = bb >> BOARD_SIZE
    down = (bb << BOARD_SIZE) & _FULL_MASK
    left = (bb & ~_COL0_MASK) >> 1
    right = (bb & ~_COL8_MASK) << 1
    return up | down | left | right


# ---------------------------------------------------------------------------
# Flat array caches: index = piece_id * (MAX_ORIENTS * BOARD_CELLS) + orient * BOARD_CELLS + pos_idx
# ---------------------------------------------------------------------------
_CACHE_SIZE = N_PIECES * MAX_ORIENTS * BOARD_CELLS


def _cache_index(piece_id: int, orient: int, pos_idx: int) -> int:
    return piece_id * (MAX_ORIENTS * BOARD_CELLS) + orient * BOARD_CELLS + pos_idx


def cache_index(piece_id: int, orient: int, pos_idx: int) -> int:
    """Public for callers that want to pass flat indices to scores_batch."""
    return _cache_index(piece_id, orient, pos_idx)


def _occ_words_to_bitboard(occ0: int, occ1: int, occ2: int) -> int:
    """Convert 3×32-bit occupancy words to single 81-bit bitboard."""
    o0 = int(occ0) & 0xFFFFFFFF
    o1 = int(occ1) & 0xFFFFFFFF
    o2 = int(occ2) & 0x1FFFF
    return o0 | (o1 << 32) | (o2 << 64)


# PLACEMENT_BB[i] = placement bitboard for cache index i (0 if invalid)
PLACEMENT_BB: list[int] = [0] * _CACHE_SIZE

# WINDOW_MASK[radius_index][i] = window bitboard for placement at i, radius 0..3
WINDOW_MASK: list[list[int]] = [[0] * _CACHE_SIZE for _ in range(4)]


def _build_flat_caches() -> None:
    """Precompute placement bitboards and window masks (radius 0..3)."""
    for piece_id in range(N_PIECES):
        for orient in range(MAX_ORIENTS):
            h = int(ORIENT_SHAPE_H[piece_id, orient])
            w = int(ORIENT_SHAPE_W[piece_id, orient])
            for pos_idx in range(BOARD_CELLS):
                idx = _cache_index(piece_id, orient, pos_idx)
                m0 = int(MASK_W0[piece_id, orient, pos_idx])
                m1 = int(MASK_W1[piece_id, orient, pos_idx])
                m2 = int(MASK_W2[piece_id, orient, pos_idx])
                if (m0 | m1 | m2) != 0:
                    PLACEMENT_BB[idx] = _occ_words_to_bitboard(m0, m1, m2)
                else:
                    PLACEMENT_BB[idx] = 0
                top = pos_idx // BOARD_SIZE
                left = pos_idx % BOARD_SIZE
                h1 = max(1, h)
                w1 = max(1, w)
                for ri, radius in enumerate(range(4)):
                    r_min = max(0, top - radius)
                    r_max = min(BOARD_SIZE, top + h1 + radius)
                    c_min = max(0, left - radius)
                    c_max = min(BOARD_SIZE, left + w1 + radius)
                    wm = 0
                    for r in range(r_min, r_max):
                        for c in range(c_min, c_max):
                            wm |= 1 << (r * BOARD_SIZE + c)
                    WINDOW_MASK[ri][idx] = wm


_build_flat_caches()

# Numpy arrays (lo/hi 64-bit) for Cython extension; built after caches exist.
PLACEMENT_LO: np.ndarray = np.zeros(_CACHE_SIZE, dtype=np.uint64)
PLACEMENT_HI: np.ndarray = np.zeros(_CACHE_SIZE, dtype=np.uint64)
WINDOW_LO: np.ndarray = np.zeros((4, _CACHE_SIZE), dtype=np.uint64)
WINDOW_HI: np.ndarray = np.zeros((4, _CACHE_SIZE), dtype=np.uint64)


def _build_cython_arrays() -> None:
    """Fill PLACEMENT_LO/HI and WINDOW_LO/HI from PLACEMENT_BB and WINDOW_MASK."""
    mask64 = 0xFFFFFFFFFFFFFFFF
    mask_hi = 0x1FFFF
    for i in range(_CACHE_SIZE):
        bb = PLACEMENT_BB[i]
        PLACEMENT_LO[i] = np.uint64(int(bb) & mask64)
        PLACEMENT_HI[i] = np.uint64((int(bb) >> 64) & mask_hi)
        for r in range(4):
            wm = WINDOW_MASK[r][i]
            WINDOW_LO[r, i] = np.uint64(int(wm) & mask64)
            WINDOW_HI[r, i] = np.uint64((int(wm) >> 64) & mask_hi)


_build_cython_arrays()

try:
    from src.mcts.packing_heuristic_cy import packing_scores_batch_cy
    _CYTHON_AVAILABLE = True
except ImportError:
    packing_scores_batch_cy = None
    _CYTHON_AVAILABLE = False


def get_placement_bitboard(piece_id: int, orient: int, top: int, left: int) -> int:
    """O(1) placement bitboard from flat array. Returns 0 if invalid."""
    pos_idx = top * BOARD_SIZE + left
    return PLACEMENT_BB[_cache_index(piece_id, orient, pos_idx)]


def get_placement_bitboard_and_window(
    piece_id: int, orient: int, top: int, left: int, radius_index: int
) -> tuple[int, int]:
    """O(1) return (placement_bb, window_mask). radius_index in 0..3."""
    pos_idx = top * BOARD_SIZE + left
    i = _cache_index(piece_id, orient, pos_idx)
    return PLACEMENT_BB[i], WINDOW_MASK[min(radius_index, 3)][i]


def occ_words_to_bitboard_for_node(occ0: int, occ1: int, occ2: int) -> int:
    """Decode current-player occupancy to bitboard once per node."""
    return _occ_words_to_bitboard(occ0, occ1, occ2)


# Scale applied to raw score (match original semantics)
_SCALE = 3.0


def _score_one(
    filled_before_bb: int,
    idx: int,
    radius_index: int,
    w_adj: float,
    w_corner: float,
    w_iso: float,
    w_front: float,
    w_area: float = 0.0,
) -> float:
    """Single placement score; idx = _cache_index(piece_id, orient, pos_idx)."""
    placed_bb = PLACEMENT_BB[idx]
    if placed_bb == 0:
        return 0.0
    window_mask = WINDOW_MASK[radius_index][idx]
    filled_after = filled_before_bb | placed_bb
    empty_after = _FULL_MASK ^ filled_after

    placed_up = placed_bb >> BOARD_SIZE
    placed_down = (placed_bb << BOARD_SIZE) & _FULL_MASK
    placed_left = (placed_bb & ~_COL0_MASK) >> 1
    placed_right = (placed_bb & ~_COL8_MASK) << 1
    adj_edges = (
        (filled_before_bb & placed_up).bit_count()
        + (filled_before_bb & placed_down).bit_count()
        + (filled_before_bb & placed_left).bit_count()
        + (filled_before_bb & placed_right).bit_count()
    )
    score = w_adj * adj_edges
    score += w_corner * (1 if (placed_bb & _CORNER_MASK) else 0)
    empty_neighbors = _neigh(empty_after)
    iso_mask = empty_after & ~empty_neighbors
    filled_neighbors = _neigh(filled_after)
    frontier_mask = empty_after & filled_neighbors
    score -= w_iso * (iso_mask & window_mask).bit_count()
    score -= w_front * (frontier_mask & window_mask).bit_count()
    if w_area != 0.0:
        score += w_area * placed_bb.bit_count()
    return score / _SCALE


def packing_heuristic_score_fast_core(
    filled_before_bb: int,
    piece_id: int,
    orient: int,
    pos_idx: int,
    radius_index: int,
    w_adj: float,
    w_corner: float,
    w_iso: float,
    w_front: float,
    w_area: float = 0.0,
) -> float:
    """Inner hot path: no dict/tuple, only bit-ops. Call with unpacked weights and radius_index 0..3."""
    idx = _cache_index(piece_id, orient, pos_idx)
    return _score_one(filled_before_bb, idx, radius_index, w_adj, w_corner, w_iso, w_front, w_area)


def packing_heuristic_scores_batch(
    filled_before_bb: int,
    indices,  # list of (piece_id, orient, pos_idx) or cache idx; or np.ndarray int32 (no alloc)
    radius_index: int,
    w_adj: float,
    w_corner: float,
    w_iso: float,
    w_front: float,
    w_area: float = 0.0,
    _use_cython: bool = True,
) -> list:
    """Compute scores for multiple placements. Uses Cython extension if available and _use_cython=True.
    w_area = area_bonus: +w_area * (cells in placed piece). Pass indices as contiguous np.ndarray int32 to avoid alloc."""
    n = len(indices)
    if n == 0:
        return []
    # Use preallocated int32 array when provided; otherwise build list and convert once
    idx_arr = None
    idx_list = None
    if isinstance(indices, np.ndarray) and indices.dtype == np.int32 and indices.ndim == 1:
        if indices.flags.c_contiguous:
            idx_arr = indices
        else:
            idx_arr = np.ascontiguousarray(indices, dtype=np.int32)
    else:
        idx_list = []
        for item in indices:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                idx_list.append(_cache_index(int(item[0]), int(item[1]), int(item[2])))
            else:
                idx_list.append(int(item))
        idx_arr = np.array(idx_list, dtype=np.int32)
    if _use_cython and _CYTHON_AVAILABLE and packing_scores_batch_cy is not None:
        filled_lo = int(filled_before_bb) & 0xFFFFFFFFFFFFFFFF
        filled_hi = (int(filled_before_bb) >> 64) & 0x1FFFF
        out = packing_scores_batch_cy(
            filled_lo, filled_hi, idx_arr, radius_index,
            w_adj, w_corner, w_iso, w_front, w_area,
            PLACEMENT_LO, PLACEMENT_HI, WINDOW_LO, WINDOW_HI, _SCALE,
        )
        return out.tolist()
    py_indices = idx_list if idx_list is not None else idx_arr.tolist()
    return _packing_heuristic_scores_batch_py(
        filled_before_bb, py_indices, radius_index, w_adj, w_corner, w_iso, w_front, w_area
    )


def _packing_heuristic_scores_batch_py(
    filled_before_bb: int,
    indices: list,  # list of int (cache indices)
    radius_index: int,
    w_adj: float,
    w_corner: float,
    w_iso: float,
    w_front: float,
    w_area: float = 0.0,
) -> list:
    """Pure Python batch scorer (fallback when Cython not built)."""
    n = len(indices)
    out = [0.0] * n
    for i in range(n):
        idx = indices[i]
        placed_bb = PLACEMENT_BB[idx]
        if placed_bb == 0:
            continue
        window_mask = WINDOW_MASK[radius_index][idx]
        filled_after = filled_before_bb | placed_bb
        empty_after = _FULL_MASK ^ filled_after
        placed_up = placed_bb >> BOARD_SIZE
        placed_down = (placed_bb << BOARD_SIZE) & _FULL_MASK
        placed_left = (placed_bb & ~_COL0_MASK) >> 1
        placed_right = (placed_bb & ~_COL8_MASK) << 1
        adj_edges = (
            (filled_before_bb & placed_up).bit_count()
            + (filled_before_bb & placed_down).bit_count()
            + (filled_before_bb & placed_left).bit_count()
            + (filled_before_bb & placed_right).bit_count()
        )
        s = w_adj * adj_edges
        s += w_corner * (1 if (placed_bb & _CORNER_MASK) else 0)
        empty_neighbors = _neigh(empty_after)
        iso_mask = empty_after & ~empty_neighbors
        filled_neighbors = _neigh(filled_after)
        frontier_mask = empty_after & filled_neighbors
        s -= w_iso * (iso_mask & window_mask).bit_count()
        s -= w_front * (frontier_mask & window_mask).bit_count()
        if w_area != 0.0:
            s += w_area * placed_bb.bit_count()
        out[i] = s / _SCALE
    return out


def packing_heuristic_score_fast(
    filled_before_bb: int,
    action: tuple,
    weights: dict,
    local_check_radius: int,
) -> float:
    """
    Compute packing score using only bit-ops (no Python loops).
    filled_before_bb = current player occupancy (81 bits).
    """
    if len(action) < 6 or int(action[0]) != 2:
        return 0.0
    piece_id = int(action[2])
    orient = int(action[3])
    top = int(action[4])
    left = int(action[5])
    pos_idx = top * BOARD_SIZE + left
    radius_index = min(max(0, local_check_radius), 3)
    w_adj = float(weights.get("adj_edges", 1.0))
    w_corner = float(weights.get("corner_bonus", 0.5))
    w_iso = float(weights.get("iso_hole_penalty", 2.0))
    w_front = float(weights.get("frontier_penalty", 0.25))
    w_area = float(weights.get("area_bonus", 0.0))
    return packing_heuristic_score_fast_core(
        filled_before_bb, piece_id, orient, pos_idx, radius_index, w_adj, w_corner, w_iso, w_front, w_area
    )


# ---------------------------------------------------------------------------
# Legacy API
# ---------------------------------------------------------------------------

def _occ_words_to_bool_grid(occ0: int, occ1: int, occ2: int) -> np.ndarray:
    """Decode 3×32-bit occupancy words into (9,9) boolean grid (True = filled)."""
    grid = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.bool_)
    for idx in range(BOARD_CELLS):
        word = idx >> 5
        bit = idx & 31
        mask = 1 << bit
        if word == 0:
            occupied = (int(occ0) & mask) != 0
        elif word == 1:
            occupied = (int(occ1) & mask) != 0
        else:
            occupied = (int(occ2) & mask) != 0
        r, c = divmod(idx, BOARD_SIZE)
        grid[r, c] = occupied
    return grid


def packing_heuristic_score(
    state: np.ndarray,
    action: tuple,
    weights: dict,
    local_check_radius: int,
) -> float:
    """Original API: decode occupancy from state then call fast path."""
    if len(action) < 6 or int(action[0]) != 2:
        return 0.0
    pl = current_player_fast(state)
    if pl == 0:
        occ0, occ1, occ2 = int(state[P0_OCC0]), int(state[P0_OCC1]), int(state[P0_OCC2])
    else:
        occ0, occ1, occ2 = int(state[P1_OCC0]), int(state[P1_OCC1]), int(state[P1_OCC2])
    filled_before_bb = _occ_words_to_bitboard(occ0, occ1, occ2)
    return packing_heuristic_score_fast(filled_before_bb, action, weights, local_check_radius)
