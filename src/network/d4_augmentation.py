"""
D4 symmetry augmentation for 32-channel gold_v2 spatial encoding.

Provides full D4 group: identity, rot90, rot180, rot270, mirror (h-flip),
mirror+rot90, mirror+rot180, mirror+rot270.

Transform indices: 0=id, 1=r90, 2=r180, 3=r270, 4=m, 5=m_r90, 6=m_r180, 7=m_r270.
Mirror = horizontal flip (r,c) -> (r, 8-c).

All 32 channels are spatial (no scalar channels):
  0-7:   board, coord, frontier, valid_7x7
  8-31:  slot×orient shape planes
  (legalTL is computed on-GPU; not stored in the state tensor)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.network.gold_v2_constants import C_SPATIAL_ENC, SLOT_ORIENT_SHAPE_BASE
from src.game.patchwork_engine import (
    PIECE_AUGMENTATION,
    PIECE_BY_ID,
    ORIENT_COUNT,
    ORIENT_SHAPE_H,
    ORIENT_SHAPE_W,
    BOARD_SIZE,
    MASK_W0,
    MASK_W1,
    MASK_W2,
)
from src.game.patchwork_engine import (
    _rotate_ccw,
    _flip_vertical,
    _flip_horizontal,
    _canonical_cells,
    _shape_dims,
)

# D4 transform names (for logging/debug)
D4_NAMES = [
    "id", "r90", "r180", "r270",
    "m", "m_r90", "m_r180", "m_r270",
]
D4_COUNT = 8

# Spatial channels to transform (gold_v2: 32 encoded channels; legalTL on GPU)
SPATIAL_CHANNELS = list(range(C_SPATIAL_ENC))
BOARD_CHANNELS = [0, 1]  # occupancy
COORD_CHANNELS = [2, 3]   # row_norm, col_norm
SLOT_ORIENT_BASE = SLOT_ORIENT_SHAPE_BASE   # shape planes 8-31
SLOT_ORIENT_CHANNELS = list(range(SLOT_ORIENT_SHAPE_BASE, C_SPATIAL_ENC))

PASS_INDEX = 0
PATCH_START = 1
BUY_START = 82
NUM_SLOTS = 3
NUM_ORIENTS = 8
BS = BOARD_SIZE


def _mask_words_to_plane(w0: int, w1: int, w2: int) -> np.ndarray:
    """Decode 3×32-bit mask words to (9,9) float32 plane. Same layout as encoder."""
    words = np.array([
        int(w0) & 0xFFFFFFFF,
        int(w1) & 0xFFFFFFFF,
        int(w2) & 0xFFFFFFFF,
    ], dtype=np.uint32)
    byte_arr = words.view(np.uint8)
    all_bits = np.unpackbits(byte_arr, bitorder="little")
    return all_bits[:81].astype(np.float32).reshape(BS, BS)


def _transform_mask_plane(plane: np.ndarray, transform_idx: int) -> np.ndarray:
    """
    Transform a 9x9 mask by D4 symmetry. Uses same convention as _transform_board_plane:
    new[r,c] = old[inv_T(r,c)] (content moves with cells).
    """
    _init_inv_index_2d()
    row_idx, col_idx = _INV_INDEX_2D[transform_idx]
    return np.ascontiguousarray(plane[row_idx, col_idx])


def _build_orient_map_for_transform(
    piece_id: int,
    transform_idx: int,
) -> Optional[List[int]]:
    """
    Build orient_map by exact mask matching.
    For each o: target = transform_mask(mask[p][o], sym); find o2 s.t. mask[p][o2] == target.
    Ensures bijection: each output used at most once, so get_orient_before_transform works.
    """
    n = int(ORIENT_COUNT[piece_id])
    if n == 0:
        return None
    candidates_per_o: List[List[int]] = []
    for o in range(n):
        mask_o = None
        for pos in range(81):
            m0 = int(MASK_W0[piece_id, o, pos])
            m1 = int(MASK_W1[piece_id, o, pos])
            m2 = int(MASK_W2[piece_id, o, pos])
            if (m0 | m1 | m2) != 0:
                mask_o = _mask_words_to_plane(m0, m1, m2)
                break
        if mask_o is None:
            return None
        target = _transform_mask_plane(mask_o, transform_idx)
        candidates: List[int] = []
        for o2 in range(n):
            for pos in range(81):
                m0 = int(MASK_W0[piece_id, o2, pos])
                m1 = int(MASK_W1[piece_id, o2, pos])
                m2 = int(MASK_W2[piece_id, o2, pos])
                if (m0 | m1 | m2) == 0:
                    continue
                mask_o2 = _mask_words_to_plane(m0, m1, m2)
                if np.array_equal(target, mask_o2):
                    candidates.append(o2)
                    break
        if not candidates:
            return None
        candidates_per_o.append(sorted(candidates))
    used: Set[int] = set()
    result: List[int] = []
    for o in range(n):
        available = [c for c in candidates_per_o[o] if c not in used]
        if not available:
            return None
        o2 = min(available)
        used.add(o2)
        result.append(o2)
    return result


def _get_orient_shapes_for_piece(piece_id: int) -> list:
    """Get oriented shapes for piece in engine order (reuse engine logic)."""
    from src.game.patchwork_engine import PIECES, _get_orient_shapes
    for p in PIECES:
        if int(p["id"]) == piece_id:
            return _get_orient_shapes(p["shape"])
    return []


def _transform_shape_r90(shape):
    return _rotate_ccw(shape)


def _transform_shape_r180(shape):
    return _rotate_ccw(_rotate_ccw(shape))


def _transform_shape_r270(shape):
    return _rotate_ccw(_rotate_ccw(_rotate_ccw(shape)))


def _transform_shape_m(shape):
    return _flip_horizontal(shape)


def _transform_shape_m_r90(shape):
    return _rotate_ccw(_flip_horizontal(shape))


def _transform_shape_m_r180(shape):
    return _flip_vertical(shape)


def _transform_shape_m_r270(shape):
    return _flip_horizontal(_rotate_ccw(_rotate_ccw(shape)))


_SHAPE_TRANSFORMS = [
    lambda s: s,
    _transform_shape_r90,
    _transform_shape_r180,
    _transform_shape_r270,
    _transform_shape_m,
    _transform_shape_m_r90,
    _transform_shape_m_r180,
    _transform_shape_m_r270,
]


def _verify_orient_map_mask_consistency(pid: int, ti: int, om: List[int]) -> None:
    """Assert: transform_mask(mask[p][o]) == mask[p][om[o]] for each o."""
    n = int(ORIENT_COUNT[pid])
    for o in range(n):
        mask_o = None
        for pos in range(81):
            m0, m1, m2 = int(MASK_W0[pid, o, pos]), int(MASK_W1[pid, o, pos]), int(MASK_W2[pid, o, pos])
            if (m0 | m1 | m2) != 0:
                mask_o = _mask_words_to_plane(m0, m1, m2)
                break
        if mask_o is None:
            continue
        target = _transform_mask_plane(mask_o, ti)
        o2 = om[o]
        found = False
        for pos in range(81):
            m0, m1, m2 = int(MASK_W0[pid, o2, pos]), int(MASK_W1[pid, o2, pos]), int(MASK_W2[pid, o2, pos])
            if (m0 | m1 | m2) == 0:
                continue
            mask_o2 = _mask_words_to_plane(m0, m1, m2)
            if np.array_equal(target, mask_o2):
                found = True
                break
        assert found, f"D4: piece_id={pid} ti={ti} o={o} -> o2={o2} mask mismatch"


def _verify_round_trip_masks() -> None:
    """Round-trip: transform then inverse_transform recovers original mask."""
    _init_inv_index_2d()
    inv_ti = {0: 0, 1: 3, 2: 2, 3: 1, 4: 4, 5: 5, 6: 6, 7: 7}
    for ti in range(D4_COUNT):
        plane = np.zeros((BS, BS), dtype=np.float32)
        plane[4, 4] = 1.0
        transformed = _transform_mask_plane(plane, ti)
        recovered = _transform_mask_plane(transformed, inv_ti[ti])
        assert np.array_equal(plane, recovered), f"D4 round-trip failed for ti={ti}"


def _build_all_orient_maps() -> Dict[int, List[List[int]]]:
    """
    Build orient_map[piece_id][transform_idx][k] = k' for k in 0..7.
    Uses exact mask matching: transform_mask(mask[p][o]) == mask[p][o'].
    For k in n..7: identity (padding orients for pieces with n<8).
    """
    _verify_round_trip_masks()
    out: Dict[int, List[List[int]]] = {}
    for pid in PIECE_BY_ID:
        pid = int(pid)
        n_orient = int(ORIENT_COUNT[pid])
        maps = []
        for ti in range(D4_COUNT):
            om_partial = _build_orient_map_for_transform(pid, ti)
            if om_partial is None:
                raise RuntimeError(
                    f"D4 orient map: piece_id={pid} transform={D4_NAMES[ti]} "
                    "cannot be derived from mask matching. FAIL FAST."
                )
            n = len(om_partial)
            if n != n_orient:
                raise RuntimeError(
                    f"D4 orient map: piece_id={pid} transform={D4_NAMES[ti]} "
                    f"expected {n_orient} entries, got {n}"
                )
            # Extend to length 8: orients n..7 map to themselves (identity for padding)
            om = list(om_partial) + list(range(n, NUM_ORIENTS))
            maps.append(om)
            _verify_orient_map_mask_consistency(pid, ti, om)
        out[pid] = maps
    return out


_ORIENT_MAPS: Dict[int, List[List[int]]] = {}
_ORIENT_MAPS_INVERSE: Dict[int, List[List[int]]] = {}


def _get_orient_maps() -> Dict[int, List[List[int]]]:
    global _ORIENT_MAPS, _ORIENT_MAPS_INVERSE
    if not _ORIENT_MAPS:
        _ORIENT_MAPS = _build_all_orient_maps()
        # Build inverse for each (pid, ti): inv_om[o] = first k with om[k]=o
        _ORIENT_MAPS_INVERSE = {}
        for pid, maps in _ORIENT_MAPS.items():
            n_orient = int(ORIENT_COUNT[pid])
            inv_maps = []
            for ti, om in enumerate(maps):
                inv_om = [-1] * NUM_ORIENTS
                for k in range(NUM_ORIENTS):
                    o = om[k]
                    if inv_om[o] < 0:
                        inv_om[o] = k
                for o in range(n_orient, NUM_ORIENTS):
                    inv_om[o] = o  # Padding: identity
                inv_maps.append(inv_om)
            _ORIENT_MAPS_INVERSE[pid] = inv_maps
    return _ORIENT_MAPS


def get_orient_after_transform(piece_id: int, transform_idx: int, orient: int) -> int:
    """Given piece_id, transform, and orient k, return orient k' after transform."""
    maps = _get_orient_maps()
    if piece_id not in maps:
        raise RuntimeError(f"D4: unknown piece_id={piece_id}. FAIL FAST.")
    om = maps[piece_id][transform_idx]
    if orient >= len(om):
        return orient  # Padding orient: identity
    return om[orient]


def get_orient_before_transform(piece_id: int, transform_idx: int, orient: int) -> int:
    """
    Inverse: for output orient o, which input orient k produces it?
    Returns first (min) preimage. For padding orients (n..7), returns orient (identity).
    """
    _get_orient_maps()  # Ensure inverse is built
    if piece_id not in _ORIENT_MAPS_INVERSE:
        raise RuntimeError(f"D4: unknown piece_id={piece_id}. FAIL FAST.")
    inv_om = _ORIENT_MAPS_INVERSE[piece_id][transform_idx]
    n_orient = int(ORIENT_COUNT[piece_id])
    if orient >= n_orient:
        return orient  # Padding orient: identity
    k = inv_om[orient]
    if k < 0:
        raise RuntimeError(
            f"D4: orient_before piece_id={piece_id} transform={transform_idx} orient={orient} "
            "has no preimage"
        )
    return k


# Position transforms: (r, c) -> (r', c') for 9x9 board
def _pos_id(r: int, c: int) -> Tuple[int, int]:
    return (r, c)


def _pos_r90(r: int, c: int) -> Tuple[int, int]:
    return (c, BS - 1 - r)


def _pos_r180(r: int, c: int) -> Tuple[int, int]:
    return (BS - 1 - r, BS - 1 - c)


def _pos_r270(r: int, c: int) -> Tuple[int, int]:
    return (BS - 1 - c, r)


def _pos_m(r: int, c: int) -> Tuple[int, int]:
    return (r, BS - 1 - c)


def _pos_m_r90(r: int, c: int) -> Tuple[int, int]:
    r2, c2 = _pos_m(r, c)
    return _pos_r90(r2, c2)


def _pos_m_r180(r: int, c: int) -> Tuple[int, int]:
    r2, c2 = _pos_m(r, c)
    r3, c3 = _pos_r90(r2, c2)
    return _pos_r90(r3, c3)


def _pos_m_r270(r: int, c: int) -> Tuple[int, int]:
    r2, c2 = _pos_m(r, c)
    r3, c3 = _pos_r90(r2, c2)
    r4, c4 = _pos_r90(r3, c3)
    return _pos_r90(r4, c4)


_POS_TRANSFORMS = [
    _pos_id, _pos_r90, _pos_r180, _pos_r270,
    _pos_m, _pos_m_r90, _pos_m_r180, _pos_m_r270,
]


def transform_position(r: int, c: int, transform_idx: int) -> Tuple[int, int]:
    return _POS_TRANSFORMS[transform_idx](r, c)


# Cache for transform_buy_top_left: (pid, orient, top, left, ti) -> (top_new, left_new)
_BUY_TL_CACHE: Dict[Tuple[int, ...], Tuple[int, int]] = {}


def transform_buy_top_left(
    piece_id: int,
    orient: int,
    top: int,
    left: int,
    transform_idx: int,
) -> Tuple[int, int]:
    """
    Dimension-aware transform of buy (top, left). Uses bulletproof method:
    - build occupied cell list from mask at (top, left)
    - transform each cell via board position transform
    - top_new, left_new = min row/col of transformed cells
    Matches encoder top-left anchoring convention.
    """
    key = (piece_id, orient, top, left, transform_idx)
    if key in _BUY_TL_CACHE:
        return _BUY_TL_CACHE[key]
    if transform_idx == 0:
        res = (top, left)
        _BUY_TL_CACHE[key] = res
        return res
    pos = top * BS + left
    m0 = int(MASK_W0[piece_id, orient, pos])
    m1 = int(MASK_W1[piece_id, orient, pos])
    m2 = int(MASK_W2[piece_id, orient, pos])
    if (m0 | m1 | m2) == 0:
        res = (top, left)
        _BUY_TL_CACHE[key] = res
        return res
    plane = _mask_words_to_plane(m0, m1, m2)
    occupied: List[Tuple[int, int]] = []
    for r in range(BS):
        for c in range(BS):
            if plane[r, c] > 0:
                occupied.append((r, c))
    if not occupied:
        res = (top, left)
        _BUY_TL_CACHE[key] = res
        return res
    transformed = [_POS_TRANSFORMS[transform_idx](r, c) for r, c in occupied]
    res = (min(p[0] for p in transformed), min(p[1] for p in transformed))
    _BUY_TL_CACHE[key] = res
    return res


def inverse_transform_idx(transform_idx: int) -> int:
    """Return transform index that undoes the given transform."""
    inv = {0: 0, 1: 3, 2: 2, 3: 1, 4: 4, 5: 5, 6: 6, 7: 7}
    return inv[transform_idx]


def _build_inverse_pos_map(transform_idx: int) -> np.ndarray:
    """Map: new_pos -> old_pos for applying transform (old content goes to new cell)."""
    arr = np.zeros(81, dtype=np.int32)
    for pos in range(81):
        r, c = pos // BS, pos % BS
        rn, cn = _POS_TRANSFORMS[transform_idx](r, c)
        arr[rn * BS + cn] = pos
    return arr


# Precomputed 2D index arrays for vectorized _transform_board_plane (module load).
# _INV_INDEX_2D[ti] = (row_idx, col_idx) s.t. out = plane[row_idx, col_idx] yields transformed 9x9.
_INV_INDEX_2D: List[Tuple[np.ndarray, np.ndarray]] = []


def _init_inv_index_2d() -> None:
    """Precompute row/col index arrays for each D4 transform."""
    global _INV_INDEX_2D
    if _INV_INDEX_2D:
        return
    for ti in range(D4_COUNT):
        inv_map = _build_inverse_pos_map(ti)
        # inv_map[new_pos] = old_pos; for advanced indexing we need (ro, co) per (rn, cn)
        old_pos_flat = inv_map  # shape (81,)
        row_idx = (old_pos_flat // BS).reshape(BS, BS).astype(np.intp)
        col_idx = (old_pos_flat % BS).reshape(BS, BS).astype(np.intp)
        _INV_INDEX_2D.append((row_idx, col_idx))


def _transform_board_plane(plane: np.ndarray, transform_idx: int) -> np.ndarray:
    """Transform a 9x9 plane by moving content. new[r,c] = old[inv_T(r,c)]. Pure vectorized."""
    _init_inv_index_2d()
    row_idx, col_idx = _INV_INDEX_2D[transform_idx]
    return np.ascontiguousarray(plane[row_idx, col_idx])


def _transform_board_plane_batch(planes: np.ndarray, transform_idx: int) -> np.ndarray:
    """Transform (B, 9, 9) planes. Same spatial perm for all batch elements."""
    _init_inv_index_2d()
    row_idx, col_idx = _INV_INDEX_2D[transform_idx]
    return np.ascontiguousarray(planes[:, row_idx, col_idx])


def _transform_coord_planes(
    row_plane: np.ndarray,
    col_plane: np.ndarray,
    transform_idx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform coord_row and coord_col under D4.
    Same spatial transform as boards: content moves with cells (new[r,c]=old[inv_T(r,c)]).
    Ensures round-trip T then inv(T) recovers exactly.
    """
    return (
        _transform_board_plane(row_plane, transform_idx),
        _transform_board_plane(col_plane, transform_idx),
    )


def _inverse_pos(r: int, c: int, transform_idx: int) -> Tuple[int, int]:
    inv_ti = inverse_transform_idx(transform_idx)
    return _POS_TRANSFORMS[inv_ti](r, c)


def transform_state(
    state: np.ndarray,
    transform_idx: int,
    slot_piece_ids: List[Optional[int]],
) -> np.ndarray:
    """
    Transform state (32, 9, 9) for gold_v2. All 32 channels spatial.
    Channels 0-7: board/coord/frontier/valid_7x7 — direct spatial transform.
    Channels 8-31: shape planes — spatial + orient permute.
    (legalTL is not stored in the state; computed on-GPU at inference time.)
    """
    out = state.copy()
    ti = transform_idx

    # 0-7: boards, coords, frontier, valid_7x7
    for ch in range(8):
        out[ch] = _transform_board_plane(state[ch], ti)

    # 8-31: slot×orient shape planes
    orient_maps = _get_orient_maps()
    for slot in range(NUM_SLOTS):
        pid = slot_piece_ids[slot] if slot < len(slot_piece_ids) else None
        n_orient = int(ORIENT_COUNT[pid]) if pid is not None else NUM_ORIENTS
        if pid is None or n_orient == 0:
            for o in range(NUM_ORIENTS):
                ch = SLOT_ORIENT_BASE + slot * NUM_ORIENTS + o
                out[ch] = _transform_board_plane(state[ch], ti)
        else:
            for o in range(NUM_ORIENTS):
                o_old = get_orient_before_transform(pid, ti, o)
                ch_src = SLOT_ORIENT_BASE + slot * NUM_ORIENTS + o_old
                ch_dst = SLOT_ORIENT_BASE + slot * NUM_ORIENTS + o
                plane = _transform_board_plane(state[ch_src], ti)
                out[ch_dst] = plane

    return out


def transform_action_vector(
    arr: np.ndarray,
    transform_idx: int,
    slot_piece_ids: List[Optional[int]],
) -> np.ndarray:
    """
    Remap policy or mask (2026) under D4 transform.
    Permutation: no renormalization needed.
    """
    out = np.zeros_like(arr)
    ti = transform_idx

    out[PASS_INDEX] = arr[PASS_INDEX]

    for old_pos in range(81):
        r, c = old_pos // BS, old_pos % BS
        rn, cn = _POS_TRANSFORMS[ti](r, c)
        new_pos = rn * BS + cn
        out[PATCH_START + new_pos] = arr[PATCH_START + old_pos]

    for old_idx in range(BUY_START, BUY_START + NUM_SLOTS * NUM_ORIENTS * 81):
        val = arr[old_idx]
        if val == 0:
            continue  # Leave output 0 (from zeros_like); bijection ensures unique target
        k = old_idx - BUY_START
        so, pos = divmod(k, 81)
        slot, orient = divmod(so, NUM_ORIENTS)
        top, left = pos // BS, pos % BS

        pid = slot_piece_ids[slot] if slot < len(slot_piece_ids) else None
        if pid is not None and orient < int(ORIENT_COUNT[pid]):
            orient_new = get_orient_after_transform(pid, ti, orient)
            top_new, left_new = transform_buy_top_left(pid, orient, top, left, ti)
        else:
            orient_new = orient
            top_new, left_new = _POS_TRANSFORMS[ti](top, left)
        pos_new = top_new * BS + left_new
        new_idx = BUY_START + (slot * NUM_ORIENTS + orient_new) * 81 + pos_new
        out[new_idx] = val  # Bijection: each target written exactly once

    return out


def apply_d4_augment(
    state: np.ndarray,
    policy: np.ndarray,
    mask: np.ndarray,
    slot_piece_ids: List[Optional[int]],
    transform_idx: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply D4 transform. Returns (state, policy, mask). Policy renormalized; mask binarized."""
    new_state = transform_state(state, transform_idx, slot_piece_ids)
    new_policy = transform_action_vector(policy, transform_idx, slot_piece_ids)
    new_mask = transform_action_vector(mask, transform_idx, slot_piece_ids)

    total = float(new_policy.sum())
    if total > 0:
        new_policy = (new_policy / total).astype(np.float32)
    new_mask = (new_mask > 0).astype(np.float32)
    return new_state, new_policy, new_mask


def transform_legalTL_planes(
    L: np.ndarray,
    transform_idx: int,
    slot_piece_ids: List[Optional[int]],
) -> np.ndarray:
    """
    Transform legalTL (24, 9, 9) using scatter-map with dimension-aware buy transform.
    For each old (slot, orient, top, left) with value v:
      compute (orient_new, top_new, left_new) via get_orient_after_transform + transform_buy_top_left
      write out[slot, orient_new, top_new, left_new] = v
    Ensures legalTL transforms exactly like the buy action space.
    """
    out = np.zeros_like(L)
    ti = transform_idx
    for slot in range(NUM_SLOTS):
        pid = slot_piece_ids[slot] if slot < len(slot_piece_ids) else None
        for orient in range(NUM_ORIENTS):
            for top in range(BS):
                for left in range(BS):
                    v = L[slot * NUM_ORIENTS + orient, top, left]
                    if v == 0:
                        continue
                    if pid is not None and orient < int(ORIENT_COUNT[pid]):
                        orient_new = get_orient_after_transform(pid, ti, orient)
                        top_new, left_new = transform_buy_top_left(
                            pid, orient, top, left, ti
                        )
                    else:
                        orient_new = orient
                        top_new, left_new = _POS_TRANSFORMS[ti](top, left)
                    out[slot * NUM_ORIENTS + orient_new, top_new, left_new] = v
    return out


# Cache: (ti, p0, p1, p2) -> perm (1944,) for legalTL flat indexing; out_flat = in_flat[perm]
_LEGALTL_PERM_CACHE: Dict[Tuple[int, ...], np.ndarray] = {}


def _get_legalTL_perm(
    transform_idx: int,
    slot_piece_ids: Tuple[Optional[int], Optional[int], Optional[int]],
) -> np.ndarray:
    """
    Cached permutation for batched legalTL transform.
    perm[dst_flat] = src_flat (input index that maps to output position dst_flat).
    LegalTL layout: (24, 9, 9) = slot*8+orient, top, left; flat = (so)*81 + top*9+left.
    """
    key = (transform_idx, slot_piece_ids[0] or -1, slot_piece_ids[1] or -1, slot_piece_ids[2] or -1)
    if key in _LEGALTL_PERM_CACHE:
        return _LEGALTL_PERM_CACHE[key]
    ti = transform_idx
    p0, p1, p2 = slot_piece_ids
    pids = [p0, p1, p2]
    perm = np.zeros(24 * 81, dtype=np.int32)  # 1944
    for k in range(1944):
        so, pos = divmod(k, 81)
        slot, orient = divmod(so, NUM_ORIENTS)
        top, left = pos // BS, pos % BS
        pid = pids[slot] if slot < len(pids) else None
        if pid is not None and orient < int(ORIENT_COUNT[pid]):
            orient_new = get_orient_after_transform(pid, ti, orient)
            top_new, left_new = transform_buy_top_left(pid, orient, top, left, ti)
        else:
            orient_new = orient
            rn, cn = _POS_TRANSFORMS[ti](top, left)
            top_new, left_new = rn, cn
        dst_flat = (slot * NUM_ORIENTS + orient_new) * 81 + top_new * BS + left_new
        perm[dst_flat] = k
    _LEGALTL_PERM_CACHE[key] = perm
    return perm


def _transform_legalTL_batch(
    L_batch: np.ndarray,
    transform_idx: int,
    slot_piece_ids: Tuple[Optional[int], Optional[int], Optional[int]],
) -> np.ndarray:
    """
    Batched legalTL transform. L_batch: (B, 24, 9, 9).
    Uses cached permutation: out_flat = in_flat[perm].
    """
    perm = _get_legalTL_perm(transform_idx, slot_piece_ids)
    # L_batch (B, 24, 9, 9) -> flat (B, 1944)
    flat = L_batch.reshape(L_batch.shape[0], -1)
    out_flat = flat[:, perm]
    return out_flat.reshape(L_batch.shape)


def get_d4_transform_tag(transform_idx: int) -> str:
    """Tag for storage (e.g. ownership flip mapping)."""
    return D4_NAMES[transform_idx]


def get_d4_transform_idx(tag: str) -> int:
    """Inverse: tag -> transform index."""
    try:
        return D4_NAMES.index(tag)
    except ValueError:
        legacy = {"none": 0, "id": 0, "v": 6, "h": 4, "vh": 2}
        return legacy.get(tag, 0)


def apply_ownership_transform(ownership: np.ndarray, transform_idx: int) -> np.ndarray:
    """
    Transform (2, 9, 9) ownership map to match transformed state.
    Uses the same spatial move as state boards.
    """
    out = np.zeros_like(ownership)
    for ch in range(2):
        out[ch] = _transform_board_plane(ownership[ch], transform_idx)
    return out


# --- Batched D4 (vectorized, for training Dataset __getitem__) ---
# Precomputed patch permutation: for each ti, src_perm[ti][new_pos] = old_pos
_PATCH_SRC_INDEX: Optional[np.ndarray] = None
# Cache: (ti, p0, p1, p2) -> buy_src_indices (1944,) for fast advanced indexing
_BUY_SRC_CACHE: Dict[Tuple[int, ...], np.ndarray] = {}


def _get_patch_src_index() -> np.ndarray:
    """Shape (8, 81): patch_src_index[ti, new_pos] = old_pos for PATCH_START+1..81."""
    global _PATCH_SRC_INDEX
    if _PATCH_SRC_INDEX is not None:
        return _PATCH_SRC_INDEX
    arr = np.zeros((D4_COUNT, 81), dtype=np.int32)
    for ti in range(D4_COUNT):
        for old_pos in range(81):
            r, c = old_pos // BS, old_pos % BS
            rn, cn = _POS_TRANSFORMS[ti](r, c)
            new_pos = rn * BS + cn
            arr[ti, new_pos] = old_pos
    _PATCH_SRC_INDEX = arr
    return arr


def _transform_state_batch_uniform(
    states: np.ndarray,
    transform_idx: int,
    slot_piece_ids: Tuple[Optional[int], Optional[int], Optional[int]],
) -> np.ndarray:
    """
    Vectorized transform_state for batch where all samples share same (ti, p0, p1, p2).
    states: (B, 32, 9, 9) for gold_v2
    """
    out = states.copy()
    ti = transform_idx
    p0, p1, p2 = slot_piece_ids

    # 0-7: boards, coords, frontier, valid_7x7
    for ch in range(8):
        out[:, ch] = _transform_board_plane_batch(states[:, ch], ti)

    # 8-31: shape planes
    for slot, pid in enumerate([p0, p1, p2]):
        n_orient = int(ORIENT_COUNT[pid]) if pid is not None else NUM_ORIENTS
        if pid is None or n_orient == 0:
            for o in range(NUM_ORIENTS):
                ch = SLOT_ORIENT_BASE + slot * NUM_ORIENTS + o
                out[:, ch] = _transform_board_plane_batch(states[:, ch], ti)
        else:
            for o in range(NUM_ORIENTS):
                o_old = get_orient_before_transform(pid, ti, o)
                ch_src = SLOT_ORIENT_BASE + slot * NUM_ORIENTS + o_old
                ch_dst = SLOT_ORIENT_BASE + slot * NUM_ORIENTS + o
                out[:, ch_dst] = _transform_board_plane_batch(states[:, ch_src], ti)

    return out


def _get_buy_src_index(
    transform_idx: int,
    slot_piece_ids: Tuple[Optional[int], Optional[int], Optional[int]],
) -> np.ndarray:
    """Cached (1944,) src indices for BUY section. dst[i] = arr[buy_start + src[i]]. Uses dimension-aware transform for buy top-left."""
    key = (transform_idx, slot_piece_ids[0] or -1, slot_piece_ids[1] or -1, slot_piece_ids[2] or -1)
    if key in _BUY_SRC_CACHE:
        return _BUY_SRC_CACHE[key]
    p0, p1, p2 = slot_piece_ids
    src = np.arange(1944, dtype=np.int32)
    for k in range(1944):
        so, pos = divmod(k, 81)
        slot, orient = divmod(so, NUM_ORIENTS)
        top, left = pos // BS, pos % BS
        pids = [p0, p1, p2]
        pid = pids[slot] if slot < len(pids) else None
        if pid is not None and orient < int(ORIENT_COUNT[pid]):
            orient_new = get_orient_after_transform(pid, transform_idx, orient)
            top_new, left_new = transform_buy_top_left(pid, orient, top, left, transform_idx)
        else:
            orient_new = orient
            rn, cn = _POS_TRANSFORMS[transform_idx](top, left)
            top_new, left_new = rn, cn
        pos_new = top_new * BS + left_new
        dst_flat = (slot * NUM_ORIENTS + orient_new) * 81 + pos_new
        src[dst_flat] = k
    _BUY_SRC_CACHE[key] = src
    return src


def _transform_action_vector_batch_uniform(
    arr: np.ndarray,
    transform_idx: int,
    slot_piece_ids: Tuple[Optional[int], Optional[int], Optional[int]],
) -> np.ndarray:
    """
    Vectorized transform_action_vector for batch with same (ti, p0, p1, p2).
    arr: (B, 2026)
    """
    out = np.zeros_like(arr)
    ti = transform_idx

    out[:, PASS_INDEX] = arr[:, PASS_INDEX]

    patch_src = _get_patch_src_index()[ti]
    out[:, PATCH_START : PATCH_START + 81] = arr[:, PATCH_START + patch_src]

    buy_src = _get_buy_src_index(ti, slot_piece_ids)
    out[:, BUY_START:] = arr[:, BUY_START + buy_src]

    return out


def apply_d4_augment_batch(
    states: np.ndarray,
    policies: np.ndarray,
    masks: np.ndarray,
    slot_ids: np.ndarray,
    transform_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Batched D4 augmentation. Groups samples by (ti, p0, p1, p2) and applies
    vectorized transforms. ~10-50x faster than per-sample loop.
    """
    n = states.shape[0]
    out_states = np.empty_like(states)
    out_policies = np.empty_like(policies)
    out_masks = np.empty_like(masks)

    # Build group key: (ti, p0, p1, p2) with -1 for None
    slot_ids_int = np.where(slot_ids >= 0, slot_ids, -1).astype(np.int32)
    keys = [
        (int(transform_indices[i]), int(slot_ids_int[i, 0]), int(slot_ids_int[i, 1]), int(slot_ids_int[i, 2]))
        for i in range(n)
    ]
    key_to_indices: Dict[tuple, List[int]] = {}
    for i, k in enumerate(keys):
        key_to_indices.setdefault(k, []).append(i)

    for (ti, p0, p1, p2), indices in key_to_indices.items():
        idx = np.array(indices)
        p0_n = None if p0 < 0 else p0
        p1_n = None if p1 < 0 else p1
        p2_n = None if p2 < 0 else p2
        slot_tuple = (p0_n, p1_n, p2_n)

        sub_states = states[idx]
        sub_policies = policies[idx]
        sub_masks = masks[idx]

        out_states[idx] = _transform_state_batch_uniform(sub_states, ti, slot_tuple)
        out_policies[idx] = _transform_action_vector_batch_uniform(
            sub_policies, ti, slot_tuple
        )
        out_masks[idx] = _transform_action_vector_batch_uniform(sub_masks, ti, slot_tuple)

    # Renormalize policies
    totals = out_policies.sum(axis=1, keepdims=True)
    np.divide(out_policies, totals, out=out_policies, where=totals > 0)
    out_masks = (out_masks > 0).astype(np.float32)

    return out_states, out_policies, out_masks


def apply_ownership_transform_batch(
    ownerships: np.ndarray, transform_indices: np.ndarray
) -> np.ndarray:
    """
    Batched ownership transform. Groups by transform index, applies vectorized
    plane transform per group.
    ownerships: (B, 2, 9, 9)
    """
    n = ownerships.shape[0]
    out = np.empty_like(ownerships)

    for ti in range(D4_COUNT):
        mask = transform_indices == ti
        if not np.any(mask):
            continue
        sub = ownerships[mask]  # (G, 2, 9, 9)
        for ch in range(2):
            out[mask, ch] = _transform_board_plane_batch(sub[:, ch], ti)

    return out


# Eager init: precompute orient maps at import so first training batch doesn't stall (0% GPU)
import logging as _d4_logging
_d4_log = _d4_logging.getLogger(__name__)
_d4_log.info("D4: precomputing orient maps (one-time)...")
_get_orient_maps()
_d4_log.info("D4: orient maps ready.")
