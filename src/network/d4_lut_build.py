"""
Fast vectorized D4 LUT build.

Precomputes (ti, pid, orient, pos) -> (orient_new, pos_new) table, then fills
LUTs with numpy operations (no per-key Python loops over 1944).
"""

from __future__ import annotations

import os
import numpy as np

from src.network.d4_constants import COMPACT_SIZE, PC_MAX
from src.network.d4_lut_cache import LEGAL_FLAT_SIZE

D4_COUNT = 8
NUM_ORIENTS = 8
BS = 9


def _build_transform_table() -> np.ndarray:
    """
    table[ti, pid_idx, orient, pos] -> (orient_new, pos_new).
    pid_idx: 0=empty, 1..33=piece 1..32 (game piece IDs 1-32; 0 reserved).
    For empty or orient >= ORIENT_COUNT: use spatial-only transform.
    """
    from src.network import d4_augmentation as _d4
    from src.game.patchwork_engine import ORIENT_COUNT

    _d4._init_inv_index_2d()
    table = np.zeros((D4_COUNT, 34, NUM_ORIENTS, 81, 2), dtype=np.int32)
    for ti in range(D4_COUNT):
        for pid in range(-1, 33):
            pid_idx = pid + 1
            for orient in range(NUM_ORIENTS):
                for pos in range(81):
                    top, left = pos // BS, pos % BS
                    if pid < 0 or orient >= int(ORIENT_COUNT[pid]):
                        rn, cn = _d4._POS_TRANSFORMS[ti](top, left)
                        orient_new = orient
                        pos_new = rn * BS + cn
                    else:
                        orient_new = _d4.get_orient_after_transform(pid, ti, orient)
                        top_new, left_new = _d4.transform_buy_top_left(pid, orient, top, left, ti)
                        pos_new = top_new * BS + left_new
                    table[ti, pid_idx, orient, pos, 0] = orient_new
                    table[ti, pid_idx, orient, pos, 1] = pos_new
    return table


def build_buy_legal_luts_fast() -> np.ndarray:
    """
    Build (COMPACT_SIZE, 1944) buy LUT using vectorized operations.
    Returns buy_lut only (legal_lut removed — legalTL computed on-GPU).
    """
    table = _build_transform_table()

    ti_arr = np.arange(COMPACT_SIZE, dtype=np.int32) // (PC_MAX ** 3)
    rem = np.arange(COMPACT_SIZE, dtype=np.int32) % (PC_MAX ** 3)
    p0c = rem // (PC_MAX ** 2)
    rem = rem % (PC_MAX ** 2)
    p1c = rem // PC_MAX
    p2c = rem % PC_MAX
    p0 = np.where(p0c == 0, -1, p0c)
    p1 = np.where(p1c == 0, -1, p1c)
    p2 = np.where(p2c == 0, -1, p2c)

    idx_range = np.arange(COMPACT_SIZE, dtype=np.int32)
    # buy: identity init (match CPU src=np.arange(1944))
    buy_lut = np.arange(LEGAL_FLAT_SIZE, dtype=np.int32)[np.newaxis, :].repeat(COMPACT_SIZE, axis=0)

    for k in range(LEGAL_FLAT_SIZE):
        so, pos = divmod(k, 81)
        slot = so // NUM_ORIENTS
        orient = so % NUM_ORIENTS
        pid = np.where(slot == 0, p0, np.where(slot == 1, p1, p2))
        pid_idx = np.clip(pid + 1, 0, 33).astype(np.int32)
        orient_new = table[ti_arr, pid_idx, orient, pos, 0]
        pos_new = table[ti_arr, pid_idx, orient, pos, 1]
        dst_flat = (slot * NUM_ORIENTS + orient_new) * 81 + pos_new
        buy_lut[idx_range, dst_flat] = k

    return buy_lut
