"""
WorkerSharedBuffer: Per-worker shared memory layout for zero-copy IPC between
MCTS workers (CPU) and the GPU inference server.

Each worker has one SharedMemory block divided into `n_slots` slots.
n_slots = parallel_leaves (max concurrent pending inference requests per worker).

Slot layout (all float32 except shop_ids int16, n_legal int32):
  Field         Shape                dtype    bytes
  spatial       (36, 9, 9)           f32      11664   [C_SPATIAL_ENC=36]
  global        (65,)                f32        260   [F_GLOBAL=65]
  track         (8, 54)              f32       1728
  shop_ids      (33,)                i16         68   (66 + 2 pad for alignment)
  shop_feats    (33, 10)             f32       1320
  mask          (2026,)              f32       8104
  legal_idxs    (2026,)              i32       8104   (max size)
  n_legal       (1,)                 i32          4
                                             ------
  SLOT_BYTES                                  31252   [computed from gold_v2_constants]

NOTE: All sizes are computed dynamically from gold_v2_constants (C_SPATIAL_ENC, F_GLOBAL, etc.)
so this docstring must be updated whenever those constants change.
"""

from __future__ import annotations

from multiprocessing.shared_memory import SharedMemory
from typing import Dict, Optional, Tuple

import numpy as np

from src.network.gold_v2_constants import (
    C_SPATIAL_ENC,
    C_TRACK,
    F_GLOBAL,
    F_SHOP,
    MAX_ACTIONS,
    NMAX,
    TRACK_LEN,
)

# Module-level max payload size seen in check_slot_write_bounds (for diagnostics; thread-safe not required).
_max_payload_nbytes_seen: int = 0


class WorkerSharedBuffer:
    """Per-worker shared memory block. GPU server reads from it without pickling."""

    _S  = C_SPATIAL_ENC * 9 * 9 * 4        # spatial  (C_SPATIAL_ENC,9,9)  f32: C_SPATIAL_ENC*324 bytes
    _G  = F_GLOBAL * 4                       # global   (F_GLOBAL,)          f32: F_GLOBAL*4 bytes
    _T  = C_TRACK * TRACK_LEN * 4           # track    (8,54)                f32:  1728 bytes
    _SI = ((NMAX * 2) + 3) & ~3             # shop_ids (33,)                 i16:    68 bytes (4-byte aligned)
    _SF = NMAX * F_SHOP * 4                  # shop_feats (33,10)             f32:  1320 bytes
    _M  = MAX_ACTIONS * 4                    # mask     (2026,)               f32:  8104 bytes
    _L  = MAX_ACTIONS * 4                    # legal_idxs (2026,)             i32:  8104 bytes (max size)
    _NL = 4                                  # n_legal  (1,)                  i32:     4 bytes
    SLOT_BYTES: int = _S + _G + _T + _SI + _SF + _M + _L + _NL  # computed from constants
    # Bounds check: payload written = fixed part + n_legal*4; must not exceed one slot.
    HEADER_BYTES: int = 0  # no separate header; slot is contiguous layout
    PAYLOAD_FIXED_BYTES: int = _S + _G + _T + _SI + _SF + _M + _NL  # computed (slot minus legal_idxs max)

    OFF_S  = 0
    OFF_G  = OFF_S  + _S
    OFF_T  = OFF_G  + _G
    OFF_SI = OFF_T  + _T
    OFF_SF = OFF_SI + _SI
    OFF_M  = OFF_SF + _SF
    OFF_L  = OFF_M  + _M
    OFF_NL = OFF_L  + _L

    def __init__(
        self,
        n_slots: Optional[int],
        worker_id: int = 0,
        create: bool = True,
        name: Optional[str] = None,
        expected_n_slots: Optional[int] = None,
    ) -> None:
        if create:
            assert n_slots is not None and n_slots > 0, "n_slots required when create=True"
            total = n_slots * self.SLOT_BYTES
            self._shm = SharedMemory(create=True, size=max(1, total))
            self.n_slots = n_slots
        else:
            assert name is not None, "name required when create=False"
            self._shm = SharedMemory(create=False, name=name)
            # SHM ATTACH VALIDATION: prevent n_slots=0 or undersized segments (can cause
            # STATUS_STACK_BUFFER_OVERRUN in child when creating numpy views).
            if self._shm.size < self.SLOT_BYTES:
                raise ValueError(
                    f"SHM buffer too small: name={name!r} size={self._shm.size} "
                    f"< SLOT_BYTES={self.SLOT_BYTES}"
                )
            derived = self._shm.size // self.SLOT_BYTES
            if derived < 1:
                raise ValueError(
                    f"SHM buffer has no full slots: name={name!r} size={self._shm.size} "
                    f"SLOT_BYTES={self.SLOT_BYTES} derived={derived}"
                )
            if expected_n_slots is not None and derived < expected_n_slots:
                raise ValueError(
                    f"SHM slot count too small: name={name!r} size={self._shm.size} "
                    f"SLOT_BYTES={self.SLOT_BYTES} derived={derived} expected_n_slots={expected_n_slots}"
                )
            self.n_slots = n_slots if n_slots is not None else derived

        self.name = self._shm.name
        self.worker_id = worker_id

    # ------------------------------------------------------------------
    # Slot base address
    # ------------------------------------------------------------------

    def _base(self, slot: int) -> int:
        return slot * self.SLOT_BYTES

    def check_slot_write_bounds(
        self,
        slot_idx: int,
        n_legal: int,
        *,
        worker_id: int = 0,
        expected_n_slots: int | None = None,
        derived_n_slots: int | None = None,
    ) -> None:
        """Raise ValueError if writing n_legal legal indices to slot_idx would exceed slot capacity."""
        derived = derived_n_slots if derived_n_slots is not None else self.n_slots
        if slot_idx < 0 or slot_idx >= derived:
            raise ValueError(
                f"SHM slot index out of range: worker_id={worker_id} slot_idx={slot_idx} "
                f"expected_n_slots={expected_n_slots} derived_n_slots={derived}"
            )
        payload_nbytes = self.PAYLOAD_FIXED_BYTES + n_legal * 4
        global _max_payload_nbytes_seen
        if payload_nbytes > _max_payload_nbytes_seen:
            _max_payload_nbytes_seen = payload_nbytes
        if payload_nbytes > self.SLOT_BYTES:
            raise ValueError(
                f"SHM slot write overflow: worker_id={worker_id} slot_idx={slot_idx} "
                f"payload_nbytes={payload_nbytes} SLOT_BYTES={self.SLOT_BYTES} HEADER_BYTES={self.HEADER_BYTES} "
                f"expected_n_slots={expected_n_slots} derived_n_slots={derived}"
            )

    # ------------------------------------------------------------------
    # Zero-copy numpy views into shared memory (no allocation)
    # ------------------------------------------------------------------

    def spatial_view(self, slot: int) -> np.ndarray:
        return np.ndarray((C_SPATIAL_ENC, 9, 9), dtype=np.float32,
                          buffer=self._shm.buf, offset=self._base(slot) + self.OFF_S)

    def global_view(self, slot: int) -> np.ndarray:
        return np.ndarray((F_GLOBAL,), dtype=np.float32,
                          buffer=self._shm.buf, offset=self._base(slot) + self.OFF_G)

    def track_view(self, slot: int) -> np.ndarray:
        return np.ndarray((C_TRACK, TRACK_LEN), dtype=np.float32,
                          buffer=self._shm.buf, offset=self._base(slot) + self.OFF_T)

    def shopids_view(self, slot: int) -> np.ndarray:
        return np.ndarray((NMAX,), dtype=np.int16,
                          buffer=self._shm.buf, offset=self._base(slot) + self.OFF_SI)

    def shopfeats_view(self, slot: int) -> np.ndarray:
        return np.ndarray((NMAX, F_SHOP), dtype=np.float32,
                          buffer=self._shm.buf, offset=self._base(slot) + self.OFF_SF)

    def mask_view(self, slot: int) -> np.ndarray:
        return np.ndarray((MAX_ACTIONS,), dtype=np.float32,
                          buffer=self._shm.buf, offset=self._base(slot) + self.OFF_M)

    def legalidxs_view(self, slot: int, n: int) -> np.ndarray:
        return np.ndarray((n,), dtype=np.int32,
                          buffer=self._shm.buf, offset=self._base(slot) + self.OFF_L)

    def write_nlegal(self, slot: int, n: int) -> None:
        np.ndarray((1,), dtype=np.int32,
                   buffer=self._shm.buf,
                   offset=self._base(slot) + self.OFF_NL)[0] = n

    def read_nlegal(self, slot: int) -> int:
        return int(np.ndarray((1,), dtype=np.int32,
                               buffer=self._shm.buf,
                               offset=self._base(slot) + self.OFF_NL)[0])

    def read_all(
        self, slot: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (spatial, global, track, shop_ids, shop_feats, mask, legal_idxs) views."""
        n = self.read_nlegal(slot)
        return (
            self.spatial_view(slot),
            self.global_view(slot),
            self.track_view(slot),
            self.shopids_view(slot),
            self.shopfeats_view(slot),
            self.mask_view(slot),
            self.legalidxs_view(slot, n),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Detach from shared memory (does not unlink/delete it)."""
        try:
            self._shm.close()
        except Exception:
            pass

    def destroy(self) -> None:
        """Detach and unlink (delete) shared memory segment."""
        try:
            self._shm.close()
            self._shm.unlink()
        except Exception:
            pass


def get_shm_safety_margin() -> tuple[int, int, int]:
    """Return (max_payload_nbytes_seen, capacity_bytes, margin_bytes). capacity = SLOT_BYTES - HEADER_BYTES."""
    capacity = WorkerSharedBuffer.SLOT_BYTES - WorkerSharedBuffer.HEADER_BYTES
    margin = capacity - _max_payload_nbytes_seen
    return (_max_payload_nbytes_seen, capacity, margin)
