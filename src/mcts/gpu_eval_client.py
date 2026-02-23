"""
GPU Eval Client (worker-side)

Used by CPU self-play workers to request batched inference from the GPUInferenceServer.

Design goals:
- Safe under Windows spawn
- Correct routing (request_id)
- Handles out-of-order responses

Protocol (dict payload):
- encoding_version: str ("gold_v2_32ch" | "full_clarity_v1")
- action_mask: (2026,) float32
- legal_idxs: (K,) int32
- Legacy (deprecated): spatial-only state — use gold_v2 instead
- Gold v2 pickle: x_spatial (32,9,9), x_global (61,), x_track (8,54), shop_ids (33,), shop_feats (33,10)
- Gold v2 SHM:   encoding_version="gold_v2_32ch", slot=int, n_legal=int (data in shared memory)

FIX CHANGELOG:
- [C2] Request IDs are now globally unique: (pid << 32) | counter.
- [C3] Multimodal support: submit_legacy, submit_multimodal, dict payload with encoding_version.
- [C4] SHM support: submit_shm sends tiny metadata; data lives in WorkerSharedBuffer.
"""

from __future__ import annotations

import logging
import os
import queue
import time
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

ENCODING_LEGACY = "full_clarity_v1"
ENCODING_GOLD_V2 = "gold_v2_32ch"


class GPUEvalClient:
    def __init__(
        self,
        req_q,
        resp_q,
        worker_id: int = 0,
        timeout_s: Optional[float] = None,
        retry_attempts: int = 3,
    ):
        self.req_q = req_q
        self.resp_q = resp_q
        self.worker_id = int(worker_id)
        self.timeout_s = timeout_s
        self.retry_attempts = int(retry_attempts)

        # [C2 FIX] Globally unique IDs: embed PID in high bits so workers
        # sharing the same resp_q never collide.
        self._pid_prefix = (os.getpid() & 0xFFFF) << 32
        self._next_counter = 1
        self._stash: Dict[int, Tuple[np.ndarray, float, float]] = {}

    def _next_id(self) -> int:
        rid = self._pid_prefix | self._next_counter
        self._next_counter += 1
        return rid

    def submit(self, state_np: np.ndarray, mask_np: np.ndarray, legal_idxs_np: np.ndarray) -> int:
        """Legacy (deprecated): enqueue spatial-only state request. For gold_v2 use submit_multimodal."""
        return self.submit_legacy(state_np, mask_np, legal_idxs_np)

    def submit_legacy(
        self,
        state_np: np.ndarray,
        mask_np: np.ndarray,
        legal_idxs_np: np.ndarray,
    ) -> int:
        """Enqueue legacy (full_clarity_v1) inference request."""
        rid = self._next_id()
        if state_np.dtype != np.float32:
            state_np = state_np.astype(np.float32, copy=False)
        mask_np = np.asarray(mask_np, dtype=np.float32)
        legal_idxs_np = np.asarray(legal_idxs_np, dtype=np.int32)
        payload = {
            "rid": rid,
            "wid": self.worker_id,
            "encoding_version": ENCODING_LEGACY,
            "state": state_np,
            "action_mask": mask_np,
            "legal_idxs": legal_idxs_np,
        }
        self.req_q.put(payload)
        return rid

    def submit_multimodal(
        self,
        x_spatial: np.ndarray,
        x_global: np.ndarray,
        x_track: np.ndarray,
        shop_ids: np.ndarray,
        shop_feats: np.ndarray,
        mask_np: np.ndarray,
        legal_idxs_np: np.ndarray,
    ) -> int:
        """Enqueue gold_v2_multimodal inference request."""
        rid = self._next_id()
        x_spatial = np.asarray(x_spatial, dtype=np.float32)
        x_global = np.asarray(x_global, dtype=np.float32)
        x_track = np.asarray(x_track, dtype=np.float32)
        shop_ids = np.asarray(shop_ids, dtype=np.int64)
        shop_feats = np.asarray(shop_feats, dtype=np.float32)
        mask_np = np.asarray(mask_np, dtype=np.float32)
        legal_idxs_np = np.asarray(legal_idxs_np, dtype=np.int32)
        payload = {
            "rid": rid,
            "wid": self.worker_id,
            "encoding_version": ENCODING_GOLD_V2,
            "x_spatial": x_spatial,
            "x_global": x_global,
            "x_track": x_track,
            "shop_ids": shop_ids,
            "shop_feats": shop_feats,
            "action_mask": mask_np,
            "legal_idxs": legal_idxs_np,
        }
        self.req_q.put(payload)
        return rid

    def submit_shm(self, slot: int, n_legal: int) -> int:
        """Enqueue SHM inference request. Data is already written to WorkerSharedBuffer slot.

        Sends only a tiny metadata dict (~100 bytes) instead of pickled arrays (~30KB).
        GPU server reads state from the shared memory buffer identified by (wid, slot).
        """
        rid = self._next_id()
        self.req_q.put({
            "rid": rid,
            "wid": self.worker_id,
            "encoding_version": ENCODING_GOLD_V2,
            "slot": slot,
            "n_legal": n_legal,
        })
        return rid

    def receive(self, rid: int) -> Tuple[np.ndarray, float, float]:
        """Block until response for rid arrives (handles out-of-order). Returns (priors, value, score)."""
        rid = int(rid)

        if rid in self._stash:
            pri, val, s = self._stash.pop(rid)
            return pri, val, s

        deadline = (time.time() + float(self.timeout_s)) if self.timeout_s is not None else None
        while True:
            if deadline is not None and time.time() >= deadline:
                raise TimeoutError("GPU inference response timed out")

            wait_s = 0.1
            if deadline is not None:
                wait_s = max(0.0, min(wait_s, deadline - time.time()))

            try:
                resp_rid, priors_legal, value, score = self.resp_q.get(timeout=wait_s)
            except queue.Empty:
                continue

            resp_rid = int(resp_rid)
            if resp_rid == rid:
                return np.asarray(priors_legal, dtype=np.float32), float(value), float(score)

            # out-of-order: stash it
            self._stash[resp_rid] = (np.asarray(priors_legal, dtype=np.float32), float(value), float(score))

    def evaluate(
        self,
        state_np: np.ndarray,
        mask_np: np.ndarray,
        legal_idxs_np: np.ndarray,
    ) -> Tuple[np.ndarray, float, float]:
        """Blocking convenience: submit_legacy + receive, with retries."""
        last_err: Optional[Exception] = None
        for attempt in range(self.retry_attempts):
            rid = self.submit_legacy(state_np, mask_np, legal_idxs_np)
            try:
                return self.receive(rid)
            except TimeoutError as e:
                last_err = e
                if attempt < self.retry_attempts - 1:
                    logger.warning(
                        "GPU inference timed out (attempt %d/%d, worker=%d), retrying...",
                        attempt + 1,
                        self.retry_attempts,
                        self.worker_id,
                    )
                    time.sleep(0.5 * (attempt + 1))
                else:
                    raise
        if last_err is not None:
            raise last_err
        raise RuntimeError("GPUEvalClient.evaluate failed unexpectedly")

    def evaluate_multimodal(
        self,
        x_spatial: np.ndarray,
        x_global: np.ndarray,
        x_track: np.ndarray,
        shop_ids: np.ndarray,
        shop_feats: np.ndarray,
        mask_np: np.ndarray,
        legal_idxs_np: np.ndarray,
    ) -> Tuple[np.ndarray, float, float]:
        """Blocking convenience: submit_multimodal + receive, with retries."""
        last_err: Optional[Exception] = None
        for attempt in range(self.retry_attempts):
            rid = self.submit_multimodal(
                x_spatial, x_global, x_track, shop_ids, shop_feats, mask_np, legal_idxs_np
            )
            try:
                return self.receive(rid)
            except TimeoutError as e:
                last_err = e
                if attempt < self.retry_attempts - 1:
                    logger.warning(
                        "GPU inference timed out (attempt %d/%d, worker=%d), retrying...",
                        attempt + 1,
                        self.retry_attempts,
                        self.worker_id,
                    )
                    time.sleep(0.5 * (attempt + 1))
                else:
                    raise
        if last_err is not None:
            raise last_err
        raise RuntimeError("GPUEvalClient.evaluate_multimodal failed unexpectedly")
