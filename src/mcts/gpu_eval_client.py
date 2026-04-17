"""
GPU Eval Client (worker-side)

Used by CPU self-play workers to request batched inference from the GPUInferenceServer.

Design goals:
- Safe under Windows spawn
- Correct routing (request_id)
- Handles out-of-order responses

Protocol (dict payload):
- encoding_version, action_mask, legal_idxs (as before)
- score_center_points: float32 (point space, for dynamic score utility)
- effective_static_w, effective_dynamic_w: float32 (DSU-gated weights)
- Gold v2 SHM: slot=int, n_legal=int (data in shared memory)

Response: (rid, priors_legal, value, mean_points, score_utility) — 5-tuple from server.
receive() returns (priors_legal, value, mean_points, score_utility) — 4 values.

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
from src.network.gold_v2_constants import ENCODING_VERSION as ENCODING_GOLD_V2


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
        self._stash: Dict[int, Tuple[np.ndarray, float, float, float]] = {}

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
        score_center_points: float = 0.0,
        effective_static_w: float = 0.0,
        effective_dynamic_w: float = 0.3,
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
            "score_center_points": float(score_center_points),
            "effective_static_w": float(effective_static_w),
            "effective_dynamic_w": float(effective_dynamic_w),
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
        score_center_points: float = 0.0,
        effective_static_w: float = 0.0,
        effective_dynamic_w: float = 0.3,
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
            "score_center_points": float(score_center_points),
            "effective_static_w": float(effective_static_w),
            "effective_dynamic_w": float(effective_dynamic_w),
        }
        self.req_q.put(payload)
        return rid

    def submit_shm(
        self,
        slot: int,
        n_legal: int,
        score_center_points: float = 0.0,
        effective_static_w: float = 0.0,
        effective_dynamic_w: float = 0.3,
    ) -> int:
        """Enqueue SHM inference request. Data is already written to WorkerSharedBuffer slot.

        Sends only a tiny metadata dict; GPU server reads state from shared memory (wid, slot).
        """
        rid = self._next_id()
        self.req_q.put({
            "rid": rid,
            "wid": self.worker_id,
            "encoding_version": ENCODING_GOLD_V2,
            "slot": slot,
            "n_legal": n_legal,
            "score_center_points": float(score_center_points),
            "effective_static_w": float(effective_static_w),
            "effective_dynamic_w": float(effective_dynamic_w),
        })
        return rid

    def receive(self, rid: int) -> Tuple[np.ndarray, float, float, float]:
        """Block until response for rid arrives (handles out-of-order).

        Returns (priors_legal, value, mean_points, score_utility) — 4 values.
        Server sends 5-tuple (rid, priors_legal, value, mean_points, score_utility).
        """
        rid = int(rid)

        if rid in self._stash:
            pri, val, mean_pts, su = self._stash.pop(rid)
            return pri, val, mean_pts, su

        deadline = (time.time() + float(self.timeout_s)) if self.timeout_s is not None else None
        while True:
            if deadline is not None and time.time() >= deadline:
                raise TimeoutError("GPU inference response timed out")

            wait_s = 0.1
            if deadline is not None:
                wait_s = max(0.0, min(wait_s, deadline - time.time()))

            try:
                resp_rid, priors_legal, value, mean_points, score_utility = self.resp_q.get(timeout=wait_s)
            except queue.Empty:
                continue

            resp_rid = int(resp_rid)
            if resp_rid == rid:
                return (
                    np.asarray(priors_legal, dtype=np.float32),
                    float(value),
                    float(mean_points),
                    float(score_utility),
                )

            # out-of-order: stash it
            self._stash[resp_rid] = (
                np.asarray(priors_legal, dtype=np.float32),
                float(value),
                float(mean_points),
                float(score_utility),
            )

    def evaluate(
        self,
        state_np: np.ndarray,
        mask_np: np.ndarray,
        legal_idxs_np: np.ndarray,
        score_center_points: float = 0.0,
        effective_static_w: float = 0.0,
        effective_dynamic_w: float = 0.3,
    ) -> Tuple[np.ndarray, float, float, float]:
        """Blocking convenience: submit_legacy + receive, with retries. Returns (priors, value, mean_points, score_utility)."""
        last_err: Optional[Exception] = None
        for attempt in range(self.retry_attempts):
            rid = self.submit_legacy(
                state_np, mask_np, legal_idxs_np,
                score_center_points, effective_static_w, effective_dynamic_w,
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
        score_center_points: float = 0.0,
        effective_static_w: float = 0.0,
        effective_dynamic_w: float = 0.3,
    ) -> Tuple[np.ndarray, float, float, float]:
        """Blocking convenience: submit_multimodal + receive, with retries. Returns (priors, value, mean_points, score_utility)."""
        last_err: Optional[Exception] = None
        for attempt in range(self.retry_attempts):
            rid = self.submit_multimodal(
                x_spatial, x_global, x_track, shop_ids, shop_feats, mask_np, legal_idxs_np,
                score_center_points, effective_static_w, effective_dynamic_w,
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
