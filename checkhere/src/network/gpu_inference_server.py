"""
GPU Inference Server for AlphaZero MCTS (Patchwork)

Production-grade design for Windows/Linux:
- Single process owns the CUDA context and model (avoids multi-process CUDA contention)
- Batches inference requests from many CPU self-play workers
- Returns priors only for legal action indices to reduce IPC overhead
- Computes score utility on GPU; returns (priors, value, mean_points, score_utility) scalars only.

Protocol (dict payload):
  encoding_version: "gold_v2_32ch" | "full_clarity_v1"
  action_mask: (2026,) float32, legal_idxs: (K,) int32
  score_center_points: float (optional, default 0.0) — center for dynamic score utility (node's to_move perspective)
  Legacy (deprecated): spatial-only state
  Gold v2: x_spatial (32,9,9), x_global (61,), x_track (8,54), shop_ids (33,), shop_feats (33,10)

Response: (rid, priors_legal, value_scalar, mean_points_scalar, score_utility_scalar)

Determinism:
- If config["deterministic"] is true, enables torch deterministic algorithms (slower).
"""

from __future__ import annotations

import logging
import math
import os
import queue
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class InferenceSettings:
    batch_size: int = 256
    max_batch_wait_ms: int = 2
    use_amp: bool = True
    amp_dtype: str = "float16"  # "float16" or "bfloat16"
    torch_compile: bool = False
    allow_tf32: bool = True


def _apply_determinism_if_requested(config: dict) -> None:
    if not bool(config.get("deterministic", False)):
        return
    # NOTE: deterministic GPU can be slower; still training-correct either way.
    # NOTE: torch.use_deterministic_algorithms(True) requires CUBLAS_WORKSPACE_CONFIG
    # env var on CUDA >= 10.2, which causes errors. Disabled for compatibility.
    # torch.use_deterministic_algorithms(True)  # Disabled - CUBLAS compatibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GPUInferenceServer:
    """Single-process GPU inference server that batches requests."""

    def __init__(self, config: dict, checkpoint_path: str, device: str = "cuda",
                 worker_shm_names: Optional[dict] = None):
        self.config = config
        self.device = torch.device(device)

        # Settings (safe defaults if config has no "inference" section)
        inf = config.get("inference", {}) or {}
        self.settings = InferenceSettings(
            batch_size=int(inf.get("batch_size", 256)),
            max_batch_wait_ms=int(inf.get("max_batch_wait_ms", 2)),
            use_amp=bool(inf.get("use_amp", True)),
            amp_dtype=str(inf.get("amp_dtype", "float16")),
            torch_compile=bool(inf.get("torch_compile", False)),
            allow_tf32=bool(inf.get("allow_tf32", True)),
        )

        _apply_determinism_if_requested(config)

        # Model
        from src.network.model import create_network, load_model_checkpoint, get_state_dict_for_inference

        self.model = create_network(config).to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = get_state_dict_for_inference(ckpt, config, for_selfplay=True)
        load_model_checkpoint(self.model, state_dict)
        self.model.eval()

        # Speed knobs
        torch.set_grad_enabled(False)

        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = self.settings.allow_tf32
            torch.backends.cudnn.allow_tf32 = self.settings.allow_tf32
            # Best throughput for fixed-size tensors (56x9x9 spatial, 2026 mask)
            if not bool(config.get("deterministic", False)):
                torch.backends.cudnn.benchmark = True

        if self.settings.torch_compile:
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
            except Exception:
                # Fallback safely if compile isn't available / fails
                pass

        self._amp_dtype = torch.float16 if self.settings.amp_dtype.lower() == "float16" else torch.bfloat16

        # Expected encoding from model (use_film => gold_v2_32ch)
        from src.network.gold_v2_constants import ENCODING_VERSION as _GV2_ENC
        self._expected_encoding = (
            _GV2_ENC if getattr(self.model, "use_film", False) else "full_clarity_v1"
        )
        logger.debug("GPU server expects encoding_version=%s", self._expected_encoding)

        # Score utility constants (KataGo-style sat() on GPU)
        mcts_cfg = (config.get("selfplay", {}) or {}).get("mcts", {}) or {}
        self._score_min = int(mcts_cfg.get("score_min", -100))
        self._score_max = int(mcts_cfg.get("score_max", 100))
        self._score_scale = float(mcts_cfg.get("score_utility_point_scale", 30.0))
        self._static_w = float(mcts_cfg.get("static_score_utility_weight", 0.0))
        self._dynamic_w = float(mcts_cfg.get("dynamic_score_utility_weight", 0.3))
        self._score_bins_t = torch.arange(
            self._score_min, self._score_max + 1,
            device=self.device, dtype=torch.float32,
        )

        # Fallback/request counters for regression diagnostics (tripwires)
        self._fallback_count = 0
        self._total_requests = 0

        # Open per-worker shared memory buffers (zero-copy IPC path).
        # Workers write encoded state into SHM; server reads via numpy views.
        self._worker_shm: Dict[int, Any] = {}
        if worker_shm_names:
            try:
                from src.mcts.shared_state_buffer import WorkerSharedBuffer
                for wid, name in worker_shm_names.items():
                    try:
                        buf = WorkerSharedBuffer(n_slots=None, worker_id=int(wid), create=False, name=name)
                        self._worker_shm[int(wid)] = buf
                    except Exception as e:
                        logger.warning("Failed to open SHM wid=%d name=%s: %s", wid, name, e)
                if self._worker_shm:
                    logger.debug("GPU server opened %d worker SHM buffers (zero-copy IPC active)",
                                len(self._worker_shm))
            except ImportError:
                logger.warning("shared_state_buffer not available — SHM IPC disabled")

    def _warmup_inference(self) -> None:
        """Run a dummy forward pass to trigger cuDNN algorithm selection.

        cudnn.benchmark=True causes PyTorch to profile convolution algorithms on
        the first real inference call, which can take 30-90 seconds for large models.
        By running a warmup before signalling "ready", workers never see that latency.
        """
        import time
        if self.device.type != "cuda":
            return
        t0 = time.time()
        logger.debug("GPU server: running cuDNN warmup inference...")
        try:
            enc = self._expected_encoding
            B = min(self.settings.batch_size, 32)  # small but representative batch

            if enc == "gold_v2_32ch":
                from src.network.gold_v2_constants import C_SPATIAL_ENC, F_GLOBAL, C_TRACK, TRACK_LEN, NMAX, F_SHOP
                states = torch.zeros((B, C_SPATIAL_ENC, 9, 9), dtype=torch.float32, device=self.device)
                masks  = torch.ones((B, 2026), dtype=torch.float32, device=self.device)
                x_global = torch.zeros((B, F_GLOBAL), dtype=torch.float32, device=self.device)
                x_track  = torch.zeros((B, C_TRACK, TRACK_LEN), dtype=torch.float32, device=self.device)
                shop_ids  = torch.zeros((B, NMAX), dtype=torch.int64, device=self.device)
                shop_feats = torch.zeros((B, NMAX, F_SHOP), dtype=torch.float32, device=self.device)
            else:
                in_ch = getattr(self.model.conv_input, "in_channels", 56)
                states = torch.zeros((B, in_ch, 9, 9), dtype=torch.float32, device=self.device)
                masks  = torch.ones((B, 2026), dtype=torch.float32, device=self.device)
                x_global = x_track = shop_ids = shop_feats = None

            with torch.inference_mode():
                if self.settings.use_amp:
                    with torch.autocast(device_type="cuda", dtype=self._amp_dtype):
                        self.model(states, masks,
                                   x_global=x_global, x_track=x_track,
                                   shop_ids=shop_ids, shop_feats=shop_feats)
                else:
                    self.model(states, masks,
                               x_global=x_global, x_track=x_track,
                               shop_ids=shop_ids, shop_feats=shop_feats)

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            logger.debug("GPU server: cuDNN warmup done in %.1fs", time.time() - t0)
        except Exception as e:
            logger.warning("GPU server warmup failed (non-fatal): %s", e)

    def _send_error(self, resp_qs, payload: Union[dict, tuple], msg: str) -> None:
        """Send error response so worker doesn't deadlock."""
        self._fallback_count += 1
        if isinstance(payload, dict):
            rid = int(payload.get("rid", 0))
            wid = int(payload.get("wid", 0))
        else:
            rid = int(payload[0]) if len(payload) > 0 else 0
            wid = int(payload[1]) if len(payload) >= 5 else 0
        legal_idxs = payload.get("legal_idxs", []) if isinstance(payload, dict) else (payload[-1] if len(payload) >= 4 else [])
        legal_idxs = np.asarray(legal_idxs, dtype=np.int32)
        n_legal = max(1, len(legal_idxs))
        fallback = np.full((n_legal,), 1.0 / n_legal, dtype=np.float32)
        if isinstance(resp_qs, (list, tuple)):
            wid = max(0, min(wid, len(resp_qs) - 1))
            resp_q = resp_qs[wid]
        else:
            resp_q = resp_qs
        resp_q.put((int(rid), fallback, np.float32(0.0), np.float32(0.0), np.float32(0.0)))
        logger.error(
            "[SERVER_FALLBACK] reason=%s batch_size=1 last_rid=%s center_stats=N/A",
            msg, rid,
        )
        logger.error("GPU inference error for rid=%s: %s", rid, msg)

    def serve(self, req_q, resp_qs, stop_evt) -> None:
        """
        Blocking loop. Intended to run inside a dedicated multiprocessing.Process.

        req_q: request queue (shared by all workers)
        resp_qs: list of per-worker response queues (fixes MPMC deadlock)
        stop_evt: event to signal shutdown
        """
        pending = []
        last_flush = time.time()
        batch_wait_s = self.settings.max_batch_wait_ms / 1000.0
        batch_count = 0
        _debug_centers = os.environ.get("PW_DEBUG_SCORE_CENTERS") or bool(self.config.get("debug_score_centers", False))

        while True:
            if stop_evt.is_set():
                break

            # Block on first item if pending is empty, else drain non-blocking
            if not pending:
                try:
                    item = req_q.get(timeout=batch_wait_s)
                    pending.append(item)
                except queue.Empty:
                    continue
                last_flush = time.time()

            # Aggressively drain queue up to batch_size (non-blocking)
            while len(pending) < self.settings.batch_size:
                try:
                    pending.append(req_q.get_nowait())
                except queue.Empty:
                    break

            now = time.time()
            should_flush = (
                len(pending) >= self.settings.batch_size
                or (now - last_flush) * 1000.0 >= self.settings.max_batch_wait_ms
            )
            if not should_flush:
                # Brief sleep to avoid busy-spin while waiting for batch to fill
                time.sleep(0.0002)
                continue

            batch = pending[: self.settings.batch_size]
            pending = pending[self.settings.batch_size :]
            last_flush = now

            try:
                parsed: List[Dict[str, Any]] = []
                for x in batch:
                    if isinstance(x, dict):
                        enc = x.get("encoding_version", "full_clarity_v1")
                        if enc != self._expected_encoding:
                            self._send_error(resp_qs, x, f"encoding_version mismatch: got {enc}, server expects {self._expected_encoding}")
                            continue
                        parsed.append(x)
                    else:
                        # Legacy tuple: (rid, wid?, state, mask, legal)
                        if len(x) == 4:
                            rid, state_np, mask_np, legal_idxs_np = x
                            wid = 0
                        elif len(x) == 5:
                            rid, wid, state_np, mask_np, legal_idxs_np = x
                        else:
                            continue
                        if self._expected_encoding != "full_clarity_v1":
                            self._send_error(resp_qs, {"rid": rid, "wid": wid}, "server expects gold_v2_32ch, got legacy tuple")
                            continue
                        # Legacy tuple has no score_center_points; support extended 6-element format if present
                        score_center = float(x[5]) if len(x) >= 6 else 0.0
                        parsed.append({
                            "rid": rid,
                            "wid": wid,
                            "encoding_version": "full_clarity_v1",
                            "state": np.asarray(state_np, dtype=np.float32),
                            "action_mask": np.asarray(mask_np, dtype=np.float32),
                            "legal_idxs": np.asarray(legal_idxs_np, dtype=np.int32),
                            "score_center_points": score_center,
                        })

                if not parsed:
                    continue

                # Build batch tensors
                rids = [p["rid"] for p in parsed]
                wids = [int(p.get("wid", 0)) for p in parsed]
                centers_list = [float(p.get("score_center_points", 0.0)) for p in parsed]
                assert len(centers_list) == len(parsed), "centers_list length must match parsed batch"

                # CENTER_STATS tripwire: assert all finite; log stats occasionally
                centers_arr = np.array(centers_list, dtype=np.float64)
                n_centers = len(centers_arr)
                count_zero = int(np.sum(centers_arr == 0))
                count_nan = int(np.sum(~np.isfinite(centers_arr)))
                count_inf = int(np.sum(np.isinf(centers_arr)))
                if count_nan > 0 or count_inf > 0:
                    logger.error(
                        "[CENTER_STATS] INVALID: min=%s mean=%s max=%s zero=%d/%d nan=%d inf=%d",
                        np.nanmin(centers_arr), np.nanmean(centers_arr), np.nanmax(centers_arr),
                        count_zero, n_centers, count_nan, count_inf,
                    )
                    raise ValueError("score_center_points must be finite (no NaN/Inf)")
                c_min, c_mean, c_max = float(np.min(centers_arr)), float(np.mean(centers_arr)), float(np.max(centers_arr))
                if batch_count % 200 == 0 or _debug_centers:
                    logger.info(
                        "[CENTER_STATS] min=%.2f mean=%.2f max=%.2f zero=%d/%d nan=%d inf=%d",
                        c_min, c_mean, c_max, count_zero, n_centers, count_nan, count_inf,
                    )
                if _debug_centers and batch_count % 200 == 0 and centers_list:
                    logger.debug("score_centers batch %d: min=%.2f max=%.2f", batch_count, c_min, c_max)
                batch_count += 1

                # Request counters for SERVER_STATS
                self._total_requests += len(parsed)
                if self._total_requests >= 1000 and self._total_requests % 1000 < len(parsed):
                    rate = self._fallback_count / max(1, self._total_requests)
                    logger.info(
                        "[SERVER_STATS] total=%d fallback=%d fallback_rate=%.6f",
                        self._total_requests, self._fallback_count, rate,
                    )
                enc = parsed[0]["encoding_version"]

                if enc == "gold_v2_32ch":
                    # Mixed-batch safe: each request is routed individually.
                    # SHM requests carry {"slot", "wid", "n_legal"} (no arrays).
                    # Pickle requests carry {"x_spatial", "x_global", ...} (full arrays).
                    # Both can appear in the same batch (e.g. root node via pickle,
                    # leaf nodes via SHM), so we dispatch per-item rather than
                    # branching on parsed[0].
                    spatial_list, global_list, track_list, si_list, sf_list, mask_list, li_list = [], [], [], [], [], [], []
                    for p in parsed:
                        wid_p = int(p.get("wid", 0))
                        if "slot" in p and self._worker_shm:
                            # SHM path — zero-copy read from shared memory
                            slot_p = int(p["slot"])
                            shm = self._worker_shm.get(wid_p)
                            if shm is not None:
                                x_s, x_g, x_t, x_si, x_sf, x_m, x_li = shm.read_all(slot_p)
                                spatial_list.append(x_s.copy())
                                global_list.append(x_g.copy())
                                track_list.append(x_t.copy())
                                si_list.append(x_si.astype(np.int64))
                                sf_list.append(x_sf.copy())
                                mask_list.append(x_m.copy())
                                li_list.append(x_li.copy())
                            else:
                                # SHM buffer missing for this worker — uniform fallback
                                n_lp = int(p.get("n_legal", 1))
                                from src.network.gold_v2_constants import C_SPATIAL_ENC, F_GLOBAL, C_TRACK, TRACK_LEN, NMAX, F_SHOP
                                spatial_list.append(np.zeros((C_SPATIAL_ENC, 9, 9), dtype=np.float32))
                                global_list.append(np.zeros(F_GLOBAL, dtype=np.float32))
                                track_list.append(np.zeros((C_TRACK, TRACK_LEN), dtype=np.float32))
                                si_list.append(np.full(NMAX, -1, dtype=np.int64))
                                sf_list.append(np.zeros((NMAX, F_SHOP), dtype=np.float32))
                                mask_list.append(np.ones(2026, dtype=np.float32) / 2026.0)
                                li_list.append(np.arange(n_lp, dtype=np.int32))
                        else:
                            # Pickle path — arrays are in the payload dict
                            spatial_list.append(p["x_spatial"].astype(np.float32))
                            global_list.append(p["x_global"].astype(np.float32))
                            track_list.append(p["x_track"].astype(np.float32))
                            si_list.append(np.asarray(p["shop_ids"], dtype=np.int64))
                            sf_list.append(p["shop_feats"].astype(np.float32))
                            mask_list.append(np.asarray(p["action_mask"], dtype=np.float32))
                            li_list.append(np.asarray(p["legal_idxs"], dtype=np.int32))

                    states_np = np.stack(spatial_list, axis=0)
                    x_global_np = np.stack(global_list, axis=0)
                    x_track_np = np.stack(track_list, axis=0)
                    shop_ids_np = np.stack(si_list, axis=0)
                    shop_feats_np = np.stack(sf_list, axis=0)
                    masks_np_arr = np.stack(mask_list, axis=0)
                    legal_list = li_list

                else:
                    states_np = np.stack([p["state"].astype(np.float32) for p in parsed], axis=0)
                    masks_np_arr = np.stack([np.asarray(p["action_mask"], dtype=np.float32) for p in parsed], axis=0)
                    legal_list = [np.asarray(p["legal_idxs"], dtype=np.int32) for p in parsed]

                non_block = self.device.type == "cuda"
                states = torch.from_numpy(states_np).to(self.device, non_blocking=non_block)
                masks = torch.from_numpy(masks_np_arr).to(self.device, non_blocking=non_block)

                x_global_t = x_track_t = shop_ids_t = shop_feats_t = None
                if enc == "gold_v2_32ch":
                    x_global_t = torch.from_numpy(x_global_np).to(self.device, non_blocking=non_block)
                    x_track_t = torch.from_numpy(x_track_np).to(self.device, non_blocking=non_block)
                    shop_ids_t = torch.from_numpy(shop_ids_np).to(self.device, non_blocking=non_block)
                    shop_feats_t = torch.from_numpy(shop_feats_np).to(self.device, non_blocking=non_block)

                with torch.inference_mode():
                    if self.device.type == "cuda" and self.settings.use_amp:
                        with torch.autocast(device_type="cuda", dtype=self._amp_dtype):
                            policy_logits, value, score = self.model(
                                states, masks,
                                x_global=x_global_t, x_track=x_track_t,
                                shop_ids=shop_ids_t, shop_feats=shop_feats_t,
                            )
                    else:
                        policy_logits, value, score = self.model(
                            states, masks,
                            x_global=x_global_t, x_track=x_track_t,
                            shop_ids=shop_ids_t, shop_feats=shop_feats_t,
                        )

                policy_probs = torch.softmax(policy_logits, dim=-1)
                value = value.squeeze(-1)  # (B,)
                score_logits_f = score.detach().float()  # (B, 201) — keep on GPU

                # Compute mean_points and score_utility on GPU (no IPC of 201-dim logits)
                p = torch.softmax(score_logits_f, dim=-1)
                bins = self._score_bins_t
                mean_points_t = (p * bins).sum(dim=-1)  # (B,)
                centers_t = torch.tensor(centers_list, device=self.device, dtype=torch.float32)
                scale = self._score_scale
                sat = lambda x: (2.0 / math.pi) * torch.atan(x)
                dynamic_util_t = (p * sat((bins.unsqueeze(0) - centers_t.unsqueeze(1)) / scale)).sum(dim=-1)
                score_utility_t = self._dynamic_w * dynamic_util_t
                if self._static_w != 0.0:
                    static_util_t = (p * sat(bins.unsqueeze(0) / scale)).sum(dim=-1)
                    score_utility_t = score_utility_t + self._static_w * static_util_t

                policy_probs_cpu = policy_probs.detach().cpu()
                value_cpu = value.detach().cpu()
                mean_points_cpu = mean_points_t.detach().cpu()
                score_utility_cpu = score_utility_t.detach().cpu()

                for i, rid in enumerate(rids):
                    wid = max(0, min(wids[i], len(resp_qs) - 1)) if isinstance(resp_qs, (list, tuple)) else 0
                    legal_idxs = legal_list[i]
                    if legal_idxs.size == 0:
                        priors_legal = np.zeros((0,), dtype=np.float32)
                    else:
                        idx_t = torch.from_numpy(legal_idxs).long()
                        priors_legal = policy_probs_cpu[i].index_select(0, idx_t).float().numpy()

                    v = float(value_cpu[i].item())
                    mean_pts = np.float32(mean_points_cpu[i].item())
                    su = np.float32(score_utility_cpu[i].item())
                    resp_q = resp_qs[wid] if isinstance(resp_qs, (list, tuple)) else resp_qs
                    resp_q.put((int(rid), priors_legal, np.float32(v), mean_pts, su))

            except Exception as e:
                logger.error("BATCH PROCESSING FAILED: %s", e, exc_info=True)
                sys.stdout.flush()
                last_rid = 0
                for x in batch:
                    try:
                        if isinstance(x, dict):
                            rid = int(x.get("rid", 0))
                            wid = int(x.get("wid", 0))
                            legal_idxs = np.asarray(x.get("legal_idxs", []), dtype=np.int32)
                        elif len(x) >= 4:
                            rid, wid = int(x[0]), int(x[1]) if len(x) == 5 else 0
                            legal_idxs = np.asarray(x[-1], dtype=np.int32)
                        else:
                            continue
                        last_rid = rid
                        self._fallback_count += 1
                        n_legal = max(1, len(legal_idxs))
                        fallback = np.full((n_legal,), 1.0 / n_legal, dtype=np.float32)
                        rq = resp_qs[max(0, min(wid, len(resp_qs) - 1))] if isinstance(resp_qs, (list, tuple)) else resp_qs
                        rq.put((rid, fallback, np.float32(0.0), np.float32(0.0), np.float32(0.0)))
                    except Exception:
                        pass
                self._total_requests += len(batch)
                logger.error(
                    "[SERVER_FALLBACK] reason=BATCH_PROCESSING_FAILED batch_size=%d last_rid=%s center_stats=N/A",
                    len(batch), last_rid,
                )



def run_gpu_inference_server(
    config: dict,
    checkpoint_path: str,
    req_q,
    resp_qs,
    stop_evt,
    ready_q=None,
    device: str = "cuda",
    worker_shm_names: Optional[dict] = None,
) -> None:
    """Entry point for multiprocessing.Process (Windows spawn safe)."""
    import logging
    import signal
    import sys

    # Ignore SIGINT — parent process handles CTRL+C via stop_evt.
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (OSError, ValueError):
        pass

    logging.basicConfig(level=logging.INFO, format='[GPU Server] %(message)s')
    logger = logging.getLogger(__name__)

    try:
        logger.debug("Initializing GPU inference server (device=%s)...", device)
        sys.stdout.flush()
        server = GPUInferenceServer(
            config=config, checkpoint_path=checkpoint_path, device=device,
            worker_shm_names=worker_shm_names,
        )
        logger.debug("GPU server initialized successfully, running warmup...")
        sys.stdout.flush()
        server._warmup_inference()

        # Signal ready to parent process (after warmup so workers never see cold-start latency)
        if ready_q is not None:
            ready_q.put("ready")

        server.serve(req_q=req_q, resp_qs=resp_qs, stop_evt=stop_evt)
    except Exception as e:
        logger.error(f"GPU server failed: {e}", exc_info=True)
        sys.stdout.flush()

        # Signal error to parent process
        if ready_q is not None:
            try:
                ready_q.put(f"error:{str(e)}")
            except:
                pass
        raise