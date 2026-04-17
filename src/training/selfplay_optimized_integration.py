"""
Selfplay Integration Layer

Provides factory function to create self-play generators compatible with main.py orchestrator.
Bridges OptimizedSelfPlayWorker with multiprocessing-safe initialization.

ARCHITECTURE:
- main.py imports create_selfplay_generator() factory
- SelfPlayGenerator orchestrates parallel self-play with multiprocessing pool
- Auto-detects when to use GPU inference server (CUDA + multi-worker)
- Workers write NPZ shards to avoid Windows IPC MemoryError
- Main process merges shards into HDF5 dataset

"""

import logging
import multiprocessing as mp
import queue
import os
import signal
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

from .run_layout import (
    ENCODING_VERSION_ATTR,
    SELFPLAY_COMPLETE_ATTR,
    SELFPLAY_EXPECTED_CHANNELS_ATTR,
    SELFPLAY_NUM_GAMES_ATTR,
    SELFPLAY_SCHEMA_VERSION,
    SELFPLAY_SCHEMA_VERSION_ATTR,
    SELFPLAY_VALUE_TARGET_TYPE_ATTR,
)
from src.network.gold_v2_constants import (
    C_SPATIAL_ENC,
    F_GLOBAL,
    C_TRACK,
    TRACK_LEN,
    NMAX,
    F_SHOP,
    ENCODING_VERSION,
)
from .selfplay_optimized import (
    OptimizedSelfPlayWorker,
    init_optimized_worker,
    play_game_optimized,
)
from src.mcts.shared_state_buffer import WorkerSharedBuffer
from src.training.replay_buffer import _validate_score_margins, SCORE_MARGIN_MAX_ABS
from src.utils.packing_metrics import aggregate_packing_over_games, aggregate_root_over_moves


def _init_worker_ignore_sigint(*args):
    """Worker initializer wrapper that ignores SIGINT.

    On Windows, CTRL+C is delivered to all processes in the console group.
    Workers should ignore it and let the parent orchestrate shutdown via
    pool.terminate().  Without this, workers crash with KeyboardInterrupt
    mid-game, potentially corrupting shared queues.
    """
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (OSError, ValueError):
        pass  # Some platforms/contexts don't support this
    init_optimized_worker(*args)


# Module-level ref for graceful CTRL+C: terminate pool before main process exits
# so workers don't hit BrokenPipeError when trying to send results through closed pipes.
_active_pool = None


def _register_pool(pool) -> None:
    global _active_pool
    _active_pool = pool


def _unregister_pool() -> None:
    global _active_pool
    _active_pool = None


def terminate_active_pool() -> None:
    """Terminate any active multiprocessing pool. Call before os._exit() on hard Ctrl+C.

    Prevents BrokenPipeError in workers when main process exits without closing the pool.
    """
    global _active_pool
    if _active_pool is not None:
        try:
            _active_pool.terminate()
            _active_pool.join(timeout=3.0)
        except Exception:
            pass
        _active_pool = None


logger = logging.getLogger(__name__)


class SelfPlayGenerator:
    """Orchestrates parallel self-play using OptimizedSelfPlayWorker."""

    def __init__(self, config: dict):
        self.config = config
        self.data_dir = Path(config["paths"].get("selfplay_dir", "data/selfplay"))

        # GPU inference server support
        self.use_gpu_server = False
        self.gpu_process = None
        self.req_q = None
        self.resp_qs = None  # Per-worker response queues (fixes MPMC deadlock)
        self.stop_evt = None

        # Shared memory buffers for zero-copy IPC (one per worker)
        self._worker_shm_bufs: Dict[int, WorkerSharedBuffer] = {}
        self._worker_shm_names: Dict[int, str] = {}
        self._expected_n_slots: Optional[int] = None  # Set when GPU server SHM is created; used by workers

    def _should_use_gpu_server(self, network_path: Optional[str]) -> bool:
        """Decide if GPU inference server should be used (Windows-safe: no parent CUDA touch).

        Uses config only so the parent never calls torch.cuda.is_available() before spawn.
        If CUDA is unavailable, the child will put ('error', ...) on ready_q and we raise.
        """
        if network_path is None:
            return False  # Bootstrap mode (pure MCTS)
        hw = (self.config.get("hardware", {}) or {}).get("device", "cpu")
        if hw != "cuda":
            return False
        num_workers = int(self.config.get("selfplay", {}).get("num_workers", 1))
        return num_workers > 1

    def _get_expected_n_slots(self) -> int:
        """Max parallel_leaves over schedule if present, else config selfplay.mcts.parallel_leaves."""
        base = int(
            self.config.get("selfplay", {}).get("mcts", {}).get("parallel_leaves", 32)
        )
        pl_schedule = self.config.get("iteration", {}).get("parallel_leaves_schedule", [])
        if pl_schedule:
            return max(int(e.get("parallel_leaves", base)) for e in pl_schedule)
        return base

    def _start_gpu_server(self, network_path: str, num_workers: int, iteration: Optional[int] = None):
        """Start GPU inference server process with per-worker response queues.

        Uses expected_n_slots (max over schedule) for SHM sizing everywhere; same config
        is passed to the child. Cleanup (SHM close+unlink) runs in finally.
        iteration: optional, for lifecycle logging (child PID is logged at info).
        """
        from src.network.gpu_inference_server import run_gpu_inference_server

        ctx = mp.get_context("spawn")
        self.req_q = ctx.Queue(maxsize=4096)
        self.resp_qs = [ctx.Queue(maxsize=4096) for _ in range(num_workers)]
        self.stop_evt = ctx.Event()
        ready_q = ctx.Queue(maxsize=1)

        expected_n_slots = self._get_expected_n_slots()
        self._worker_shm_bufs = {}
        self._worker_shm_names = {}
        started_ok = False

        try:
            for wid in range(num_workers):
                try:
                    buf = WorkerSharedBuffer(
                        n_slots=expected_n_slots, worker_id=wid, create=True
                    )
                    self._worker_shm_bufs[wid] = buf
                    self._worker_shm_names[wid] = buf.name
                except Exception as e:
                    logger.warning("Failed to create SHM buffer for worker %d: %s", wid, e)
                    self._worker_shm_bufs = {}
                    self._worker_shm_names = {}
                    break
            if not self._worker_shm_names:
                raise RuntimeError(
                    "Failed to create any SHM buffers for GPU server; check parallel_leaves / schedule."
                )
            logger.debug(
                "Created %d shared memory buffers (expected_n_slots=%d × %d bytes each)",
                num_workers, expected_n_slots, WorkerSharedBuffer.SLOT_BYTES,
            )

            self.gpu_process = ctx.Process(
                target=run_gpu_inference_server,
                args=(self.config, network_path, self.req_q, self.resp_qs, self.stop_evt, ready_q),
                kwargs={
                    "worker_shm_names": self._worker_shm_names,
                    "expected_n_slots": expected_n_slots,
                },
                daemon=False,
            )
            self.gpu_process.start()
            logger.info(
                "[LIFECYCLE] GPU server process started parent_pid=%s child_pid=%s iter=%s",
                os.getpid(), self.gpu_process.pid, iteration,
            )
            logger.debug("Waiting for GPU server to initialize...")

            gpu_ready_timeout = float(
                self.config.get("selfplay", {}).get("gpu_server_ready_timeout_s", 180)
            )
            status = None
            poll_interval = 1.0  # Check child liveness every second
            deadline = time.time() + gpu_ready_timeout
            try:
                while time.time() < deadline:
                    if not self.gpu_process.is_alive():
                        exitcode = getattr(self.gpu_process, "exitcode", None)
                        raise RuntimeError(
                            "GPU server child process exited before signalling ready. "
                            "This usually indicates a crash during CUDA/model init (e.g. nvcuda64.dll). "
                            f"Child exit code: {exitcode}. "
                            "Check GPU_SERVER_BREADCRUMB_LOG or child stdout for last breadcrumb."
                        )
                    try:
                        status = ready_q.get(timeout=min(poll_interval, max(0.1, deadline - time.time())))
                        break
                    except queue.Empty:
                        continue
                if status is None:
                    if self.gpu_process.is_alive():
                        self.gpu_process.terminate()
                        self.gpu_process.join(timeout=5)
                    exitcode = getattr(self.gpu_process, "exitcode", None)
                    msg = (
                        f"GPU server did not signal ready within {gpu_ready_timeout:.0f}s (timeout). "
                        "Increase selfplay.gpu_server_ready_timeout_s to wait longer."
                    )
                    if exitcode is not None:
                        msg += f" Child exit code: {exitcode}."
                    raise RuntimeError(msg)
            except RuntimeError:
                raise

            if status == "ready":
                started_ok = True
                self._expected_n_slots = expected_n_slots
                logger.debug("GPU inference server ready!")
            elif isinstance(status, tuple) and len(status) >= 3 and status[0] == "error":
                _, err_msg, err_tb = status[0], status[1], status[2]
                raise RuntimeError(
                    f"GPU server initialization failed: {err_msg}\n\nChild traceback:\n{err_tb}"
                )
            elif isinstance(status, str) and status.startswith("error:"):
                raise RuntimeError(f"GPU server initialization failed: {status[6:]}")
            else:
                raise RuntimeError(f"GPU server sent unexpected status: {type(status)} {status!r}")
        except Exception:
            if self.gpu_process and self.gpu_process.is_alive():
                self.gpu_process.terminate()
                self.gpu_process.join(timeout=5)
            raise
        finally:
            # On startup failure: full cleanup (queues, process ref, SHM) so next cycle does not leak
            if not started_ok:
                self._stop_gpu_server()

    def _stop_gpu_server(self):
        """Stop GPU inference server, close queues, and destroy shared memory buffers.

        Lifecycle: ensures child is joined, queues are closed (Windows pipe leak avoidance),
        SHM destroyed, and references cleared so the next cycle starts clean.
        """
        pid, exitcode = None, None
        if self.gpu_process:
            pid = getattr(self.gpu_process, "pid", None)
            if self.gpu_process.is_alive():
                logger.debug("Stopping GPU inference server...")
                if self.stop_evt is not None:
                    self.stop_evt.set()
                self.gpu_process.join(timeout=5)
                if self.gpu_process.is_alive():
                    logger.warning("GPU server didn't stop gracefully, terminating...")
                    self.gpu_process.terminate()
                    self.gpu_process.join(timeout=2)
            exitcode = getattr(self.gpu_process, "exitcode", None)
            logger.info(
                "[LIFECYCLE] GPU server stopped child_pid=%s exitcode=%s",
                pid, exitcode,
            )
            self.gpu_process = None

        # Close queues to release pipe handles (Windows: avoids accumulation across cycles)
        if self.req_q is not None:
            try:
                self.req_q.close()
                if hasattr(self.req_q, "cancel_join_thread"):
                    self.req_q.cancel_join_thread()
            except Exception as e:
                logger.debug("Failed to close req_q: %s", e)
            self.req_q = None
        if self.resp_qs is not None:
            for q in self.resp_qs:
                try:
                    q.close()
                    if hasattr(q, "cancel_join_thread"):
                        q.cancel_join_thread()
                except Exception as e:
                    logger.debug("Failed to close resp_q: %s", e)
            self.resp_qs = None
        self.stop_evt = None

        # Destroy shared memory buffers (must be done after GPU server exits)
        for wid, buf in list(self._worker_shm_bufs.items()):
            try:
                buf.destroy()
            except Exception as e:
                logger.debug("Failed to destroy SHM buffer for worker %d: %s", wid, e)
        self._worker_shm_bufs.clear()
        self._worker_shm_names.clear()
        self._expected_n_slots = None

    def generate(
        self,
        iteration: int,
        network_path: Optional[str] = None,
        output_dir: Optional[Path] = None,
        num_games_override: Optional[int] = None,
    ) -> Tuple[str, Dict]:
        """Generate self-play data for iteration.

        Args:
            iteration: Iteration number.
            network_path: Path to model checkpoint for inference.
            output_dir: If provided, write HDF5 to output_dir/selfplay.h5 (for staging).
            num_games_override: If set, use this many games instead of schedule (e.g. adaptive-games).

        Returns:
            (data_path, stats): Path to HDF5 file and game statistics dict
        """
        logger.debug("Generating self-play data for iteration %d", iteration)

        num_games = int(num_games_override) if num_games_override is not None else self._get_num_games(iteration)

        if output_dir is not None:
            output_dir = Path(output_dir)
            data_path = output_dir / "selfplay.h5"
            if data_path.exists():
                try:
                    with h5py.File(data_path, "r") as f:
                        actual_games = int(f.attrs.get("num_games", 0))
                        states_key = "spatial_states" if "spatial_states" in f else "states"
                        n_pos = int(f[states_key].shape[0]) if states_key in f else 0
                    if actual_games >= num_games or n_pos >= num_games * 20:
                        logger.info(
                            "Reusing preserved self-play: iter%03d (%d games, %d positions) — skipping regeneration",
                            iteration, actual_games, n_pos,
                        )
                        return str(data_path), {
                            "num_games": actual_games,
                            "num_positions": n_pos,
                            "avg_game_length": 0,
                            "games_per_minute": 0,
                            "avg_policy_entropy": 0.0,
                            "avg_top1_prob": 0.0,
                            "avg_num_legal": 0.0,
                            "avg_redundancy": 0.0,
                        }
                except Exception as e:
                    logger.debug("Could not reuse existing selfplay.h5: %s", e)
        num_workers = min(
            self.config["selfplay"]["num_workers"],
            os.cpu_count() or 1,
            num_games,
        )
        num_workers = max(1, num_workers)

        # MAJOR FIX: Apply temperature and MCTS schedules based on iteration
        iteration_config = self._apply_iteration_schedules(iteration)

        # Start GPU server if beneficial
        use_gpu_server = self._should_use_gpu_server(network_path)
        if use_gpu_server:
            logger.info("[LIFECYCLE] starting GPU server for iteration %s", iteration)
            self._start_gpu_server(network_path, num_workers, iteration)
            self.use_gpu_server = True

        try:
            # Log schedule changes
            temp = iteration_config["selfplay"]["mcts"]["temperature"]
            sims = iteration_config["selfplay"]["mcts"]["simulations"]
            logger.debug(
                "Generating %d games with %d workers (temp=%.2f, sims=%d)%s",
                num_games,
                num_workers,
                temp,
                sims,
                " (GPU server mode)" if use_gpu_server else " (local mode)",
            )
            start_time = time.time()

            if output_dir is not None:
                output_dir.mkdir(parents=True, exist_ok=True)
                data_path = output_dir / "selfplay.h5"
            else:
                self.data_dir.mkdir(parents=True, exist_ok=True)
                data_path = self.data_dir / f"iteration_{iteration:03d}.h5"

            # Parallel generation with shard mode (Windows-safe)
            game_summaries = self._generate_parallel(
                num_games, num_workers, iteration, network_path, data_path, iteration_config
            )

            generation_time = time.time() - start_time
            stats = self._compute_stats(game_summaries, generation_time)

            logger.debug(
                f"Generated {len(game_summaries)} games "
                f"({stats.get('num_positions', 0)} positions) in {generation_time:.1f}s"
            )
            return str(data_path), stats

        finally:
            if self.use_gpu_server:
                self._stop_gpu_server()
                self.use_gpu_server = False

    def _run_game_batch(
        self,
        num_games: int,
        num_workers: int,
        iteration: int,
        network_path: Optional[str],
        shard_dir: Path,
        iteration_config: dict,
        game_idx_offset: int = 0,
        base_seed: int = 42,
    ) -> List[Dict]:
        """Run a batch of games with a single model. Shards go to shard_dir."""
        ctx = mp.get_context("spawn")

        game_args = [
            (game_idx_offset + i, iteration, base_seed + iteration * 1_000_000 + game_idx_offset + i)
            for i in range(num_games)
        ]

        pool = ctx.Pool(
            processes=num_workers,
            initializer=_init_worker_ignore_sigint,
            initargs=(
                network_path,
                iteration_config,
                self.req_q,
                self.resp_qs,
                "shard",
                str(shard_dir),
                self._worker_shm_names or None,
                self._expected_n_slots,  # for SHM attach validation in worker
            ),
        )
        _register_pool(pool)

        summaries = []
        batch_start = time.time()

        try:
            for result in pool.imap_unordered(play_game_optimized, game_args):
                if result:
                    summaries.append(result)
                    done = len(summaries)
                    if done % 50 == 0:
                        elapsed = time.time() - batch_start
                        gpm = (done / elapsed) * 60.0 if elapsed > 0 else 0
                        logger.info(
                            "    Batch progress: %d/%d games (%.1f games/min)",
                            done, num_games, gpm,
                        )
            pool.close()
        except KeyboardInterrupt:
            logger.warning("Batch interrupted after %d/%d games", len(summaries), num_games)
            pool.terminate()
            raise
        finally:
            _unregister_pool()
            pool.join()

        return summaries

    def _generate_parallel(
        self, num_games, num_workers, iteration, network_path, output_path, iteration_config
    ):
        """Parallel game generation with shard mode.

        Args:
            iteration_config: Config with schedules applied for this iteration
        """
        ctx = mp.get_context("spawn")

        # Use shard mode to avoid Windows MemoryError on large game returns.
        # Put shards alongside output so staging cleanup removes them.
        output_parent = Path(output_path).parent
        shard_dir = output_parent / f"iter_{iteration:03d}_shards"
        # Ensure stale shards from interrupted runs are removed before reuse.
        if shard_dir.exists():
            shutil.rmtree(shard_dir, ignore_errors=True)
        shard_dir.mkdir(exist_ok=True)

        # MAJOR FIX: Vary seeds across iterations for diversity
        base_seed = int(self.config.get("seed", 42))
        game_args = [(i, iteration, base_seed + iteration * 1_000_000 + i) for i in range(num_games)]

        pool = ctx.Pool(
            processes=num_workers,
            initializer=_init_worker_ignore_sigint,
            initargs=(
                network_path,
                iteration_config,
                self.req_q,
                self.resp_qs,
                "shard",
                str(shard_dir),
                self._worker_shm_names or None,
                self._expected_n_slots,  # for SHM attach validation in worker
            ),
        )
        _register_pool(pool)

        summaries = []
        selfplay_start_time = time.time()

        interrupted = False
        try:
            for result in pool.imap_unordered(play_game_optimized, game_args):
                if result:
                    summaries.append(result)
                    if len(summaries) % 50 == 0:
                        elapsed = time.time() - selfplay_start_time
                        games_per_min = (len(summaries) / elapsed) * 60.0 if elapsed > 0 else 0
                        logger.info(
                            f"{len(summaries)}/{num_games} ({games_per_min:.1f} games/min)"
                        )
            pool.close()
        except TimeoutError as e:
            if self.use_gpu_server and self.gpu_process and not self.gpu_process.is_alive():
                raise RuntimeError(
                    "GPU inference server process died during self-play. "
                    "Check for OOM, CUDA errors, or GPU driver issues."
                ) from e
            raise
        except KeyboardInterrupt:
            interrupted = True
            logger.warning(
                f"Self-play interrupted after {len(summaries)}/{num_games} games. "
                f"Terminating workers..."
            )
            pool.terminate()
            raise
        finally:
            _unregister_pool()
            pool.join()

        # Merge shards into final HDF5
        self._merge_shards(shard_dir, output_path, summaries)

        # MINOR FIX: Clean up shard files after successful merge
        try:
            shutil.rmtree(shard_dir)
            logger.debug("Shard directory cleaned up")
        except Exception as e:
            logger.warning(f"Failed to clean up shards: {e}")

        return summaries

    def _merge_shards(self, shard_dir, output_path, summaries):
        """Merge NPZ shards into single HDF5. Supports gold_v2_multimodal schema."""
        shard_files = sorted(shard_dir.glob("*.npz"))
        exp_ch = int(self.config.get("network", {}).get("input_channels", 56))
        is_gold_v2 = exp_ch in (56, 60)

        if not shard_files:
            logger.warning("No shard files found, creating empty HDF5")
            with h5py.File(output_path, "w") as f:
                if is_gold_v2:
                    f.create_dataset("spatial_states", shape=(0, C_SPATIAL_ENC, 9, 9), dtype=np.float32)
                    f.create_dataset("global_states", shape=(0, F_GLOBAL), dtype=np.float32)
                    f.create_dataset("track_states", shape=(0, C_TRACK, TRACK_LEN), dtype=np.float32)
                    f.create_dataset("shop_ids", shape=(0, NMAX), dtype=np.int16)
                    f.create_dataset("shop_feats", shape=(0, NMAX, F_SHOP), dtype=np.float32)
                else:
                    f.create_dataset("states", shape=(0, exp_ch, 9, 9), dtype=np.float32)
                f.create_dataset("action_masks", shape=(0, 2026), dtype=np.float32)
                f.create_dataset("policies", shape=(0, 2026), dtype=np.float32)
                f.create_dataset("values", shape=(0,), dtype=np.float32)
                f.create_dataset("ownerships", shape=(0, 2, 9, 9), dtype=np.float32)
                f.attrs["num_games"] = 0
                f.attrs["num_positions"] = 0
            return

        # Detect schema from first non-empty shard
        with np.load(shard_files[0]) as first:
            has_multimodal = "spatial_states" in first

        logger.debug(f"Merging {len(shard_files)} shards into {output_path}")
        sp_cfg = self.config.get("selfplay", {}) or {}
        compression = sp_cfg.get("hdf5_compression", "lzf")
        if compression is not None:
            compression = str(compression).lower()
        compression_level = int(sp_cfg.get("hdf5_compression_level", 4))
        if compression in ("none", "", "null"):
            compression = None
        compression_opts = compression_level if compression == "gzip" else None

        with h5py.File(output_path, "w") as out:
            if has_multimodal:
                states_ds = out.create_dataset(
                    "spatial_states",
                    shape=(0, C_SPATIAL_ENC, 9, 9),
                    maxshape=(None, C_SPATIAL_ENC, 9, 9),
                    dtype=np.float32,
                    chunks=(256, C_SPATIAL_ENC, 9, 9),
                    compression=compression,
                    compression_opts=compression_opts,
                )
                global_ds = out.create_dataset(
                    "global_states",
                    shape=(0, F_GLOBAL),
                    maxshape=(None, F_GLOBAL),
                    dtype=np.float16,
                    chunks=(512, F_GLOBAL),
                    compression=compression,
                    compression_opts=compression_opts,
                )
                track_ds = out.create_dataset(
                    "track_states",
                    shape=(0, C_TRACK, TRACK_LEN),
                    maxshape=(None, C_TRACK, TRACK_LEN),
                    dtype=np.float16,
                    chunks=(512, C_TRACK, TRACK_LEN),
                    compression=compression,
                    compression_opts=compression_opts,
                )
                shop_ids_ds = out.create_dataset(
                    "shop_ids",
                    shape=(0, NMAX),
                    maxshape=(None, NMAX),
                    dtype=np.int16,
                    chunks=(512, NMAX),
                    compression=compression,
                    compression_opts=compression_opts,
                )
                shop_feats_ds = out.create_dataset(
                    "shop_feats",
                    shape=(0, NMAX, F_SHOP),
                    maxshape=(None, NMAX, F_SHOP),
                    dtype=np.float16,
                    chunks=(256, NMAX, F_SHOP),
                    compression=compression,
                    compression_opts=compression_opts,
                )
            else:
                states_ds = out.create_dataset(
                    "states",
                    shape=(0, exp_ch, 9, 9),
                    maxshape=(None, exp_ch, 9, 9),
                    dtype=np.float32,
                    chunks=(256, exp_ch, 9, 9),
                    compression=compression,
                    compression_opts=compression_opts,
                )
            masks_ds = out.create_dataset(
                "action_masks",
                shape=(0, 2026),
                maxshape=(None, 2026),
                dtype=np.float32,
                chunks=(256, 2026),
                compression=compression,
                compression_opts=compression_opts,
            )
            policies_ds = out.create_dataset(
                "policies",
                shape=(0, 2026),
                maxshape=(None, 2026),
                dtype=np.float32,
                chunks=(256, 2026),
                compression=compression,
                compression_opts=compression_opts,
            )
            values_ds = out.create_dataset(
                "values",
                shape=(0,),
                maxshape=(None,),
                dtype=np.float32,
                chunks=(1024,),
                compression=compression,
                compression_opts=compression_opts,
            )
            score_margins_ds = out.create_dataset(
                "score_margins",
                shape=(0,),
                maxshape=(None,),
                dtype=np.float32,
                chunks=(1024,),
                compression=compression,
                compression_opts=compression_opts,
            )
            ownerships_ds = out.create_dataset(
                "ownerships",
                shape=(0, 2, 9, 9),
                maxshape=(None, 2, 9, 9),
                dtype=np.float32,
                chunks=(256, 2, 9, 9),
                compression=compression,
                compression_opts=compression_opts,
            )
            slot_piece_ids_ds = None
            canonical_mode = None
            scores_mode = None

            total_pos = 0
            for i, shard_path in enumerate(shard_files):
                if i % 100 == 0 and i > 0:
                    logger.debug(f"Merged {i}/{len(shard_files)} shards ({total_pos} positions)")

                with np.load(shard_path) as shard:
                    n = shard["states"].shape[0]
                    if n == 0:
                        continue

                    shard_multimodal = "spatial_states" in shard
                    if shard_multimodal != has_multimodal:
                        raise ValueError(
                            f"Cannot mix multimodal and legacy shards: "
                            f"{shard_path} has_multimodal={shard_multimodal}"
                        )
                    has_slot_ids = "slot_piece_ids" in shard
                    has_scores = "score_margins" in shard
                    score_loss_weight = float(
                        (self.config.get("training", {}) or {}).get("score_loss_weight", 0.0)
                    )
                    if score_loss_weight > 0 and not has_scores:
                        raise ValueError(
                            f"Shard {shard_path} has no score_margins but training has score_loss_weight > 0. "
                            "Regenerate self-play with score head enabled."
                        )
                    if canonical_mode is None:
                        canonical_mode = has_slot_ids
                    elif canonical_mode != has_slot_ids:
                        raise ValueError(
                            f"Cannot mix canonical and pre-augmented shards: "
                            f"{shard_path} has slot_piece_ids={has_slot_ids}"
                        )
                    if scores_mode is None:
                        scores_mode = has_scores

                    new_size = total_pos + n
                    if has_multimodal:
                        states_ds.resize(new_size, axis=0)
                        global_ds.resize(new_size, axis=0)
                        track_ds.resize(new_size, axis=0)
                        shop_ids_ds.resize(new_size, axis=0)
                        shop_feats_ds.resize(new_size, axis=0)
                    else:
                        states_ds.resize(new_size, axis=0)
                    masks_ds.resize(new_size, axis=0)
                    policies_ds.resize(new_size, axis=0)
                    values_ds.resize(new_size, axis=0)
                    score_margins_ds.resize(new_size, axis=0)
                    ownerships_ds.resize(new_size, axis=0)

                    if has_multimodal:
                        states_ds[total_pos:new_size] = shard["spatial_states"]
                        global_ds[total_pos:new_size] = shard["global_states"].astype(np.float16)
                        track_ds[total_pos:new_size] = shard["track_states"].astype(np.float16)
                        shop_ids_ds[total_pos:new_size] = shard["shop_ids"]
                        shop_feats_ds[total_pos:new_size] = shard["shop_feats"].astype(np.float16)
                    else:
                        states_ds[total_pos:new_size] = shard["states"]
                    masks_ds[total_pos:new_size] = shard["action_masks"]
                    policies_ds[total_pos:new_size] = shard["policies"]
                    values_ds[total_pos:new_size] = shard["values"]
                    if has_scores:
                        sc = np.asarray(shard["score_margins"], dtype=np.float32)
                        _validate_score_margins(
                            sc,
                            max_abs=SCORE_MARGIN_MAX_ABS,
                            source=str(shard_path),
                        )
                        score_margins_ds[total_pos:new_size] = sc
                    else:
                        score_margins_ds[total_pos:new_size] = np.zeros(n, dtype=np.float32)
                    ownerships_ds[total_pos:new_size] = shard["ownerships"]

                    if has_slot_ids:
                        if slot_piece_ids_ds is None:
                            slot_piece_ids_ds = out.create_dataset(
                                "slot_piece_ids",
                                shape=(0, 3),
                                maxshape=(None, 3),
                                dtype=np.int16,
                                chunks=(512, 3),
                                compression=compression,
                                compression_opts=compression_opts,
                            )
                        slot_piece_ids_ds.resize(new_size, axis=0)
                        slot_piece_ids_ds[total_pos:new_size] = shard["slot_piece_ids"]

                    total_pos = new_size

            out.attrs["num_games"] = len(summaries)
            out.attrs["num_positions"] = total_pos
            out.attrs[SELFPLAY_COMPLETE_ATTR] = True
            out.attrs[SELFPLAY_NUM_GAMES_ATTR] = len(summaries)
            out.attrs[SELFPLAY_SCHEMA_VERSION_ATTR] = SELFPLAY_SCHEMA_VERSION
            out.attrs[SELFPLAY_EXPECTED_CHANNELS_ATTR] = C_SPATIAL_ENC if has_multimodal else exp_ch
            out.attrs[ENCODING_VERSION_ATTR] = ENCODING_VERSION if has_multimodal else "full_clarity_v1"
            out.attrs[SELFPLAY_VALUE_TARGET_TYPE_ATTR] = "dual_head"
            if has_multimodal:
                out.attrs["C_spatial"] = C_SPATIAL_ENC
                out.attrs["C_track"] = C_TRACK
                out.attrs["F_global"] = F_GLOBAL
                out.attrs["F_shop"] = F_SHOP
                out.attrs["Nmax"] = NMAX

        logger.debug(f"Merge complete: {total_pos} positions from {len(summaries)} games")

        # Shard cleanup already done in _merge_shards() method

    def _get_num_games(self, iteration: int) -> int:
        """Determine number of games for iteration."""
        if iteration == 0:
            return self.config["selfplay"]["bootstrap"]["games"]

        # Check for dynamic schedule
        schedule = self.config.get("iteration", {}).get("games_schedule", [])
        num_games = self.config["selfplay"]["games_per_iteration"]

        for entry in sorted(schedule, key=lambda x: x["iteration"], reverse=True):
            if iteration >= entry["iteration"]:
                num_games = entry["games"]
                break

        return num_games

    def _apply_iteration_schedules(self, iteration: int, quiet: bool = False) -> dict:
        """Apply temperature and MCTS schedules based on iteration.

        MAJOR FIX: Implement temperature_schedule and mcts_schedule application.
        Returns a modified config copy with schedules applied.
        quiet: if True, skip logging (used when caller just needs effective values).
        """
        import copy
        config = copy.deepcopy(self.config)

        # Apply temperature schedule
        temp_schedule = self.config.get("iteration", {}).get("temperature_schedule", [])
        for entry in sorted(temp_schedule, key=lambda x: x["iteration"], reverse=True):
            if iteration >= entry["iteration"]:
                config["selfplay"]["mcts"]["temperature"] = entry["temperature"]
                break

        # Apply MCTS simulations schedule
        mcts_schedule = self.config.get("iteration", {}).get("mcts_schedule", [])
        for entry in sorted(mcts_schedule, key=lambda x: x["iteration"], reverse=True):
            if iteration >= entry["iteration"]:
                config["selfplay"]["mcts"]["simulations"] = entry["simulations"]
                break

        # Apply Dirichlet alpha schedule
        alpha_schedule = self.config.get("iteration", {}).get("dirichlet_alpha_schedule", [])
        for entry in sorted(alpha_schedule, key=lambda x: x["iteration"], reverse=True):
            if iteration >= entry["iteration"]:
                config["selfplay"]["mcts"]["root_dirichlet_alpha"] = entry["alpha"]
                break

        # Apply noise weight (epsilon) schedule
        noise_schedule = self.config.get("iteration", {}).get("noise_weight_schedule", [])
        for entry in sorted(noise_schedule, key=lambda x: x["iteration"], reverse=True):
            if iteration >= entry["iteration"]:
                config["selfplay"]["mcts"]["root_noise_weight"] = entry["weight"]
                break

        # Apply parallel_leaves schedule (batch width for MCTS inference)
        pl_schedule = self.config.get("iteration", {}).get("parallel_leaves_schedule", [])
        for entry in sorted(pl_schedule, key=lambda x: x["iteration"], reverse=True):
            if iteration >= entry["iteration"]:
                config["selfplay"]["mcts"]["parallel_leaves"] = int(entry["parallel_leaves"])
                if not quiet:
                    logger.debug("[SCHEDULE] iter=%d parallel_leaves=%d", iteration, config["selfplay"]["mcts"]["parallel_leaves"])
                break

        # Bootstrap override: use bootstrap.mcts_simulations for iteration 0
        bootstrap_cfg = config["selfplay"].get("bootstrap", {})
        if iteration == 0 and bootstrap_cfg.get("use_pure_mcts"):
            bootstrap_sims = bootstrap_cfg.get("mcts_simulations")
            if bootstrap_sims is not None:
                config["selfplay"]["mcts"]["simulations"] = bootstrap_sims

        if not quiet:
            mcts_cfg = config["selfplay"]["mcts"]
            logger.debug(
                f"Schedule: temp={mcts_cfg['temperature']:.2f}  sims={mcts_cfg['simulations']}  "
                f"alpha={mcts_cfg['root_dirichlet_alpha']}  noise={mcts_cfg['root_noise_weight']}  "
                f"parallel_leaves={mcts_cfg.get('parallel_leaves', '?')}"
            )

        return config

    def _compute_stats(self, summaries, generation_time) -> Dict:
        """Compute statistics from game summaries."""
        empty_beat_humans = {
            "selfplay_avg_final_empty_squares_mean": 0.0,
            "selfplay_avg_final_empty_components_mean": 0.0,
            "selfplay_avg_final_isolated_1x1_holes_mean": 0.0,
            "selfplay_p50_final_empty_squares_mean": 0.0,
            "selfplay_p90_final_empty_squares_mean": 0.0,
            "selfplay_p50_final_empty_components_mean": 0.0,
            "selfplay_p90_final_empty_components_mean": 0.0,
            "selfplay_p50_final_isolated_1x1_holes_mean": 0.0,
            "selfplay_p90_final_isolated_1x1_holes_mean": 0.0,
            "selfplay_avg_final_empty_squares_abs_diff": 0.0,
            "selfplay_avg_final_empty_components_abs_diff": 0.0,
            "selfplay_avg_final_isolated_1x1_holes_abs_diff": 0.0,
            "selfplay_avg_root_legal_count": 0.0,
            "selfplay_avg_root_expanded_count": 0.0,
            "selfplay_avg_root_expanded_ratio": 0.0,
            "selfplay_p90_root_legal_count": 0.0,
            "selfplay_p90_root_expanded_ratio": 0.0,
            "selfplay_frac_games_vs_packer": 0.0,
            "selfplay_nn_vs_packer_winrate": 0.0,
            "selfplay_avg_final_empty_components_mean_vs_packer": 0.0,
            "selfplay_avg_final_isolated_1x1_holes_mean_vs_packer": 0.0,
            "selfplay_avg_packing_ordering_enabled": 0.0,
            "selfplay_avg_packing_score_top1": 0.0,
        }
        if not summaries:
            return {
                "num_games": 0,
                "num_positions": 0,
                "avg_game_length": 0,
                "p0_wins": 0,
                "p1_wins": 0,
                "generation_time": generation_time,
                "games_per_minute": 0,
                "avg_policy_entropy": 0.0,
                "avg_top1_prob": 0.0,
                "avg_num_legal": 0.0,
                "avg_redundancy": 0.0,
                "unique_positions": 0,
                "avg_root_q": 0.0,
                **empty_beat_humans,
            }

        game_lengths = [s.get("game_length", 0) for s in summaries]
        winners = [s.get("winner", -1) for s in summaries]
        num_positions = sum(s.get("num_positions", 0) for s in summaries)

        # Policy collapse canary metrics
        entropies = [s.get("avg_policy_entropy", 0) for s in summaries if s.get("avg_policy_entropy")]
        top1_probs = [s.get("avg_top1_prob", 0) for s in summaries if s.get("avg_top1_prob")]
        num_legals = [s.get("avg_num_legal", 0) for s in summaries if s.get("avg_num_legal")]

        # Position redundancy (Oracle Part 2)
        redundancies = [s.get("redundancy", 0) for s in summaries if "redundancy" in s]
        unique_pos = sum(s.get("unique_positions", 0) for s in summaries)

        # Root Q stats
        root_qs = [s.get("avg_root_q", 0) for s in summaries if "avg_root_q" in s]

        # Packing / quilt quality (beat-humans): per-game (e0, e1, c0, c1, i0, i1)
        packing_per_game = []
        for s in summaries:
            e0 = s.get("empty_squares_p0", 0)
            e1 = s.get("empty_squares_p1", 0)
            c0 = s.get("empty_components_p0", 0)
            c1 = s.get("empty_components_p1", 0)
            i0 = s.get("isolated_1x1_holes_p0", 0)
            i1 = s.get("isolated_1x1_holes_p1", 0)
            packing_per_game.append((e0, e1, c0, c1, i0, i1))
        packing_stats = aggregate_packing_over_games(packing_per_game) if packing_per_game else empty_beat_humans

        # Root search health: flatten lists across all games
        all_root_legal = []
        all_root_expanded = []
        for s in summaries:
            all_root_legal.extend(s.get("root_legal_counts", []))
            all_root_expanded.extend(s.get("root_expanded_counts", []))
        root_stats = aggregate_root_over_moves(all_root_legal, all_root_expanded)

        # Opponent mix (vs packer): fraction of games and NN win rate in those games
        vs_packer_games = [s for s in summaries if s.get("vs_packer")]
        nn_wons = [s.get("nn_won") for s in vs_packer_games if "nn_won" in s]
        frac_vs_packer = len(vs_packer_games) / len(summaries) if summaries else 0.0
        nn_vs_packer_wr = float(np.mean(nn_wons)) if nn_wons else 0.0
        # Packing stats in vs_packer games only (per-game mean across both players)
        comps_vs_packer = [
            (s.get("empty_components_p0", 0) + s.get("empty_components_p1", 0)) / 2.0
            for s in vs_packer_games
        ]
        isol_vs_packer = [
            (s.get("isolated_1x1_holes_p0", 0) + s.get("isolated_1x1_holes_p1", 0)) / 2.0
            for s in vs_packer_games
        ]
        mean_comp_vs_packer = float(np.mean(comps_vs_packer)) if comps_vs_packer else 0.0
        mean_isol_vs_packer = float(np.mean(isol_vs_packer)) if isol_vs_packer else 0.0

        return {
            "num_games": len(summaries),
            "num_positions": num_positions,
            "avg_game_length": float(np.mean(game_lengths)) if game_lengths else 0,
            "p0_wins": winners.count(0),
            "p1_wins": winners.count(1),
            "generation_time": generation_time,
            "games_per_minute": len(summaries) / (generation_time / 60.0) if generation_time > 0 else 0,
            # Policy collapse canary — watch these in TensorBoard
            "avg_policy_entropy": float(np.mean(entropies)) if entropies else 0.0,
            "avg_top1_prob": float(np.mean(top1_probs)) if top1_probs else 0.0,
            "avg_num_legal": float(np.mean(num_legals)) if num_legals else 0.0,
            # Position redundancy (Oracle Part 2): 0 = fully unique, >0.5 = worrying
            "avg_redundancy": float(np.mean(redundancies)) if redundancies else 0.0,
            "unique_positions": int(unique_pos),
            # Root Q value stats
            "avg_root_q": float(np.mean(root_qs)) if root_qs else 0.0,
            **packing_stats,
            **root_stats,
            "selfplay_frac_games_vs_packer": frac_vs_packer,
            "selfplay_nn_vs_packer_winrate": nn_vs_packer_wr,
            "selfplay_avg_final_empty_components_mean_vs_packer": mean_comp_vs_packer,
            "selfplay_avg_final_isolated_1x1_holes_mean_vs_packer": mean_isol_vs_packer,
            "selfplay_avg_packing_ordering_enabled": 1.0 if (self.config.get("selfplay", {}).get("mcts", {}).get("packing_ordering", {}).get("enabled", False)) else 0.0,
            "selfplay_avg_packing_score_top1": float(np.mean([s.get("packing_score_top1", 0.0) for s in summaries if "packing_score_top1" in s])) if any("packing_score_top1" in s for s in summaries) else 0.0,
        }


def create_selfplay_generator(config: dict) -> SelfPlayGenerator:
    """Factory function for main.py compatibility.

    Args:
        config: Full training configuration dict

    Returns:
        SelfPlayGenerator instance ready to generate training data
    """
    return SelfPlayGenerator(config)
