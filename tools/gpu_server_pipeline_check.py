#!/usr/bin/env python3
"""
Fast diagnostic: validate GPU-server startup pipeline (SHM + spawn + ready handshake)
without running full self-play. Stress per-iteration boundary by repeatedly starting
and stopping the GPU server child. Outputs a single report for pasting to ChatGPT.

Usage:
  python tools/gpu_server_pipeline_check.py --config configs/config_best.yaml --cycles 100 --timeout 30
  python tools/gpu_server_pipeline_check.py --checkpoint path/to/iteration_005.pt --cycles 50
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import queue
import sys
import time
from pathlib import Path

import numpy as np

# Project root
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_config(config_path: Path) -> dict:
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get_expected_n_slots(config: dict) -> int:
    """Same logic as production SelfPlayGenerator._get_expected_n_slots."""
    base = int(
        config.get("selfplay", {}).get("mcts", {}).get("parallel_leaves", 32)
    )
    pl_schedule = config.get("iteration", {}).get("parallel_leaves_schedule", [])
    if pl_schedule:
        return max(int(e.get("parallel_leaves", base)) for e in pl_schedule)
    return base


def _resolve_checkpoint(checkpoint_arg: str | None, config: dict) -> Path:
    """Return checkpoint path: --checkpoint, else checkpoints/, else newest committed."""
    if checkpoint_arg:
        p = Path(checkpoint_arg)
        if p.is_absolute():
            if not p.exists():
                raise FileNotFoundError(f"Checkpoint not found: {p}")
            return p
        p = REPO_ROOT / p
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return p

    paths_config = config.get("paths", {})
    ckpt_dir = Path(paths_config.get("checkpoints_dir", "checkpoints"))
    if not ckpt_dir.is_absolute():
        ckpt_dir = REPO_ROOT / ckpt_dir

    # Prefer best.pt, then latest_model.pt
    for name in ("best.pt", "latest_model.pt"):
        cand = ckpt_dir / name
        if cand.exists():
            return cand

    # Newest in runs/*/committed/iter_*/*.pt
    runs_dir = REPO_ROOT / "runs"
    if runs_dir.exists():
        best_pt: Path | None = None
        best_mtime = 0.0
        for committed_dir in runs_dir.rglob("committed"):
            for pt in committed_dir.rglob("*.pt"):
                if pt.is_file():
                    m = pt.stat().st_mtime
                    if m > best_mtime:
                        best_mtime = m
                        best_pt = pt
        if best_pt is not None:
            return best_pt

    raise FileNotFoundError(
        "No checkpoint found. Use --checkpoint PATH, or ensure checkpoints/best.pt, "
        "checkpoints/latest_model.pt, or runs/*/committed/**/*.pt exists."
    )


def _make_synthetic_states(n: int = 4):
    """Build n minimal gold_v2 state tuples for burst (queue path)."""
    from src.network.gold_v2_constants import (
        C_SPATIAL_ENC,
        F_GLOBAL,
        C_TRACK,
        TRACK_LEN,
        NMAX,
        F_SHOP,
        MAX_ACTIONS,
    )
    states = []
    for _ in range(n):
        x_spatial = np.zeros((C_SPATIAL_ENC, 9, 9), dtype=np.float32)
        x_global = np.zeros(F_GLOBAL, dtype=np.float32)
        x_track = np.zeros((C_TRACK, TRACK_LEN), dtype=np.float32)
        shop_ids = np.zeros(NMAX, dtype=np.int64)
        shop_feats = np.zeros((NMAX, F_SHOP), dtype=np.float32)
        mask = np.zeros(MAX_ACTIONS, dtype=np.float32)
        legal_idxs = np.array([0, 1, 2], dtype=np.int32)  # pass + 2 patch actions
        mask[0] = 1.0
        mask[1] = 1.0
        mask[2] = 1.0
        states.append((x_spatial, x_global, x_track, shop_ids, shop_feats, mask, legal_idxs))
    return states


def _run_burst(
    req_q,
    resp_qs: list,
    burst_workers: int,
    burst_requests: int,
    receive_timeout_s: float = 60.0,
) -> tuple[float, int, int]:
    """Send burst_requests via GPUEvalClient (submit_multimodal), receive all, verify. Returns (req_per_s, timeouts, malformed)."""
    from src.mcts.gpu_eval_client import GPUEvalClient

    n_workers = min(burst_workers, len(resp_qs))
    clients = [
        GPUEvalClient(req_q, resp_qs[wid], worker_id=wid, timeout_s=receive_timeout_s)
        for wid in range(n_workers)
    ]
    synthetic = _make_synthetic_states()
    t0 = time.perf_counter()
    rids: list[tuple[int, GPUEvalClient]] = []
    for i in range(burst_requests):
        wid = i % n_workers
        st = synthetic[i % len(synthetic)]
        rid = clients[wid].submit_multimodal(
            st[0], st[1], st[2], st[3], st[4], st[5], st[6],
            score_center_points=0.0, effective_static_w=0.0, effective_dynamic_w=0.3,
        )
        rids.append((rid, clients[wid]))
    submit_done = time.perf_counter() - t0

    timeouts = 0
    malformed = 0
    for rid, client in rids:
        try:
            priors, value, mean_pts, score_util = client.receive(rid)
            if not isinstance(priors, np.ndarray) or priors.dtype != np.float32:
                malformed += 1
            if not isinstance(value, (float, np.floating)):
                malformed += 1
            if not isinstance(mean_pts, (float, np.floating)):
                malformed += 1
            if not isinstance(score_util, (float, np.floating)):
                malformed += 1
            if len(priors) != 3:  # we sent legal_idxs length 3
                malformed += 1
        except TimeoutError:
            timeouts += 1
        except Exception:
            malformed += 1

    total_s = time.perf_counter() - t0
    req_per_s = burst_requests / total_s if total_s > 0 else 0.0
    return req_per_s, timeouts, malformed


def run_cycle(
    ctx: mp.context.SpawnContext,
    config: dict,
    checkpoint_path: Path,
    num_workers: int,
    expected_n_slots: int,
    timeout_s: float,
    cycle: int,
    report_lines: list[str],
    verbose_cycles: set[int],
    burst_requests: int = 0,
    burst_workers: int = 12,
) -> tuple[bool, str | None, int | None, dict | None]:
    """
    One cycle: create SHM, spawn GPU server, wait for ready, optionally run burst, stop, cleanup.
    Returns (success, error_msg_or_traceback, child_exitcode, burst_stats or None).
    """
    from src.mcts.shared_state_buffer import WorkerSharedBuffer
    from src.network.gpu_inference_server import run_gpu_inference_server

    req_q = ctx.Queue(maxsize=4096)
    resp_qs = [ctx.Queue(maxsize=4096) for _ in range(num_workers)]
    stop_evt = ctx.Event()
    ready_q = ctx.Queue(maxsize=1)

    worker_shm_bufs: dict[int, WorkerSharedBuffer] = {}
    worker_shm_names: dict[int, str] = {}
    gpu_process = None
    child_exitcode: int | None = None
    burst_stats: dict | None = None

    try:
        # Create SHM exactly like production
        for wid in range(num_workers):
            buf = WorkerSharedBuffer(
                n_slots=expected_n_slots, worker_id=wid, create=True
            )
            worker_shm_bufs[wid] = buf
            worker_shm_names[wid] = buf.name

        # Validate derived >= expected_n_slots (should always hold after creation)
        slot_bytes = WorkerSharedBuffer.SLOT_BYTES
        for wid, buf in worker_shm_bufs.items():
            derived = buf._shm.size // slot_bytes
            if derived < expected_n_slots:
                raise ValueError(
                    f"Cycle {cycle} wid={wid}: derived={derived} < expected_n_slots={expected_n_slots} "
                    f"(size={buf._shm.size} SLOT_BYTES={slot_bytes})"
                )

        # Compact log line for this cycle (first 3 + every 25th)
        if cycle in verbose_cycles:
            parts = [f"cycle={cycle} expected_n_slots={expected_n_slots} SLOT_BYTES={slot_bytes}"]
            for wid, buf in worker_shm_bufs.items():
                d = buf._shm.size // slot_bytes
                parts.append(f"wid{wid}: name={buf.name!r} size={buf._shm.size} derived={d}")
            line = " | ".join(parts)
            print(line)
            report_lines.append(line)

        # Spawn GPU server (same entrypoint and args as production)
        gpu_process = ctx.Process(
            target=run_gpu_inference_server,
            args=(config, str(checkpoint_path), req_q, resp_qs, stop_evt, ready_q),
            kwargs={
                "worker_shm_names": worker_shm_names,
                "expected_n_slots": expected_n_slots,
            },
            daemon=False,
        )
        gpu_process.start()

        try:
            status = ready_q.get(timeout=timeout_s)
        except queue.Empty:
            child_exitcode = getattr(gpu_process, "exitcode", None)
            return False, None, child_exitcode, None

        if status != "ready":
            if isinstance(status, tuple) and len(status) >= 3 and status[0] == "error":
                _, err_msg, err_tb = status[0], status[1], status[2]
                return False, f"{err_msg}\n\n{err_tb}", getattr(gpu_process, "exitcode", None), None
            return False, f"Unexpected status: {type(status)} {status!r}", getattr(gpu_process, "exitcode", None), None

        # Server is ready: run burst if requested
        if burst_requests > 0 and gpu_process.is_alive():
            req_per_s, timeouts, malformed = _run_burst(
                req_q, resp_qs, burst_workers, burst_requests, receive_timeout_s=120.0
            )
            burst_stats = {"req_per_s": req_per_s, "timeouts": timeouts, "malformed": malformed}

        # Stop server cleanly (production path)
        stop_evt.set()
        gpu_process.join(timeout=5)
        if gpu_process.is_alive():
            gpu_process.terminate()
            gpu_process.join(timeout=2)
        return True, None, None, burst_stats

    finally:
        if gpu_process and gpu_process.is_alive():
            stop_evt.set()
            gpu_process.join(timeout=2)
            if gpu_process.is_alive():
                gpu_process.terminate()
                gpu_process.join(timeout=1)
        for buf in worker_shm_bufs.values():
            try:
                buf.destroy()
            except Exception:
                pass
        worker_shm_bufs.clear()
        worker_shm_names.clear()


def build_report(
    config_path: Path,
    config: dict,
    num_workers: int,
    parallel_leaves: int,
    pl_schedule: list,
    expected_n_slots: int,
    slot_bytes: int,
    first_cycle_shm_lines: list[str],
    pass_count: int,
    fail_count: int,
    failure_detail: str | None,
    duration_s: float,
    burst_stats: dict | None = None,
    shm_margin: tuple[int, int, int] | None = None,
) -> str:
    """Build the markdown report content."""
    lines = [
        "# GPU Server Pipeline Check Report",
        "",
        "## Config summary",
        f"- Config file: `{config_path}`",
        f"- `selfplay.num_workers`: {num_workers}",
        f"- `selfplay.mcts.parallel_leaves`: {parallel_leaves}",
        f"- `iteration.parallel_leaves_schedule`: {pl_schedule!r}",
        "",
        "## Expected slot computation",
        f"- Logic: max(schedule) if schedule else base.",
        f"- **expected_n_slots**: {expected_n_slots}",
        f"- **WorkerSharedBuffer.SLOT_BYTES**: {slot_bytes}",
        "",
        "## First cycle SHM details",
    ]
    for ln in first_cycle_shm_lines:
        lines.append(f"- {ln}")
    lines.extend([
        "",
        "## GPU server attach logs (from gpu_inference_server)",
        "- Child logs one line per worker after attach, e.g.:",
        "- `[GPU Server] SHM attach wid=0 name=wnsm_XXXX size=1916928 SLOT_BYTES=29940 derived=64 expected_n_slots=64`",
        "- (On Windows, child size may be page-rounded, e.g. 1916928 vs parent 1916160; derived=64 in both.)",
        "- Parent-side first cycle: see First cycle SHM details above.",
        "",
    ])
    if burst_stats is not None:
        req_per_s = burst_stats.get("req_per_s", 0.0)
        timeouts = burst_stats.get("timeouts", 0)
        malformed = burst_stats.get("malformed", 0)
        burst_ok = timeouts == 0 and malformed == 0
        lines.extend([
            "## Burst (live request path)",
            f"- **Request throughput**: {req_per_s:.1f} req/s",
            f"- **Timeouts**: {timeouts}",
            f"- **Malformed responses**: {malformed}",
            f"- **Burst PASS**: " + ("yes" if burst_ok else "no"),
            "",
        ])
    if shm_margin is not None:
        max_seen, capacity_bytes, margin_bytes = shm_margin
        lines.extend([
            "## SHM safety margin (tightest observed)",
            f"- **max_payload_nbytes_seen**: {max_seen}",
            f"- **capacity_bytes** = SLOT_BYTES - HEADER_BYTES: {capacity_bytes}",
            f"- **margin_bytes** = capacity_bytes - max_payload_nbytes_seen: {margin_bytes}",
            "",
        ])
    overall_pass = pass_count > 0 and fail_count == 0 and (burst_stats is None or (burst_stats.get("timeouts", 0) == 0 and burst_stats.get("malformed", 0) == 0))
    lines.extend([
        "## Result",
        f"- **PASS/FAIL**: " + ("PASS" if overall_pass else "FAIL"),
        f"- **Pass**: {pass_count}",
        f"- **Fail**: {fail_count}",
        f"- **Duration (s)**: {duration_s:.2f}",
        "",
    ])
    if failure_detail:
        lines.extend([
            "## Failure detail",
            "```",
            failure_detail,
            "```",
            "",
        ])
    lines.append("---")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate GPU server startup pipeline (SHM + spawn + ready) without full self-play."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "config_best.yaml",
        help="Config YAML path",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=100,
        help="Number of start/stop cycles",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for ready per cycle",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path; else auto-resolved from checkpoints/ or runs/.../committed/",
    )
    parser.add_argument(
        "--burst_requests",
        type=int,
        default=20000,
        help="After ready, send this many inference requests (0 = skip burst)",
    )
    parser.add_argument(
        "--burst_workers",
        type=int,
        default=12,
        help="Number of GPUEvalClient workers for burst (round-robin)",
    )
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    config = _load_config(config_path)
    num_workers = int(config.get("selfplay", {}).get("num_workers", 1))
    parallel_leaves = int(
        config.get("selfplay", {}).get("mcts", {}).get("parallel_leaves", 32)
    )
    pl_schedule = config.get("iteration", {}).get("parallel_leaves_schedule", [])
    expected_n_slots = _get_expected_n_slots(config)

    from src.mcts.shared_state_buffer import WorkerSharedBuffer
    slot_bytes = WorkerSharedBuffer.SLOT_BYTES

    print("Config summary:")
    print(f"  selfplay.num_workers: {num_workers}")
    print(f"  selfplay.mcts.parallel_leaves: {parallel_leaves}")
    print(f"  iteration.parallel_leaves_schedule: {pl_schedule}")
    print(f"  expected_n_slots (max(schedule) else base): {expected_n_slots}")
    print(f"  WorkerSharedBuffer.SLOT_BYTES: {slot_bytes}")

    try:
        checkpoint_path = _resolve_checkpoint(args.checkpoint, config)
    except FileNotFoundError as e:
        print(f"Checkpoint: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"  checkpoint: {checkpoint_path}")

    # Cycles to print in detail: first 3 and every 25th
    verbose_cycles = {0, 1, 2}
    for i in range(25, args.cycles, 25):
        verbose_cycles.add(i)

    report_lines: list[str] = []
    first_cycle_shm_lines: list[str] = []
    pass_count = 0
    fail_count = 0
    failure_detail: str | None = None
    burst_stats_agg: dict | None = None

    ctx = mp.get_context("spawn")
    start = time.perf_counter()

    for cycle in range(args.cycles):
        ok, err_msg, exitcode, burst_stats = run_cycle(
            ctx, config, checkpoint_path, num_workers, expected_n_slots,
            args.timeout, cycle, report_lines, verbose_cycles,
            burst_requests=args.burst_requests,
            burst_workers=args.burst_workers,
        )
        if cycle == 0 and report_lines:
            first_cycle_shm_lines = list(report_lines)
        if burst_stats is not None and burst_stats_agg is None:
            burst_stats_agg = burst_stats
        if ok:
            pass_count += 1
        else:
            fail_count += 1
            if failure_detail is None:
                if err_msg:
                    failure_detail = err_msg
                else:
                    failure_detail = f"Timeout or crash; child exitcode: {exitcode}"
        if not ok:
            print(f"Cycle {cycle} FAIL: exitcode={exitcode} err={err_msg[:200] if err_msg else 'timeout'}")
            break
    else:
        # All passed
        pass_count = args.cycles
        fail_count = 0

    duration_s = time.perf_counter() - start

    # SHM safety margin: from check_slot_write_bounds if any; else burst payload (synthetic 3 legal)
    from src.mcts.shared_state_buffer import get_shm_safety_margin, WorkerSharedBuffer
    max_seen, capacity_bytes, margin_bytes = get_shm_safety_margin()
    if max_seen == 0 and args.burst_requests > 0:
        burst_payload = WorkerSharedBuffer.PAYLOAD_FIXED_BYTES + 3 * 4  # synthetic states use 3 legal
        max_seen = burst_payload
        margin_bytes = capacity_bytes - max_seen
    shm_margin = (max_seen, capacity_bytes, margin_bytes)

    # If we never ran cycle 0 verbose (e.g. fail before first print), fill first-cycle from first run
    if not first_cycle_shm_lines and report_lines:
        first_cycle_shm_lines = report_lines[:num_workers + 1]

    report_content = build_report(
        config_path=config_path,
        config=config,
        num_workers=num_workers,
        parallel_leaves=parallel_leaves,
        pl_schedule=pl_schedule,
        expected_n_slots=expected_n_slots,
        slot_bytes=slot_bytes,
        first_cycle_shm_lines=first_cycle_shm_lines,
        pass_count=pass_count,
        fail_count=fail_count,
        failure_detail=failure_detail,
        duration_s=duration_s,
        burst_stats=burst_stats_agg,
        shm_margin=shm_margin,
    )

    report_path = REPO_ROOT / "tools" / "GPU_SERVER_PIPELINE_REPORT.md"
    report_path.write_text(report_content, encoding="utf-8")
    print(f"\nReport written to: {report_path}")

    burst_fail = False
    if burst_stats_agg and (burst_stats_agg.get("timeouts", 0) > 0 or burst_stats_agg.get("malformed", 0) > 0):
        burst_fail = True

    if fail_count > 0:
        print("FAIL")
        sys.exit(1)
    if burst_fail:
        print("FAIL (burst: timeouts or malformed responses)")
        sys.exit(1)
    print("PASS")
    print(f"  {pass_count} cycles in {duration_s:.2f}s")
    if burst_stats_agg:
        print(f"  burst: {burst_stats_agg.get('req_per_s', 0):.1f} req/s, 0 timeouts, 0 malformed")
    print(f"  SHM safety: max_payload_nbytes_seen={max_seen} capacity_bytes={capacity_bytes} margin_bytes={margin_bytes}")


if __name__ == "__main__":
    main()
