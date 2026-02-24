"""
Main Training Orchestrator — Production Grade

Coordinates the complete AlphaZero training loop:
1. Self-play data generation (using best/latest model)
2. Network training (with replay buffer, KataGo warm restart)
3. Evaluation (SPRT-based gating with anti-regression floor)
4. Promotion (best_model.pt only updated when gate passes)
5. Iteration

Transactional iteration commits:
  - All outputs for iteration N go to runs/<run_id>/staging/iter_N/
  - Only at the end (after eval + gating) do we atomically commit to committed/iter_N/
  - All file writes use .tmp + os.replace. Commit = atomic rename + run_state update.
  - run_state.json is updated ONLY at commit — never mid-iteration
  - On resume: reconcile run_state with filesystem, discard partial staging, start at last+1

Checkpoint semantics:
  - iteration_XXX.pt: per-iteration snapshot (in committed/iter_N/)
  - latest_model.pt: always the most recently trained model (crash recovery)
  - best_model_iterXXX.pt: the gate-promoted model for self-play generation
  - metadata.jsonl: append-only log with config hash + file hashes

Key design decisions (KataGo-aligned):
  - Fresh optimizer/scheduler per iteration by default (resume_*=false); clean iter-boundary workflow
  - SPRT-based model gating (statistically rigorous, saves compute)
  - Anti-regression floor: even force-accepted models are blocked if below floor
  - KataGo Dual-Head: value=win/loss/tie, score_margin=raw integer
  - Playout cap randomization for data diversity
"""

import argparse
import contextlib
import datetime
import warnings
import hashlib
import io
import json
import logging
import ctypes
import gc
import os
import platform
import random
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import Dict, Optional

# Suppress PyTorch nested-tensor prototype warning (TransformerEncoder with src_key_padding_mask)
warnings.filterwarnings(
    "ignore",
    message="The PyTorch API of nested tensors is in prototype stage",
    category=UserWarning,
)
# Suppress Flash Attention unavailable warning (Windows PyTorch binary; falls back to math attention)
warnings.filterwarnings(
    "ignore",
    message=".*Torch was not compiled with flash attention.*",
    category=UserWarning,
)

# Graceful shutdown: first SIGINT/SIGTERM = stop after current iteration commits;
# second = exit immediately, unless commit is in progress (then wait for it).
_shutdown_requested = False
_hard_stop_requested = False
_commit_in_progress = False


def _handle_shutdown_signal(signum: int, frame) -> None:
    global _shutdown_requested, _hard_stop_requested
    logger = logging.getLogger(__name__)
    if _hard_stop_requested:
        sys.stdout.flush()
        sys.stderr.flush()
        terminate_active_pool()  # Gracefully kill workers before exit (prevents BrokenPipeError)
        os._exit(1)
    if _shutdown_requested:
        _hard_stop_requested = True
        if _commit_in_progress:
            logger.warning(
                "[SHUTDOWN] Hard stop requested but commit in progress; "
                "exiting immediately after commit completes."
            )
            sys.stdout.flush()
            sys.stderr.flush()
            return  # Do NOT os._exit; let commit finish, then loop will break
        logger.warning("[SHUTDOWN] Second interrupt — exiting immediately.")
        sys.stdout.flush()
        sys.stderr.flush()
        terminate_active_pool()  # Gracefully kill workers before exit (prevents BrokenPipeError)
        os._exit(1)
    _shutdown_requested = True
    logger.warning(
        "[SHUTDOWN] Stop requested (Ctrl+C); will exit after current iteration commits. "
        "Press Ctrl+C again to exit immediately."
    )
    sys.stdout.flush()
    sys.stderr.flush()


def _request_shutdown() -> None:
    """Set shutdown flag (e.g. from STOP_AFTER_ITERATION file)."""
    global _shutdown_requested
    _shutdown_requested = True


def _get_lr_phase_info(config: dict, iteration: int) -> tuple[int, int, float]:
    """Return (phase_start_iter, phase_end_iter, base_lr) for the current phase."""
    entries = sorted(
        config.get("iteration", {}).get("lr_schedule", []) or [{"iteration": 0, "lr": 0.001}],
        key=lambda x: x["iteration"],
    )
    phase_start = 0
    phase_end = 999999
    base_lr = float(config.get("training", {}).get("learning_rate", 0.001))
    for i, ent in enumerate(entries):
        if iteration >= ent["iteration"]:
            phase_start = ent["iteration"]
            base_lr = float(ent.get("lr", base_lr))
            phase_end = entries[i + 1]["iteration"] if i + 1 < len(entries) else 999999
    return phase_start, phase_end, base_lr


import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

# ── Colorama: pretty terminal output (graceful fallback if not installed) ────
try:
    from colorama import Fore, Style, init as _colorama_init
    _colorama_init(autoreset=True)
    _CLR_BAR   = "\033[38;5;208m"  # Orange (xterm-256 index 208)
    _CLR_HDR   = Fore.CYAN + Style.BRIGHT
    _CLR_DIM   = Style.DIM
    _CLR_VAL   = Fore.YELLOW
    _CLR_OK    = Fore.GREEN + Style.BRIGHT
    _CLR_FAIL  = Fore.RED + Style.BRIGHT
    _CLR_SECT  = Fore.CYAN
    _CLR_RST   = Style.RESET_ALL
except ImportError:
    _CLR_BAR = _CLR_HDR = _CLR_DIM = _CLR_VAL = _CLR_OK = _CLR_FAIL = _CLR_SECT = _CLR_RST = ""

_HBAR = "━" * 100


def _fmt_k(n: int) -> str:
    """Format a large integer as e.g. '301k' or '1.2M'."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n // 1_000}k"
    return str(n)


def _print_iter_header(
    iteration: int,
    total_iterations: int,
    best_label: str,
    buf_positions: int,
    rejections: int,
    applied_settings: dict,
) -> None:
    """Print the colorized iteration header directly to stdout (bypasses logger timestamps)."""
    sp = applied_settings.get("selfplay", {})
    tr = applied_settings.get("training", {})
    sims  = int(sp.get("simulations", 0))
    temp  = float(sp.get("temperature", 1.0))
    alpha = float(sp.get("dirichlet_alpha", 0.0))
    eps   = float(sp.get("noise_weight", 0.0))
    cpuct = float(sp.get("cpuct", 1.5))
    q_wt  = float(sp.get("q_value_weight", 0.0))
    lr    = float(tr.get("lr", 0.0))
    games = int(sp.get("games", 0))

    bar  = _CLR_BAR + _HBAR + _CLR_RST
    hdr  = (
        f"  {_CLR_HDR}ITER {iteration:03d}/{total_iterations}{_CLR_RST}"
        f"  ·  {_CLR_DIM}best={_CLR_RST}{_CLR_VAL}{best_label}{_CLR_RST}"
        f"  ·  {_CLR_DIM}buf={_CLR_RST}{_CLR_VAL}{_fmt_k(buf_positions)} pos{_CLR_RST}"
        f"  ·  {_CLR_DIM}rejects={_CLR_RST}{_CLR_VAL}{rejections}{_CLR_RST}"
    )
    params = (
        f"  {_CLR_DIM}sims={_CLR_RST}{_CLR_VAL}{sims}{_CLR_RST}"
        f"  {_CLR_DIM}temp={_CLR_RST}{_CLR_VAL}{temp:.2f}{_CLR_RST}"
        f"  {_CLR_DIM}α={_CLR_RST}{_CLR_VAL}{alpha:.2f}{_CLR_RST}"
        f"  {_CLR_DIM}ε={_CLR_RST}{_CLR_VAL}{eps:.2f}{_CLR_RST}"
        f"  {_CLR_DIM}cpuct={_CLR_RST}{_CLR_VAL}{cpuct:.2f}{_CLR_RST}"
        f"  {_CLR_DIM}q_wt={_CLR_RST}{_CLR_VAL}{q_wt:.2f}{_CLR_RST}"
        f"  {_CLR_DIM}LR={_CLR_RST}{_CLR_VAL}{lr:.2e}{_CLR_RST}"
        f"  {_CLR_DIM}games={_CLR_RST}{_CLR_VAL}{games}{_CLR_RST}"
    )
    print(f"\n{bar}\n{hdr}\n{params}\n{bar}", flush=True)


def _print_auto_resume(
    last_comm: int,
    next_iter: int,
    best_label: str,
    best_model_path: str,
    global_step: int,
    rejections: int,
    elo_count: int,
) -> None:
    """Print the colorized auto-resume banner directly to stdout."""
    bar = _CLR_BAR + _HBAR + _CLR_RST
    hdr = (
        f"  {_CLR_HDR}AUTO-RESUME{_CLR_RST}"
        f"  ·  {_CLR_DIM}last=iter{last_comm:03d}{_CLR_RST}"
        f"  ·  {_CLR_DIM}next=iter{next_iter:03d}{_CLR_RST}"
        f"  ·  {_CLR_DIM}best={best_label}{_CLR_RST}"
        f"  ·  {_CLR_DIM}step={global_step:,}{_CLR_RST}"
    )
    lines = [
        f"  {_CLR_DIM}Last committed :{_CLR_RST} iter{last_comm:03d}",
        f"  {_CLR_DIM}Next iteration :{_CLR_RST} iter{next_iter:03d}",
        f"  {_CLR_DIM}Best model     :{_CLR_RST} {best_label}  ({best_model_path})",
        f"  {_CLR_DIM}Global step    :{_CLR_RST} {global_step:,}",
        f"  {_CLR_DIM}Rejections     :{_CLR_RST} {rejections}",
        f"  {_CLR_DIM}Elo ratings    :{_CLR_RST} {elo_count} players restored",
        f"  {_CLR_DIM}Replay buffer  :{_CLR_RST} will be restored from state file",
    ]
    print(f"\n{bar}\n{hdr}\n" + "\n".join(lines) + f"\n{bar}\n", flush=True)


def _print_section(label: str) -> None:
    """Print a colorized [N/3] section header directly to stdout."""
    print(f"{_CLR_SECT}{label}{_CLR_RST}", flush=True)


def _print_iter_summary(
    iteration: int,
    accepted: bool,
    iter_time: float,
    wr_best_str: str,
    margin_best: float,
    wr_mcts_str: str,
    train_metrics: dict,
) -> None:
    """Print the colorized iteration summary directly to stdout."""
    status_clr = _CLR_OK if accepted else _CLR_FAIL
    status_str = "✓ ACCEPTED" if accepted else "✗ REJECTED"
    loss = train_metrics.get("total_loss", 0)
    pol  = train_metrics.get("policy_accuracy", 0) * 100
    print(
        f"\n  {_CLR_HDR}ITER {iteration:03d}{_CLR_RST}  {status_clr}{status_str}{_CLR_RST}"
        f"  {_CLR_DIM}time={_CLR_RST}{iter_time:.0f}s"
        f"  {_CLR_DIM}WR(best)={_CLR_RST}{wr_best_str}"
        f"  {_CLR_DIM}margin={_CLR_RST}{margin_best:+.1f}pts"
        f"  {_CLR_DIM}WR(mcts)={_CLR_RST}{wr_mcts_str}"
        f"  {_CLR_DIM}loss={_CLR_RST}{loss:.4f}"
        f"  {_CLR_DIM}pol={_CLR_RST}{pol:.1f}%"
        f"\n  {_CLR_SECT}{_HBAR}{_CLR_RST}",
        flush=True,
    )

from src.training.evaluation import Evaluator, EloTracker
from src.training.league import LeagueManager, LeagueConfig, GateResult
from src.training.replay_buffer import ReplayBuffer
from src.training.run_layout import (
    acquire_run_lock,
    cleanup_staging,
    cleanup_stale_tmp_files,
    commit_iteration,
    committed_dir,
    get_run_id,
    get_run_root,
    max_committed_iteration,
    reconcile_run_state,
    staging_dir,
)
from src.training.selfplay_optimized_integration import (
    create_selfplay_generator,
    terminate_active_pool,
)
from src.training.trainer import train_iteration
from src.network.model import create_network


class _NoOpSummaryWriter:
    """Drop-in no-op when logging.tensorboard.enabled=false."""

    def add_scalar(self, *args, **kwargs):  # noqa: D102
        pass

    def flush(self):  # noqa: D102
        pass

    def close(self):  # noqa: D102
        pass


os.makedirs("logs", exist_ok=True)
# Use UTF-8-safe stream for console (avoids UnicodeEncodeError on Windows cp1252)
_safe_stdout = io.TextIOWrapper(
    sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
) if hasattr(sys.stdout, "buffer") else sys.stdout
# Staged training log: NO file handler at startup. FileHandler is added per-iteration
# to staging/iter_N/training.log; only appended to permanent log at commit.
_log_file_handler: Optional[logging.FileHandler] = None

# ── Compact log formatter: 'YYYY-MM-DD HH:MM:SS - [TAG] - message' ──────────
_LOG_TAGS = {
    "__main__":                                        "MAIN",
    "src.training.selfplay_optimized_integration":     "SELFPLAY",
    "src.training.trainer":                            "TRAIN",
    "src.training.evaluation":                         "EVAL",
    "src.training.replay_buffer":                      "BUFFER",
    "src.training.run_layout":                         "COMMIT",
    "src.training.league":                             "LEAGUE",
    "src.network.gpu_inference_server":                "GPU",
    "src.network.model":                               "MODEL",
    "src.mcts.alphazero_mcts_optimized":               "MCTS",
}


class _TidyFormatter(logging.Formatter):
    """Compact formatter: no milliseconds, short module tag."""
    def format(self, record: logging.LogRecord) -> str:
        record.shorttag = _LOG_TAGS.get(record.name, record.name.rsplit(".", 1)[-1].upper()[:12])
        return super().format(record)


_TIDY_FMT = _TidyFormatter(
    fmt="%(asctime)s - [%(shorttag)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_console_handler = logging.StreamHandler(_safe_stdout)
_console_handler.setFormatter(_TIDY_FMT)
_console_handler.setLevel(logging.INFO)
_root_logger = logging.getLogger()
_root_logger.setLevel(logging.INFO)
_root_logger.addHandler(_console_handler)

logger = logging.getLogger(__name__)


def _attach_staging_log_handler(staging_path: Path) -> None:
    """Add FileHandler to write training log to staging. Only staged data; committed at iteration end."""
    global _log_file_handler
    _detach_staging_log_handler()
    # Use pathlib throughout; resolve to absolute for Windows compatibility
    staging_dir = Path(staging_path).resolve()
    log_path = staging_dir / "training.log"
    # Ensure parent is a directory (not a file) before opening the log file
    staging_dir.mkdir(parents=True, exist_ok=True)
    if staging_dir.exists() and not staging_dir.is_dir():
        raise RuntimeError(f"Staging path exists but is not a directory: {staging_dir}")
    _log_file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
    _log_file_handler.setFormatter(_TIDY_FMT)
    _log_file_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(_log_file_handler)


def _detach_staging_log_handler() -> None:
    """Remove and close staging log FileHandler. Call before commit so file is flushed."""
    global _log_file_handler
    if _log_file_handler is not None:
        try:
            logging.getLogger().removeHandler(_log_file_handler)
            _log_file_handler.close()
        except Exception:
            pass
        _log_file_handler = None


@contextlib.contextmanager
def _staging_log_context(staging_path: Path):
    """Ensure staging log handler is attached during iteration and always detached on exit (incl. exceptions)."""
    _attach_staging_log_handler(staging_path)
    try:
        yield
    finally:
        _detach_staging_log_handler()


def _append_staged_log_to_permanent(committed_path: Path, permanent_log_path: Path) -> None:
    """Append committed iteration's training.log to the permanent log. Only runs after successful commit."""
    src = committed_path / "training.log"
    if not src.exists():
        return
    try:
        permanent_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(src, "r", encoding="utf-8") as f:
            content = f.read()
        with open(permanent_log_path, "a", encoding="utf-8") as f:
            f.write(content)
            if content and not content.endswith("\n"):
                f.write("\n")
    except OSError as e:
        logger.warning("Failed to append staged training log to %s: %s", permanent_log_path, e)


def _shallow_diff(old_cfg: dict, new_cfg: dict, keys: list) -> dict:
    """Return shallow diff of selected top-level keys (for config_mismatch provenance)."""
    diff = {}
    for k in keys:
        if k not in old_cfg and k not in new_cfg:
            continue
        o = old_cfg.get(k)
        n = new_cfg.get(k)
        if o != n:
            diff[k] = {"old": o, "new": n}
    return diff


def _check_config_mismatch(
    run_root: Path,
    run_state_path: Path,
    current_config: dict,
    current_config_hash: str,
    allow_mismatch: bool,
    cli_args: list,
) -> None:
    """
    If run_state exists and config_hash differs: warn, write config_mismatch.json, abort unless allowed.
    """
    if not run_state_path.exists():
        return
    try:
        with open(run_state_path, "r", encoding="utf-8-sig") as f:
            state = json.load(f)
    except Exception as e:
        logger.warning("Could not read run_state for config check: %s", e)
        return
    old_hash = state.get("config_hash")
    if old_hash is None:
        return
    if str(old_hash).strip() == str(current_config_hash).strip():
        return

    # Mismatch: write provenance, warn, possibly abort
    mismatch_path = run_root / "config_mismatch.json"
    schedule_keys = [
        "iteration",
        "selfplay",
        "training",
        "replay_buffer",
        "evaluation",
        "paths",
    ]
    try:
        old_cfg = {}
        logs_dir = current_config.get("paths", {}).get("logs_dir", "logs")
        config_snap = Path(logs_dir) / "config_snapshot.yaml"
        if config_snap.exists():
            with open(config_snap, "r") as f:
                old_cfg = yaml.safe_load(f) or {}
        diff = _shallow_diff(old_cfg, current_config, schedule_keys)
    except Exception:
        diff = {}

    mismatch_record = {
        "old_hash": str(old_hash),
        "new_hash": str(current_config_hash),
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "run_id": state.get("run_id", "unknown"),
        "run_root": str(run_root),
        "cli_args": list(cli_args),
        "diff_keys": diff,
    }
    try:
        mismatch_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = mismatch_path.with_suffix(mismatch_path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(mismatch_record, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, mismatch_path)
    except Exception as e:
        logger.warning("Could not write config_mismatch.json: %s", e)

    # Loud warning block
    logger.error("")
    logger.error("=" * 70)
    logger.error("  CONFIG MISMATCH - Run fork / silent behavior change prevented")
    logger.error("=" * 70)
    logger.error("  run_id       : %s", state.get("run_id"))
    logger.error("  run_root     : %s", run_root)
    logger.error("  old hash     : %s", old_hash)
    logger.error("  new hash     : %s", current_config_hash)
    logger.error("  mismatch file: %s", mismatch_path)
    logger.error("  To proceed despite mismatch, pass --allow-config-mismatch")
    logger.error("=" * 70)
    logger.error("")

    if not allow_mismatch:
        sys.exit(1)


def _file_hash(path: str, algo: str = "sha256") -> Optional[str]:
    """Compute hash of a file for metadata tracking."""
    try:
        h = hashlib.new(algo)
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    except Exception:
        return None


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _step_schedule_lookup(schedule: list, iteration: int, value_key: str, base_value) -> float:
    """Step schedule: select last entry with entry.iteration <= iteration."""
    if not schedule:
        return base_value
    for entry in sorted(schedule, key=lambda x: x["iteration"], reverse=True):
        if iteration >= entry["iteration"]:
            return float(entry.get(value_key, base_value))
    return base_value


def _get_window_iterations_for_iteration(config: dict, iteration: int) -> int:
    """Replay buffer window size from schedule or base. Always >= 1."""
    rb = config.get("replay_buffer", {}) or {}
    base = int(rb.get("window_iterations", 5))
    sched = config.get("iteration", {}).get("window_iterations_schedule", [])
    return max(1, int(_step_schedule_lookup(sched, iteration, "window_iterations", base)))


def _get_num_games_for_iteration(config: dict, iteration: int) -> int:
    """Number of games for iteration (bootstrap or schedule)."""
    if iteration == 0:
        return int(config.get("selfplay", {}).get("bootstrap", {}).get("games", 200))
    sched = config.get("iteration", {}).get("games_schedule", [])
    base = int(config.get("selfplay", {}).get("games_per_iteration", 400))
    return int(_step_schedule_lookup(sched, iteration, "games", base))


def _apply_q_value_weight_and_cpuct_schedules(config: dict, iteration: int) -> tuple:
    """Apply q_value_weight and cpuct schedules; mutate config. Returns (qvw, cpuct)."""
    sp = config.get("selfplay", {}) or {}
    mcts = sp.get("mcts", {}) or {}
    eval_cfg = config.get("evaluation", {}) or {}
    iter_cfg = config.get("iteration", {}) or {}

    qvw_base = float(sp.get("q_value_weight", 0.0))
    qvw_sched = iter_cfg.get("q_value_weight_schedule", [])
    qvw = float(_step_schedule_lookup(qvw_sched, iteration, "q_value_weight", qvw_base))
    qvw = max(0.0, min(1.0, qvw))
    config["selfplay"]["q_value_weight"] = qvw

    cpuct_base = float(mcts.get("cpuct", 1.5))
    cpuct_sched = iter_cfg.get("cpuct_schedule", [])
    cpuct = float(_step_schedule_lookup(cpuct_sched, iteration, "cpuct", cpuct_base))
    cpuct = max(0.1, min(5.0, cpuct))
    config["selfplay"]["mcts"]["cpuct"] = cpuct

    lock_eval = bool(eval_cfg.get("lock_eval_cpuct_to_selfplay", True))
    if lock_eval and "eval_mcts" in eval_cfg:
        config["evaluation"]["eval_mcts"]["cpuct"] = cpuct

    return qvw, cpuct


def _assert_encoding_config_consistent(config: dict, device: torch.device) -> None:
    """Fail-fast if api_url is enabled (external HTTP API not wired for gold_v2).

    GPU server (queue-based eval_client) is supported for gold_v2_multimodal.
    api_url refers to external HTTP API which is not implemented.
    """
    sp = config.get("selfplay", {}) or {}
    api_url = sp.get("api_url")
    api_enabled = api_url is not None and str(api_url).strip() not in ("", "null", "none")
    if api_enabled:
        enc = str((config.get("data", {}) or {}).get("encoding_version", "") or "").strip().lower()
        if enc == "gold_v2_multimodal":
            raise RuntimeError(
                "gold_v2_multimodal with api_url (external HTTP API) not implemented; "
                f"config has api_url={api_url!r}. Use api_url=null for queue-based GPU server."
            )


class AlphaZeroTrainer:
    """Main training coordinator.

    Checkpoint semantics:
      - best_model.pt: gate-promoted model, used for self-play generation.
        NEVER deleted by cleanup. Updated only when a new model passes the gate.
      - latest_model.pt: always the most recently trained model. Used for
        crash recovery. Updated every iteration regardless of gate outcome.
      - iteration_XXX.pt: per-iteration snapshots, subject to rotation.
    """

    def __init__(
        self,
        config_path: str,
        cli_run_dir: Optional[str] = None,
        cli_run_id: Optional[str] = None,
        allow_config_mismatch: bool = False,
        cli_args: Optional[list] = None,
        cli_iterations: Optional[int] = None,
    ):
        self.config_path = str(config_path)
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        if cli_iterations is not None:
            self.config.setdefault("iteration", {})["max_iterations"] = cli_iterations
            logger.info("CLI override: max_iterations=%d", cli_iterations)

        logger.info("Loaded config from %s", config_path)

        # Apply log level from config
        log_cfg = self.config.get("logging", {}) or {}
        log_level = str(log_cfg.get("log_level", "INFO")).upper()
        logging.getLogger().setLevel(getattr(logging, log_level, logging.INFO))

        # Compute config hash for metadata tracking
        self._config_hash = hashlib.sha256(
            json.dumps(self.config, sort_keys=True).encode()
        ).hexdigest()[:12]

        if self.config["hardware"]["device"] == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
            logger.info(f"Using GPU: {gpu_name} ({vram_mb:.0f} MB VRAM)")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU")

        # Transactional run layout: runs/<run_id>/staging|committed
        # Deterministic run_dir/run_id = same command resumes seamlessly
        self.run_root = get_run_root(self.config, cli_run_id=cli_run_id, cli_run_dir=cli_run_dir)
        self.run_id = (
            get_run_id(self.config, cli_run_id=cli_run_id)
            if not cli_run_dir
            else Path(cli_run_dir).name
        )
        acquire_run_lock(self.run_root)  # creates run_root; staging/committed created in iteration loop
        cleanup_stale_tmp_files(self.run_root)
        logger.info("Run root: %s (run_id=%s)", self.run_root, self.run_id)

        # Config mismatch guard: abort if resuming with different config (prevents silent behavior change)
        self.run_state_path = self.run_root / "run_state.json"
        _check_config_mismatch(
            self.run_root,
            self.run_state_path,
            self.config,
            self._config_hash,
            allow_mismatch=allow_config_mismatch,
            cli_args=cli_args or [],
        )

        self._snapshot_run_metadata()

        # Fail-fast: api_url (external API) not supported for gold_v2
        _assert_encoding_config_consistent(self.config, self.device)

        self.selfplay_generator = create_selfplay_generator(self.config)
        self.evaluator = Evaluator(self.config, self.device)
        self.elo_tracker = EloTracker(
            initial_rating=self.config["evaluation"]["elo"]["initial_rating"],
            k_factor=self.config["evaluation"]["elo"]["k_factor"],
        )

        # Replay buffer: state lives in run_root; only committed paths are persisted
        self.replay_buffer = ReplayBuffer(
            self.config,
            state_path=self.run_root / "replay_state.json",
        )

        # League system: PFSP, payoff matrix, anchor-based gating
        self.league_enabled = bool(self.config.get("league", {}).get("enabled", False))
        self.league: Optional[LeagueManager] = None
        if self.league_enabled:
            self.league = LeagueManager(self.config, self.run_root)
            logger.info("League system enabled (PFSP + robustness gating)")

        # Permanent TensorBoard log dir — events are only copied here at commit time
        tb_cfg = (self.config.get("logging", {}) or {}).get("tensorboard", {}) or {}
        self.tb_log_dir = Path(
            tb_cfg.get("log_dir") or str(Path(self.config["paths"]["logs_dir"]) / "tensorboard")
        )
        self.tb_log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = None  # Created per-iteration in staging; committed at iteration end

        self.current_iteration = 0
        self.global_step = 0  # Monotonic TensorBoard step counter across all iterations
        self.best_model_path: Optional[str] = None  # Gate-promoted model for selfplay
        self.best_model_iteration: Optional[int] = None  # Which iteration produced the best
        self.latest_model_path: Optional[str] = None  # Always latest trained
        self.latest_checkpoint: Optional[str] = None
        self.iteration_history = []
        self.consecutive_rejections = 0
        self.last_committed_iteration: int = -1  # -1 = none committed yet
        self.elo_state_path = self.run_root / "elo_state.json"
        # Health-line debug info: set during _try_auto_resume or at startup
        self._health_fs_last_committed: Optional[int] = None
        self._health_run_state_last_committed: Optional[int] = None
        self._health_reconcile_advanced: bool = False
        self.metadata_path = Path(self.config["paths"]["logs_dir"]) / "metadata.jsonl"
        self.env_info_path = Path(self.config["paths"]["logs_dir"]) / "environment.json"
        self.config_snapshot_path = Path(self.config["paths"]["logs_dir"]) / "config_snapshot.yaml"
        self.ckpt_dir = Path(self.config["paths"]["checkpoints_dir"])

    def _collect_applied_settings(self, iteration: int) -> dict:
        """Collect realized schedule values for provenance (after _apply_iteration_schedules).
        Uses selfplay generator's schedule-applied config for temp/sims/alpha/noise (they are
        applied there, not in main's config).
        """
        # Selfplay schedules (temp, sims, dirichlet, noise) are applied in selfplay_generator;
        # q_value_weight and cpuct are applied in main's _apply_iteration_schedules (shared config).
        sp_config = self.selfplay_generator._apply_iteration_schedules(iteration, quiet=True)
        sp = sp_config.get("selfplay", {}) or {}
        mcts = sp.get("mcts", {}) or {}

        train = self.config.get("training", {}) or {}
        rb = self.config.get("replay_buffer", {}) or {}
        league = self.config.get("league", {}) or {}
        amp_dtype = "bfloat16" if (train.get("amp_dtype") or "").lower() == "bfloat16" else "float16"

        recency_window = 0.0
        if league.get("enabled"):
            recency_window = float(league.get("recency_newest_window", 0.15))

        return {
            "selfplay": {
                "games": _get_num_games_for_iteration(self.config, iteration),
                "temperature": float(mcts.get("temperature", 1.0)),
                "policy_target_mode": str(sp.get("policy_target_mode", "visits")).lower()
                    or "visits",
                "dirichlet_alpha": float(mcts.get("root_dirichlet_alpha", 0.0)),
                "noise_weight": float(mcts.get("root_noise_weight", 0.0)),
                "cpuct": float(mcts.get("cpuct", 1.5)),
                "simulations": int(mcts.get("simulations", 0)),
                "q_value_weight": float(sp.get("q_value_weight", 0.0)),
                "score_utility_weight": float(mcts.get("score_utility_weight", 0.02)),
            },
            "training": {
                "lr": float(train.get("learning_rate", 0.0)),
                "q_value_weight": float(sp.get("q_value_weight", 0.0)),
                "batch_size": int(train.get("batch_size", 0)),
                "amp_dtype": amp_dtype,
            },
            "replay": {
                "window_iterations": self.replay_buffer.window_size,
                "max_size": self.replay_buffer.max_size,
                "newest_fraction": float(self.replay_buffer.newest_fraction),
                "recency_window": recency_window,
            },
        }

    def _apply_iteration_schedules(self, iteration: int) -> None:
        """Apply q_value_weight and cpuct schedules (mutates self.config)."""
        _apply_q_value_weight_and_cpuct_schedules(self.config, iteration)

    def _log_health_at_startup(
        self, start_iteration: int, startup_train_base: Optional[str], resume_checkpoint: Optional[str]
    ) -> None:
        """Log health at process start (stub for diagnostics)."""
        logger.debug(
            "Health: start_iter=%d train_base=%s resume=%s",
            start_iteration, startup_train_base, resume_checkpoint,
        )

    def _log_health_at_iteration_start(self, iteration: int, train_base: Optional[str]) -> None:
        """Log health at iteration start (stub for diagnostics)."""
        logger.debug("Health: iter=%d train_base=%s", iteration, train_base)

    def _snapshot_run_metadata(self) -> None:
        logs_dir = Path(self.config["paths"]["logs_dir"])
        logs_dir.mkdir(parents=True, exist_ok=True)
        cfg_snapshot = logs_dir / "config_snapshot.yaml"
        env_snapshot = logs_dir / "environment.json"

        with open(cfg_snapshot, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.config, f, sort_keys=False)

        env = {
            "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
            "python": sys.version,
            "platform": platform.platform(),
            "executable": sys.executable,
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "seed": self.config.get("seed", None),
            "config_path": self.config_path,
        }
        _atomic_write_json(env_snapshot, env)
        logger.info("Wrote config/environment snapshots to logs directory")

    def _save_run_state(self, last_committed_iteration: int) -> None:
        """Persist run state. Call ONLY at commit time (never mid-iteration)."""
        state = {
            "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
            "last_committed_iteration": int(last_committed_iteration),
            "best_iteration": self.best_model_iteration,
            "best_model_path": self.best_model_path,
            "latest_model_path": self.latest_model_path,
            "global_step": int(self.global_step),
            "consecutive_rejections": int(self.consecutive_rejections),
            "config_hash": self._config_hash,
            "run_id": self.run_id,
            "seed": int(self.config.get("seed", 42)),
        }
        _atomic_write_json(self.run_state_path, state)

    def _repair_run_state_from_committed(
        self, state: dict, from_iter: int, to_iter: int
    ) -> None:
        """
        Repair run_state when filesystem has committed iters (from_iter+1..to_iter)
        that run_state didn't record (crash-after-move-before-run_state).
        Replays each commit: copy checkpoints, update replay buffer, advance state.
        """
        for i in range(from_iter + 1, to_iter + 1):
            comm_path = committed_dir(self.run_root, i)
            manifest_path = comm_path / "commit_manifest.json"
            ckpt_path = comm_path / f"iteration_{i:03d}.pt"
            selfplay_path = comm_path / "selfplay.h5"
            if not manifest_path.exists() or not ckpt_path.exists():
                logger.warning("Repair: iter%03d missing manifest or checkpoint, skipping", i)
                continue
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            accepted = manifest.get("accepted", False)
            best_iter = manifest.get("best_model_iteration")
            state["global_step"] = int(manifest.get("global_step", state.get("global_step", 0)))
            state["consecutive_rejections"] = int(
                manifest.get("consecutive_rejections", state.get("consecutive_rejections", 0))
            )
            # latest_model.pt advances every iter
            latest_dest = self._atomic_copy_checkpoint(ckpt_path, "latest_model.pt")
            state["latest_model_path"] = str(latest_dest)
            state["latest_checkpoint"] = str(latest_dest)
            # best_model.pt advances only when manifest says ACCEPT
            save_best = bool((self.config.get("training", {}) or {}).get("save_best", True))
            if accepted and best_iter == i:
                if save_best:
                    best_dest = self._atomic_copy_checkpoint(ckpt_path, "best_model.pt")
                    state["best_model_path"] = str(best_dest)
                else:
                    state["best_model_path"] = str(ckpt_path)
                state["best_iteration"] = i
                if save_best:
                    self._save_best_stamped(ckpt_path, i)
            # Replay consecutive_rejections from manifest (already done above)
            if selfplay_path.exists():
                try:
                    import h5py
                    with h5py.File(selfplay_path, "r") as f:
                        states_key = "spatial_states" if "spatial_states" in f else "states"
                        n_pos = int(f[states_key].shape[0])
                except Exception:
                    n_pos = 0
                # Use scheduled window size for iteration i (eviction uses correct historical value)
                self.replay_buffer.window_size = _get_window_iterations_for_iteration(self.config, i)
                self.replay_buffer.finalize_iteration_for_commit(
                    i, str(selfplay_path), n_pos
                )
            # Copy TensorBoard events for repaired iteration (no-op for old iters without TB data)
            self._commit_tensorboard_events(comm_path)
        state["last_committed_iteration"] = to_iter
        state["timestamp_utc"] = datetime.datetime.utcnow().isoformat() + "Z"
        if "config_hash" not in state:
            state["config_hash"] = self._config_hash
        if "run_id" not in state:
            state["run_id"] = self.run_id
        if "seed" not in state:
            state["seed"] = int(self.config.get("seed", 42))
        _atomic_write_json(self.run_state_path, state)

        # Restore Elo from last repaired manifest
        last_manifest_path = committed_dir(self.run_root, to_iter) / "commit_manifest.json"
        if last_manifest_path.exists():
            with open(last_manifest_path, "r", encoding="utf-8") as f:
                last_manifest = json.load(f)
            if "elo_ratings" in last_manifest:
                elo_state = {
                    "initial_rating": self.elo_tracker.initial_rating,
                    "k_factor": self.elo_tracker.k_factor,
                    "ratings": {k: float(v) for k, v in last_manifest["elo_ratings"].items()},
                }
                _atomic_write_json(self.elo_state_path, elo_state)
                self.elo_tracker.ratings = dict(elo_state["ratings"])

        logger.info("Repaired run_state up to iter%03d (latest, best, replay, Elo)", to_iter)

    def _try_auto_resume(self, start_iteration: int, resume_checkpoint: Optional[str]) -> tuple[int, Optional[str]]:
        if resume_checkpoint is not None:
            return start_iteration, resume_checkpoint

        auto_resume = bool(self.config.get("iteration", {}).get("auto_resume", True))
        if not auto_resume:
            return start_iteration, None
        if start_iteration != 0:
            return start_iteration, None
        if not self.run_state_path.exists():
            return start_iteration, None

        try:
            with open(self.run_state_path, "r", encoding="utf-8-sig") as f:
                state = json.load(f)
            # New schema: last_committed_iteration. Legacy: next_iteration -> last = next - 1
            last_comm = state.get("last_committed_iteration")
            if last_comm is not None:
                last_comm = int(last_comm)
            else:
                next_it = int(state.get("next_iteration", 0))
                last_comm = max(-1, next_it - 1)

            # Reconcile with filesystem: if crash occurred after move but before run_state write,
            # committed/ may have iter_N that run_state doesn't know about
            old_last_comm = last_comm
            last_comm, needs_repair = reconcile_run_state(self.run_root, last_comm)
            self._health_fs_last_committed = last_comm
            self._health_run_state_last_committed = old_last_comm
            self._health_reconcile_advanced = needs_repair
            # Set window_size before restore so eviction uses correct schedule
            self.replay_buffer.window_size = _get_window_iterations_for_iteration(
                self.config, last_comm + 1
            )
            if needs_repair:
                self.replay_buffer.restore_state()
                self._repair_run_state_from_committed(state, old_last_comm, last_comm)
            self.last_committed_iteration = last_comm

            self.best_model_path = state.get("best_model_path")
            bi = state.get("best_iteration")
            self.best_model_iteration = bi if bi is not None else state.get("best_model_iteration")
            self.latest_model_path = state.get("latest_model_path")
            self.latest_checkpoint = state.get("latest_checkpoint") or self.latest_model_path
            self.global_step = int(state.get("global_step", 0))
            self.consecutive_rejections = int(state.get("consecutive_rejections", 0))

            # Staging already discarded at train() startup (before _try_auto_resume)

            # window_size already set above before needs_repair block; ensure correct for restore at train()
            self.replay_buffer.window_size = _get_window_iterations_for_iteration(
                self.config, last_comm + 1
            )

            # Restore Elo tracker ratings
            self.elo_tracker.load_state(self.elo_state_path)

            # Restore league state
            if self.league is not None:
                restored = self.league.load_state()
                if not restored and self.best_model_path and self.best_model_iteration is not None:
                    # League state file doesn't exist but we have a best model:
                    # seed the pool so league can start working
                    mid = self.league.model_id(self.best_model_iteration)
                    self.league.promote(mid, self.best_model_path)

            next_iter = last_comm + 1
            checkpoint = self.best_model_path or self.latest_model_path or self.latest_checkpoint
            if checkpoint and Path(checkpoint).exists():
                if self.global_step == 0:
                    try:
                        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
                        self.global_step = int(ckpt.get("global_step", 0))
                    except Exception:
                        pass

                best_label = (
                    f"iter{self.best_model_iteration:03d}"
                    if self.best_model_iteration is not None else "N/A"
                )
                _print_auto_resume(
                    last_comm=last_comm,
                    next_iter=next_iter,
                    best_label=best_label,
                    best_model_path=str(self.best_model_path),
                    global_step=self.global_step,
                    rejections=self.consecutive_rejections,
                    elo_count=len(self.elo_tracker.ratings),
                )
                logger.info(
                    "AUTO-RESUME: last=iter%03d next=iter%03d best=%s step=%d rejects=%d",
                    last_comm, next_iter, best_label, self.global_step, self.consecutive_rejections,
                )
                return next_iter, checkpoint
        except Exception as e:
            logger.warning("Failed to auto-resume from %s: %s", self.run_state_path, e)
        return start_iteration, None

    def train(
        self,
        start_iteration: int = 0,
        resume_checkpoint: Optional[str] = None,
    ):
        """Run complete AlphaZero training pipeline.

        Checkpoint flow:
          - Self-play always uses best_model_path (the gate-promoted model)
          - Training warm-starts from best_model_path (model weights only)
          - latest_model.pt is ALWAYS updated after training (crash recovery)
          - best_model.pt is updated ONLY when the gate accepts the new model
          - Rejected models do NOT become the generator
        """
        logger.info("Starting AlphaZero training pipeline")
        # Always discard partial staging on startup so each iteration restarts from self-play.
        # (Previously this ran only when auto-resuming; without run_state.json it was skipped,
        # leaving staging/iter_N from interrupted runs and causing self-play reuse.)
        last_comm = max_committed_iteration(self.run_root)
        cleanup_staging(self.run_root, last_comm, config=self.config)
        start_iteration, resume_checkpoint = self._try_auto_resume(start_iteration, resume_checkpoint)

        # Attach staging log handler for first iteration (captures all logs until commit)
        first_staging = staging_dir(self.run_root, start_iteration)
        first_staging.mkdir(parents=True, exist_ok=True)
        _attach_staging_log_handler(first_staging)

        # Health-line: ensure fs_last is set even for fresh runs (no auto-resume)
        if self._health_fs_last_committed is None:
            self._health_fs_last_committed = max_committed_iteration(self.run_root)

        # Restore replay buffer sliding window on resume
        if start_iteration > 0:
            self.replay_buffer.restore_state()

        # Health line at process start
        startup_train_base = (
            resume_checkpoint
            if resume_checkpoint
            else (self.best_model_path if start_iteration > 0 else None)
        )
        self._log_health_at_startup(start_iteration, startup_train_base, resume_checkpoint)

        if resume_checkpoint:
            logger.info("Resuming from checkpoint: %s", resume_checkpoint)
            self.latest_checkpoint = resume_checkpoint
            if self.best_model_path is None:
                self.best_model_path = resume_checkpoint
                self.best_model_iteration = start_iteration - 1 if start_iteration > 0 else None

        total_iterations = self.config["iteration"]["max_iterations"]

        stop_file = self.run_root / "STOP_AFTER_ITERATION"
        for iteration in range(start_iteration, total_iterations):
            if stop_file.exists():
                _request_shutdown()
                try:
                    stop_file.unlink()
                except OSError:
                    pass
                logger.info("[SHUTDOWN] Stop requested by STOP_AFTER_ITERATION file; will exit after commit.")
            if _shutdown_requested:
                logger.info(
                    "[SHUTDOWN] Exiting cleanly after commit. Last committed: iter%03d.",
                    self.last_committed_iteration,
                )
                break
            iter_start = time.time()
            self.current_iteration = iteration
            # Apply window_iterations_schedule at breakpoints (before self-play/train)
            self.replay_buffer.window_size = _get_window_iterations_for_iteration(self.config, iteration)

            staging_path = staging_dir(self.run_root, iteration)
            staging_path.mkdir(parents=True, exist_ok=True)

            with _staging_log_context(staging_path):
                # Apply iteration schedules (q_value_weight, cpuct) and collect for provenance
                self._apply_iteration_schedules(iteration)
                applied_settings = self._collect_applied_settings(iteration)

                # Transactional TensorBoard: stage events here, commit to permanent dir at iteration end
                tb_enabled = bool((self.config.get("logging", {}) or {}).get("tensorboard", {}).get("enabled", True))
                if tb_enabled:
                    self.writer = SummaryWriter(log_dir=str(staging_path / "tensorboard"))
                else:
                    self.writer = _NoOpSummaryWriter()

                best_label = f"iter{self.best_model_iteration:03d}" if self.best_model_iteration is not None else "None"
                _print_iter_header(
                    iteration=iteration,
                    total_iterations=total_iterations,
                    best_label=best_label,
                    buf_positions=self.replay_buffer.total_positions,
                    rejections=self.consecutive_rejections,
                    applied_settings=applied_settings,
                )
                logger.info(
                    "ITER %03d/%d  best=%s  rejects=%d  buf=%d pos",
                    iteration, total_iterations, best_label,
                    self.consecutive_rejections, self.replay_buffer.total_positions,
                )

                if iteration == start_iteration and resume_checkpoint:
                    selfplay_model = resume_checkpoint
                    train_base = resume_checkpoint
                else:
                    selfplay_model = self.best_model_path
                    train_base = self.best_model_path

                self._log_health_at_iteration_start(iteration, train_base)

                sp_label = f"iter{self.best_model_iteration:03d}" if (selfplay_model and self.best_model_iteration is not None) else "random"
                _print_section(f"\n[1/3] SELF-PLAY  (generator: {sp_label})")

                data_path, selfplay_stats = self._generate_selfplay_data(
                    iteration, selfplay_model, output_dir=staging_path
                )
                logger.info(
                    "[1/3] Self-play complete: %d games, %d positions, %.1f games/min",
                    selfplay_stats.get("num_games", 0),
                    selfplay_stats.get("num_positions", 0),
                    selfplay_stats.get("games_per_minute", 0),
                )

                train_label = f"iter{self.best_model_iteration:03d}" if (train_base and self.best_model_iteration is not None) else "scratch"
                _print_section(f"\n[2/3] TRAINING  (warm-start: {train_label})")
                checkpoint_path, train_metrics, global_step = self._train_network(
                    iteration, data_path, train_base, staging_dir=staging_path
                )
                self.global_step = global_step
                logger.info(
                    "[2/3] Training complete: loss=%.4f  pol_acc=%.1f%%  val_mse=%.4f  grad_norm=%.3f",
                    train_metrics.get("total_loss", 0),
                    train_metrics.get("policy_accuracy", 0) * 100,
                    train_metrics.get("value_mse", 0),
                    train_metrics.get("grad_norm", 0),
                )

                self.latest_checkpoint = checkpoint_path
                if self.best_model_path is None:
                    self.best_model_path = checkpoint_path
                    self.best_model_iteration = iteration
                    # Bootstrap: seed the league pool with the first model
                    if self.league is not None:
                        mid = self.league.model_id(iteration)
                        self.league.promote(mid, checkpoint_path)

                # Only print EVALUATION section header when actual evaluation games will run
                _eval_cfg_now = self.config.get("evaluation", {}) or {}
                _eval_ngm = int(_eval_cfg_now.get("games_per_eval") or _eval_cfg_now.get("games_vs_pure_mcts", 0))
                _eval_ngb = int(_eval_cfg_now.get("games_vs_best") or _eval_ngm)
                _sprt_on  = bool((_eval_cfg_now.get("sprt", {}) or {}).get("enabled", False))
                _micro_on = bool((_eval_cfg_now.get("micro_gate", {}) or {}).get("enabled", True))
                _eval_active = _eval_ngm > 0 or (iteration > 0 and (_eval_ngb > 0 or _sprt_on or _micro_on))
                if _eval_active:
                    _print_section("\n[3/3] EVALUATION")
                eval_results = self._evaluate_model(iteration, checkpoint_path)

                # ── TensorBoard: iteration-level metrics ──────────────────────────────
                # Eval metrics
                if "elo_rating" in eval_results:
                    self.writer.add_scalar("eval/elo_rating", eval_results["elo_rating"], iteration)
                if "vs_pure_mcts" in eval_results:
                    vm = eval_results["vs_pure_mcts"]
                    self.writer.add_scalar("eval/win_rate_vs_mcts", vm["win_rate"], iteration)
                    self.writer.add_scalar("eval/score_margin_vs_mcts", vm.get("avg_model_score_margin", 0), iteration)
                    self.writer.add_scalar("eval/game_length_vs_mcts", vm.get("avg_game_length", 0), iteration)
                    # P0/P1 position-bias signal
                    _vm_results = vm.get("results", [])
                    if _vm_results:
                        _p0 = [r for r in _vm_results if r.get("model_plays_first", False)]
                        _p1 = [r for r in _vm_results if not r.get("model_plays_first", False)]
                        if _p0:
                            self.writer.add_scalar("eval/win_rate_as_p0_vs_mcts", sum(1 for r in _p0 if r["model_won"]) / len(_p0), iteration)
                        if _p1:
                            self.writer.add_scalar("eval/win_rate_as_p1_vs_mcts", sum(1 for r in _p1 if r["model_won"]) / len(_p1), iteration)
                if "vs_previous_best" in eval_results:
                    vb = eval_results["vs_previous_best"]
                    self.writer.add_scalar("eval/win_rate_vs_best", vb["win_rate"], iteration)
                    self.writer.add_scalar("eval/score_margin_vs_best", vb.get("avg_model_score_margin", 0), iteration)
                    # P0/P1 position-bias signal vs best model
                    _vb_results = vb.get("results", [])
                    if _vb_results:
                        _p0b = [r for r in _vb_results if r.get("model_plays_first", False)]
                        _p1b = [r for r in _vb_results if not r.get("model_plays_first", False)]
                        if _p0b:
                            self.writer.add_scalar("eval/win_rate_as_p0_vs_best", sum(1 for r in _p0b if r["model_won"]) / len(_p0b), iteration)
                        if _p1b:
                            self.writer.add_scalar("eval/win_rate_as_p1_vs_best", sum(1 for r in _p1b if r["model_won"]) / len(_p1b), iteration)

                # Self-play stats
                self.writer.add_scalar("selfplay/games_per_min", selfplay_stats.get("games_per_minute", 0), iteration)
                self.writer.add_scalar("selfplay/num_positions", selfplay_stats.get("num_positions", 0), iteration)

                # Replay buffer health
                self.writer.add_scalar("buffer/total_positions", self.replay_buffer.total_positions, iteration)
                self.writer.add_scalar("buffer/num_iterations", self.replay_buffer.num_iterations, iteration)

                # Training diagnostics (epoch averages)
                for k, v in train_metrics.items():
                    self.writer.add_scalar(f"iter/{k}", v, iteration)

                # --- League-based evaluation & gating ---
                if self.league_enabled and self.league is not None and iteration > 0:
                    accepted = self._league_evaluate_and_gate(
                        iteration, checkpoint_path, eval_results, staging_path, global_step
                    )
                else:
                    # Original gating logic (non-league or iteration 0)
                    accepted = self._original_gate(iteration, checkpoint_path, eval_results)

                iter_time = time.time() - iter_start

                summary = {
                    "iteration": iteration,
                    "selfplay_stats": selfplay_stats,
                    "train_metrics": train_metrics,
                    "eval_results": eval_results,
                    "accepted": accepted,
                    "best_model": self.best_model_path,
                    "latest_model": self.latest_model_path,
                    "latest_checkpoint": self.latest_checkpoint,
                    "iteration_time_s": iter_time,
                    "replay_buffer_positions": self.replay_buffer.total_positions,
                    "replay_buffer_iterations": self.replay_buffer.num_iterations,
                    "applied_settings": applied_settings,
                }
                # Write summary to staging (included in commit)
                self._save_iteration_summary(iteration, summary, output_dir=staging_path)
                mem_summary = dict(summary)
                for eval_key in ("vs_previous_best", "vs_pure_mcts"):
                    if eval_key in mem_summary.get("eval_results", {}):
                        mem_summary["eval_results"] = dict(mem_summary["eval_results"])
                        mem_summary["eval_results"][eval_key] = {
                            k: v for k, v in mem_summary["eval_results"][eval_key].items()
                            if k != "results"
                        }
                self.iteration_history.append(mem_summary)

                # Close staging TensorBoard writer before commit (flush + release file handles)
                if self.writer is not None:
                    self.writer.close()
                    self.writer = None

                # ATOMIC COMMIT: move staging -> committed, update all state
                self._commit_iteration(
                    iteration,
                    checkpoint_path,
                    selfplay_stats,
                    accepted,
                    summary,
                    train_metrics,
                    eval_results,
                    iter_time,
                    global_step,
                )

        if _shutdown_requested:
            msg = f"  STOPPED GRACEFULLY  (last committed: iter{self.last_committed_iteration:03d})"
            clr = _CLR_FAIL
        else:
            msg = "  TRAINING COMPLETE"
            clr = _CLR_OK
        print(f"\n{_CLR_BAR}{_HBAR}{_CLR_RST}\n{clr}{msg}{_CLR_RST}\n{_CLR_BAR}{_HBAR}{_CLR_RST}\n", flush=True)
        logger.info(msg.strip())

    def _atomic_copy_checkpoint(self, src: Path, dest_name: str) -> Path:
        """Copy checkpoint atomically to checkpoints dir."""
        dest = self.ckpt_dir / dest_name
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        shutil.copy2(src, tmp)
        os.replace(tmp, dest)
        return dest

    def _save_best_stamped(self, src: Path, iteration: int) -> None:
        """Save iteration-stamped best model copy (traceability)."""
        stamped = self.ckpt_dir / f"best_model_iter{iteration:03d}.pt"
        try:
            shutil.copy2(src, stamped)
            logger.info("Saved best model: %s", stamped.name)
        except Exception as e:
            logger.warning("Failed to save stamped best model: %s", e)

    def _save_best_model(self, checkpoint_path: str, iteration: Optional[int] = None):
        """Copy checkpoint to best_model.pt (gate-promoted, used for selfplay generation).

        Also saves a timestamped copy best_model_iterXXX.pt for traceability.
        The canonical best_model.pt is NEVER deleted by checkpoint cleanup.
        """
        ckpt_dir = Path(self.config["paths"]["checkpoints_dir"])

        # Save canonical best_model.pt (always overwritten)
        best_path = ckpt_dir / "best_model.pt"
        tmp_best = best_path.with_suffix(".tmp")
        shutil.copy2(checkpoint_path, tmp_best)
        os.replace(tmp_best, best_path)
        self.best_model_path = str(best_path)

        # Save iteration-stamped copy for traceability (never cleaned up)
        it = iteration if iteration is not None else self.current_iteration
        self.best_model_iteration = it
        stamped_name = f"best_model_iter{it:03d}.pt"
        stamped_path = ckpt_dir / stamped_name
        try:
            shutil.copy2(checkpoint_path, stamped_path)
        except Exception as e:
            logger.warning(f"Failed to save stamped best model: {e}")

        logger.info(f"Saved best model: {stamped_name}")

    def _append_metadata(self, iteration: int, selfplay_stats: dict,
                         train_metrics: dict, eval_results: dict,
                         accepted: bool, iter_time: float, global_step: int):
        """Append one JSONL line per iteration to metadata.jsonl.

        Includes config hash and file hashes for full reproducibility tracking.
        """
        def _safe(obj):
            if isinstance(obj, dict):
                return {k: _safe(v) for k, v in obj.items() if k != "results"}
            if isinstance(obj, (list, tuple)):
                return [_safe(v) for v in obj[:5]]  # truncate large lists
            if hasattr(obj, "item"):
                return obj.item()
            return obj

        entry = {
            "iteration": iteration,
            "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
            "config_hash": self._config_hash,
            "config_path": self.config_path,
            "best_model_hash": _file_hash(self.best_model_path) if self.best_model_path else None,
            "accepted": accepted,
            "global_step": global_step,
            "iter_time_s": round(iter_time, 1),
            "train": _safe(train_metrics),
            "selfplay": _safe(selfplay_stats),
            "eval_vs_best_wr": None,
            "eval_vs_best_margin": None,
            "eval_vs_mcts_wr": None,
            "best_model": self.best_model_path,
            "replay_positions": self.replay_buffer.total_positions,
            "consecutive_rejections": self.consecutive_rejections,
        }
        if "vs_previous_best" in eval_results:
            entry["eval_vs_best_wr"] = eval_results["vs_previous_best"].get("win_rate")
            entry["eval_vs_best_margin"] = eval_results["vs_previous_best"].get("avg_model_score_margin")
            sprt = eval_results["vs_previous_best"].get("sprt")
            if sprt:
                entry["sprt"] = {
                    "accept": sprt.get("accept"),
                    "reject": sprt.get("reject"),
                    "llr": sprt.get("llr"),
                    "games": sprt.get("games_played"),
                }
        if "vs_pure_mcts" in eval_results:
            entry["eval_vs_mcts_wr"] = eval_results["vs_pure_mcts"].get("win_rate")
        if "elo_rating" in eval_results:
            entry["elo"] = eval_results["elo_rating"]

        try:
            with open(self.metadata_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write metadata: {e}")

    def _generate_selfplay_data(
        self, iteration: int, network_path: Optional[str], output_dir: Optional[Path] = None
    ) -> tuple:
        # Always use standard self-play (current best model vs itself).
        # League diversity is handled in evaluation/gating only — not in data generation.
        return self.selfplay_generator.generate(iteration, network_path, output_dir=output_dir)

    def _train_network(
        self,
        iteration: int,
        data_path: str,
        previous_checkpoint: Optional[str],
        staging_dir: Optional[Path] = None,
    ) -> tuple:
        result = train_iteration(
            iteration,
            data_path,
            self.config,
            self.device,
            previous_checkpoint,
            replay_buffer=self.replay_buffer,
            writer=self.writer,
            global_step_offset=self.global_step,
            iteration_output_dir=staging_dir,
            merged_output_path=str(staging_dir / "merged_training.h5") if staging_dir else None,
        )
        # Release training dataset memory (PatchworkDataset numpy arrays) back to the OS
        # before self-play starts. With 2-3 GB datasets, Python's C heap retains freed pages
        # unless explicitly trimmed, leaving only ~700 MB available for the GPU server subprocess.
        gc.collect()
        if sys.platform == "win32":
            # SetProcessWorkingSetSize(-1, -1) trims the process working set on Windows,
            # returning freed heap pages to the OS immediately (equivalent to malloc_trim on Linux).
            ctypes.windll.kernel32.SetProcessWorkingSetSize(
                ctypes.windll.kernel32.GetCurrentProcess(), -1, -1
            )
        return result

    def _evaluate_model(self, iteration: int, model_path: str) -> dict:
        eval_results = {}

        eval_cfg = self.config.get("evaluation", {}) or {}
        num_games_mcts = int(eval_cfg.get("games_per_eval") or eval_cfg.get("games_vs_pure_mcts", 0))
        num_games_best = int(eval_cfg.get("games_vs_best") or num_games_mcts)

        # Optional: skip redundant pure-MCTS evaluation after early iterations
        # (useful once the model consistently crushes pure MCTS).
        skip_after = self.config["evaluation"].get("skip_pure_mcts_after_iter", None)
        run_pure_mcts = True
        if skip_after is not None:
            try:
                run_pure_mcts = int(iteration) <= int(skip_after)
            except Exception:
                run_pure_mcts = True

        if run_pure_mcts and num_games_mcts > 0:
            logger.info(f"[EVAL] Playing {num_games_mcts} games vs pure MCTS...")
            pure_mcts_results = self.evaluator.evaluate_vs_baseline(
                model_path, baseline_type="pure_mcts", num_games=num_games_mcts
            )
            eval_results["vs_pure_mcts"] = pure_mcts_results
            logger.info(
                f"[EVAL] vs pure MCTS: WR={pure_mcts_results['win_rate']:.1%}  "
                f"avg_margin={pure_mcts_results.get('avg_model_score_margin', 0):.1f}pts  "
                f"avg_len={pure_mcts_results['avg_game_length']:.0f} moves"
            )
        elif not run_pure_mcts and skip_after is not None:
            logger.debug(
                f"[EVAL] Skipping pure MCTS (iter {iteration} > skip_after={skip_after})"
            )

        if self.config["evaluation"]["elo"]["enabled"] and "vs_pure_mcts" in eval_results:
            player_id = f"iteration_{iteration:03d}"
            for result in eval_results["vs_pure_mcts"]["results"]:
                outcome = 1.0 if result["model_won"] is True else 0.0
                self.elo_tracker.update(player_id, "pure_mcts", outcome)

            eval_results["elo_rating"] = self.elo_tracker.get_rating(player_id)
            logger.info(f"[EVAL] Elo: {eval_results['elo_rating']:.0f}")

        if iteration > 0 and self.best_model_path:
            # Use SPRT if configured, otherwise fall back to micro-gate / fixed-game eval
            sprt_cfg = self.config.get("evaluation", {}).get("sprt", {}) or {}
            micro_cfg = self.config.get("evaluation", {}).get("micro_gate", {}) or {}

            if bool(sprt_cfg.get("enabled", False)):
                prev_best_results = self.evaluator.evaluate_sprt(
                    model_path, self.best_model_path, sprt_cfg
                )
            elif bool(micro_cfg.get("enabled", True)):
                prev_best_results = self._run_micro_gate(model_path, self.best_model_path, micro_cfg)
            else:
                if num_games_best > 0:
                    logger.info(f"[EVAL] Playing {num_games_best} fixed games vs previous best...")
                prev_best_results = self.evaluator.evaluate_vs_baseline(
                    model_path,
                    baseline_type="previous_best",
                    baseline_path=self.best_model_path,
                    num_games=num_games_best,
                )
            eval_results["vs_previous_best"] = prev_best_results

            wr = prev_best_results.get("win_rate", 0)
            margin = prev_best_results.get("avg_model_score_margin", 0)
            sprt_info = prev_best_results.get("sprt", {})
            sprt_str = ""
            if sprt_info:
                sprt_str = (
                    f"  SPRT={'ACCEPT' if sprt_info.get('accept') else 'REJECT' if sprt_info.get('reject') else 'INCONCLUSIVE'}  "
                    f"LLR={sprt_info.get('llr', 0):.3f}"
                )
            total_games = prev_best_results.get("total_games", 0)
            if total_games > 0:
                logger.info(
                    f"[EVAL] vs previous best: WR={wr:.1%}  "
                    f"avg_margin={margin:.1f}pts  "
                    f"games={total_games}"
                    f"{sprt_str}"
                )

        return eval_results

    @staticmethod
    def _merge_eval_stats(accumulated: dict, new_stats: dict) -> dict:
        merged = {
            "model_wins": int(accumulated.get("model_wins", 0)) + int(new_stats.get("model_wins", 0)),
            "baseline_wins": int(accumulated.get("baseline_wins", 0)) + int(new_stats.get("baseline_wins", 0)),
            "ties": int(accumulated.get("ties", 0)) + int(new_stats.get("ties", 0)),
            "results": list(accumulated.get("results", [])) + list(new_stats.get("results", [])),
        }
        total_games = merged["model_wins"] + merged["baseline_wins"] + merged["ties"]
        merged["total_games"] = int(total_games)
        merged["win_rate"] = float(merged["model_wins"] / total_games) if total_games > 0 else 0.0
        # Recompute score margin stats
        margins = [r.get("model_score_margin", 0) for r in merged["results"]]
        merged["avg_model_score_margin"] = float(np.mean(margins)) if margins else 0.0
        return merged

    def _run_micro_gate(self, model_path: str, baseline_path: str, micro_cfg: dict) -> dict:
        """Legacy micro-gate (adaptive fixed-game evaluation).
        
        Falls back to this when SPRT is not enabled.
        """
        start_games = int(micro_cfg.get("start_games", 20))
        max_games = int(micro_cfg.get("max_games", 80))
        step_games = int(micro_cfg.get("step_games", 20))
        threshold = float(micro_cfg.get("anti_regression_threshold", 0.45))

        logger.info(
            "[EVAL] Micro-gate vs previous best "
            "(start=%s max=%s step=%s threshold=%.3f)",
            start_games,
            max_games,
            step_games,
            threshold,
        )

        running = {"model_wins": 0, "baseline_wins": 0, "ties": 0, "results": []}
        decisive = False
        accept_now = False

        while int(running.get("total_games", 0)) < max_games:
            remaining = max_games - int(running.get("total_games", 0))
            batch_games = min(step_games, remaining)
            batch = self.evaluator.evaluate_vs_baseline(
                model_path,
                baseline_type="previous_best",
                baseline_path=baseline_path,
                num_games=batch_games,
            )
            running = self._merge_eval_stats(running, batch)
            n = int(running["total_games"])
            w = int(running["model_wins"])
            win_rate = float(running["win_rate"])

            logger.info(
                "[EVAL] Micro-gate: games=%s  wins=%s  WR=%.3f",
                n, w, win_rate,
            )

            if n < start_games:
                continue

            rem = max_games - n
            # optimistic/pessimistic bounds if all remaining games are wins/losses
            best_possible = float((w + rem) / max_games)
            worst_possible = float(w / max_games)

            if best_possible < threshold:
                decisive = True
                accept_now = False
                logger.info("[EVAL] Micro-gate early reject: cannot reach threshold.")
                break
            if worst_possible >= threshold:
                decisive = True
                accept_now = True
                logger.info("[EVAL] Micro-gate early accept: guaranteed above threshold.")
                break

        running["micro_gate"] = {
            "enabled": True,
            "start_games": start_games,
            "max_games": max_games,
            "step_games": step_games,
            "anti_regression_threshold": threshold,
            "decisive": decisive,
            "early_accept": bool(accept_now),
        }
        return running

    def _original_gate(self, iteration: int, checkpoint_path: str, eval_results: dict) -> bool:
        """Original gating logic: threshold/SPRT vs best + anti-regression floor.

        Used when league is disabled or for iteration 0.
        Returns True if model is accepted (promoted to best).
        """
        accepted = self._should_accept_model(eval_results)
        above_floor = self._check_anti_regression_floor(eval_results)

        if accepted and above_floor:
            self.consecutive_rejections = 0
            self.best_model_path = checkpoint_path
            self.best_model_iteration = iteration
            logger.debug("[GATE] >>> MODEL ACCEPTED — will promote to best at commit (iter%03d) <<<", iteration)
            # Register in league pool if league is active
            if self.league is not None:
                mid = self.league.model_id(iteration)
                self.league.promote(mid, checkpoint_path)
            return True
        elif accepted and not above_floor:
            self.consecutive_rejections += 1
            logger.warning(
                f"[GATE] Model passed gate but BLOCKED by anti-regression floor. "
                f"Consecutive rejections: {self.consecutive_rejections}"
            )
            # Still add to pool even if not promoted
            if self.league is not None:
                mid = self.league.model_id(iteration)
                self.league.add_to_pool(mid, checkpoint_path)
            return False
        else:
            self.consecutive_rejections += 1
            max_rejections = self.config["evaluation"].get(
                "max_consecutive_rejections", 5
            )
            if self.consecutive_rejections >= max_rejections:
                if above_floor:
                    logger.info(
                        "[GATE] Force-accepting iter%03d after %d consecutive rejections "
                        "(above anti-regression floor).",
                        iteration,
                        self.consecutive_rejections,
                    )
                    self.best_model_path = checkpoint_path
                    self.best_model_iteration = iteration
                    self.consecutive_rejections = 0
                    if self.league is not None:
                        mid = self.league.model_id(iteration)
                        self.league.promote(mid, checkpoint_path)
                    return True
                else:
                    logger.warning(
                        f"[GATE] Would force-accept after {self.consecutive_rejections} "
                        f"rejections, but model is BELOW anti-regression floor. "
                        f"Keeping current best. Resetting rejection counter."
                    )
                    self.consecutive_rejections = 0
                    if self.league is not None:
                        mid = self.league.model_id(iteration)
                        self.league.add_to_pool(mid, checkpoint_path)
                    return False
            else:
                best_ref = f"iter{self.best_model_iteration:03d}" if self.best_model_iteration is not None else "None"
                logger.info(
                    f"[GATE] Model REJECTED (best={best_ref} unchanged). "
                    f"Consecutive rejections: {self.consecutive_rejections}"
                )
                # Still add rejected models to pool for diversity
                if self.league is not None:
                    mid = self.league.model_id(iteration)
                    self.league.add_to_pool(mid, checkpoint_path)
                return False

    def _league_evaluate_and_gate(
        self,
        iteration: int,
        checkpoint_path: str,
        eval_results: dict,
        staging_path: Path,
        global_step: int,
    ) -> bool:
        """League-based evaluation and promotion gating.

        1) Evaluate candidate vs best_model
        2) Evaluate candidate vs anchor suite
        3) Apply robustness-based promotion gate
        4) Update payoff matrix
        5) Log diagnostics
        6) Save payoff CSV and diagnostics JSON

        Returns True if candidate is promoted.
        """
        league = self.league
        candidate_id = league.model_id(iteration)
        league.add_to_pool(candidate_id, checkpoint_path)
        league.main_id = candidate_id

        # Step 1: Get vs-best winrate from existing evaluation
        vs_best_wr = 0.5
        if "vs_previous_best" in eval_results:
            vs_best_wr = eval_results["vs_previous_best"]["win_rate"]

        # Update payoff: candidate vs best
        if league.best_id and "vs_previous_best" in eval_results:
            total = eval_results["vs_previous_best"].get("total_games", 0)
            wins = eval_results["vs_previous_best"].get("model_wins", 0)
            if total > 0:
                league.payoff.record_result(candidate_id, league.best_id, wins, total)

        # Step 2: Evaluate vs anchor suite
        anchors = league.refresh_anchors(iteration)
        suite_winrates: Dict[str, float] = {}

        if anchors:
            suite = {}
            for aid in anchors:
                path = league.get_checkpoint_path(aid)
                if path and Path(path).exists() and aid != candidate_id:
                    suite[aid] = path

            if suite:
                logger.info("[LEAGUE] Evaluating candidate vs %d anchors...", len(suite))
                suite_eval_games = league.config.suite_eval_games
                suite_winrates = self.evaluator.evaluate_vs_suite(
                    checkpoint_path, suite, games_per_opponent=suite_eval_games
                )

                # Update payoff matrix with suite results
                for opp_id, wr in suite_winrates.items():
                    games = suite_eval_games
                    wins = int(round(wr * games))
                    league.payoff.record_result(candidate_id, opp_id, wins, games)

        # Sync PFSP from payoff
        league.update_pfsp_from_payoff(candidate_id)

        # Step 3: Apply promotion gate
        gate_result = league.evaluate_candidate(candidate_id, vs_best_wr, suite_winrates)

        # Log gate result
        logger.info("")
        logger.info("=" * 60)
        logger.info("  LEAGUE GATE RESULT: %s", "PASS" if gate_result.passed else "FAIL")
        logger.info("=" * 60)
        logger.info("  vs best: %.1f%% (threshold: %.1f%%)",
                     gate_result.vs_best_wr * 100, league.config.gate_threshold * 100)
        if suite_winrates:
            logger.info("  suite mean: %.1f%% (threshold: %.1f%%)",
                         gate_result.suite_mean_wr * 100, league.config.suite_threshold * 100)
            logger.info("  suite worst: %.1f%% vs %s (threshold: %.1f%%)",
                         gate_result.suite_worst_wr * 100,
                         gate_result.suite_worst_opponent,
                         league.config.worst_threshold * 100)
        if gate_result.regression_alerts:
            logger.warning("  REGRESSION ALERTS: %s", gate_result.regression_alerts)
        logger.info("=" * 60)

        # Log to TensorBoard
        self.writer.add_scalar("league/vs_best_wr", gate_result.vs_best_wr, global_step)
        if suite_winrates:
            self.writer.add_scalar("league/suite_mean_wr", gate_result.suite_mean_wr, global_step)
            self.writer.add_scalar("league/suite_worst_wr", gate_result.suite_worst_wr, global_step)

        # Step 4: Promotion decision
        above_floor = self._check_anti_regression_floor(eval_results)
        accepted = False

        if gate_result.passed and above_floor:
            self.consecutive_rejections = 0
            self.best_model_path = checkpoint_path
            self.best_model_iteration = iteration
            league.promote(candidate_id, checkpoint_path)
            accepted = True
            logger.info("[LEAGUE GATE] >>> MODEL PROMOTED (iter%03d) — passed robustness gate <<<", iteration)
        elif gate_result.passed and not above_floor:
            self.consecutive_rejections += 1
            logger.warning(
                "[LEAGUE GATE] Passed robustness gate but BLOCKED by anti-regression floor. "
                "Consecutive rejections: %d", self.consecutive_rejections
            )
        elif vs_best_wr >= league.config.gate_threshold and not gate_result.passed:
            # Beats best but fails robustness: add to pool but don't promote
            self.consecutive_rejections += 1
            logger.info(
                "[LEAGUE GATE] Candidate beats best (%.1f%%) but FAILS robustness "
                "(suite_mean=%.1f%%, suite_worst=%.1f%%). Added to pool, not promoted.",
                vs_best_wr * 100, gate_result.suite_mean_wr * 100, gate_result.suite_worst_wr * 100
            )
        else:
            self.consecutive_rejections += 1
            max_rejections = self.config["evaluation"].get("max_consecutive_rejections", 5)
            if self.consecutive_rejections >= max_rejections and above_floor:
                logger.info(
                    "[LEAGUE GATE] Force-accepting iter%03d after %d consecutive rejections.",
                    iteration, self.consecutive_rejections,
                )
                self.best_model_path = checkpoint_path
                self.best_model_iteration = iteration
                self.consecutive_rejections = 0
                league.promote(candidate_id, checkpoint_path)
                accepted = True
            else:
                best_ref = f"iter{self.best_model_iteration:03d}" if self.best_model_iteration is not None else "None"
                logger.info(
                    "[LEAGUE GATE] Model REJECTED (best=%s unchanged). "
                    "Consecutive rejections: %d", best_ref, self.consecutive_rejections
                )

        # Step 5: Log diagnostics
        eval_results["league_gate"] = {
            "passed": gate_result.passed,
            "vs_best_wr": gate_result.vs_best_wr,
            "suite_mean_wr": gate_result.suite_mean_wr,
            "suite_worst_wr": gate_result.suite_worst_wr,
            "suite_worst_opponent": gate_result.suite_worst_opponent,
            "regression_alerts": gate_result.regression_alerts,
            "suite_winrates": suite_winrates,
        }
        diag = league.log_diagnostics(iteration, candidate_id)
        self.writer.add_scalar("league/cycles", diag.get("cycles_in_200_triples", 0), global_step)
        self.writer.add_scalar("league/exploitability", diag.get("exploitability_proxy", 0), global_step)
        self.writer.add_scalar("league/pool_size", diag.get("pool_size", 0), global_step)

        # Step 6: Save reports
        logs_dir = Path(self.config["paths"]["logs_dir"])
        league.save_payoff_csv(logs_dir / "payoff_matrix.csv")
        league.save_diagnostics_json(logs_dir / "league_diagnostics.jsonl", iteration, candidate_id)

        return accepted

    def _should_accept_model(self, eval_results: dict) -> bool:
        """Determine if the trained model should replace best_model.pt.

        Decision hierarchy:
          1. Iteration 0: always accept (bootstrap).
          2. Evaluation disabled: always accept.
          3. SPRT result if available: accept/reject overrides win-rate threshold.
          4. Win rate >= threshold.
          5. Anti-regression floor: even if forced, block models below hard floor.
        """
        win_rate_threshold = self.config["evaluation"]["win_rate_threshold"]
        # Hard floor: models below this win rate are NEVER promoted, even on force-accept
        anti_regression_floor = float(
            self.config.get("evaluation", {}).get("anti_regression_floor", 0.35)
        )

        # [M2 FIX] Always accept the first model — bootstrap data trains a weak
        # net, but it MUST become latest_model_path so subsequent iterations have
        # a valid baseline for evaluation and self-play.
        if self.current_iteration == 0:
            logger.info("[GATE] Iteration 0: auto-accepting model (bootstrap).")
            return True

        # [C2 FIX] When evaluation is disabled (games_per_eval/games_vs_* <= 0),
        # auto-accept every model so latest_model_path stays current.
        eval_cfg = self.config.get("evaluation", {}) or {}
        num_eval_games = int(
            eval_cfg.get("games_per_eval")
            or max(eval_cfg.get("games_vs_pure_mcts", 0), eval_cfg.get("games_vs_best", 0))
            or 0
        )
        if num_eval_games <= 0:
            logger.debug("[GATE] Evaluation disabled (games_per_eval <= 0): auto-accepting.")
            return True

        if "vs_previous_best" in eval_results:
            prev_best_wr = eval_results["vs_previous_best"]["win_rate"]

            # SPRT decision overrides simple threshold
            sprt_info = eval_results["vs_previous_best"].get("sprt", {})
            if sprt_info:
                if sprt_info.get("accept"):
                    logger.info(f"[GATE] SPRT ACCEPT (WR={prev_best_wr:.1%}, LLR={sprt_info['llr']:.3f})")
                    return True
                if sprt_info.get("reject"):
                    logger.info(f"[GATE] SPRT REJECT (WR={prev_best_wr:.1%}, LLR={sprt_info['llr']:.3f})")
                    return False
                # Inconclusive: fall through to threshold check
                logger.info(f"[GATE] SPRT inconclusive; falling back to WR threshold")

            return prev_best_wr >= win_rate_threshold

        # If pure-MCTS eval was skipped, fall back to accepting.
        if "vs_pure_mcts" in eval_results:
            pure_mcts_wr = eval_results["vs_pure_mcts"]["win_rate"]
            return pure_mcts_wr >= win_rate_threshold

        logger.info("[GATE] No evaluation baselines were run; defaulting to accept.")
        return True

    def _check_anti_regression_floor(self, eval_results: dict) -> bool:
        """Check if model is above the hard anti-regression floor.

        Returns True if model is ABOVE the floor (safe to promote).
        Returns False if model is BELOW the floor (must NOT promote even on force-accept).
        """
        floor = float(
            self.config.get("evaluation", {}).get("anti_regression_floor", 0.35)
        )
        if "vs_previous_best" not in eval_results:
            return True  # Can't check, assume OK
        wr = eval_results["vs_previous_best"]["win_rate"]
        if wr < floor:
            logger.warning(
                f"[GATE] ANTI-REGRESSION FLOOR: WR={wr:.1%} < floor={floor:.1%}. "
                f"Model will NOT be promoted even on force-accept."
            )
            return False
        return True

    def _save_iteration_summary(
        self, iteration: int, summary: dict, output_dir: Optional[Path] = None
    ):
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            elif hasattr(obj, "item"):
                return obj.item()
            return obj

        summary = convert_types(summary)
        if output_dir is not None:
            summary_path = output_dir / f"iteration_{iteration:03d}.json"
        else:
            summary_path = Path(self.config["paths"]["logs_dir"]) / f"iteration_{iteration:03d}.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = summary_path.with_suffix(summary_path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, summary_path)
        logger.debug("Saved iteration summary to %s", summary_path)

    def _commit_tensorboard_events(self, committed_path: Path) -> None:
        """Copy TensorBoard event files from a committed iteration dir to permanent log dir.

        Called after atomic commit so events only appear in logs/tensorboard/
        once an iteration has fully completed.  Safe to call on old committed
        dirs that don't have a tensorboard/ subdirectory (no-op).
        """
        committed_tb = committed_path / "tensorboard"
        if not committed_tb.exists():
            return
        self.tb_log_dir.mkdir(parents=True, exist_ok=True)
        for item in committed_tb.iterdir():
            dest = self.tb_log_dir / item.name
            try:
                if item.is_file():
                    shutil.copy2(item, dest)
                elif item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
            except Exception as e:
                logger.warning("Failed to copy TensorBoard event %s: %s", item.name, e)
        logger.debug("Committed TensorBoard events to %s", self.tb_log_dir)

    def _commit_iteration(
        self,
        iteration: int,
        checkpoint_path: str,
        selfplay_stats: dict,
        accepted: bool,
        summary: dict,
        train_metrics: dict,
        eval_results: dict,
        iter_time: float,
        global_step: int,
    ) -> None:
        """
        Atomically commit iteration: move staging -> committed, update run_state,
        replay buffer, best/latest checkpoints, Elo, metadata.
        """
        global _commit_in_progress
        _commit_in_progress = True
        try:
            self._commit_iteration_impl(
                iteration, checkpoint_path, selfplay_stats, accepted,
                summary, train_metrics, eval_results, iter_time, global_step,
            )
        finally:
            _commit_in_progress = False
            if _hard_stop_requested:
                logger.info("[SHUTDOWN] Commit complete; exiting immediately.")
                sys.exit(0)

    def _commit_iteration_impl(
        self,
        iteration: int,
        checkpoint_path: str,
        selfplay_stats: dict,
        accepted: bool,
        summary: dict,
        train_metrics: dict,
        eval_results: dict,
        iter_time: float,
        global_step: int,
    ) -> None:
        """Implementation of _commit_iteration (called with _commit_in_progress set)."""
        committed_path = committed_dir(self.run_root, iteration)
        staging_path = staging_dir(self.run_root, iteration)

        # Log summary to staging file before detach (so it's included in the iteration log)
        wr_best = eval_results.get("vs_previous_best", {}).get("win_rate", -1)
        wr_mcts = eval_results.get("vs_pure_mcts", {}).get("win_rate", -1)
        margin_best = eval_results.get("vs_previous_best", {}).get("avg_model_score_margin", 0)
        wr_best_str = f"{wr_best:.1%}" if wr_best >= 0 else "N/A"
        wr_mcts_str = f"{wr_mcts:.1%}" if wr_mcts >= 0 else "N/A"
        _print_iter_summary(
            iteration=iteration,
            accepted=accepted,
            iter_time=iter_time,
            wr_best_str=wr_best_str,
            margin_best=margin_best,
            wr_mcts_str=wr_mcts_str,
            train_metrics=train_metrics,
        )
        logger.debug(
            "ITER %03d SUMMARY  |  %s  |  time=%.0fs  |  WR(best)=%s  margin=%.1fpts  |  WR(mcts)=%s  |  loss=%.4f  pol_acc=%.1f%%",
            iteration,
            "ACCEPTED" if accepted else "REJECTED",
            iter_time,
            wr_best_str,
            margin_best,
            wr_mcts_str,
            train_metrics.get("total_loss", 0),
            train_metrics.get("policy_accuracy", 0) * 100,
        )
        _detach_staging_log_handler()

        applied_settings = summary.get("applied_settings", {})
        manifest = {
            "iteration": iteration,
            "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
            "accepted": accepted,
            "best_model_iteration": self.best_model_iteration,
            "consecutive_rejections": self.consecutive_rejections,
            "global_step": global_step,
            "num_positions": selfplay_stats.get("num_positions", 0),
            "elo_ratings": self.elo_tracker.ratings,
            "applied_settings": applied_settings,
        }
        commit_iteration(self.run_root, iteration, manifest)

        # Append staged training log to permanent log (only completed iterations)
        permanent_log = Path(
            self.config.get("logging", {}).get("log_file")
            or str(Path(self.config.get("paths", {}).get("logs_dir", "logs")) / "training.log")
        )
        _append_staged_log_to_permanent(committed_path, permanent_log)

        # Update best/latest in checkpoints dir (stable paths for self-play and eval)
        src_ckpt = committed_path / f"iteration_{iteration:03d}.pt"
        latest_dest = self._atomic_copy_checkpoint(src_ckpt, "latest_model.pt")
        self.latest_model_path = str(latest_dest)
        self.latest_checkpoint = str(latest_dest)
        save_best = bool((self.config.get("training", {}) or {}).get("save_best", True))
        if accepted and self.best_model_iteration == iteration:
            if save_best:
                best_dest = self._atomic_copy_checkpoint(src_ckpt, "best_model.pt")
                self.best_model_path = str(best_dest)
                self._save_best_stamped(src_ckpt, iteration)
            else:
                self.best_model_path = str(src_ckpt)  # Use committed iter checkpoint for self-play

        # Replay buffer: point to committed selfplay path, persist
        committed_selfplay = committed_path / "selfplay.h5"
        self.replay_buffer.finalize_iteration_for_commit(
            iteration, str(committed_selfplay), selfplay_stats.get("num_positions", 0)
        )

        self.last_committed_iteration = iteration
        _atomic_write_json(self.run_state_path, {
            "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
            "last_committed_iteration": iteration,
            "best_iteration": self.best_model_iteration,
            "best_model_path": self.best_model_path,
            "latest_model_path": self.latest_model_path,
            "global_step": global_step,
            "consecutive_rejections": self.consecutive_rejections,
            "config_hash": self._config_hash,
            "run_id": self.run_id,
            "seed": int(self.config.get("seed", 42)),
        })
        self.elo_tracker.save_state(self.elo_state_path)
        # Persist league state at commit time
        if self.league is not None:
            self.league.save_state()
        self._append_metadata(
            iteration, selfplay_stats, train_metrics, eval_results, accepted, iter_time, global_step
        )

        # Copy TensorBoard events from committed staging to permanent log dir
        self._commit_tensorboard_events(committed_path)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Patchwork AlphaZero Training")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to configuration file")
    parser.add_argument("--start-iteration", type=int, default=0, help="Iteration to start from")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Run directory (e.g. runs/patchwork_overnight). Same dir = same run = seamless resume.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run ID (e.g. patchwork_overnight). Overrides config paths.run_id for deterministic resume.",
    )
    parser.add_argument(
        "--allow-config-mismatch",
        action="store_true",
        default=False,
        help="If run_state exists with different config hash, allow resume (default: abort).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Override max_iterations for short runs (e.g. --iterations 1).",
    )

    args = parser.parse_args()

    # Graceful shutdown: first Ctrl+C = stop after iteration commit; second = hard exit.
    signal.signal(signal.SIGINT, _handle_shutdown_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_shutdown_signal)

    trainer = AlphaZeroTrainer(
        args.config,
        cli_run_dir=args.run_dir,
        cli_run_id=args.run_id,
        allow_config_mismatch=args.allow_config_mismatch,
        cli_args=sys.argv,
        cli_iterations=args.iterations,
    )

    try:
        trainer.train(args.start_iteration, args.resume)
    except KeyboardInterrupt:
        logger.info("")
        logger.info("Training interrupted by user.")
        logger.info(
            "Transactional design: no state saved mid-iteration. "
            "On next launch, partial staging will be discarded and iter%03d "
            "will restart from self-play.",
            trainer.current_iteration,
        )
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Close staging TensorBoard writer if still open (interrupted iteration).
        # Events are NOT committed for interrupted iterations — staging cleanup
        # on next resume will discard them.
        if hasattr(trainer, "writer") and trainer.writer is not None:
            try:
                trainer.writer.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()