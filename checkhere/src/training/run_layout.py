"""
Transactional Run Layout for AlphaZero Training

Provides per-iteration staging and commit directories so that:
- All outputs for iteration N are written to staging/iter_N/ first
- Only at the end of a successful iteration do we atomically commit
- On resume, partial staging is discarded and we restart from the last committed iteration

All file writes use .tmp + os.replace for atomicity. Commit = atomic rename
(staging -> committed) + atomic run_state update.

Directory layout:
  runs/<run_id>/staging/iter_<N>/   - outputs produced during iteration N (discarded if interrupted)
  runs/<run_id>/committed/iter_<N>/ - finalized outputs after successful commit
  runs/<run_id>/run_state.json      - authoritative global state (updated only at commit)
"""

import atexit
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)

_LOCK_PATH: Optional[Path] = None

# Test hook: callbacks run during slow-commit window (PATCHWORK_SLOW_COMMIT_FOR_TEST=1)
_commit_test_callbacks: list = []


def register_commit_test_callback(cb):
    """Register callback to run during commit (test-only). Callback receives no args."""
    _commit_test_callbacks.append(cb)


def clear_commit_test_callbacks() -> None:
    """Clear all commit test callbacks (test teardown)."""
    _commit_test_callbacks.clear()


def _run_commit_test_callbacks() -> None:
    for cb in _commit_test_callbacks:
        try:
            cb()
        except Exception as e:
            logger.warning("Commit test callback failed: %s", e)

ITER_DIR_FMT = "iter_{:03d}"
COMMIT_MANIFEST = "commit_manifest.json"

# HDF5 selfplay schema (gold_v2_multimodal)
SELFPLAY_SCHEMA_VERSION = 3
SELFPLAY_COMPLETE_ATTR = "selfplay_complete"
SELFPLAY_NUM_GAMES_ATTR = "selfplay_num_games_written"
SELFPLAY_SCHEMA_VERSION_ATTR = "selfplay_schema_version"
SELFPLAY_EXPECTED_CHANNELS_ATTR = "expected_channels"
SELFPLAY_SCORE_SCALE_ATTR = "score_scale"
SELFPLAY_VALUE_TARGET_TYPE_ATTR = "value_target_type"
ENCODING_VERSION_ATTR = "encoding_version"


def get_run_id(config: dict, cli_run_id: Optional[str] = None) -> str:
    """
    Derive or read run_id. Deterministic for same config = same run = seamless resume.
    Priority: CLI --run-id > config paths.run_id > config hash.
    """
    if cli_run_id:
        return str(cli_run_id).strip()
    paths = config.get("paths", {})
    explicit = paths.get("run_id") or paths.get("run_dir") or paths.get("run_name")
    if explicit:
        return str(explicit).strip()
    cfg_hash = _config_hash(config)
    return f"run_{cfg_hash}"


def _find_existing_run_root(base: Path) -> Optional[Path]:
    """If exactly one subdir has run_state.json, return it (for resume without explicit run-id)."""
    if not base.exists():
        return None
    found = []
    for item in base.iterdir():
        if item.is_dir() and (item / "run_state.json").exists():
            found.append(item)
    return found[0] if len(found) == 1 else None


def get_run_root(
    config: dict,
    cli_run_id: Optional[str] = None,
    cli_run_dir: Optional[str] = None,
) -> Path:
    """
    Get the root directory for this run. Deterministic = same run every time = seamless resume.
    Priority: CLI --run-dir (full path) > config paths.run_dir > run_root/run_id.
    When user did NOT pass --run-id and no paths.run_id: if exactly one existing run_state
    in runs/, use that (keep existing run root; avoid forking on config change).
    """
    if cli_run_dir:
        return Path(cli_run_dir.strip())
    paths = config.get("paths", {})
    run_dir = paths.get("run_dir")
    if run_dir:
        p = Path(str(run_dir).strip())
        if p.is_absolute() or "/" in str(run_dir) or "\\" in str(run_dir):
            return p
    base = Path(paths.get("run_root", "runs"))
    run_id = get_run_id(config, cli_run_id)
    candidate = base / run_id
    # Keep existing run root when run_state exists and user didn't explicitly pass --run-id
    if cli_run_id is None and not (paths.get("run_id") or paths.get("run_dir") or paths.get("run_name")):
        existing = _find_existing_run_root(base)
        if existing is not None:
            return existing
    return candidate


def staging_dir(run_root: Path, iteration: int) -> Path:
    """Path to staging directory for iteration N."""
    return run_root / "staging" / ITER_DIR_FMT.format(iteration)


def committed_dir(run_root: Path, iteration: int) -> Path:
    """Path to committed directory for iteration N."""
    return run_root / "committed" / ITER_DIR_FMT.format(iteration)


def is_iter_committed(run_root: Path, iteration: int) -> bool:
    """True if iter_N has been successfully committed."""
    path = committed_dir(run_root, iteration)
    manifest = path / COMMIT_MANIFEST
    return manifest.exists()


def atomic_write_json(path: Path, payload: dict) -> None:
    """Write JSON atomically: .tmp then os.replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_file(path: Path, write_fn) -> None:
    """
    Atomically write a file. write_fn receives (tmp_path) and should write content.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    write_fn(tmp)
    os.replace(tmp, path)


def atomic_copy_file(src: Path, dst: Path) -> None:
    """Copy file atomically: copy to dst.tmp then replace."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def _config_hash(config: dict) -> str:
    """Stable hash of config for run identification."""
    import hashlib
    canonical = json.dumps(config, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def max_committed_iteration(run_root: Path) -> int:
    """
    Scan committed/ for the highest N with commit_manifest.json.
    Used to reconcile run_state with filesystem (crash-after-move-before-run_state edge case).
    Returns -1 if no committed iterations exist.
    """
    committed_base = run_root / "committed"
    if not committed_base.exists():
        return -1
    max_n = -1
    for item in committed_base.iterdir():
        if not item.is_dir() or not item.name.startswith("iter_"):
            continue
        try:
            n = int(item.name.split("_")[1])
            if (item / COMMIT_MANIFEST).exists() and n > max_n:
                max_n = n
        except (ValueError, IndexError):
            continue
    return max_n


def reconcile_run_state(run_root: Path, last_from_state: int) -> tuple[int, bool]:
    """
    Reconcile run_state with filesystem. If committed/ has iter_N not reflected in state
    (crash-after-move-before-run_state edge case), return max_on_disk > last_from_state.
    Returns (effective_last_committed, state_needs_repair).
    """
    max_on_disk = max_committed_iteration(run_root)
    if max_on_disk <= last_from_state:
        return last_from_state, False
    logger.info(
        "Reconciling run_state: filesystem has iter%03d committed but run_state had %d; will repair.",
        max_on_disk,
        last_from_state,
    )
    return max_on_disk, True


def cleanup_stale_tmp_files(run_root: Path) -> int:
    """Remove stale .tmp files from crash/interrupt. Returns count removed."""
    run_root = Path(run_root)
    if not run_root.exists():
        return 0
    removed = 0
    for path in run_root.rglob("*.tmp"):
        try:
            path.unlink()
            removed += 1
            logger.debug("Removed stale tmp: %s", path)
        except OSError as e:
            logger.warning("Could not remove stale tmp %s: %s", path, e)
    if removed:
        logger.info("Cleaned %d stale .tmp files from run directory", removed)
    return removed


def acquire_run_lock(run_root: Path) -> None:
    """
    Acquire exclusive lock on run directory. Prevents two processes from using same run.
    Uses O_EXCL for atomic create so two processes cannot both think they got it.
    Aborts with RuntimeError if lock is held by another live process.
    """
    global _LOCK_PATH
    lock_path = run_root / ".lock"
    run_root.mkdir(parents=True, exist_ok=True)

    for attempt in range(5):
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            try:
                with open(lock_path, "r", encoding="utf-8") as f:
                    pid = int(f.read().strip())
            except (ValueError, OSError):
                logger.warning("Stale or invalid .lock file, removing")
                lock_path.unlink(missing_ok=True)
                continue
            if _process_alive(pid):
                raise RuntimeError(
                    f"Run directory {run_root} is locked by process {pid}. "
                    "Another training run may be active. Exit the other process or use a different --run-dir."
                )
            logger.info("Removing stale lock from exited process %d", pid)
            lock_path.unlink(missing_ok=True)
            continue
        except OSError as e:
            raise RuntimeError(f"Failed to create lock file in {run_root}: {e}") from e

        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(str(os.getpid()))
                f.flush()
                os.fsync(f.fileno())
        except OSError:
            os.close(fd)
            lock_path.unlink(missing_ok=True)
            raise

        _LOCK_PATH = lock_path
        atexit.register(_release_run_lock)
        return

    raise RuntimeError(
        f"Could not acquire lock on {run_root} after retries. "
        "Another process may be racing to start."
    )


def _release_run_lock() -> None:
    global _LOCK_PATH
    if _LOCK_PATH and _LOCK_PATH.exists():
        try:
            _LOCK_PATH.unlink()
            logger.debug("Released run lock: %s", _LOCK_PATH)
        except OSError:
            pass
        _LOCK_PATH = None


def _process_alive(pid: int) -> bool:
    """Check if process with given PID is running. Cross-platform (Unix + Windows)."""
    if pid <= 0:
        return False
    if os.name == "nt":
        return _process_alive_win(pid)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _process_alive_win(pid: int) -> bool:
    """Windows: use OpenProcess to check existence (os.kill(pid,0) may not work reliably)."""
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.windll.kernel32
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid)
        if handle:
            k32.CloseHandle(handle)
            return True
        return False
    except Exception:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def _get_expected_games_for_iteration(config: dict, iteration: int) -> int:
    """Expected number of games for iteration (bootstrap or schedule)."""
    if iteration == 0:
        return int(config.get("selfplay", {}).get("bootstrap", {}).get("games", 200))
    schedule = config.get("iteration", {}).get("games_schedule", [])
    base = int(config.get("selfplay", {}).get("games_per_iteration", 400))
    for entry in sorted(schedule, key=lambda x: x["iteration"], reverse=True):
        if iteration >= entry["iteration"]:
            return int(entry["games"])
    return base


def _staging_has_complete_selfplay(staging_path: Path, iteration: int, config: Optional[dict] = None) -> bool:
    """True if selfplay.h5 exists with num_games >= expected or legacy fallback."""
    h5_path = staging_path / "selfplay.h5"
    if not h5_path.exists():
        return False
    try:
        import h5py
        with h5py.File(h5_path, "r") as f:
            states_key = "spatial_states" if "spatial_states" in f else "states"
            if states_key not in f:
                return False
            num_games = int(f.attrs.get(SELFPLAY_NUM_GAMES_ATTR, f.attrs.get("num_games", 0)))
            complete = f.attrs.get(SELFPLAY_COMPLETE_ATTR, None)
            if config is not None:
                expected = _get_expected_games_for_iteration(config, iteration)
                if complete is True:
                    return num_games >= expected
                if complete is False:
                    return False
            if complete is True:
                return True
            n_pos = int(f[states_key].shape[0])
            if config is not None:
                expected = _get_expected_games_for_iteration(config, iteration)
                return n_pos >= expected * 20 or num_games >= expected
            return num_games > 0 or n_pos > 0
    except Exception:
        return False


def get_staging_cleanup_plan(
    run_root: Path, last_committed: int, config: Optional[dict] = None
) -> list[tuple[int, str, str]]:
    """
    Return what would happen during cleanup_staging without mutating.
    Returns [(iter_num, 'preserve'|'delete', reason), ...] for each staging dir.
    """
    staging_base = run_root / "staging"
    result: list[tuple[int, str, str]] = []
    if not staging_base.exists():
        return result

    for item in staging_base.iterdir():
        if not item.is_dir() or not item.name.startswith("iter_"):
            continue
        try:
            iter_num = int(item.name.split("_")[1])
        except (ValueError, IndexError):
            continue

        if is_iter_committed(run_root, iter_num):
            result.append((iter_num, "preserve", "already committed"))
            continue

        result.append((iter_num, "delete", "staging discarded; iteration will restart from beginning"))
    return result


def cleanup_staging(
    run_root: Path, last_committed: int, config: Optional[dict] = None
) -> None:
    """
    Remove any staging/iter_* that does not have a corresponding committed marker.
    These are partial iterations from an interrupted run. Always discard — on resume,
    the iteration starts from the beginning (self-play -> train -> eval -> commit).
    """
    staging_base = run_root / "staging"
    if not staging_base.exists():
        return

    for item in staging_base.iterdir():
        if not item.is_dir() or not item.name.startswith("iter_"):
            continue
        try:
            iter_num = int(item.name.split("_")[1])
        except (ValueError, IndexError):
            continue

        if is_iter_committed(run_root, iter_num):
            continue

        logger.info(
            "Found partial staging for iter%03d; discarding and restarting iter%03d from self-play.",
            iter_num,
            iter_num,
        )
        try:
            shutil.rmtree(item)
        except OSError as e:
            logger.warning("Failed to remove staging %s: %s", item, e)


def write_commit_manifest(staging_path: Path, manifest: Dict[str, Any]) -> Path:
    """Write commit_manifest.json to staging directory."""
    manifest_path = staging_path / COMMIT_MANIFEST
    atomic_write_json(manifest_path, manifest)
    return manifest_path


def _same_filesystem(path_a: Path, path_b: Path) -> bool:
    """True if both paths are on the same filesystem/mount (atomic rename safe)."""
    try:
        sa = os.stat(path_a)
        sb = os.stat(path_b)
        return sa.st_dev == sb.st_dev
    except OSError:
        return False


def _commit_via_copy_then_delete(staging_path: Path, committed_path: Path) -> None:
    """Cross-filesystem fallback: copy tree then remove source (not atomic).
    Fsyncs each copied file, then the committed dir and its parent, so a crash
    doesn't leave a directory that looks moved but isn't durable.
    """
    shutil.copytree(staging_path, committed_path, dirs_exist_ok=False)
    for root, _dirs, files in os.walk(committed_path, topdown=True):
        for name in files:
            p = Path(root) / name
            try:
                with open(p, "ab") as f:
                    f.flush()
                    os.fsync(f.fileno())
            except OSError:
                pass
    # Sync directory entries so parent links to new iter dir are durable
    for d in (committed_path, committed_path.parent):
        try:
            fd = os.open(str(d), os.O_RDONLY)
            os.fsync(fd)
            os.close(fd)
        except OSError:
            pass
    shutil.rmtree(staging_path)


def commit_iteration(
    run_root: Path, iteration: int, manifest: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Atomically commit iteration N: write manifest, then rename staging/iter_N -> committed/iter_N.

    Cross-filesystem check: if staging/ and committed/ are on different mounts, shutil.move is
    non-atomic (copy+delete). We detect this and use explicit copy+fsync+delete fallback, recording
    commit_method in manifest.
    """
    staging_path = staging_dir(run_root, iteration)
    committed_path = committed_dir(run_root, iteration)

    if not staging_path.exists():
        raise RuntimeError(f"Cannot commit iter{iteration:03d}: staging directory missing")

    committed_path.parent.mkdir(parents=True, exist_ok=True)

    if committed_path.exists():
        shutil.rmtree(committed_path)

    # Check same filesystem for atomic rename
    staging_base = run_root / "staging"
    committed_base = run_root / "committed"
    staging_base.mkdir(parents=True, exist_ok=True)
    committed_base.mkdir(parents=True, exist_ok=True)

    if _same_filesystem(staging_base, committed_base):
        manifest["commit_method"] = "rename"
    else:
        manifest["commit_method"] = "copy"
        logger.warning(
            "Staging and committed on different filesystems; commit uses copy+delete (non-atomic)."
        )

    # Move staging -> committed first. All artifacts in place before done marker.
    if manifest["commit_method"] == "rename":
        shutil.move(str(staging_path), str(committed_path))
    else:
        _commit_via_copy_then_delete(staging_path, committed_path)

    # Test hook: slow commit to exercise double-SIGINT during commit (PATCHWORK_SLOW_COMMIT_FOR_TEST=1)
    if os.environ.get("PATCHWORK_SLOW_COMMIT_FOR_TEST") == "1":
        _run_commit_test_callbacks()
        time.sleep(2.0)

    # Done marker written LAST: temp -> fsync -> atomic rename -> fsync dir.
    # If manifest write fails, committed/ has artifacts but no marker = resume ignores (safe).
    write_commit_manifest(committed_path, manifest)
    for d in (committed_path, committed_path.parent):
        try:
            fd = os.open(str(d), os.O_RDONLY)
            os.fsync(fd)
            os.close(fd)
        except OSError:
            pass

    if manifest["commit_method"] == "rename":
        logger.debug("Committed iter%03d successfully; last_committed_iteration updated.", iteration)
    else:
        logger.debug("Committed iter%03d successfully (copy); last_committed_iteration updated.", iteration)

    return manifest
