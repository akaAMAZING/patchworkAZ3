"""
D4 LUT disk cache: versioned filenames, atomic writes, file lock, fast build.

- Cache filename: d4_buy_lut_pc33_v5.npy
- Metadata sidecar: d4_lut_meta_pc33_v5.json
- Old unversioned/versioned caches are ignored (version mismatch).
- legal_lut removed in v5: legalTL is computed on-GPU, not stored in state.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Optional, Tuple

import numpy as np

from src.network.d4_constants import COMPACT_SIZE, PC_MAX

# Bump when LUT format/structure changes (v5: legal_lut removed; only buy_lut kept)
LUT_VERSION = 5
LEGAL_FLAT_SIZE = 24 * 81  # 1944 — flat size used for buy_lut (same dimension)


def _get_cache_dir() -> str:
    cache_dir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    return os.path.normpath(os.path.join(cache_dir, "patchworkaz"))


def _versioned_basename(suffix: str) -> str:
    return f"d4_{suffix}_pc{PC_MAX}_v{LUT_VERSION}"


def _unversioned_globs() -> tuple:
    """Basenames of old unversioned/versioned caches (ignored)."""
    return (
        "d4_buy_lut.npy", "d4_legal_lut.npy",
        "d4_buy_lut_pc22.npy", "d4_legal_lut_pc22.npy",
        # v4 and earlier had separate legal_lut file; now removed
    )


def get_lut_paths() -> Tuple[str, str]:
    """Return (buy_path, meta_path)."""
    lut_dir = _get_cache_dir()
    base = _versioned_basename
    return (
        os.path.join(lut_dir, base("buy_lut") + ".npy"),
        os.path.join(lut_dir, base("lut_meta") + ".json"),
    )


def _try_acquire_lock(lut_dir: str) -> Optional[object]:
    """Best-effort file lock. Returns handle to close, or None if lock failed."""
    lock_path = os.path.join(lut_dir, f".d4_lut_build_pc{PC_MAX}_v{LUT_VERSION}.lock")
    try:
        if sys.platform == "win32":
            import msvcrt
            fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                return fd
            except OSError:
                os.close(fd)
                return None
        else:
            import fcntl
            fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return fd
            except (OSError, BlockingIOError):
                os.close(fd)
                return None
    except Exception:
        return None


def _release_lock(handle: object) -> None:
    try:
        if sys.platform == "win32":
            import msvcrt
            msvcrt.locking(handle, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_UN)
    except Exception:
        pass
    finally:
        try:
            os.close(handle)
        except Exception:
            pass


def _get_git_hash() -> str:
    try:
        import subprocess
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=2, cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        )
        if r.returncode == 0 and r.stdout:
            return r.stdout.strip()[:12]
    except Exception:
        pass
    return ""


def _write_atomic(path: str, arr: np.ndarray) -> None:
    """Write array to temp file then replace."""
    import tempfile
    path = os.path.abspath(os.path.normpath(path))
    lut_dir = os.path.dirname(path)
    os.makedirs(lut_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".npy", dir=lut_dir)
    try:
        os.close(fd)
        np.save(tmp, arr)
        os.replace(tmp, path)
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        raise


def _save_metadata(meta_path: str) -> None:
    meta = {
        "pc_max": PC_MAX,
        "lut_version": LUT_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dtype": "int32",
        "shape": [COMPACT_SIZE, LEGAL_FLAT_SIZE],
        "git": _get_git_hash(),
    }
    lut_dir = os.path.dirname(meta_path)
    os.makedirs(lut_dir, exist_ok=True)
    tmp = meta_path + ".tmp." + str(os.getpid())
    try:
        with open(tmp, "w") as f:
            json.dump(meta, f, indent=2)
        os.replace(tmp, meta_path)
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


def load_luts_if_valid() -> Optional[np.ndarray]:
    """
    Load versioned buy LUT if present and valid. Returns buy_lut or None.
    Old unversioned/versioned caches are never loaded (version mismatch).
    """
    buy_path, meta_path = get_lut_paths()
    if not os.path.exists(buy_path):
        return None
    try:
        buy = np.load(buy_path)
        if buy.shape != (COMPACT_SIZE, LEGAL_FLAT_SIZE):
            return None
        if buy.dtype != np.int32:
            return None
        return buy.astype(np.int32)
    except Exception:
        return None


def build_and_save_luts(verbose: bool = True) -> np.ndarray:
    """Build buy LUT (vectorized), save atomically, return buy_lut."""
    from src.network.d4_lut_build import build_buy_legal_luts_fast
    buy_path, meta_path = get_lut_paths()
    lut_dir = os.path.dirname(buy_path)
    os.makedirs(lut_dir, exist_ok=True)
    lock_handle = _try_acquire_lock(lut_dir)
    if lock_handle is not None:
        try:
            loaded = load_luts_if_valid()
            if loaded is not None:
                if verbose:
                    print("D4 LUT: loaded from cache")
                return loaded
            if verbose:
                print("D4 LUT: building (this may take a few seconds)...")
            t0 = time.perf_counter()
            buy_arr = build_buy_legal_luts_fast()
            t1 = time.perf_counter()
            if verbose:
                print(f"D4 LUT: built in {t1 - t0:.1f}s, saving...")
            _write_atomic(buy_path, buy_arr)
            _save_metadata(meta_path)
            if verbose:
                print("D4 LUT: saved")
            return buy_arr
        finally:
            _release_lock(lock_handle)
    else:
        if verbose:
            print("D4 LUT: another process is building; waiting for cache...")
        for _ in range(120):
            loaded = load_luts_if_valid()
            if loaded is not None:
                if verbose:
                    print("D4 LUT: loaded after wait")
                return loaded
            time.sleep(0.5)
        raise RuntimeError("D4 LUT: timed out waiting for another process to build")
