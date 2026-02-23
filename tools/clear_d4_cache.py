#!/usr/bin/env python3
"""
Clear D4 LUT disk cache.

Lists cache directory, shows what would be deleted. Use --yes to delete.
Removes old unversioned caches by default. Use --all to remove all D4 caches.
"""

from __future__ import annotations

import argparse
import os


def _get_cache_dir() -> str:
    cache_dir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    return os.path.join(cache_dir, "patchworkaz")


def _find_d4_files(lut_dir: str, include_versioned: bool) -> list:
    """Find D4 cache files. include_versioned=True adds versioned (pc*_v*.npy)."""
    if not os.path.isdir(lut_dir):
        return []
    result = []
    for name in os.listdir(lut_dir):
        path = os.path.join(lut_dir, name)
        if not os.path.isfile(path):
            continue
        if name.startswith(".d4_lut_build") and name.endswith(".lock"):
            result.append(path)
            continue
        if not name.startswith("d4_"):
            continue
        is_versioned = "_v" in name and "pc" in name
        if is_versioned:
            if include_versioned:
                result.append(path)
        else:
            result.append(path)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Clear D4 LUT cache")
    ap.add_argument("--yes", action="store_true", help="Actually delete; default is dry-run")
    ap.add_argument("--all", action="store_true", help="Remove all D4 caches including versioned")
    args = ap.parse_args()
    lut_dir = _get_cache_dir()
    include_versioned = args.all
    files = _find_d4_files(lut_dir, include_versioned)
    if not files:
        print(f"No D4 cache files found in {lut_dir}")
        return
    print("Would delete:" if not args.yes else "Deleting:")
    for p in sorted(files):
        try:
            sz = os.path.getsize(p)
            print(f"  {p} ({sz} bytes)")
        except OSError:
            print(f"  {p}")
    if not args.yes:
        print("\nRun with --yes to delete")
        return
    for p in files:
        try:
            os.remove(p)
            print(f"Removed {p}")
        except OSError as e:
            print(f"Failed to remove {p}: {e}")


if __name__ == "__main__":
    main()
