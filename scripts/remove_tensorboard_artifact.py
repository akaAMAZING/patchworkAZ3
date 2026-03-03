#!/usr/bin/env python3
"""
Remove TensorBoard "snap to origin" artifact from existing logs.

The artifact is caused by one or more event files that contain train/val scalars
with step values 0, 10, 20, ... (from an iteration that logged with wrong global_step)
after the run had already reached high steps (e.g. 22k+). This script finds those
event files and moves them to a backup folder so charts show a single continuous
step sequence.

Usage:
  python scripts/remove_tensorboard_artifact.py [path_to_tensorboard_dir]
  Default: logs/tensorboard

Backup: logs/tensorboard_artifact_backup/
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TB = ROOT / "logs" / "tensorboard"
BACKUP_SUFFIX = "_artifact_backup"

# Only step-based tags can produce the artifact (train/*, val/*)
STEP_TAGS_PREFIX = ("train/", "val/")

# Once we've seen steps above this, a file with max_step below (global_max - THRESHOLD) is artifact
MIN_GLOBAL_BEFORE_ARTIFACT = 10_000
STEP_GAP_THRESHOLD = 5_000


def get_max_step_for_file(event_path: Path) -> int | None:
    """Return max step across train/ and val/ tags in this event file, or None if unreadable."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return None
    try:
        ea = EventAccumulator(str(event_path))
        ea.Reload()
        max_step = -1
        for tag in ea.Tags().get("scalars", []):
            if not any(tag.startswith(p) for p in STEP_TAGS_PREFIX):
                continue
            for e in ea.Scalars(tag):
                if e.step > max_step:
                    max_step = int(e.step)
        return max_step if max_step >= 0 else None
    except Exception:
        return None


def collect_event_files(tb_dir: Path) -> list[Path]:
    """List event files, sorted by modification time (oldest first)."""
    if not tb_dir.exists():
        return []
    files = list(tb_dir.glob("events.out.tfevents.*"))
    return sorted(files, key=lambda p: p.stat().st_mtime)


def find_artifact_files(tb_dir: Path) -> list[Path]:
    """
    Find event files that contain low-step train/val data written after high-step data.
    Those are the ones causing the chart to "snap to origin".
    """
    files = collect_event_files(tb_dir)
    if not files:
        return []

    # (path, max_step) in time order
    file_max_steps: list[tuple[Path, int | None]] = []
    for p in files:
        ms = get_max_step_for_file(p)
        file_max_steps.append((p, ms))

    global_max = 0
    artifact_paths: list[Path] = []
    for path, max_step in file_max_steps:
        if max_step is not None and max_step > global_max:
            global_max = max_step
        # File has step-based tags but max step is way behind what we've already seen
        if (
            max_step is not None
            and global_max >= MIN_GLOBAL_BEFORE_ARTIFACT
            and max_step < global_max - STEP_GAP_THRESHOLD
        ):
            artifact_paths.append(path)
    return artifact_paths


def main() -> None:
    tb_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TB
    tb_dir = tb_dir.resolve()
    if not tb_dir.exists():
        print(f"Directory does not exist: {tb_dir}")
        sys.exit(1)

    artifact_files = find_artifact_files(tb_dir)
    if not artifact_files:
        print("No artifact event files found. Charts are already clean or path has no events.")
        return

    backup_dir = tb_dir.parent / (tb_dir.name + BACKUP_SUFFIX)
    backup_dir.mkdir(parents=True, exist_ok=True)
    print(f"Moving {len(artifact_files)} artifact event file(s) to {backup_dir}")
    for p in artifact_files:
        dest = backup_dir / p.name
        # Avoid overwrite: if dest exists, use a numeric suffix
        if dest.exists():
            n = 0
            while dest.exists():
                n += 1
                dest = backup_dir / f"{p.stem}.{n}{p.suffix}" if p.suffix else backup_dir / f"{p.stem}.{n}"
        p.rename(dest)
        print(f"  {p.name} -> {backup_dir.name}/")
    print("Done. Restart TensorBoard or refresh the page to see clean charts.")
    print(f"(To restore artifact files, move them back from {backup_dir})")


if __name__ == "__main__":
    main()
