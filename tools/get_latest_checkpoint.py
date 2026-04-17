#!/usr/bin/env python3
"""Print the path to the latest checkpoint (highest committed iter, or checkpoints/latest_model.pt).
Used by launch_gui.bat to always use the latest model."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUN_ROOT = REPO / "runs" / "patchwork_production"
COMMITTED = RUN_ROOT / "committed"
FALLBACK = REPO / "checkpoints" / "latest_model.pt"


def main() -> None:
    if not COMMITTED.exists():
        print(str(FALLBACK))
        return
    iters = []
    for d in COMMITTED.iterdir():
        if d.is_dir() and d.name.startswith("iter_"):
            try:
                n = int(d.name.split("_")[1])
                ckpt = d / f"iteration_{n:03d}.pt"
                if ckpt.exists():
                    iters.append((n, str(ckpt)))
            except (ValueError, IndexError):
                continue
    if iters:
        iters.sort(key=lambda x: -x[0])
        print(iters[0][1])
    else:
        print(str(FALLBACK))


if __name__ == "__main__":
    main()
    sys.exit(0)
