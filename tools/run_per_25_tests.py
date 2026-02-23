#!/usr/bin/env python
"""
Run the per-25-iterations test suite.

1. Per-iteration distributions (legal count, perplexity, entropy)
2. Value stats and calibration

Outputs to --out-dir (e.g. logs/analysis_iter25/). See TESTS_PER_25_ITERS.md for full docs.

Run from repo root:
    python tools/run_per_25_tests.py --config configs/config_best.yaml --run-dir runs/patchwork_production --last-iter 24 --out-dir logs/analysis_iter25
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description="Run per-25-iters analysis suite")
    ap.add_argument("--config", default="configs/config_best.yaml", help="Config file")
    ap.add_argument("--run-dir", type=str, default="runs/patchwork_production", help="Run directory")
    ap.add_argument("--last-iter", type=int, required=True, help="Last iteration (e.g. 24 for 'at iter 25')")
    ap.add_argument("--out-dir", type=str, required=True, help="Output directory (e.g. logs/analysis_iter25)")
    ap.add_argument("--max-samples", type=int, default=20000, help="Max samples for calibration")
    ap.add_argument("--no-calibration", action="store_true", help="Skip value calibration (faster)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    iters = f"0-{args.last_iter}"
    run_dir = args.run_dir
    if not Path(run_dir).is_absolute():
        run_dir = str(REPO_ROOT / run_dir)

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path

    # 1. Distributions (default; no --no-distributions)
    print("=== 1. Per-iteration distributions ===")
    cmd1 = [
        sys.executable,
        str(REPO_ROOT / "tools" / "analyze_iteration_metrics.py"),
        "--config", str(cfg_path),
        "--run-dir", run_dir,
        "--iters", iters,
        "--out-dir", str(out_dir),
        "--no-display",
    ]
    r1 = subprocess.run(cmd1)
    if r1.returncode != 0:
        sys.exit(r1.returncode)

    # 2. Value stats + calibration (skip distributions)
    print("\n=== 2. Value stats and calibration ===")
    cmd2 = [
        sys.executable,
        str(REPO_ROOT / "tools" / "analyze_iteration_metrics.py"),
        "--config", str(cfg_path),
        "--run-dir", run_dir,
        "--iters", iters,
        "--no-distributions",
        "--value-stats",
        "--out-dir", str(out_dir),
        "--max-samples", str(args.max_samples),
    ]
    if not args.no_calibration:
        cmd2.append("--calibration")
    r2 = subprocess.run(cmd2)
    if r2.returncode != 0:
        sys.exit(r2.returncode)

    print(f"\nDone. Outputs in {out_dir}")
    print("  distributions.csv, value_stats.csv, calibration.csv")
    print("  iter_distributions.png")
    if not args.no_calibration:
        print("  calibration_iterN.png")
    print("\nSee TESTS_PER_25_ITERS.md for champion-vs-field eval (run manually).")


if __name__ == "__main__":
    main()
