"""
quick_diag/run_diag.py
======================
Launcher for the KL diagnostic test suite.

Runs scaled-down training (iter 0-6) with flat vs hierarchical policy heads
to determine whether the hierarchical factorization is the primary driver of
elevated KL divergence in the production run.

USAGE (from project root):
    python quick_diag/run_diag.py --mode flat
    python quick_diag/run_diag.py --mode hier
    python quick_diag/run_diag.py --mode both    # flat first, then hier
    python quick_diag/run_diag.py --mode both --no-clean  # keep prior runs
    python quick_diag/run_diag.py --report       # print report only (no run)

OPTIONS:
    --mode flat|hier|both   Which config(s) to run (default: both)
    --no-clean              Skip wiping prior run directories before starting
    --report                Print comparison report after run completes
    --dry-run               Print the commands that would be run, do not execute
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIAG_DIR     = PROJECT_ROOT / "quick_diag"

CONFIGS = {
    "flat": DIAG_DIR / "config_flat.yaml",
    "hier": DIAG_DIR / "config_hier.yaml",
}

# Directories to wipe before a fresh run (keeps iteration data pristine)
CLEAN_TARGETS = {
    "flat": [
        PROJECT_ROOT / "quick_diag" / "runs"          / "qd_flat",
        PROJECT_ROOT / "quick_diag" / "checkpoints_flat",
        PROJECT_ROOT / "quick_diag" / "logs_flat",
        PROJECT_ROOT / "quick_diag" / "data_flat",
    ],
    "hier": [
        PROJECT_ROOT / "quick_diag" / "runs"          / "qd_hier",
        PROJECT_ROOT / "quick_diag" / "checkpoints_hier",
        PROJECT_ROOT / "quick_diag" / "logs_hier",
        PROJECT_ROOT / "quick_diag" / "data_hier",
    ],
}


def wipe(mode: str) -> None:
    for target in CLEAN_TARGETS[mode]:
        if target.exists():
            print(f"  [clean] removing {target.relative_to(PROJECT_ROOT)}")
            shutil.rmtree(target)


def run_training(mode: str, dry_run: bool = False) -> int:
    cfg = CONFIGS[mode]
    cmd = [sys.executable, "-m", "src.training.main", "--config", str(cfg)]
    print(f"\n{'='*60}")
    print(f"  RUNNING: qd_{mode}  ({cfg.name})")
    print(f"  CMD:     {' '.join(cmd)}")
    print(f"{'='*60}\n")

    if dry_run:
        print("  [dry-run] skipping execution")
        return 0

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="KL diagnostic runner — flat vs hierarchical policy head"
    )
    parser.add_argument(
        "--mode",
        choices=["flat", "hier", "both"],
        default="both",
        help="Which policy head variant to run (default: both)",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not wipe prior run directories (use for resuming)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print comparison report after training completes",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    args = parser.parse_args()

    modes = ["flat", "hier"] if args.mode == "both" else [args.mode]

    for mode in modes:
        if not args.no_clean:
            print(f"\nCleaning prior {mode} run data...")
            wipe(mode)

        rc = run_training(mode, dry_run=args.dry_run)
        if rc != 0:
            print(f"\n[ERROR] Training exited with code {rc} for mode='{mode}'")
            sys.exit(rc)

    if args.report or args.mode == "both":
        print("\n\nGenerating comparison report...")
        report_script = DIAG_DIR / "report.py"
        subprocess.run([sys.executable, str(report_script)], cwd=str(PROJECT_ROOT))


if __name__ == "__main__":
    main()
