#!/usr/bin/env python
"""
Resume plan: transparent display of what the next training launch will do.

Prints:
  - last_committed_iteration
  - next_iteration_to_run
  - which checkpoint will be used for selfplay (best_model vs resume semantics)
  - whether staging will be deleted (and why)

USAGE:
    python tools/resume_plan.py --run-dir runs/<run_id>
    python tools/resume_plan.py --run-dir runs/patchwork_overnight --config configs/config_best.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml

from src.training.run_layout import (
    get_staging_cleanup_plan,
    max_committed_iteration,
    reconcile_run_state,
)


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Show what the next training launch will do (resume plan)"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Run directory (e.g. runs/patchwork_overnight)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_best.yaml",
        help="Config path (for staging preservation logic)",
    )
    args = parser.parse_args()

    run_root = Path(args.run_dir)
    if not run_root.is_absolute():
        run_root = REPO_ROOT / run_root

    run_state_path = run_root / "run_state.json"
    if not run_state_path.exists():
        print(f"ERROR: {run_state_path} not found. No run to resume.")
        return 1

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    config = load_config(config_path) if config_path.exists() else None

    with open(run_state_path, encoding="utf-8-sig") as f:
        state = json.load(f)

    last_from_state = state.get("last_committed_iteration")
    if last_from_state is not None:
        last_from_state = int(last_from_state)
    else:
        next_it = int(state.get("next_iteration", 0))
        last_from_state = max(-1, next_it - 1)

    last_from_fs = max_committed_iteration(run_root)
    last_comm, needs_repair = reconcile_run_state(run_root, last_from_state)
    next_iter = last_comm + 1

    best_path = state.get("best_model_path")
    latest_path = state.get("latest_model_path")
    latest_ckpt = state.get("latest_checkpoint") or latest_path
    selfplay_checkpoint = best_path or latest_path or latest_ckpt
    best_iter = state.get("best_iteration") or state.get("best_model_iteration")

    # Checkpoint source: best_model > latest_model > explicit resume
    if best_path and Path(best_path).exists():
        ckpt_source = "best_model"
    elif latest_path and Path(latest_path).exists():
        ckpt_source = "latest_model"
    elif latest_ckpt and Path(latest_ckpt).exists():
        ckpt_source = "latest_checkpoint"
    else:
        ckpt_source = "N/A (checkpoint missing)"

    plan = get_staging_cleanup_plan(run_root, last_comm, config)
    staging_deletes = [(i, r) for (i, a, r) in plan if a == "delete"]
    staging_preserves = [(i, r) for (i, a, r) in plan if a == "preserve" and "already committed" not in r]

    print()
    print("=" * 60)
    print("  RESUME PLAN")
    print("=" * 60)
    print(f"  run_dir                           : {run_root}")
    print(f"  filesystem_last_committed_iteration: {last_from_fs}")
    print(f"  run_state_last_committed_iteration : {last_from_state}")
    reconcile_will_advance = last_from_fs > last_from_state
    print(f"  reconcile_run_state will advance  : {reconcile_will_advance}")
    if reconcile_will_advance:
        print(f"    -> run_state advances to iter{last_from_fs:03d} (filesystem ahead)")
    print(f"  next_iteration_to_run             : {next_iter}")
    print(f"  checkpoint_for_selfplay           : {ckpt_source}")
    if selfplay_checkpoint:
        print(f"    path                            : {selfplay_checkpoint}")
    best_iter_display = best_iter if best_iter is not None else "N/A"
    train_base_iter = best_iter if (best_path and Path(best_path).exists()) else last_comm
    print(f"  best_model_origin_iteration       : {best_iter_display}")
    print(f"  train_base (model weights source) : same as checkpoint_for_selfplay")
    print(f"  optimizer_state_source_iteration : {train_base_iter} (must match train_base)")
    print(f"  optimizer_matches_train_base      : True (enforced at runtime)")
    print()
    if needs_repair:
        print()
        print("  [REPAIR] run_state will be reconciled with filesystem (crash-after-move edge case)")
    print()
    if staging_deletes:
        print("  Staging WILL BE DELETED (and why):")
        for iter_num, reason in staging_deletes:
            print(f"    iter{iter_num:03d}: {reason}")
    else:
        print("  staging_deleted: no (no partial iterations)")
    if staging_preserves:
        print()
        print("  Staging PRESERVED (resume will skip selfplay for these):")
        for iter_num, reason in staging_preserves:
            print(f"    iter{iter_num:03d}: {reason}")
    print()
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
