#!/usr/bin/env python
"""
Minimal E2E smoke test for Patchwork AlphaZero training.

- Runs 1 iteration with tiny sims/games/workers, commits checkpoint.
- Simulates interrupt/resume by running again (auto-resume runs iter 1).
- Asserts schedules changed between iteration 0 and 1.
- Asserts LR at iteration 1 equals scheduled iter_lr (phase boundary).

Usage:
  python -m tools.smoke_e2e
  python -m tools.smoke_e2e --config configs/config_smoke_e2e.yaml

Exit: 0 on pass, 1 on fail.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_REL = "configs/config_smoke_e2e.yaml"


def _run_main(
    config_path: str,
    iterations: int,
    run_id: str = "smoke_e2e",
    allow_config_mismatch: bool = False,
) -> subprocess.CompletedProcess:
    """Run src.training.main with given config and iteration cap."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    cmd = [
        sys.executable,
        "-m",
        "src.training.main",
        "--config",
        str(REPO_ROOT / config_path),
        "--run-id",
        run_id,
        "--iterations",
        str(iterations),
    ]
    if allow_config_mismatch:
        cmd.append("--allow-config-mismatch")
    return subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=300)


def _run_root_from_config(config_path: str, run_id: str) -> Path:
    """Infer run root from config paths (run_root/run_id)."""
    import yaml
    with open(REPO_ROOT / config_path) as f:
        cfg = yaml.safe_load(f)
    run_root = cfg.get("paths", {}).get("run_root", "artifacts/smoke_e2e/runs")
    return REPO_ROOT / run_root / run_id


def _committed_dir(run_root: Path, iteration: int) -> Path:
    return run_root / "committed" / f"iter_{iteration:03d}"


def main() -> int:
    parser = argparse.ArgumentParser(description="E2E smoke: 1 iter + resume + schedule/LR asserts")
    parser.add_argument("--config", type=str, default=CONFIG_REL, help="Config path (relative to repo)")
    parser.add_argument("--run-id", type=str, default="smoke_e2e", help="Run ID for resume")
    args = parser.parse_args()
    config_path = args.config
    run_id = args.run_id
    run_root = _run_root_from_config(config_path, run_id)

    # Clean previous smoke run so we start fresh
    if run_root.exists():
        import shutil
        shutil.rmtree(run_root, ignore_errors=True)

    # Run 1: complete iteration 0 only and commit
    print("Smoke E2E: run 1 — complete iter 0 and commit...")
    r1 = _run_main(config_path, iterations=1, run_id=run_id)
    if r1.returncode != 0:
        print("Run 1 failed:", r1.stdout[-2000:] if len(r1.stdout) > 2000 else r1.stdout)
        print("stderr:", r1.stderr[-1000:] if len(r1.stderr) > 1000 else r1.stderr)
        return 1

    run_state_path = run_root / "run_state.json"
    if not run_state_path.exists():
        print("FAIL: run_state.json not found after run 1")
        return 1
    with open(run_state_path) as f:
        state = json.load(f)
    if state.get("last_committed_iteration") != 0:
        print("FAIL: expected last_committed_iteration=0, got", state.get("last_committed_iteration"))
        return 1

    # Run 2: auto-resume and complete iteration 1 (allow config mismatch: we pass --iterations 2 so hash differs)
    print("Smoke E2E: run 2 — auto-resume and complete iter 1...")
    r2 = _run_main(config_path, iterations=2, run_id=run_id, allow_config_mismatch=True)
    if r2.returncode != 0:
        print("Run 2 failed:", r2.stdout[-2000:] if len(r2.stdout) > 2000 else r2.stdout)
        print("stderr:", r2.stderr[-1000:] if len(r2.stderr) > 1000 else r2.stderr)
        return 1

    with open(run_state_path) as f:
        state = json.load(f)
    if state.get("last_committed_iteration") != 1:
        print("FAIL: expected last_committed_iteration=1 after run 2, got", state.get("last_committed_iteration"))
        return 1

    # Load commit manifests for iter 0 and iter 1
    manifest0_path = _committed_dir(run_root, 0) / "commit_manifest.json"
    manifest1_path = _committed_dir(run_root, 1) / "commit_manifest.json"
    if not manifest0_path.exists() or not manifest1_path.exists():
        print("FAIL: missing commit_manifest for iter 0 or 1")
        return 1
    with open(manifest0_path) as f:
        manifest0 = json.load(f)
    with open(manifest1_path) as f:
        manifest1 = json.load(f)

    applied0 = manifest0.get("applied_settings", {})
    applied1 = manifest1.get("applied_settings", {})

    # Assert schedules changed between iteration 0 and 1
    sp0 = applied0.get("selfplay", {})
    sp1 = applied1.get("selfplay", {})
    tr0 = applied0.get("training", {})
    tr1 = applied1.get("training", {})

    errors = []
    if sp0.get("games") == sp1.get("games"):
        errors.append("games schedule: expected different games for iter 0 vs 1")
    if sp0.get("simulations") == sp1.get("simulations"):
        errors.append("mcts_schedule: expected different simulations for iter 0 vs 1")
    if sp0.get("temperature") == sp1.get("temperature"):
        errors.append("temperature_schedule: expected different temperature for iter 0 vs 1")
    if sp0.get("cpuct") == sp1.get("cpuct"):
        errors.append("cpuct_schedule: expected different cpuct for iter 0 vs 1")

    # Assert LR at iteration 1 equals scheduled iter_lr (0.0005 in config_smoke_e2e)
    lr1 = tr1.get("lr")
    if lr1 is None:
        errors.append("training.lr missing in applied_settings for iter 1")
    else:
        expected_lr1 = 0.0005
        if abs(float(lr1) - expected_lr1) > 1e-7:
            errors.append(f"LR at iter 1: expected {expected_lr1}, got {lr1}")

    if errors:
        for e in errors:
            print("FAIL:", e)
        return 1

    print("Smoke E2E: all checks passed (schedules differ iter0/iter1, LR at boundary = iter_lr).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
