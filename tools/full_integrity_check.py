#!/usr/bin/env python
"""
Full Integrity Check — One-Command Validation of Entire Patchwork AZ Pipeline

Validates the complete pipeline after changes to bf16 AMP, FiLM, network size,
state channels 14->16, and schedule logic for cpuct and q_value_weight.

CRITICAL SAFETY: All artifacts go under tmp_dir. NEVER overwrites production.
- Overrides run_root, checkpoints_dir, logs_dir, log_file, run_id
- Exit 0 on success, nonzero on failure

USAGE:
    python tools/full_integrity_check.py --config configs/config_best.yaml \\
        --device auto --tmp_dir /tmp/patchwork_integrity --run_slow false --run_e2e true
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
_USE_COLOR = sys.stdout.isatty() and os.name != "nt" or os.environ.get("FORCE_COLOR")


def _green(s: str) -> str:
    return f"\033[92m{s}\033[0m" if _USE_COLOR else s


def _red(s: str) -> str:
    return f"\033[91m{s}\033[0m" if _USE_COLOR else s


def _yellow(s: str) -> str:
    return f"\033[93m{s}\033[0m" if _USE_COLOR else s


# ---------------------------------------------------------------------------
# BF16 Fallback (MUST BE GRACEFUL)
# ---------------------------------------------------------------------------
def resolve_effective_amp(config: dict, device: torch.device) -> tuple[bool, str]:
    """
    Resolve effective AMP and dtype. Returns (use_amp, effective_dtype_str).
    - If config requests amp_dtype=bfloat16:
      - CUDA + is_bf16_supported(): keep bfloat16
      - CUDA but not bf16: fall back to float16
      - CPU: disable AMP (use float32)
    """
    train_cfg = config.get("training", {}) or {}
    use_amp = train_cfg.get("use_amp", False)
    amp_dtype_str = str(train_cfg.get("amp_dtype", "bfloat16")).lower()

    if not use_amp:
        return False, "none"

    if amp_dtype_str == "bfloat16":
        if device.type == "cuda":
            if torch.cuda.is_bf16_supported():
                return True, "bfloat16"
            return True, "float16"
        return False, "float32"

    return use_amp, amp_dtype_str


def apply_amp_fallback(config: dict, use_amp: bool, effective_dtype: str) -> None:
    """Apply effective AMP settings to config in-place."""
    if "training" not in config:
        config["training"] = {}
    config["training"]["use_amp"] = use_amp
    if effective_dtype == "none":
        config["training"]["amp_dtype"] = "float32"
    elif effective_dtype == "float32":
        config["training"]["amp_dtype"] = "float32"
        config["training"]["use_amp"] = False
    else:
        config["training"]["amp_dtype"] = effective_dtype


# ---------------------------------------------------------------------------
# Path Overrides (CRITICAL SAFETY)
# ---------------------------------------------------------------------------
def apply_tmp_dir_overrides(config: dict, tmp_dir: Path, run_id: str = "integrity_smoke") -> None:
    """Override all production paths to tmp_dir. Call BEFORE any training/selfplay/eval."""
    tmp_dir = Path(tmp_dir)
    runs_dir = tmp_dir / "runs"
    ckpt_dir = tmp_dir / "checkpoints"
    logs_dir = tmp_dir / "logs"

    runs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    if "paths" not in config:
        config["paths"] = {}
    config["paths"]["run_root"] = str(runs_dir)
    config["paths"]["checkpoints_dir"] = str(ckpt_dir)
    config["paths"]["logs_dir"] = str(logs_dir)
    config["paths"]["run_id"] = run_id

    if "logging" not in config:
        config["logging"] = {}
    config["logging"]["log_file"] = str(logs_dir / "integrity.log")


# ---------------------------------------------------------------------------
# Step S1 — Compile-all (syntax validation)
# ---------------------------------------------------------------------------
def step_s1_compile_all() -> dict:
    """Run python -m compileall on src/, tools/, tests/. Fail on any syntax error."""
    t0 = time.perf_counter()
    dirs = [str(REPO_ROOT / "src"), str(REPO_ROOT / "tools"), str(REPO_ROOT / "tests")]
    result = subprocess.run(
        [sys.executable, "-m", "compileall", "-q"] + dirs,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    passed = result.returncode == 0
    if passed:
        print(_green("  [PASS] compileall (src/, tools/, tests/)"))
    else:
        # Show compile errors (often on stderr or stdout)
        err_msg = (result.stderr or result.stdout or "").strip()
        if err_msg:
            print(_red(f"  [FAIL] compileall:\n{err_msg[:800]}"))
        else:
            print(_red(f"  [FAIL] compileall: syntax error in one or more files (exit {result.returncode})"))
    return {
        "step": "S1_compileall",
        "pass": passed,
        "duration_s": time.perf_counter() - t0,
        "returncode": result.returncode,
    }


# ---------------------------------------------------------------------------
# Step S2 — Import sweep (best-effort)
# ---------------------------------------------------------------------------
# Modules allowed to fail import (optional deps, platform-specific, etc.)
IMPORT_SWEEP_ALLOWLIST: frozenset[str] = frozenset()


def step_s2_import_sweep() -> dict:
    """Import all Python modules under src/. Fail on import errors unless in allowlist."""
    t0 = time.perf_counter()
    src_root = REPO_ROOT / "src"
    modules: list[str] = []
    for path in sorted(src_root.rglob("*.py")):
        if "__pycache__" in path.parts or path.name.startswith("test_"):
            continue
        rel = path.relative_to(REPO_ROOT)
        mod_name = str(rel).replace(os.sep, ".").replace(".py", "")
        modules.append(mod_name)

    ok: list[str] = []
    failed: list[tuple[str, str]] = []
    for mod in modules:
        try:
            importlib.import_module(mod)
            ok.append(mod)
        except Exception as e:
            failed.append((mod, str(e)))
    allowed_failures = [(m, err) for m, err in failed if m in IMPORT_SWEEP_ALLOWLIST]
    unexpected_failures = [(m, err) for m, err in failed if m not in IMPORT_SWEEP_ALLOWLIST]
    passed = len(unexpected_failures) == 0

    print(f"  Modules imported: {len(ok)} ok, {len(allowed_failures)} allowed failures, {len(unexpected_failures)} unexpected")
    if unexpected_failures:
        for m, err in unexpected_failures[:5]:
            print(_red(f"    FAIL {m}: {err[:120]}"))
        if len(unexpected_failures) > 5:
            print(_red(f"    ... and {len(unexpected_failures) - 5} more"))
        print(_red("  [FAIL] Import sweep"))
    else:
        print(_green("  [PASS] Import sweep"))

    return {
        "step": "S2_import_sweep",
        "pass": passed,
        "duration_s": time.perf_counter() - t0,
        "imported_ok": len(ok),
        "allowed_failures": len(allowed_failures),
        "unexpected_failures": len(unexpected_failures),
        "failed_modules": [m for m, _ in unexpected_failures],
    }


# ---------------------------------------------------------------------------
# Step 0 — Environment + Config Sanity
# ---------------------------------------------------------------------------
def step0_environment_config(config_path: str, tmp_dir: Path, device_str: str) -> tuple[dict, dict]:
    """
    Load config, apply tmp_dir overrides, validate assertions, resolve AMP.
    Returns (config, report_dict).
    """
    t0 = time.perf_counter()
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    apply_tmp_dir_overrides(config, tmp_dir)

    device = torch.device("cpu")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    use_amp, effective_dtype = resolve_effective_amp(config, device)
    apply_amp_fallback(config, use_amp, effective_dtype)

    print(f"  Device chosen:        {device}")
    print(f"  PyTorch version:       {torch.__version__}")
    print(f"  CUDA available:       {torch.cuda.is_available()}")
    bf16_ok = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
    print(f"  BF16 supported:       {bf16_ok}")
    print(f"  AMP enabled:          {use_amp}")
    print(f"  Effective amp dtype:  {effective_dtype}")

    # Assertions (56 channels: gold_v2)
    net = config.get("network", {})
    assert net.get("input_channels") == 56, f"network.input_channels must be 56 (gold_v2), got {net.get('input_channels')}"

    report = {
        "step": "0_environment",
        "pass": True,
        "duration_s": time.perf_counter() - t0,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "bf16_supported": bf16_ok,
        "amp_enabled": use_amp,
        "effective_amp_dtype": effective_dtype,
    }
    return config, report


# ---------------------------------------------------------------------------
# Step 1 — Unit Tests (pytest)
# ---------------------------------------------------------------------------
def step1_unit_tests(run_slow: bool) -> dict:
    """Run pytest -q -m 'not slow'; if run_slow also run -m 'slow'."""
    t0 = time.perf_counter()
    passed = True
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-m", "not slow"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    fast_ok = result.returncode == 0
    if not fast_ok:
        passed = False
        print(_red(f"  [FAIL] pytest fast tests: {result.stderr[:500]}"))
    else:
        print(_green("  [PASS] pytest -m 'not slow'"))

    slow_ok = True
    if run_slow:
        result_slow = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-m", "slow"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=600,
        )
        slow_ok = result_slow.returncode == 0
        if not slow_ok:
            passed = False
            print(_red(f"  [FAIL] pytest slow tests"))
        else:
            print(_green("  [PASS] pytest -m 'slow'"))

    return {
        "step": "1_unit_tests",
        "pass": passed,
        "duration_s": time.perf_counter() - t0,
        "fast_ok": fast_ok,
        "slow_ok": slow_ok,
    }


# ---------------------------------------------------------------------------
# Step 2 — deep_preflight
# ---------------------------------------------------------------------------
def step2_deep_preflight(config_path: str, tmp_dir: Path, device_str: str, run_e2e: bool) -> dict:
    """Run deep_preflight Steps A–D; if run_e2e, also Step E."""
    t0 = time.perf_counter()
    from tools.deep_preflight import run_all_steps

    checklist, failed_step = run_all_steps(
        config_path,
        device_str,
        str(tmp_dir),
        skip_e=not run_e2e,
        run_f=False,
    )
    passed = failed_step is None
    for name, (status, val) in checklist.items():
        if status == "PASS":
            t_str = f" ({val:.3f}s)" if isinstance(val, (int, float)) else ""
            print(_green(f"  [PASS] {name}{t_str}"))
        elif status == "FAIL":
            print(_red(f"  [FAIL] {name}: {val}"))
        else:
            print(f"  [SKIP] {name}")

    return {
        "step": "2_deep_preflight",
        "pass": passed,
        "duration_s": time.perf_counter() - t0,
        "failed_step": failed_step,
        "checklist": {k: v[0] for k, v in checklist.items()},
    }


# ---------------------------------------------------------------------------
# Step 3 — Schedule Propagation Verification
# ---------------------------------------------------------------------------
def step3_schedule_verification(config: dict, tmp_dir: Path) -> dict:
    """Verify _apply_iteration_schedules produces correct values at breakpoints."""
    t0 = time.perf_counter()

    config_path = tmp_dir / "integrity_config_step3.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    from src.training.main import AlphaZeroTrainer, _get_window_iterations_for_iteration

    trainer = AlphaZeroTrainer(str(config_path), cli_run_dir=str(tmp_dir / "runs" / "integrity_schedule"))
    trainer.run_root = tmp_dir / "runs" / "integrity_schedule"
    trainer.run_id = "integrity_schedule"

    iterations = [0, 60, 120, 200]
    iter_cfg = config.get("iteration", {}) or {}
    q_sched = iter_cfg.get("q_value_weight_schedule", [])
    cpuct_sched = iter_cfg.get("cpuct_schedule", [])
    win_sched = iter_cfg.get("window_iterations_schedule", [])
    lock_eval = config.get("evaluation", {}).get("lock_eval_cpuct_to_selfplay", True)

    def expected_q(iter_val: int) -> float:
        base = config.get("selfplay", {}).get("q_value_weight", 0.4)
        for e in sorted(q_sched, key=lambda x: x["iteration"], reverse=True):
            if iter_val >= e["iteration"]:
                return max(0.0, min(1.0, float(e.get("q_value_weight", base))))
        return base

    def expected_cpuct(iter_val: int) -> float:
        base = config.get("selfplay", {}).get("mcts", {}).get("cpuct", 1.5)
        for e in sorted(cpuct_sched, key=lambda x: x["iteration"], reverse=True):
            if iter_val >= e["iteration"]:
                return max(0.1, min(5.0, float(e.get("cpuct", base))))
        return base

    def expected_window(iter_val: int) -> int:
        base = config.get("replay_buffer", {}).get("window_iterations", 5)
        if not win_sched:
            return int(base)
        for e in sorted(win_sched, key=lambda x: x["iteration"], reverse=True):
            if iter_val >= e["iteration"]:
                return int(e.get("window_iterations", base))
        return int(base)

    table = []
    for it in iterations:
        trainer._apply_iteration_schedules(it)
        # window_size is applied separately (main does it at iteration start); must sync for this test
        trainer.replay_buffer.window_size = _get_window_iterations_for_iteration(trainer.config, it)
        q = trainer.config["selfplay"]["q_value_weight"]
        cpuct = trainer.config["selfplay"]["mcts"]["cpuct"]
        exp_q = expected_q(it)
        exp_cpuct = expected_cpuct(it)
        assert abs(q - exp_q) < 1e-6, f"iter {it}: q_value_weight {q} != expected {exp_q}"
        assert abs(cpuct - exp_cpuct) < 1e-6, f"iter {it}: cpuct {cpuct} != expected {exp_cpuct}"
        if lock_eval:
            eval_cpuct = trainer.config["evaluation"]["eval_mcts"]["cpuct"]
            assert abs(eval_cpuct - cpuct) < 1e-6, f"iter {it}: lock_eval but eval cpuct {eval_cpuct} != selfplay {cpuct}"
        win = trainer.replay_buffer.window_size
        exp_win = expected_window(it)
        assert win == exp_win, f"iter {it}: window_size {win} != expected {exp_win}"
        table.append((it, q, cpuct, win))

    print("  Iteration -> effective cpuct / q_value_weight / window_iterations:")
    for row in table:
        it, q, c = row[0], row[1], row[2]
        win = row[3] if len(row) > 3 else "-"
        print(f"    iter {it:3d}: cpuct={c:.3f}  q_value_weight={q:.3f}  window_iterations={win}")
    print(_green("  [PASS] Schedule propagation verified"))

    return {
        "step": "3_schedule",
        "pass": True,
        "duration_s": time.perf_counter() - t0,
        "table": [{"iter": r[0], "cpuct": r[1], "q_value_weight": r[2], "window_iterations": r[3] if len(r) > 3 else None} for r in table],
    }


# ---------------------------------------------------------------------------
# Step 4 — Mini End-to-End Smoke (2 iterations)
# ---------------------------------------------------------------------------
def step4_mini_e2e(config: dict, tmp_dir: Path, device_str: str = "auto") -> dict:
    """Run 2 iterations through production entrypoint with overrides."""
    t0 = time.perf_counter()

    cfg = copy.deepcopy(config)
    cfg["iteration"]["auto_resume"] = False
    cfg["iteration"]["max_iterations"] = 2
    cfg["iteration"]["games_schedule"] = [{"iteration": 0, "games": 8}]
    cfg["iteration"]["window_iterations_schedule"] = [{"iteration": 0, "window_iterations": 8}]
    cfg["selfplay"]["games_per_iteration"] = 8
    # CPU-only: use 1 worker (local mode). GPU: up to 4 for speed
    if device_str == "cpu" or (device_str == "auto" and not torch.cuda.is_available()):
        cfg["hardware"]["device"] = "cpu"
        cfg["selfplay"]["num_workers"] = 1
    else:
        cfg["selfplay"]["num_workers"] = min(cfg["selfplay"].get("num_workers", 1), 4)
    cfg["selfplay"]["bootstrap"]["games"] = 8
    cfg["selfplay"]["mcts"]["simulations"] = 16
    pl = cfg["selfplay"]["mcts"].get("parallel_leaves", 24)
    cfg["selfplay"]["mcts"]["parallel_leaves"] = min(pl, 8)
    cfg["training"]["epochs_per_iteration"] = 1
    cfg["evaluation"]["games_vs_best"] = 8
    cfg["paths"]["run_id"] = "integrity_smoke"
    cfg["paths"]["run_root"] = str(tmp_dir / "runs")
    cfg["paths"]["checkpoints_dir"] = str(tmp_dir / "checkpoints")
    cfg["paths"]["logs_dir"] = str(tmp_dir / "logs")
    cfg["logging"]["log_file"] = str(tmp_dir / "logs" / "integrity.log")
    (tmp_dir / "runs").mkdir(parents=True, exist_ok=True)
    (tmp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (tmp_dir / "logs").mkdir(parents=True, exist_ok=True)

    config_path = tmp_dir / "integrity_e2e_config.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    from src.training.main import AlphaZeroTrainer

    trainer = AlphaZeroTrainer(str(config_path), cli_run_dir=str(tmp_dir / "runs" / "integrity_smoke"))
    trainer.train(start_iteration=0, resume_checkpoint=None)

    ckpt_dir = tmp_dir / "checkpoints"
    best_pt = ckpt_dir / "best_model.pt"
    latest_pt = ckpt_dir / "latest_model.pt"
    assert ckpt_dir.exists(), "checkpoints dir must exist"
    assert best_pt.exists() or latest_pt.exists(), "At least one checkpoint must exist"
    assert str(ckpt_dir.resolve()).startswith(str(Path(tmp_dir).resolve())), "Checkpoints must be under tmp_dir"

    run_root = tmp_dir / "runs" / "integrity_smoke"
    committed = run_root / "committed"
    h5_files = list(committed.glob("**/selfplay.h5")) if committed.exists() else []
    if h5_files:
        import h5py
        with h5py.File(h5_files[0], "r") as f:
            states_key = "spatial_states" if "spatial_states" in f else "states"
            states = f[states_key][:]
            masks = f["action_masks"][:]
        from src.network.gold_v2_constants import C_SPATIAL
        assert states.shape[1:] == (C_SPATIAL, 9, 9), f"states shape should be (N,{C_SPATIAL},9,9), got {states.shape}"
        assert masks.shape[1] == 2026, f"masks shape should be (N,2026), got {masks.shape}"
        assert (masks.sum(axis=1) > 0).all(), "Every mask must have sum > 0"

    from src.network.model import create_network
    net = create_network(cfg)
    ckpt_path = latest_pt if latest_pt.exists() else best_pt
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    from src.network.model import load_model_checkpoint
    load_model_checkpoint(net, ckpt["model_state_dict"])
    net.eval()
    from tools.deep_preflight import _dummy_gold_v2_batch
    st, xg, xt, si, sf = _dummy_gold_v2_batch(2, torch.device("cpu"))
    dummy_m = torch.ones(2, 2026)
    with torch.no_grad():
        pl, v, _ = net.forward(st, dummy_m, x_global=xg, x_track=xt, shop_ids=si, shop_feats=sf)
    assert torch.isfinite(pl).any(dim=1).all(), "Forward must produce at least one finite logit per sample"
    assert torch.isfinite(v).all(), "Forward values must be finite"

    print(_green("  [PASS] Mini E2E smoke (2 iterations)"))
    return {
        "step": "4_mini_e2e",
        "pass": True,
        "duration_s": time.perf_counter() - t0,
        "checkpoints_under_tmp": True,
    }


# ---------------------------------------------------------------------------
# Step 5 — Checkpoint Cycle
# ---------------------------------------------------------------------------
def step5_checkpoint_cycle(tmp_dir: Path, config: dict) -> dict:
    """Verify at least one checkpoint exists, load and run forward."""
    t0 = time.perf_counter()
    ckpt_dir = tmp_dir / "checkpoints"
    best_pt = ckpt_dir / "best_model.pt"
    latest_pt = ckpt_dir / "latest_model.pt"
    ckpt_path = latest_pt if latest_pt.exists() else best_pt
    assert ckpt_path.exists(), "No checkpoint found under tmp_dir/checkpoints"

    from src.network.model import create_network, load_model_checkpoint
    net = create_network(config)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    load_model_checkpoint(net, ckpt["model_state_dict"])
    net.eval()
    from tools.deep_preflight import _dummy_gold_v2_batch
    st, xg, xt, si, sf = _dummy_gold_v2_batch(2, torch.device("cpu"))
    dummy_m = torch.ones(2, 2026)
    with torch.no_grad():
        pl, v, _ = net.forward(st, dummy_m, x_global=xg, x_track=xt, shop_ids=si, shop_feats=sf)
    assert torch.isfinite(pl).any(dim=1).all(), "Each sample must have at least one finite policy logit"
    assert torch.isfinite(v).all(), "Value outputs must be finite"

    print(_green("  [PASS] Checkpoint cycle verified"))
    return {
        "step": "5_checkpoint_cycle",
        "pass": True,
        "duration_s": time.perf_counter() - t0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_full_integrity(
    config_path: str,
    tmp_dir: str,
    device: str = "auto",
    run_slow: bool = False,
    run_e2e: bool = True,
) -> dict:
    """Run all integrity steps. Returns report dict. Raises on failure."""
    tmp = Path(tmp_dir)
    tmp.mkdir(parents=True, exist_ok=True)

    report = {"steps": [], "overall_pass": True, "timings": {}}

    print("\n" + "=" * 60)
    print("FULL INTEGRITY CHECK — Patchwork AZ Pipeline")
    print("=" * 60)

    # Step S1 — Compile-all
    print("\n[Step S1] Compile-all (syntax validation)")
    r_s1 = step_s1_compile_all()
    report["steps"].append(r_s1)
    if not r_s1["pass"]:
        report["overall_pass"] = False
    report["timings"]["step_s1"] = r_s1["duration_s"]

    # Step S2 — Import sweep
    print("\n[Step S2] Import sweep")
    r_s2 = step_s2_import_sweep()
    report["steps"].append(r_s2)
    if not r_s2["pass"]:
        report["overall_pass"] = False
    report["timings"]["step_s2"] = r_s2["duration_s"]

    # Step 0
    print("\n[Step 0] Environment + Config Sanity")
    config, r0 = step0_environment_config(config_path, tmp, device)
    report["steps"].append(r0)
    report["config"] = {k: v for k, v in config.items() if k not in ("paths", "logging")}
    report["timings"]["step0"] = r0["duration_s"]

    # Step 1
    print("\n[Step 1] Unit Tests (pytest)")
    r1 = step1_unit_tests(run_slow)
    report["steps"].append(r1)
    if not r1["pass"]:
        report["overall_pass"] = False
    report["timings"]["step1"] = r1["duration_s"]

    # Step 2 — config already has tmp_dir overrides and AMP fallback from Step 0
    print("\n[Step 2] Deep Preflight")
    cfg_path = tmp / "integrity_deep_preflight_config.yaml"
    cfg_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    r2 = step2_deep_preflight(str(cfg_path), tmp, device, run_e2e)
    report["steps"].append(r2)
    if not r2["pass"]:
        report["overall_pass"] = False
    report["timings"]["step2"] = r2["duration_s"]

    # Step 3
    print("\n[Step 3] Schedule Propagation Verification")
    r3 = step3_schedule_verification(config, tmp)
    report["steps"].append(r3)
    report["timings"]["step3"] = r3["duration_s"]

    # Step 4 (only if run_e2e)
    if run_e2e:
        print("\n[Step 4] Mini End-to-End Smoke (2 iterations)")
        try:
            r4 = step4_mini_e2e(config, tmp, device)
            report["steps"].append(r4)
            report["timings"]["step4"] = r4["duration_s"]
        except Exception as e:
            report["steps"].append({"step": "4_mini_e2e", "pass": False, "error": str(e)})
            report["overall_pass"] = False
            raise

        # Step 5
        print("\n[Step 5] Checkpoint Cycle")
        r5 = step5_checkpoint_cycle(tmp, config)
        report["steps"].append(r5)
        report["timings"]["step5"] = r5["duration_s"]
    else:
        report["steps"].append({"step": "4_mini_e2e", "pass": True, "skipped": True})
        report["steps"].append({"step": "5_checkpoint_cycle", "pass": True, "skipped": True})

    # Summary
    report_path = tmp / "integrity_report.json"
    safe_report = {}
    for k, v in report.items():
        if k == "config":
            continue
        if isinstance(v, (dict, list, str, int, float, bool, type(None))):
            try:
                json.dumps(v)
                safe_report[k] = v
            except TypeError:
                safe_report[k] = str(v)
    report_path.write_text(json.dumps(safe_report, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    if report["overall_pass"]:
        print(_green("FULL INTEGRITY CHECK: PASS"))
    else:
        print(_red("FULL INTEGRITY CHECK: FAIL"))
    print(f"Report: {report_path}")
    print("=" * 60 + "\n")

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Full integrity check — validate entire Patchwork AZ pipeline")
    parser.add_argument("--config", required=True, help="Config YAML path")
    parser.add_argument("--device", default="auto", help="Device: auto|cpu|cuda")
    parser.add_argument("--tmp_dir", required=True, help="Temp directory for all artifacts (never touches production)")
    parser.add_argument("--run_slow", default="false", help="If true, also run pytest -m slow")
    parser.add_argument("--run_e2e", default="true", help="If true, run Steps 4–5 (mini E2E smoke)")
    args = parser.parse_args()

    run_slow = str(args.run_slow).lower() in ("1", "true", "yes")
    run_e2e = str(args.run_e2e).lower() in ("1", "true", "yes")

    try:
        report = run_full_integrity(
            args.config,
            args.tmp_dir,
            device=args.device,
            run_slow=run_slow,
            run_e2e=run_e2e,
        )
        return 0 if report["overall_pass"] else 1
    except Exception as e:
        print(_red(f"\nFULL INTEGRITY CHECK FAILED: {e}"))
        return 1


if __name__ == "__main__":
    sys.exit(main())
