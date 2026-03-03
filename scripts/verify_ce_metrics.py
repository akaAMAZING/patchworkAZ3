#!/usr/bin/env python3
"""
Sanity verification for CE/entropy metrics:
- Run one epoch of training with ce_identity_debug_steps=10.
- Verify returned avg_metrics contains new keys.
- Print first ~10 [CE_DEBUG] lines from captured log.

This script does NOT modify runs/.../committed; all outputs go to a scratch
directory (runs/_scratch/verify_ce_metrics_* or system temp).
"""
import sys
import logging
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED_DIR = REPO_ROOT / "runs" / "patchwork_production" / "committed"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml
import torch

# Capture [CE_DEBUG] lines
ce_debug_lines = []


class CEDebugFilter(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        if "[CE_DEBUG]" in msg:
            ce_debug_lines.append(msg)


def main():
    # Load config
    config_path = REPO_ROOT / "configs" / "config_best.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["training"]["ce_identity_debug_steps"] = 10
    config["training"]["epochs_per_iteration"] = 1  # one epoch only

    # Attach handler to capture CE_DEBUG (trainer uses src.training.trainer logger)
    trainer_logger = logging.getLogger("src.training.trainer")
    handler = CEDebugFilter()
    handler.setLevel(logging.INFO)
    trainer_logger.addHandler(handler)
    trainer_logger.setLevel(logging.INFO)

    # Paths: read-only use of committed for data and checkpoint (warm start)
    data_path = str(COMMITTED_DIR / "iter_044" / "selfplay.h5")
    prev_ckpt = str(COMMITTED_DIR / "iter_043" / "iteration_043.pt")
    if not Path(data_path).exists():
        data_path = str(COMMITTED_DIR / "iter_036" / "selfplay.h5")
        prev_ckpt = str(COMMITTED_DIR / "iter_035" / "iteration_035.pt") if (COMMITTED_DIR / "iter_035" / "iteration_035.pt").exists() else None
    if not Path(data_path).exists():
        print("No committed selfplay.h5 found; skipping training run.")
        return 1
    if prev_ckpt and not Path(prev_ckpt).exists():
        prev_ckpt = None

    # Guard: never write into committed; use scratch dir only
    scratch_base = REPO_ROOT / "runs" / "_scratch"
    out_dir = (scratch_base / "verify_ce_metrics_run").resolve()
    if COMMITTED_DIR.exists():
        committed_resolved = str(COMMITTED_DIR.resolve())
        out_str = str(out_dir)
        if out_str.startswith(committed_resolved) or out_str == committed_resolved:
            raise RuntimeError("Guard: output dir must not be under runs/.../committed.")
    out_dir.mkdir(parents=True, exist_ok=True)
    iteration_output_dir = out_dir / "staging_iter0"
    iteration_output_dir.mkdir(parents=True, exist_ok=True)

    print("Note: This script does not modify runs/.../committed; all outputs go to a scratch directory.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from src.training.trainer import train_iteration

    print("Running one training epoch (ce_identity_debug_steps=10)...")
    checkpoint_path, avg_metrics, global_step, last_lr = train_iteration(
        0,
        data_path,
        config,
        device,
        previous_checkpoint=prev_ckpt,
        replay_buffer=None,
        writer=None,
        global_step_offset=0,
        iteration_output_dir=iteration_output_dir,
        merged_output_path=None,
        force_resume_optimizer_state=False,
        force_resume_scheduler_state=False,
        force_resume_scaler_state=False,
        force_resume_ema=False,
    )
    trainer_logger.removeHandler(handler)

    # Report CE_DEBUG lines
    print("\n--- [CE_DEBUG] lines (first ~10) ---")
    for line in ce_debug_lines[:12]:
        print(line)
    print(f"(total captured: {len(ce_debug_lines)})\n")

    # Verify new keys in avg_metrics
    required = ["target_entropy", "policy_cross_entropy", "ce_minus_policy_entropy", "approx_identity_check"]
    missing = [k for k in required if k not in avg_metrics]
    if missing:
        print(f"FAIL: avg_metrics missing keys: {missing}")
        return 1
    print("avg_metrics contains: target_entropy, policy_cross_entropy, ce_minus_policy_entropy, approx_identity_check")
    print(f"  approx_identity_check = {avg_metrics['approx_identity_check']:.6e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
