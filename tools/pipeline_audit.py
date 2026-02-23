#!/usr/bin/env python
"""
Pipeline Audit — prints key runtime-resolved settings to prevent doc/code drift.

Run after loading config (optionally with schedules applied for a given iteration)
to verify PIPELINE.md ground truth matches actual behavior.

Usage:
  python tools/pipeline_audit.py --config configs/config_best.yaml
  python tools/pipeline_audit.py --config configs/config_best.yaml --iteration 25

Outputs:
  - training AMP dtype, scaler enabled, allow_tf32, matmul_precision
  - EMA enabled, use_for_selfplay, use_for_eval
  - replay merge: target score_scale, source scale behavior
  - policy_target_mode, action_selection_temperature semantics
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _step_schedule_lookup(schedule: list, iteration: int, value_key: str, base_value) -> float:
    if not schedule:
        return base_value
    for entry in sorted(schedule, key=lambda x: x["iteration"], reverse=True):
        if iteration >= entry["iteration"]:
            return float(entry.get(value_key, base_value))
    return base_value


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline audit — key runtime settings")
    parser.add_argument("--config", default="configs/config_best.yaml", help="Config path")
    parser.add_argument("--iteration", type=int, default=0, help="Iteration for schedule lookup")
    args = parser.parse_args()

    import yaml
    config_path = REPO_ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        sys.exit(1)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    iter_cfg = config.get("iteration", {}) or {}
    train_cfg = config.get("training", {}) or {}
    sp_cfg = config.get("selfplay", {}) or {}
    mcts_cfg = sp_cfg.get("mcts", {}) or {}
    inf_cfg = config.get("inference", {}) or {}

    # Apply iteration schedules (same logic as main)
    temp_schedule = iter_cfg.get("temperature_schedule", [])
    effective_temp = _step_schedule_lookup(temp_schedule, args.iteration, "temperature", mcts_cfg.get("temperature", 1.0))
    temp_threshold = int(mcts_cfg.get("temperature_threshold", 15))

    amp_dtype = str(train_cfg.get("amp_dtype", "float16")).lower()
    use_amp = bool(train_cfg.get("use_amp", False))
    scaler_enabled = amp_dtype != "bfloat16"
    allow_tf32 = bool(train_cfg.get("allow_tf32") or inf_cfg.get("allow_tf32", True))
    matmul_precision = train_cfg.get("matmul_precision") or inf_cfg.get("matmul_precision") or "default"

    ema_cfg = train_cfg.get("ema", {}) or {}
    ema_enabled = bool(ema_cfg.get("enabled", False))
    ema_use_for_selfplay = bool(ema_cfg.get("use_for_selfplay", True))
    ema_use_for_eval = bool(ema_cfg.get("use_for_eval", True))

    # Dual-Head: no score_scale; value=win/loss/tie, score_margin=raw. Legacy configs may have score_scale.
    score_scale = sp_cfg.get("score_scale")
    target_score_scale = float(score_scale) if score_scale is not None else None
    policy_target_mode = str(sp_cfg.get("policy_target_mode", "visits")).lower()
    if policy_target_mode not in ("visits", "visits_temperature_shaped"):
        policy_target_mode = "visits"

    policy_target_temp = effective_temp if policy_target_mode == "visits_temperature_shaped" else None

    print("=" * 60)
    print("  PIPELINE AUDIT (iteration=%d)" % args.iteration)
    print("=" * 60)
    print()
    print("[TRAINING AMP]")
    print("  use_amp:           %s" % use_amp)
    print("  amp_dtype:         %s" % amp_dtype)
    print("  scaler_enabled:    %s (False for bf16)" % scaler_enabled)
    print("  allow_tf32:        %s" % allow_tf32)
    print("  matmul_precision:  %s" % matmul_precision)
    print()
    print("[EMA]")
    print("  enabled:           %s" % ema_enabled)
    print("  use_for_selfplay:  %s" % (ema_use_for_selfplay if ema_enabled else "N/A"))
    print("  use_for_eval:      %s" % (ema_use_for_eval if ema_enabled else "N/A"))
    print("  weights_for_actor: %s" % ("EMA" if (ema_enabled and ema_use_for_selfplay) else "raw"))
    print()
    print("[REPLAY MERGE]")
    if target_score_scale is not None:
        print("  target_score_scale: %.1f (from config selfplay.score_scale)" % target_score_scale)
        print("  source:            read from each HDF5 attr (LEGACY_SCORE_SCALE if absent)")
        print("  rescaling:         atanh/tanh when source_scale != target_scale")
    else:
        print("  value_target:      dual_head (value=win/loss/tie, score_margin=raw)")
        print("  score_scale:       N/A (Dual-Head; no rescaling)")
        print("  legacy_reject:     HDF5 without score_margins rejected when score_loss_weight>0")
    print()
    resume_opt = bool(train_cfg.get("resume_optimizer_state", True))
    resume_sched = bool(train_cfg.get("resume_scheduler_state", True))
    resume_scaler = bool(train_cfg.get("resume_scaler_state", True))
    lr_schedule = str(train_cfg.get("lr_schedule", "cosine_warmup"))
    print("[OPTIMIZER/SCHEDULER RESUME]")
    print("  resume_optimizer_state:  %s" % resume_opt)
    print("  resume_scheduler_state: %s" % resume_sched)
    print("  resume_scaler_state:    %s (only when amp_dtype=float16)" % resume_scaler)
    print("  lr_schedule:            %s" % lr_schedule)
    print("  optimizer_state_source: train_base (same checkpoint as model weights; enforced at runtime)")
    print()
    print("[POLICY TARGET]")
    print("  policy_target_mode:      %s" % policy_target_mode)
    print("  action_selection_temp:   %.2f (early) -> 0 (after move %d)" % (effective_temp, temp_threshold))
    if policy_target_temp is not None:
        print("  policy_target_temp:       %.2f (legacy shaped)" % policy_target_temp)
    else:
        print("  policy_target_temp:       N/A (visits mode = raw counts)")
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
