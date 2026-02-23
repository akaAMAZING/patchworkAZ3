#!/usr/bin/env python3
"""Verify actual LR at specific global steps (cosine_warmup_per_phase schedule)."""
import math

def lr_at_step(step: int, base_lr: float = 0.0016, warmup_steps: int = 200,
               phase_total_steps: int = 80 * 260, min_lr_ratio: float = 0.02) -> float:
    """Compute LR from cosine_warmup_per_phase formula (trainer.py)."""
    if step < warmup_steps:
        return base_lr * (step / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, phase_total_steps - warmup_steps)
    progress = min(progress, 1.0)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    ratio = min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    return base_lr * ratio

# Config: base_lr=0.0016, warmup=200, min_lr=0.000032 => min_lr_ratio=0.02
# phase_total_steps = 80 iters * ~260 steps/iter ≈ 20800 (early phase)
base = 0.0016
warmup = 200
min_ratio = 0.000032 / 0.0016  # 0.02
phase = 80 * 260  # approximate

print("LR at key steps (base_lr=0.0016, warmup=200, phase=80*260):")
for s in [0, 61, 122, 200, 318, 500, 1220, 1828]:
    lr = lr_at_step(s, base, warmup, phase, min_ratio)
    phase_progress = (s - warmup) / max(1, phase - warmup) if s >= warmup else 0
    print(f"  step {s:5d}: LR = {lr:.4e}  (warmup done: {s >= warmup}, phase_progress={phase_progress:.2%})")
