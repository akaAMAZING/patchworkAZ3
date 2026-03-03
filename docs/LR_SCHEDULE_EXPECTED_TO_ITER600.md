# LR schedule: expected behavior to iter 600

From `configs/config_best.yaml`:

```yaml
lr_schedule:
  - iteration: 0    lr: 0.0016
  - iteration: 39   lr: 0.0012   # HOLD
  - iteration: 90   lr: 0.0012
  - iteration: 100  lr: 0.0008
  - iteration: 250  lr: 0.0008   # HOLD
  - iteration: 260  lr: 0.0004
  - iteration: 400  lr: 0.0004   # HOLD
  - iteration: 500  lr: 0.00025
```

Phases are between consecutive entries. For each phase we decide: **hold** (same LR at start and end) or **step-down** (cosine decay to next LR for a seamless merge). Step-up (next LR higher) → hold in current phase; next phase warms up to its peak.

---

## Expected action by phase (to iter 600)

| Phase (iters)   | Peak LR  | Next LR   | Type      | Expected behavior |
|-----------------|----------|-----------|-----------|-------------------|
| 0 → 39          | 0.0016   | 0.0012    | step-down | Warmup to 0.0016, then cosine 0.0016 → 0.0012 over the phase. At iter 39, LR = 0.0012. |
| 39 → 90         | 0.0012   | 0.0012    | **hold**  | Warmup to 0.0012, then flat 0.0012 for the whole phase. |
| 90 → 100        | 0.0012   | 0.0008    | step-down | Cosine 0.0012 → 0.0008. At iter 100, LR = 0.0008. |
| 100 → 250       | 0.0008   | 0.0008    | **hold**  | Flat 0.0008. |
| 250 → 260       | 0.0008   | 0.0004    | step-down | Cosine 0.0008 → 0.0004. At iter 260, LR = 0.0004. |
| 260 → 400       | 0.0004   | 0.0004    | **hold**  | Flat 0.0004. |
| 400 → 500       | 0.0004   | 0.00025   | step-down | Cosine 0.0004 → 0.00025. At iter 500, LR = 0.00025. |
| 500 → 600+      | 0.00025  | (none)    | **hold**  | No next entry → `phase_end_lr = iter_lr` → hold at 0.00025 from iter 500 onward. |

So to iter 600:

- **0–38**: cosine 0.0016 → 0.0012  
- **39–89**: hold 0.0012  
- **90–99**: cosine 0.0012 → 0.0008  
- **100–249**: hold 0.0008  
- **250–259**: cosine 0.0008 → 0.0004  
- **260–399**: hold 0.0004  
- **400–499**: cosine 0.0004 → 0.00025  
- **500–600**: hold 0.00025  

---

## How the code implements it

**Phase selection** (`trainer.py` `_create_scheduler`, `cosine_warmup_per_phase`):

- `lr_entries` sorted by `iteration`.
- For `current_iteration`, loop: `if self.current_iteration >= ent["iteration"]` → set `iter_lr`, `phase_start_iter`, `phase_end_iter`, `phase_end_lr` from current and next entry. Last matching entry wins → correct phase and next LR.

**Hold vs step-down:**

- `phase_is_hold = (phase_end_lr == iter_lr)` → hold; else step-down or step-up.
- Step-down: `phase_end_lr < iter_lr` → `end_ratio = max(min_lr_ratio, phase_end_lr / iter_lr)`.
- Step-up: `phase_end_lr > iter_lr` → `end_ratio = 1.0` (hold).
- No next entry: `phase_end_iter = 999999`, `phase_end_lr = iter_lr` → hold. So iter 500+ holds at 0.00025. ✓

**Progress within phase:**

- `step` from the scheduler is the **global** step (cumulative over all iters).
- Progress must be **within the current phase** so that warmup is at phase start and cosine runs over the phase length.
- Implementation: `phase_start_step = phase_start_iter * self.total_train_steps`, then `step_in_phase = step - phase_start_step` (clamped to `[0, phase_total_steps]`). Warmup uses `step_in_phase < warmup_steps`; decay uses `(step_in_phase - warmup_steps) / (phase_total_steps - warmup_steps)`. So progress 0→1 across the phase. ✓

**Warmup:**

- Once per phase, at the start: `step_in_phase < warmup_steps` → linear 0 → 1 (i.e. 0 → peak LR). ✓

**Lambda:**

- After warmup: if hold → return `1.0`. Else `progress` → `cosine_decay = 0.5*(1 + cos(π*progress))` (1→0), then `end_ratio + (1 - end_ratio) * cosine_decay` so LR goes from peak to `peak * end_ratio` = next phase LR at end. ✓

So the code matches the expected behavior above through iter 600 (and beyond for the 0.00025 hold).
