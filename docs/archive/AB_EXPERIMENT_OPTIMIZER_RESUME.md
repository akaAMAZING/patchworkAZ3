# A/B Experiment: Optimizer Resume vs Warm Restart

Prove that resuming optimizer/scheduler state across iterations improves strength/sample efficiency.

## Setup

Two short runs (50 iterations), identical seeds/sims/config except:

### A) Baseline (legacy)
- `resume_optimizer_state: false`
- `resume_scheduler_state: false`
- `lr_schedule: cosine_warmup` (per-iteration warm restart)

### B) New (optimizer continuation)
- `resume_optimizer_state: true`
- `resume_scheduler_state: true`
- `lr_schedule: cosine_warmup_per_phase`

## Commands

```bash
# Baseline run
python -m src.training.main --config configs/config_best.yaml --run-id ab_baseline_$(date +%Y%m%d)

# Edit config: set resume_*=false, lr_schedule=cosine_warmup for baseline
# Then for B: set resume_*=true, lr_schedule=cosine_warmup_per_phase
```

Or use two config files:
- `configs/config_ab_baseline.yaml` (resume_*=false, cosine_warmup)
- `configs/config_ab_resume.yaml` (resume_*=true, cosine_warmup_per_phase)

## Metrics to Compare

| Metric | At iter 25 | At iter 50 |
|--------|------------|------------|
| Frozen ladder Elo | | |
| Stability (fewer regressions around LR drops) | | |
| Wall-clock time per iteration | | |

## Success Criteria

- **B** has meaningfully higher Elo at 25/50 OR reaches same Elo sooner
- If not: revert cleanly (`resume_optimizer_state: false`, `lr_schedule: cosine_warmup`)

## Notes

- Ensure identical `seed` and `games_per_iteration` for both runs
- Frozen ladder must be enabled: `evaluation.frozen_ladder.enabled: true`
- Run for at least 50 iterations to observe phase boundaries (LR drop at iter 80 in config_best)
