# PATCHWORK AZ — Training TODO & Milestone Checks

---

## Pending Actions

- [ ] **NOW (before iter31)** — Run ladder backfill, then decide LR phase timing
  - `python scripts/run_ladder_eval.py --backfill`
  - If trend > 50 Elo/step: push LR phase to [iter40→0.0016, iter50→0.0012]
  - If trend 30-50 Elo/step: push by 5 iters
  - If trend 15-30 Elo/step (matches iter29 single data point): leave as-is
  - Always shift packing_alpha and dynamic_score_utility_weight by same amount as LR phase

- [ ] **iter ~80** — Run T1 round-robin tournament
  - `python scripts/run_tournament.py` (build script around iter70)
  - Sample: every 10 iters → [iter10, 20, 30, 40, 50, 60, 70, 80] = 28 pairs × 40 games ≈ 1.5 hrs
  - Purpose: validate ladder Elo accuracy, get payoff matrix for league seeding

- [ ] **iter ~90-100** — League activation (deferred from iter79 — wait for T1 data)
  - Use T1 payoff matrix to pick opponent pool (want opponents at 35-48% WR vs current)
  - Config keys: `league.enabled: true`
  - Enable exploiter at iter 150 (after epochs raise), not iter 120

- [ ] **iter ~150** — Raise `epochs_per_iteration` from 2 → 3
  - Also flip `league.exploiter_enabled: true` at this point

- [ ] **iter ~180-210** — T2 round-robin tournament
  - Sample every 15 iters → ~13 checkpoints = 78 pairs × 60 games ≈ 4 hrs
  - Purpose: re-seed league pool, check for skill regressions before final LR phases

---

## Ladder Elo — decision rule
- Primary signal: `python scripts/run_ladder_eval.py --print-only`
- TensorBoard: `ladder/step_gap_elo` and `ladder/cumulative_elo`
- LR still doing work: step gap > 15 Elo consistently
- LR exhausted: step gap < 15 Elo for 2 consecutive steps
- Manual anchor eval (vs iter192): `python scripts/run_ladder_eval.py --anchor --iter-a <N> --games 40`

---

## Milestone Checks

### iter 40 (or adjusted phase boundary)
- [ ] Verify `pack_α` drops to `0.30` in terminal header
- [ ] LR drops to `0.0012` — confirm in header
- [ ] Check packing quality trend in tensorboard (`selfplay_avg_final_isolated_1x1_holes_mean`)
- [ ] Ladder step gap at this point — if still > 40 Elo, note for next LR phase timing

### iter 80
- [ ] Temperature drops to `0.80` — confirm in header
- [ ] Run T1 tournament — see Pending Actions above
- [ ] Review ladder/cumulative_elo trend in TensorBoard

### iter 100
- [ ] `pack_α` drops to `0.15` — verify in header
- [ ] LR drops to `0.0008`
- [ ] pol_acc should be meaningfully above iter016 baseline (was 67.9%)
- [ ] League should be active by now

### iter 150
- [ ] Raise epochs to 3
- [ ] Enable league exploiter
- [ ] `pack_α` drops to `0.08`
- [ ] Review val_loss trend — should still be declining

### iter 200
- [ ] `pack_α` drops to `0.04` — heuristic effectively off, NN owns packing
- [ ] Run T2 tournament — see Pending Actions above
- [ ] Benchmark vs iter192 anchor with 100 games for clean Elo estimate

### iter 300
- [ ] LR drops to `0.0006`
- [ ] Confirm epochs = 3
- [ ] Consider BGA bot test if Elo vs anchor looks strong

---

## Future Architecture (v3 consideration)
- 7×7 bonus prediction head (binary — hardest thing for network to learn implicitly)
- Opponent board threat head (scalar — defensive play gap)
- League with exploiter for robustness
