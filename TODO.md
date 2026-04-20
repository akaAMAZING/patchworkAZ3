# PATCHWORK AZ — Training TODO & Milestone Checks

---

## Pending Actions

- [ ] **NOW (before iter31)** — Run ladder backfill, then decide LR phase timing
  - `python scripts/run_ladder_eval.py --backfill`
  - If trend > 50 Elo/step: push LR phase to [iter40→0.0016, iter50→0.0012]
  - If trend 30-50 Elo/step: push by 5 iters
  - If trend 15-30 Elo/step (matches iter29 single data point): leave as-is
  - Always shift packing_alpha and dynamic_score_utility_weight by same amount as LR phase

- [ ] **iter ~60-70** — Run T1 round-robin tournament
  - `python scripts/run_tournament.py`
  - Sample every 10 iters: [iter10, 20, 30, 40, 50, 60, 70]
  - 40-60 games per pair
  - Purpose:
    - validate ladder ranking vs full payoff matrix
    - detect non-transitivity / matchup pockets
    - check for forgetting against older checkpoints before enabling league

- [ ] **T1 decision gate** — Only enable league if tournament data justifies it
  - Turn league on only if one or more are true:
    - ladder ordering disagrees materially with tournament ordering
    - current checkpoint has bad matchup pockets vs older checkpoints
    - evidence of forgetting / regression appears
    - payoff matrix is clearly non-transitive enough that pure ladder is missing signal
  - If T1 looks mostly transitive and ladder agrees, keep league off for now

- [ ] **iter ~100-120** — Optional league activation only if T1 supports it
  - Keep this evidence-gated, not calendar-gated
  - If enabled, start small:
    - no exploiter yet
    - use past-checkpoint / PFSP exposure conservatively
  - Goal opponent pool: roughly 35-48% WR vs current

- [ ] **iter ~140-160** — Run T2 tournament if league is still under consideration
  - Re-check:
    - ladder accuracy
    - matchup non-transitivity
    - regression/forgetting
    - whether league is actually likely to add value

- [ ] **iter ~150+** — Enable exploiter only if league is already on and showing value
  - Do not enable exploiter by default
  - Use only after core training is clearly stable
  - Prefer after q_value_weight has reached its mid/late setting and the run is past the early high-plasticity phase

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
