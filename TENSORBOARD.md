# Patchwork AlphaZero — TensorBoard Field Manual (Exhaustive)

This is the **operator-grade** reference for every TensorBoard scalar you currently log (post-cleanup), with:

- **Definition** (what it is mathematically / how it's computed)
- **What it represents** (the system variable it measures)
- **Target behavior / ranges** (practical, not fake precision)
- **Red flags** (patterns that indicate trouble)
- **Actions** (what to change, in what order, and why)
- **Cross-metric diagnostics** (how to confirm a hypothesis)

**Assumptions baked in:**

- Policy action space size is **2026** → max uniform entropy over *all* actions is **log(2026) ≈ 7.61**.
- Illegal actions are masked (logits = -inf) before softmax/log-softmax.
- Policy targets are MCTS visit distributions normalized over legal actions.
- "Train" metrics are computed on batches used for SGD updates; "Val" on held-out replay batches (same distribution family).
- Iter metrics are epoch-averaged values emitted per iteration.

> If you change what you log, update this doc immediately. A clean metric surface is a competitive advantage.

---

## Quick dashboard: what to watch first

If you can only look at **6 charts**:

1. `iter/kl_divergence`
2. `iter/policy_entropy` **and** `iter/target_entropy` (same panel)
3. `selfplay/unique_positions` **and** `selfplay/avg_redundancy`
4. `train/grad_norm`
5. `selfplay/games_per_min` **or** `selfplay/generation_time`
6. `iter/value_mse`

These catch >90% of real failures early.

---

# 1) TRAIN METRICS (step-level)

## `train/learning_rate`

**Definition:** Current optimizer LR (after schedule).

**Represents:** Update step size; the primary "energy injector" into training.

**Target behavior / ranges:**

- Should match your configured schedule.
- Small discontinuities only where your schedule intentionally steps.

**Red flags:**

- Unexpected spikes/drops → schedule bug or wrong phase.
- LR changes coinciding with KL/grad_norm explosions.

**Actions:**

- If unintended: verify schedule selection and "current iteration → LR" mapping.
- If intended but destabilizing: smooth transition (glide) or reduce step magnitude.

**Cross-check:** A destabilizing LR change almost always shows up within 1–3k steps as:

- `train/grad_norm` spikes
- `iter/kl_divergence` increases next iteration
- Selfplay quality degrades (redundancy up)

---

## `train/total_loss`

**Definition:** Weighted sum of component losses (policy + value + optional heads if present in your run).

**Represents:** Coarse training progress; *not* a decision-grade metric by itself (weights can hide issues).

**Target behavior / ranges:**

- Downward trend early; later it may plateau.
- Noise is normal with large batch RL-style targets.

**Red flags:**

- Sudden sustained increase → instability.
- Flatline from the start → data/target issue or LR too low.
- "Looks fine" while other metrics break → component cancellation (total_loss masks problems).

**Actions:** Diagnose using component losses + KL + entropy; don't act on total_loss alone.

**Cross-check:** If total_loss worsens but `train/policy_loss` and `train/value_loss` improve, weights are likely changing effective emphasis (or aux head changed).

---

## `train/policy_loss`

**Definition:** Cross-entropy CE(target π, model p). Common form:  
CE(π, p) = −𝔼_π[log p(a)] = −∑_a π(a) log p(a)

**Represents:** How well the network's policy matches MCTS targets (supervision signal quality + model fit).

**Target behavior / ranges:**

- Should decrease early; later may plateau as targets sharpen.
- Absolute magnitude depends on target entropy and KL; don't fixate on a universal numeric "good".

**Red flags:**

- Rising while `iter/kl_divergence` rising → the policy is losing alignment to MCTS.
- Decreasing but selfplay diversity collapsing → overconfident policy (may be "learning the wrong thing").

**Actions** (in order):

1. If it rises with KL: reduce LR 10–20% or hold LR longer.
2. If it rises without KL: check data pipeline, target normalization, action masking.
3. If it decreases but selfplay degenerates: increase exploration (noise/alpha/temp) before changing LR.

**Cross-check:** Compare with `iter/target_entropy` and `iter/kl_divergence`: CE ≈ H(target) + KL(target‖policy). If CE rises because target entropy rises, that's not necessarily bad (targets got broader).

---

## `train/value_loss`

**Definition:** Mean squared error between predicted value and target value:  
MSE = 𝔼[(v_θ − v_target)²]

**Represents:** Value head fit/calibration. Value drift can quietly break MCTS.

**Target behavior / ranges:**

- Typically decreases more slowly than policy.
- Plateau is normal once value becomes "good enough" for search.

**Red flags:**

- Oscillation increases after schedule changes (LR/sims/score utility).
- Sustained increase with stable policy metrics → value head instability or target shift.

**Actions:**

- First: lower LR modestly (5–15%) if value is unstable.
- If instability coincides with score utility changes: revisit score/value loss weights or score utility ramp speed.
- If value remains noisy: consider more conservative temperature/noise changes (value targets depend on selfplay distribution).

**Cross-check:** Watch `selfplay/avg_root_q` and game length: value miscalibration often shifts root Q and changes game length distribution.

---

## `train/grad_norm`

**Definition:** Norm of gradients (before clipping, if clipping exists).

**Represents:** Optimization stability and effective step magnitude (LR × grad_norm).

**Target behavior / ranges:**

- Should settle into a stable band after initial transient.
- A "good" band is the one that historically produced progress without spikes for your run.
- Use first stable window as baseline; don't chase a universal number.

**Red flags:**

- Repeated spikes → LR too high, targets changing too fast, or batch outliers.
- Trend upward over many steps → accumulating instability.
- Trend to near-zero → learning stalled, dead grads, or LR too low.

**Actions:**

- Spikes: reduce LR, increase grad clipping strictness, or reduce schedule shock (temp/sims jumps).
- Near-zero: verify gradients exist (no masking bug), consider LR increase only if KL is low and training stagnates.

**Cross-check:** If grad_norm spikes but KL doesn't: could be value head causing it; check value_loss/MSE.

---

## `train/policy_entropy`

**Definition:** Shannon entropy of model policy distribution (masked), computed from log-softmax:  
H(p) = −∑_a p(a) log p(a)

**Represents:** Policy sharpness / confidence. A collapse detector.

**Target behavior / ranges:**

- **Upper bound:** log(#legal actions). Global upper bound is 7.61, but with masking it's log(num_legal).
- Healthy training typically shows a slow, controlled decrease as policy improves.
- Entropy should not crater abruptly unless you intentionally lower temperature and sims are high and stable.

**Red flags:**

- Rapid entropy collapse (down sharply over a few iterations) **plus** redundancy rising → policy collapse / exploration failure.
- Entropy spikes suddenly → instability or target distribution shift.
- Entropy decreasing while KL increasing → policy getting sharper in the wrong direction (misalignment).

**Actions** (in order):

1. If collapse: increase exploration (Dirichlet noise weight / alpha) and/or slow temperature decay.
2. Reduce LR if KL/grad_norm also show instability.
3. Pause sims increases (higher sims can amplify a collapsing prior).

**Cross-check:** Compare to `iter/target_entropy`: if H(p) ≪ H(target), network is overconfident relative to search targets.

---

## `train/policy_accuracy`

**Definition:** Top-1 accuracy vs target argmax (usually argmax of π).

**Represents:** Whether the model's top prediction matches the most-visited MCTS move.

**Target behavior / ranges:**

- Should rise early; later may plateau.
- In large action spaces, absolute values may remain "not huge" even when learning is real—interpret as trend.

**Red flags:**

- Flat for many iterations while loss decreases → model improving calibration but not ranking; or target multimodality.
- Drops sharply with stable loss → ranking issue or data drift.

**Actions:**

- Use alongside top5 (below) to understand ranking vs calibration.
- If both top1/top5 flatten and KL is low: consider capacity, LR too low, or targets too noisy.

---

## `train/policy_top5_accuracy`

**Definition:** Whether the target argmax action is within the model's top-5 predicted actions.

**Represents:** **Ranking quality** independent of exact ordering; early-learning signal.

**Target behavior / ranges:**

- Often increases earlier than top-1.
- Should be ≥ top-1 by definition.

**Red flags:**

- Declines while policy_loss stays stable → ranking degradation (subtle regression).
- Train improves but val does not → replay imbalance / too many epochs / skew to newest data.

**Actions:**

- If train ≫ val: increase replay window, reduce epochs per iteration, reduce newest_fraction pressure.
- If both stagnate with low KL: LR might be too low or model capacity bottleneck.

---

# 2) VALIDATION METRICS (held-out replay)

All `val/*` metrics mirror their `train/*` definitions, computed on non-updated batches.

## How to interpret train vs val in selfplay RL

Validation here is not "generalization to a different distribution." It is a **leak detector** and **overfit-to-recent detector** within replay.

**Expected:**

- Train and val trends are similar.
- Train is slightly better than val.
- Big gaps are informative.

### Primary red-flag patterns

- **Train improves, Val flat/worse** → replay imbalance, too many epochs, or newest_fraction dominance.
- **Val much noisier** → LR too high, unstable targets, or sampling issues.

### Primary fixes (in order)

1. Increase replay window iterations / reduce newest_fraction pressure.
2. Reduce epochs_per_iteration.
3. Lower LR if instability is present in KL/grad_norm.

(Apply one change at a time; observe 2–5 iterations.)

**Val metrics logged:** `val/total_loss`, `val/policy_loss`, `val/value_loss`, `val/policy_entropy`, `val/policy_accuracy`, `val/policy_top5_accuracy`

---

# 3) ITERATION METRICS (epoch-averaged, per iteration)

## `iter/kl_divergence`

**Definition:** KL(target π ‖ policy p) (per batch, averaged).

**Represents:** **Alignment pressure:** how hard the model must move to match search.

**Target behavior / ranges:**

- Should be stable or gently decreasing over time.
- "Good" depends on your training regime; use stable windows as baseline.

**Red flags:**

- Sustained climb over multiple iterations.
- Spikes coinciding with schedule changes (LR, temp, sims, score utility).

**Actions** (in order):

1. Hold LR (do not increase) until KL stabilizes.
2. Reduce LR 10–20% if KL remains high for ~3–5 iterations.
3. If spike after temperature drop: slow the temperature schedule or increase noise.
4. If spike after sims increase: pause sims increases until KL normalizes.

**Cross-check:** If KL rises but target_entropy also rises: targets got broader; verify whether the broader targets are expected (e.g., higher temperature or more noise).

---

## `iter/policy_entropy` and `iter/target_entropy`

See definitions above.

### The most important derived indicator: the entropy gap

Compute mentally: **Gap = H(policy) − H(target)**

**Interpretation:**

- **Gap ≈ 0:** healthy tracking of sharpness
- **Gap ≪ 0:** policy too sharp / overconfident
- **Gap ≫ 0:** policy too diffuse / underfitting

**Actions:**

- Gap ≪ 0: increase exploration or reduce LR (if also unstable).
- Gap ≫ 0: increase training pressure (slightly higher LR if safe) or increase epochs (careful).

---

## `iter/policy_cross_entropy`

Same as policy CE, logged explicitly for decomposition reasoning.

**Target behavior:** Should roughly track H(target) + KL. If it doesn't, check `iter/approx_identity_check`.

---

## `iter/value_mse`

Same as value_loss but directly MSE (useful when value_loss naming/weighting might change).

**Red flags/actions:** Same as value_loss, but treat MSE as the decision-grade value signal.

---

## `iter/step_skip_rate`

**Definition:** Fraction of steps skipped due to AMP overflow / numerical issues.

**Target:** **0**.

**Red flags:**

- >0 for more than a tiny blip → training is numerically unstable.
- Increases after LR change → LR too high.

**Actions:**

1. Lower LR (first choice).
2. Use more conservative AMP dtype/scaler parameters.
3. Reduce schedule shocks (temperature/sims increases) that change target hardness.

---

## `iter/approx_identity_check`

**Definition:** Mean over samples of |H(π_i) + KL(π_i‖p_i) − CE(π_i, p_i)|.

**Represents:** A **math consistency alarm**. If this is large, something is wrong in masking/log-prob usage or logging.

**Target behavior / ranges:**

- Typically ~1e-7 to 1e-4.
- It can increase slightly with aggressive clamping, low precision, or unusual masking—but should remain tiny.

**Red flags:**

- Sustained >1e-3
- Sudden jump after refactor

**Actions:** Verify: logits masking, target normalization, use of log-softmax for CE/entropy/KL, and that the same tensors feed each term.

---

# 4) SELFPLAY METRICS (per iteration)

Selfplay metrics are the earliest warning system for distribution collapse and search quality issues.

## `selfplay/games_per_min`

**Definition:** Throughput (games generated per minute).

**Represents:** System performance + cost of search.

**Target behavior:** Decreases as sims increase; otherwise stable.

**Red flags:**

- Sudden step down not explained by schedule.
- Gradual erosion → data loader / CPU contention / parallel leaves issues.

**Actions:**

- If schedule-related: accept or compensate by reducing games or parallel leaves.
- If not schedule-related: profile selfplay workers, CPU affinity, dataloader contention.

---

## `selfplay/num_positions`

**Definition:** Positions generated per iteration.

**Represents:** Data volume. Lower volume means higher variance SGD.

**Red flags:** Unexpected drop → selfplay failure, early termination, throughput regression.

**Actions:** Check generation_time and games_per_min first. Confirm games count and avg_game_length.

---

## `selfplay/num_games`

**Definition:** Games generated per iteration.

**Represents:** Primary sample count.

**Red flags:** Drop larger than schedule expects → perf regression or selfplay stopping early.

---

## `selfplay/avg_game_length`

**Definition:** Average moves/plies per game.

**Represents:** Distributional shift; can indicate degeneracy.

**Red flags:**

- Rapid shrink: resignation/termination bug, collapsed strategies.
- Rapid growth: indecisive play; policy too diffuse or value too uncertain.

**Actions:**

- If shrink + redundancy up: increase exploration.
- If growth + entropy high: consider more sims or modest temperature reduction (only if stable).

---

## `selfplay/unique_positions`

**Definition:** Count of unique positions encountered (or your system's notion of uniqueness).

**Represents:** Diversity; the antidote to replay overfitting.

**Target behavior:** Should scale with num_positions; ratio should remain healthy.

**Red flags:**

- Downtrend while num_positions stable → repetition / collapse.
- Downtrend coinciding with entropy collapse → strong collapse signature.

**Actions:** Increase exploration: noise weight, Dirichlet alpha, temperature, reduce q_value_weight aggressiveness.

---

## `selfplay/avg_redundancy`

**Definition:** How repetitive positions are on average.

**Represents:** Collapse / cycling / low exploration.

**Target behavior:** Should remain stable or decrease as policy improves (depends on game).

**Red flags:**

- Sustained increase over iterations.
- Spikes after temperature drop or noise decrease.

**Actions:**

1. Increase noise (weight or alpha).
2. Slow temperature decay.
3. If KL also rises: reduce LR.

---

## `selfplay/avg_policy_entropy`

**Definition:** Average entropy of the search policy at root (or model prior, depending on your pipeline), as computed in selfplay stats.

**Represents:** Search uncertainty / diversity at decision time.

**Red flags:**

- Collapsing to near-zero early.
- Diverging from train/iter entropy trends unexpectedly.

**Actions:**

- If root entropy collapses while model entropy is fine: search settings are too exploitative (cpuct, q weight, noise).
- If both collapse: increase exploration globally + reduce LR if unstable.

---

## `selfplay/avg_top1_prob`

**Definition:** Average probability mass on the most likely action (root).

**Represents:** Peakedness. Complement to entropy.

**Healthy:** Increases gradually as policy sharpens.

**Red flags:** Jumps abruptly (often means determinism/collapse).

**Actions:** Increase exploration; slow temperature decay.

---

## `selfplay/avg_num_legal`

**Definition**: average count of legal moves.

**Represents**: branching factor / constraint level.

**Red flags**: - Sudden shift can indicate rules/encoding bug, or
distributional shift in positions. - Long-term shrink can indicate
policy steering into constrained states.

**Actions**: - If sudden: check mask generation and engine legality
encoding. - If gradual + collapse signals: increase exploration.

---

## `selfplay/avg_root_q`

**Definition:** Average root Q value from search.

**Represents:** Value calibration + utility scaling; also detects bias.

**Target behavior:** Should be stable around a plausible equilibrium for your scoring scheme.

**Red flags:**

- Persistent drift (up or down).
- Drift coincides with value MSE rising.

**Actions:**

- Check value targets and loss scaling.
- Lower LR slightly; value instability often responds to LR reductions.
- If you recently changed dynamic score utility: slow ramp or reduce score weight.

---

## `selfplay/generation_time`

**Definition:** Time spent generating selfplay for an iteration (or per selfplay batch).

**Represents:** Cost and scheduling pressure.

**Red flags:**

- Increases without a schedule reason.
- Increases + games_per_min decreases unexpectedly.

**Actions:** Profile selfplay; check sims/leaves schedule; check CPU saturation.

---

# 5) BUFFER METRICS (per iteration)

## `buffer/total_positions`

**Definition:** Positions stored in replay (may cap at max_size).

**Represents:** Buffer fullness / capacity.

**Red flags:**

- Not reaching cap when expected → ingestion bug.
- Constantly at cap is fine; then diversity depends on window/newest_fraction.

**Actions:**

- If not filling: check selfplay generation and replay insertion.
- If filling too fast and cycling: increase window_iterations or adjust newest_fraction.

---

## `buffer/num_iterations`

**Definition:** Number of iterations represented in replay buffer.

**Represents:** Temporal diversity.

**Target behavior:** Should follow your configured window/retention behavior.

**Red flags:** Much smaller than expected → retention bug or over-aggressive trimming.

**Actions:** Verify replay window logic and max_size interactions.

---

# 6) RED FLAGS --- COMPLETE PLAYBOOK

This section is organized by **symptom → diagnosis → fix**.

## A) Instability (optimization)

### Symptom

- `iter/kl_divergence` spikes or trends upward
- `train/grad_norm` spikes
- `iter/step_skip_rate` > 0

### Likely causes

- LR too high for current target hardness
- Schedule shock: temperature drop, sims increase, noise decrease, score utility ramp

### Fix (order matters)

1. **Hold** the schedule (pause further temp/sims/noise changes).
2. Reduce LR 10–20% (or slow next LR step).
3. If AMP overflow: lower LR first; only then adjust scaler/dtype.

---

## B) Policy collapse / degeneracy

### Symptom

- `iter/policy_entropy` drops rapidly
- `selfplay/avg_redundancy` rises
- `selfplay/unique_positions` falls
- `selfplay/avg_top1_prob` jumps

### Likely causes

- Exploration too low (noise/alpha too low, temp too low)
- Q-weight / cpuct too exploitative
- Too fast temperature decay

### Fix (fast)

1. Increase exploration **immediately** (noise weight up, alpha up, or temp up).
2. Pause sims increases (high sims amplifies a strong prior).
3. If KL also unstable: reduce LR.

---

## C) Underfitting / "not learning"

### Symptom

- `train/policy_loss` flat for many iterations
- `train/policy_accuracy` and top5 flat
- KL very low and stable

### Likely causes

- LR too low
- Targets too easy / too deterministic
- Capacity bottleneck

### Fix

1. Verify targets still have entropy (`iter/target_entropy` not near 0).
2. If KL is very low and everything is flat, **increase LR modestly** (5–10%).
3. Consider increasing epochs_per_iteration (careful: watch train/val gap).
4. If still stuck: capacity or feature issues.

---

## D) Replay imbalance / overfit-to-recent

### Symptom

- Train improves, Val does not
- Selfplay diversity decreasing
- Unique positions fraction down

### Fix

1. Increase replay window_iterations.
2. Reduce epochs_per_iteration.
3. Reduce newest_fraction pressure (if applicable).

---

## E) Value drift

### Symptom

- `iter/value_mse` rising
- `selfplay/avg_root_q` drifting
- `selfplay/avg_game_length` shifts

### Fix

1. Lower LR slightly (5–15%).
2. Check score utility ramp and score/value loss weights.
3. Increase data volume (more games or positions) if variance is high.

---

## F) Performance regression

### Symptom

- `selfplay/games_per_min` down unexpectedly
- `selfplay/generation_time` up unexpectedly

### Fix

1. Confirm sims/leaves schedule didn't change.
2. Profile selfplay worker utilization (CPU bound vs GPU bound).
3. Check dataloader contention and pin_memory/persistent_workers settings.

---

# 7) WHEN TO CHANGE SCHEDULES (DECISION RULES)

These rules assume you only change one major knob at a time and observe for 2–5 iterations.

## Lower LR when

- `iter/kl_divergence` is elevated and not improving for ~3–5 iterations, **or**
- `train/grad_norm` spike frequency increases, **or**
- `iter/step_skip_rate` > 0, **or**
- Entropy and KL become more volatile after a schedule change.

**Typical move:** −10% to −20% (or shift the next planned drop earlier).

## Hold LR (do nothing) when

- KL is high but trending down
- Selfplay quality stable
- Losses still improving

## Increase LR (rare) when

- KL is very low and stable
- Train/val both flat
- Selfplay is healthy (diverse)
- You've been stagnant for ~8–12 iterations

**Typical move:** +5% to +10%.

---

## Increase MCTS sims when

- KL is low/stable
- Policy_loss is improving smoothly
- Selfplay diversity is stable (unique_positions not falling; redundancy not rising)
- Value_mse stable

**Do not** increase sims during instability—sims amplifies bad priors.

---

## Lower temperature when

- Policy and target entropies are close and stable
- Redundancy is not rising
- KL is low/stable

If temperature drops cause redundancy spikes or entropy collapse → revert/slow the schedule and increase noise.

---

## Reduce noise / alpha when

- Selfplay redundancy is low and stable for many iterations
- Unique_positions remains healthy
- You want more exploitation

If reduction causes collapse: revert immediately.

---

## Increase replay window when

- Train/val gap appears
- Redundancy increases
- You hit replay max_size and start cycling too fast

---

# 8) FAQ (the stuff that burns time in the moment)

### "Policy loss and policy entropy are close—is that a bug?"

Not necessarily. CE(target, policy) can be close to H(policy) when KL is moderate and entropies are similar. Use `iter/target_entropy`, `iter/kl_divergence`, and `iter/approx_identity_check`. If identity_check is tiny, math/logging is consistent.

### "What is a 'good' entropy value?"

There is no universal number. The correct target is: not collapsing abruptly, and maintaining healthy diversity in selfplay. Upper bound is log(#legal), and the baseline is whatever produced good selfplay historically.

### "Val looks the same as train—is val useless?"

Val is your "replay skew detector." If train and val always overlap, it's still useful as a safety check. If they diverge, it's extremely informative.

### "Which metric detects collapse earliest?"

Usually selfplay: `avg_redundancy` up + `unique_positions` down, followed closely by `policy_entropy` collapse.

### "Which metric is the best 'stop touching knobs' signal?"

`iter/kl_divergence` stabilizing in a healthy band while selfplay diversity stays healthy.

### Train metrics "snap to the beginning of the chart" (e.g. around steps 22k–23k)

**Symptom:** `train/*` scalars show a discontinuity where the line jumps from the current step range (e.g. 22870) back to step 0 (or a low step), then continues, so the plot appears to snap to the left and sometimes shows a vertical spike.

**Cause:** Each iteration creates a new TensorBoard event file. Train metrics use a **run-wide** `global_step` (monotonic across iterations). When the trainer resumes optimizer/scheduler from a checkpoint, it used to overwrite `global_step` with the checkpoint’s value. If that checkpoint is from an **older** iteration (e.g. `best_model` from many iters ago) or has a lower step (e.g. 0), the next event file contained steps 0, 10, 20, … TensorBoard merges all event files in `logs/tensorboard/`, so the same tag gets two step sequences (…, 22870 and 0, 10, …), producing the snap/spike.

**Fix (in code):** The trainer now keeps the run-wide step monotonic: when loading optimizer/scheduler state it sets `global_step = max(global_step_offset, checkpoint_global_step)` so the step never goes backwards. Existing runs that already wrote the bad event file will still show the artifact; new runs will not.

---

## Appendix: Metric list (canonical)

**Train:**  
`train/learning_rate`, `train/total_loss`, `train/policy_loss`, `train/value_loss`, `train/grad_norm`, `train/policy_entropy`, `train/policy_accuracy`, `train/policy_top5_accuracy`

**Val:**  
`val/total_loss`, `val/policy_loss`, `val/value_loss`, `val/policy_entropy`, `val/policy_accuracy`, `val/policy_top5_accuracy`

**Iter:**  
`iter/kl_divergence`, `iter/policy_entropy`, `iter/target_entropy`, `iter/policy_cross_entropy`, `iter/value_mse`, `iter/step_skip_rate`, `iter/approx_identity_check`

**Selfplay:**  
`selfplay/games_per_min`, `selfplay/num_positions`, `selfplay/num_games`, `selfplay/avg_game_length`, `selfplay/unique_positions`, `selfplay/avg_redundancy`, `selfplay/avg_policy_entropy`, `selfplay/avg_top1_prob`, `selfplay/avg_num_legal`, `selfplay/avg_root_q`, `selfplay/generation_time`, plus beat-humans metrics below (`selfplay/selfplay_avg_final_*`, `selfplay/selfplay_p50_*`, `selfplay/selfplay_avg_final_*_abs_diff`, `selfplay/selfplay_avg_root_*`, `selfplay/selfplay_p90_root_*`)

**Buffer:**  
`buffer/total_positions`, `buffer/num_iterations`

---

# 9) BEAT-HUMANS METRICS (selfplay)

Low-noise observability that correlates with **human strength**: packing quality (terminal quilts) and search health (PW tractability). Computed at game end for both players; definitions match `tools/ab_test_*` and `src/utils/packing_metrics.py`. Logged under `selfplay/<key>` each iteration; also in committed iteration JSON and `metadata.jsonl`. No per-player P0/P1 series—aggregates and asymmetry only.

---

## `selfplay/selfplay_avg_final_empty_components_mean`

**Definition:** Average number of separate empty regions (connected components of empty cells, 4-neighbor BFS) over all players and games at terminal state.

**Represents:** Fragmentation of final quilts; humans punish fragmentation more than raw empties.

**Target behavior / ranges:**

- Downward trend over iterations (even if slowly).
- Plateaus are okay if isolated holes are still dropping.

**Red flags:**

- It rises while win rate vs self stays fine → drifting back into "button meta."

**Actions:**

- Tighten PW slightly (reduce expanded ratio), or increase sims for play/eval, or add/strengthen packer-style opponents later.

**Cross-check:** If components rise, check `selfplay_avg_final_isolated_1x1_holes_mean` and `selfplay_avg_final_empty_squares_mean`; focus on components/isolated over raw empties.

---

## `selfplay/selfplay_p50_final_empty_components_mean`

**Definition:** Median (50th percentile) of per-game mean empty components (mean across the two players in each game).

**Represents:** Typical-game fragmentation level.

**Target behavior:** Should trend down with the mean; gap between p50 and p90 is a tail-risk indicator.

---

## `selfplay/selfplay_p90_final_empty_components_mean`

**Definition:** 90th percentile of per-game mean empty components ("worst 10% games" fragmentation).

**Represents:** Tail risk—those games that lose to humans with messy quilts.

**Target behavior:**

- Should drop more clearly than the mean over time.
- Ideally the gap between p90 and mean shrinks.

**Red flags:**

- Mean improves but p90 doesn't → still producing occasional catastrophic quilts.

**Actions:** Don't change DSU; adjust PW (looser/tighter) or later add a stronger human proxy.

---

## `selfplay/selfplay_avg_final_isolated_1x1_holes_mean`

**Definition:** Average count of 1×1 empty cavities with **zero** empty neighbors (4-neighbor) at terminal state, over all players and games.

**Represents:** Dead squares that are almost always unfillable late game; strong signal of packing mistakes.

**Target behavior:**

- Downward trend; one of the fastest indicators that packing is becoming human-like.

**Red flags:**

- Stays high even when empties drop → improving fill but still creating "traps."

**Actions:** Later, feasibility-aware ordering helps; first see what training does with PW.

---

## `selfplay/selfplay_p50_final_isolated_1x1_holes_mean`

**Definition:** Median of per-game mean isolated 1×1 holes.

**Represents:** Typical-game isolated-hole count.

---

## `selfplay/selfplay_p90_final_isolated_1x1_holes_mean`

**Definition:** 90th percentile of per-game mean isolated 1×1 holes.

**Represents:** "Catastrophe rate" gauge—worst 10% tail of isolated holes.

**Target behavior:** Strong downtrend.

**Red flags:**

- Flat p90 while mean improves → occasional blow-ups still exist.

**Actions:** Aim for stability; let PW-trained policy mature without adding new knobs.

---

## `selfplay/selfplay_avg_final_empty_squares_mean`

**Definition:** Average number of empty squares at terminal state over all players and games.

**Represents:** Sanity check; correlates with score but not as strongly with human losses as fragmentation.

**Target behavior:** Gentle downtrend. If this drops but components/isolated don't, you may be filling but still badly fragmented.

**Red flags:**

- Empties improve but fragmentation worsens → "filled more, but packed worse."

**Actions:** Focus on components/isolated; empties alone can be misleading.

---

## `selfplay/selfplay_p50_final_empty_squares_mean` / `selfplay/selfplay_p90_final_empty_squares_mean`

**Definition:** Median and 90th percentile of per-game mean empty squares.

**Represents:** Distribution and tail of raw empty count; p90 is tail risk.

**Target behavior:** Same interpretation as other p50/p90 metrics—p90 should improve to reduce catastrophe rate.

---

## `selfplay/selfplay_avg_final_empty_squares_abs_diff` / `selfplay/selfplay_avg_final_empty_components_abs_diff` / `selfplay/selfplay_avg_final_isolated_1x1_holes_abs_diff`

**Definition:** Mean over games of |P0 − P1| for that metric (empties, components, isolated holes).

**Represents:** Asymmetry—detects systematic bias (one side consistently packs worse). Can indicate exploitable bias, first-player advantage, or consistent planning failure.

**Target behavior:** Generally flat or decreasing.

**Red flags:**

- Sudden increase → policy/search may be skewing one side's packing decisions.

**Actions:** Check whether one player (P0/P1) is consistently worse; if yes, log per-player temporarily to diagnose.

---

## `selfplay/selfplay_avg_root_expanded_ratio`

**Definition:** Mean of (expanded_children / max(legal_actions, 1)) per model move when MCTS runs. PW limits how many root children are expanded.

**Represents:** Search depth vs breadth dial—how much of the legal action space is actually considered at root.

**Target behavior:**

- Stable ratio after PW is introduced.
- If legal rises over time, expanded shouldn't rise proportionally; PW should keep ratio bounded.

**Red flags:**

- **Too high:** Ratio creeping upward → drifting toward "old wide search," shallow again.
- **Too low:** Extremely low ratio → over-pruning, risk of missing tactics (could hurt win rate).

**Actions:**

- If too high → reduce k_root or k_sqrt_coef.
- If too low and win rate drops → increase k_root a bit.

**Cross-check:** Rule of thumb—enough breadth that win rate doesn't drop, tight enough to deepen search and reduce fragmentation.

---

## `selfplay/selfplay_avg_root_legal_count`

**Definition:** Average number of legal actions at root, over all model moves in selfplay.

**Represents:** Context for expanded ratio; reflects board/shop state distribution (you don't control it directly).

**Target behavior:** Stable or decreasing as packing improves (tighter quilts → fewer placements). If it trends upward a lot, positions are more open / more empty space → consistent with button meta.

**Cross-check:** If legal↑ and fragmentation↑, you're not packing tight.

---

## `selfplay/selfplay_avg_root_expanded_count`

**Definition:** Average number of root children expanded (PW-limited) per model move.

**Represents:** Raw breadth of search; use with legal count to interpret ratio.

---

## `selfplay/selfplay_p90_root_legal_count` / `selfplay/selfplay_p90_root_expanded_ratio`

**Definition:** 90th percentile of root legal count and of expanded ratio across all model moves.

**Represents:** Tail positions where legal count or expansion ratio is highest; watch for spikes that indicate very wide or very narrow search in some positions.

---

### Healthy training picture (beat-humans)

Over ~5–20 iterations with PW you want:

- `selfplay_avg_final_empty_components_mean` slowly ↓  
- `selfplay_p90_final_empty_components_mean` ↓ more noticeably  
- `selfplay_avg_final_isolated_1x1_holes_mean` ↓  
- `selfplay_p90_final_isolated_1x1_holes_mean` ↓ (catastrophe killer)  
- `selfplay_avg_final_empty_squares_mean` ↓ slightly (or flat early, then ↓)  
- `selfplay_avg_root_expanded_ratio` stable  
- `selfplay_avg_root_legal_count` stable or ↓ (often decreases if packing improves)

---

**End.**
