# Position count analysis: ~35k vs ~37.8k per 900 games

**Purpose:** Explain why current implementation outputs ~35,000 positions per 900 games while the previous tanh setup and the `checkme/` 201-bin setup output ~37,800. Output is for debugging and for ChatGPT to reason about root cause.

---

## 1. Summary of the gap

| Metric | Expected (tanh / checkme 201bin) | Current (iter_006 / iter_007) |
|--------|----------------------------------|-------------------------------|
| Positions per 900 games | ~37,800 | ~35,300–35,900 |
| Implied positions per game | 37,800 / 900 = **42.0** | 35,387 / 900 ≈ **39.3** (iter_006), 35,924 / 900 ≈ **39.9** (iter_007) |
| Avg game length (moves) | ~42 | ~39.3–39.9 |

**Conclusion:** The position gap is **not** from dropping or filtering positions. It comes from **shorter average game length** in the current run: about **2 fewer moves per game** (~40 vs ~42). So we store one position per move as before, but games end earlier.

---

## 2. How positions are counted (current vs checkme)

- **Per game:** Each move in the loop: encode state → MCTS → store **one** position (canonical) → `apply_action` → `move_count += 1`. After the loop, `game_length = move_count` and `num_positions = len(states) = move_count`. So **num_positions per game = game_length** (one stored position per move).
- **Integration:** `num_positions = sum(s.get("num_positions", 0) for s in summaries)` and per-shard summary uses `num_positions: int(states.shape[0])`. No filtering or de-duplication of positions in the count; every stored state is counted.
- **checkme** uses the same pattern: one append per move when `store_canonical_only` is true; no difference in “when we store” that would reduce count.

So the only way to get fewer total positions with the same number of games is **fewer moves per game** (lower `avg_game_length`).

---

## 3. Code differences that might affect game length (not position counting)

### 3.1 State hashing (affects only `unique_positions` / redundancy)

- **Current** (`src/training/selfplay_optimized.py`): `seen_state_hashes.add(hash(enc.tobytes()))` — hash of **encoded** state.
- **checkme** (`checkme/src/training/selfplay_optimized.py`): `seen_state_hashes.add(hash(st.tobytes()))` — hash of **engine** state `st`.

This only changes redundancy / unique_positions. It does **not** change how many positions are stored or the game length.

### 3.2 Policy target for PCR “fast” moves

- **Current:** No special case for “fast” moves. Every move gets a policy from visit counts; then `pi = (pi * mask).astype(...)` and renormalize. So we always have a non-zero policy and **always append** one position.
- **checkme:** When `use_full_sims` is False (PCR fast move), they set `pi = np.zeros(2026)`. They still **append** the position (states, masks, policies, etc.); only the policy target is zeros. So both codebases store one position per move. No count difference from this.

### 3.3 Dynamic score utility weight (likely relevant to game length)

- **Current config** (`config_best.yaml`): `dynamic_score_utility_weight_schedule` ramps **0.0 → 0.1 (iter 10) → 0.2 (iter 25) → 0.30 (iter 40)**. So at iter 6 and 7, **dynamic_score_utility_weight = 0.0**.
- **Committed manifests** (iter_006, iter_007): `"dynamic_score_utility_weight": 0.0` in `applied_settings.selfplay`.
- **checkme:** Uses `dynamic_score_utility_weight` with default 0.3 and a schedule. If checkme did **not** use a ramp from 0 and instead used 0.3 from iter 0, then MCTS would “push” for margin from the start, which can change move choices and possibly **lengthen** games (e.g. more “point-seeking” play). So: **hypothesis — current ramp (0 for early iters) leads to shorter games (~40 moves) vs constant 0.3 (~42 moves).**

### 3.4 WIN_FIRST

- **Current:** WIN_FIRST is enabled (win-first root selection, DSU gating). This changes which move is chosen at the root when ahead/behind. It could make play more “result-oriented” (e.g. secure win sooner) and thus **shorten** games.
- **checkme:** Need to confirm if WIN_FIRST existed and was enabled there. If checkme had no WIN_FIRST, that could also explain longer games there.

### 3.5 Terminal condition and max length

- Both use `while (not terminal_fast(st)) and (move_count < self.max_game_length)` and the same `max_game_length` (200). No evidence of a bug that would terminate one move early; last move is stored then applied, then loop exits on next iteration. So terminal/max_length logic is not the cause of the count gap.

---

## 4. Hypotheses for root cause (for ChatGPT to refine)

1. **Dynamic score utility ramp (0 for iters 0–9):** With `dynamic_score_utility_weight = 0`, MCTS ignores score and only uses value. That may yield a different policy (e.g. “just win” instead of “win by more”), leading to ~2 fewer moves per game. **Suggestion:** Compare with a run that uses 0.3 from iter 0 (or from iter 1) and see if positions return to ~37,800.
2. **WIN_FIRST:** If checkme did not use WIN_FIRST, enabling it in current could shorten games. **Suggestion:** Temporarily disable WIN_FIRST for one iteration and compare positions/game length.
3. **Interaction:** WIN_FIRST + no score utility (0.0) might together favor “short” wins and fewer moves per game.

---

## 5. Current training run — committed metrics (great detail)

From `runs/patchwork_production/committed/` (iter_006 and iter_007).

### iter_006

**commit_manifest.json**
- iteration: 6  
- timestamp_utc: 2026-03-02T01:20:28.706362Z  
- accepted: true  
- best_model_iteration: 6  
- consecutive_rejections: 0  
- global_step: 1890  
- **num_positions: 35,387**  
- elo_ratings: {}  
- applied_settings.selfplay: games=900, temperature=1.0, policy_target_mode=visits, dirichlet_alpha=0.1, noise_weight=0.25, cpuct=1.45, simulations=192, q_value_weight=0.2, static_score_utility_weight=0.0, **dynamic_score_utility_weight=0.0**, parallel_leaves=64  
- applied_settings.training: lr=0.0016, q_value_weight=0.2, batch_size=1024, amp_dtype=bfloat16  
- applied_settings.replay: window_iterations=8, max_size=300000, newest_fraction=0.25, recency_window=0.0  

**iteration_006.json**
- **selfplay_stats:** num_games=900, **num_positions=35,387**, **avg_game_length=39.318888888888885**, p0_wins=511, p1_wins=389, generation_time=1632.59 s, games_per_minute=33.08, avg_policy_entropy=3.30, avg_top1_prob=0.228, avg_num_legal=151.84, avg_redundancy=0.0937, unique_positions=32,022, avg_root_q=0.00697  
- **train_metrics:** policy_loss=3.006, value_loss=0.288, score_loss=3.626, ownership_loss=0.334, total_loss=3.708, policy_accuracy=0.433, policy_top5_accuracy=0.633, value_mse=0.288, grad_norm=0.603, policy_entropy=3.007, kl_divergence=0.276, ownership_accuracy=0.828, step_skip_rate=0.0  
- **eval_results.vs_previous_best:** total_games=0 (eval skipped or disabled)  
- replay_buffer_positions=258,026, replay_buffer_iterations=7  
- iteration_time_s=2103.98  

### iter_007

**commit_manifest.json**
- iteration: 7  
- timestamp_utc: 2026-03-02T01:52:50.101037Z  
- accepted: true  
- best_model_iteration: 7  
- consecutive_rejections: 0  
- global_step: 2420  
- **num_positions: 35,924**  
- applied_settings.selfplay: same as iter_006, **dynamic_score_utility_weight=0.0**  
- applied_settings.training/replay: same as iter_006  

**iteration_007.json**
- **selfplay_stats:** num_games=900, **num_positions=35,924**, **avg_game_length=39.91555555555556**, p0_wins=513, p1_wins=387, generation_time=1638.51 s, games_per_minute=32.96, avg_policy_entropy=3.22, avg_top1_prob=0.246, avg_num_legal=147.49, avg_redundancy=0.0914, unique_positions=32,584, avg_root_q=0.00464  
- **train_metrics:** policy_loss=3.050, value_loss=0.292, score_loss=3.622, ownership_loss=0.332, total_loss=3.754, policy_accuracy=0.435, policy_top5_accuracy=0.633, value_mse=0.292, grad_norm=0.577, policy_entropy=3.050, kl_divergence=0.269, ownership_accuracy=0.830, step_skip_rate=0.0  
- **eval_results.vs_previous_best:** total_games=0  
- replay_buffer_positions=293,950, replay_buffer_iterations=8  
- iteration_time_s=1938.06  

### Metrics summary table (current run)

| Iter | num_positions | num_games | avg_game_length | p0_wins | p1_wins | games/min | unique_positions | avg_redundancy | policy_loss | value_loss | score_loss | total_loss | pol_acc | value_mse | grad_norm | replay_positions |
|------|----------------|-----------|------------------|---------|---------|-----------|------------------|----------------|-------------|------------|------------|------------|---------|-----------|-----------|-------------------|
| 6    | 35,387         | 900       | 39.32            | 511     | 389     | 33.08     | 32,022           | 0.094          | 3.006       | 0.288      | 3.626      | 3.708     | 0.433   | 0.288    | 0.603    | 258,026           |
| 7    | 35,924         | 900       | 39.92            | 513     | 387     | 32.96     | 32,584           | 0.091         | 3.050       | 0.292      | 3.622      | 3.754     | 0.435   | 0.292    | 0.577    | 293,950           |

---

## 6. What to change to test (suggestions for ChatGPT/user)

1. **Restore ~37,800 positions:** If the hypothesis is “no score utility → shorter games,” try raising dynamic score utility earlier, e.g. set the schedule so `dynamic_score_utility_weight` is 0.3 from iter 0 (or from iter 1) and see if avg_game_length returns to ~42 and positions to ~37,800.
2. **Align with checkme:** Confirm in checkme whether `dynamic_score_utility_weight` was 0.3 from the start and whether WIN_FIRST was disabled; then mirror that in current for an A/B comparison.
3. **Optional consistency fix:** In current, change `seen_state_hashes.add(hash(enc.tobytes()))` to `seen_state_hashes.add(hash(st.tobytes()))` so redundancy/unique_positions are defined the same way as in checkme (engine state), if you want identical semantics. This does **not** fix the position count; it only affects diagnostics.

---

## 7. Files and locations (for code inspection)

- Position storage (one per move): `src/training/selfplay_optimized.py` — loop around lines 386–519, append when `store_canonical_only` (lines 482–495).  
- Per-game num_positions: `int(states.shape[0])` in `selfplay_optimized_integration.py` (shard summary, ~line 857).  
- Stats aggregation: `selfplay_optimized_integration.py` — `_compute_stats`, `num_positions = sum(s.get("num_positions", 0) for s in summaries)` (~line 864).  
- checkme equivalent: `checkme/src/training/selfplay_optimized.py` (game loop and append logic); `checkme/src/training/selfplay_optimized_integration.py` if present.  
- Schedule: `configs/config_best.yaml` — `dynamic_score_utility_weight_schedule` (iter 0 → 0.0, iter 10 → 0.1, iter 25 → 0.2, iter 40 → 0.30).

End of analysis.
