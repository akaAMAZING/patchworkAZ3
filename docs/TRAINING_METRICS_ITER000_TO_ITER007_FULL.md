# Full training metrics: iter 000 → iter 007 (bootstrap to now)

**Run:** `patchwork_production`  
**Sources:** `logs/metadata.jsonl`, `runs/patchwork_production/run_state.json`, `replay_state.json`, `elo_state.json`, `committed/iter_XXX/commit_manifest.json`, `committed/iter_XXX/iteration_XXX.json`

---

## 1. Master list of ALL tracked metrics

Every metric name that appears in this run (by category).

### Run / global state
| Key | Description |
|-----|-------------|
| `timestamp_utc` | ISO UTC when run state or commit was written |
| `last_committed_iteration` | Latest committed iter index |
| `best_iteration` | Iteration of current best model |
| `best_model_path` | Path to best checkpoint |
| `latest_model_path` | Path to latest checkpoint |
| `global_step` | Cumulative training step count |
| `consecutive_rejections` | Rejections in a row (gate) |
| `config_hash` | Hash of config for reproducibility |
| `run_id` | Run identifier |
| `seed` | RNG seed |

### Commit manifest (per iteration, in committed/iter_XXX/)
| Key | Description |
|-----|-------------|
| `iteration` | Iteration index |
| `timestamp_utc` | Commit time (UTC) |
| `accepted` | Whether model was accepted by gate |
| `best_model_iteration` | Best iter after this commit |
| `consecutive_rejections` | Value after commit |
| `global_step` | Global step after this iter |
| `num_positions` | Positions from this iter's self-play |
| `elo_ratings` | Elo state (dict) |
| `applied_settings` | Selfplay/training/replay settings used |
| `commit_method` | e.g. "rename" |

### Self-play stats (per iteration)
| Key | Description |
|-----|-------------|
| `num_games` | Number of games generated |
| `num_positions` | Total positions (states) stored |
| `avg_game_length` | Mean moves per game |
| `p0_wins` | Wins by player 0 |
| `p1_wins` | Wins by player 1 |
| `generation_time` | Self-play wall time (seconds) |
| `games_per_minute` | Throughput (games/min) |
| `avg_policy_entropy` | Mean policy entropy over moves (canary) |
| `avg_top1_prob` | Mean top-1 action probability (canary) |
| `avg_num_legal` | Mean number of legal actions per position |
| `avg_redundancy` | 1 - unique/total positions (0 = all unique) |
| `unique_positions` | Count of distinct positions (by hash) |
| `avg_root_q` | Mean MCTS root Q (value + score_utility) |

### Train metrics (per iteration, epoch-averaged)
| Key | Description |
|-----|-------------|
| `policy_loss` | Cross-entropy policy loss |
| `value_loss` | MSE value loss |
| `score_loss` | Score head loss (e.g. CE over bins) |
| `ownership_loss` | Auxiliary ownership loss |
| `total_loss` | Weighted total loss |
| `policy_accuracy` | Fraction correct top-1 policy |
| `policy_top5_accuracy` | Fraction top-5 policy |
| `value_mse` | Same as value_loss (MSE) |
| `grad_norm` | Gradient norm (e.g. L2) |
| `policy_entropy` | Mean policy entropy (train) |
| `kl_divergence` | KL(policy \|\| target) |
| `ownership_accuracy` | Ownership head accuracy |
| `step_skip_rate` | Fraction of steps skipped (e.g. NaN guard) |

### Eval results (per iteration; when eval runs)
| Key | Description |
|-----|-------------|
| `vs_previous_best` | Dict: model vs previous best |
| `vs_pure_mcts` | Dict: model vs pure MCTS |
| `model_wins` | Wins by candidate model |
| `baseline_wins` | Wins by baseline |
| `ties` | Tie count |
| `total_games` | Eval games played |
| `win_rate` | model_wins / total_games |
| `avg_game_length` | Mean game length in eval |
| `avg_score_diff` | Mean score difference (raw) |
| `avg_model_score_margin` | Mean (model_score - baseline_score) |
| `avg_win_margin` | Mean margin when model wins |
| `avg_loss_margin` | Mean margin when model loses |
| `results` | Per-game list (excluded from metadata.jsonl) |
| `sprt` | If SPRT: accept, reject, llr, games_played |

### Replay buffer
| Key | Description |
|-----|-------------|
| `replay_buffer_positions` | Total positions in buffer after iter |
| `replay_buffer_iterations` | Number of iters in sliding window |
| `replay_state` (file) | Per-iter: iteration, path, positions |

### Metadata.jsonl (per-iteration line)
| Key | Description |
|-----|-------------|
| `iteration` | Iter index |
| `timestamp_utc` | When metadata line was written |
| `config_hash` | Config hash |
| `config_path` | Config file path |
| `best_model_hash` | File hash of best checkpoint |
| `accepted` | Gate accepted |
| `global_step` | After this iter |
| `iter_time_s` | Full iteration wall time (s) |
| `train` | Train metrics dict (no "results") |
| `selfplay` | Self-play stats dict |
| `eval_vs_best_wr` | Win rate vs best (if ran) |
| `eval_vs_best_margin` | Avg margin vs best |
| `eval_vs_mcts_wr` | Win rate vs pure MCTS (if ran) |
| `best_model` | Path to best model |
| `replay_positions` | Replay buffer total positions |
| `consecutive_rejections` | After this iter |
| `sprt` | SPRT result if used |
| `elo` | Elo rating if tracked |

### Applied settings (snapshot per iter)
| Key | Description |
|-----|-------------|
| `applied_settings.selfplay` | games, temperature, policy_target_mode, dirichlet_alpha, noise_weight, cpuct, simulations, q_value_weight, static_score_utility_weight, dynamic_score_utility_weight, parallel_leaves |
| `applied_settings.training` | lr, q_value_weight, batch_size, amp_dtype |
| `applied_settings.replay` | window_iterations, max_size, newest_fraction, recency_window |

---

## 2. Run state (current)

**File:** `runs/patchwork_production/run_state.json`

```json
{
  "timestamp_utc": "2026-03-02T01:52:50.189934Z",
  "last_committed_iteration": 7,
  "best_iteration": 7,
  "best_model_path": "runs\\patchwork_production\\committed\\iter_007\\iteration_007.pt",
  "latest_model_path": "checkpoints\\latest_model.pt",
  "global_step": 2420,
  "consecutive_rejections": 0,
  "config_hash": "7e809564bb76",
  "run_id": "patchwork_production",
  "seed": 42
}
```

---

## 3. Replay buffer state (per-iteration positions)

**File:** `runs/patchwork_production/replay_state.json`

| iteration | path | positions |
|-----------|------|-----------|
| 0 | runs\patchwork_production\committed\iter_000\selfplay.h5 | 38,191 |
| 1 | runs\patchwork_production\committed\iter_001\selfplay.h5 | 38,544 |
| 2 | runs\patchwork_production\committed\iter_002\selfplay.h5 | 37,322 |
| 3 | runs\patchwork_production\committed\iter_003\selfplay.h5 | 36,927 |
| 4 | runs\patchwork_production\committed\iter_004\selfplay.h5 | 36,300 |
| 5 | runs\patchwork_production\committed\iter_005\selfplay.h5 | 35,355 |
| 6 | runs\patchwork_production\committed\iter_006\selfplay.h5 | 35,387 |
| 7 | runs\patchwork_production\committed\iter_007\selfplay.h5 | 35,924 |

**Cumulative (after each iter):** 38,191 → 76,735 → 114,057 → 150,984 → 187,284 → 222,639 → 258,026 → 293,950

---

## 4. Elo state

**File:** `runs/patchwork_production/elo_state.json`

```json
{
  "initial_rating": 1500,
  "k_factor": 32,
  "ratings": {}
}
```
(Elo disabled in config; ratings empty.)

---

## 5. Full metrics table: iter 000 → iter 007

All values below from `logs/metadata.jsonl` (and, for iter 6–7, committed iteration JSON where needed). Evaluation was disabled (games_vs_best=0, games_vs_pure_mcts=0), so eval fields are 0 or null.

### 5.1 Self-play (every metric)

| iter | num_games | num_positions | avg_game_length | p0_wins | p1_wins | generation_time_s | games_per_min | avg_policy_entropy | avg_top1_prob | avg_num_legal | avg_redundancy | unique_positions | avg_root_q |
|------|-----------|--------------|-----------------|---------|---------|-------------------|---------------|--------------------|---------------|---------------|---------------|------------------|------------|
| 0 | 900 | 38,191 | 42.434 | 433 | 467 | 718.99 | 75.11 | 2.650 | 0.355 | 97.93 | 0.149 | 32,377 | 0.000 |
| 1 | 900 | 38,544 | 42.827 | 455 | 445 | 1679.53 | 32.15 | 2.500 | 0.358 | 99.93 | 0.148 | 32,705 | 0.00041 |
| 2 | 900 | 37,322 | 41.469 | 492 | 408 | 1616.78 | 33.40 | 2.522 | 0.356 | 104.38 | 0.145 | 31,802 | 0.00124 |
| 3 | 900 | 36,927 | 41.030 | 482 | 418 | 1581.79 | 34.14 | 2.599 | 0.337 | 109.13 | 0.138 | 31,736 | 0.00406 |
| 4 | 900 | 36,300 | 40.333 | 506 | 394 | 1618.73 | 33.36 | 2.868 | 0.286 | 120.04 | 0.125 | 31,652 | 0.00949 |
| 5 | 900 | 35,355 | 39.283 | 477 | 423 | 1604.50 | 33.66 | 3.222 | 0.231 | 140.48 | 0.102 | 31,684 | 0.01069 |
| 6 | 900 | 35,387 | 39.319 | 511 | 389 | 1632.59 | 33.08 | 3.297 | 0.228 | 151.84 | 0.094 | 32,022 | 0.00697 |
| 7 | 900 | 35,924 | 39.916 | 513 | 387 | 1638.51 | 32.96 | 3.221 | 0.246 | 147.49 | 0.091 | 32,584 | 0.00464 |

### 5.2 Training (every metric)

| iter | policy_loss | value_loss | score_loss | ownership_loss | total_loss | policy_accuracy | policy_top5_accuracy | value_mse | grad_norm | policy_entropy | kl_divergence | ownership_accuracy | step_skip_rate |
|------|-------------|------------|------------|----------------|------------|-----------------|----------------------|-----------|-----------|----------------|---------------|--------------------|-----------------|
| 0 | 2.678 | 0.464 | 4.257 | 0.558 | 3.651 | 0.351 | 0.477 | 0.464 | 4.364 | 2.677 | 0.101 | 0.755 | 0.0 |
| 1 | 2.751 | 0.320 | 3.735 | 0.347 | 3.497 | 0.382 | 0.538 | 0.320 | 2.149 | 2.748 | 0.238 | 0.824 | 0.0 |
| 2 | 2.754 | 0.301 | 3.649 | 0.334 | 3.469 | 0.413 | 0.590 | 0.301 | 1.591 | 2.753 | 0.266 | 0.829 | 0.0 |
| 3 | 2.784 | 0.296 | 3.600 | 0.331 | 3.489 | 0.428 | 0.619 | 0.296 | 1.053 | 2.783 | 0.281 | 0.830 | 0.0 |
| 4 | 2.838 | 0.289 | 3.602 | 0.332 | 3.538 | 0.434 | 0.635 | 0.289 | 0.772 | 2.838 | 0.281 | 0.829 | 0.0 |
| 5 | 2.933 | 0.288 | 3.621 | 0.334 | 3.633 | 0.434 | 0.637 | 0.288 | 0.686 | 2.934 | 0.280 | 0.828 | 0.0 |
| 6 | 3.006 | 0.288 | 3.626 | 0.334 | 3.708 | 0.433 | 0.633 | 0.288 | 0.603 | 3.007 | 0.276 | 0.828 | 0.0 |
| 7 | 3.050 | 0.292 | 3.623 | 0.333 | 3.754 | 0.435 | 0.633 | 0.292 | 0.577 | 3.050 | 0.269 | 0.830 | 0.0 |

### 5.3 Global step, time, replay, gate

| iter | global_step | iter_time_s | replay_positions | consecutive_rejections | accepted | best_model_hash |
|------|-------------|-------------|------------------|------------------------|----------|-----------------|
| 0 | 70 | 822.5 | 38,191 | 0 | true | 8814f63ec4f486fb |
| 1 | 208 | 1819.8 | 76,735 | 0 | true | 815e12cb9006e743 |
| 2 | 414 | 1771.3 | 114,057 | 0 | true | 0b112add0d441abe |
| 3 | 686 | 1745.8 | 150,984 | 0 | true | 96eff384bbb21f61 |
| 4 | 1,024 | 1800.9 | 187,284 | 0 | true | 9957f9b7a3b2fbe0 |
| 5 | 1,426 | 1794.0 | 222,639 | 0 | true | 3f60ed5dd91022f4 |
| 6 | 1,890 | 2104.0 | 258,026 | 0 | true | de91c1c2c0257a98 |
| 7 | 2,420 | 1938.1 | 293,950 | 0 | true | 075308b1a9ed4287 |

### 5.4 Eval (this run: eval disabled)

| iter | eval_vs_best_wr | eval_vs_best_margin | eval_vs_mcts_wr |
|------|------------------|----------------------|-----------------|
| 0 | null | null | null |
| 1 | 0.0 | 0.0 | null |
| 2 | 0.0 | 0.0 | null |
| 3 | 0.0 | 0.0 | null |
| 4 | 0.0 | 0.0 | null |
| 5 | 0.0 | 0.0 | null |
| 6 | 0.0 | 0.0 | null |
| 7 | 0.0 | 0.0 | null |

(Eval vs best ran but with 0 games; vs MCTS not run.)

---

## 6. Iteration summary JSON (iter 6 & 7) — extra fields

For iter 6 and 7 the full `iteration_XXX.json` in committed also includes:

- **iteration_time_s** (wall): 2103.98 (iter 6), 1938.06 (iter 7)
- **replay_buffer_iterations**: 7 (iter 6), 8 (iter 7)
- **eval_results.vs_previous_best**: model_wins, baseline_wins, ties, total_games, win_rate, avg_game_length, avg_score_diff, avg_model_score_margin, avg_win_margin, avg_loss_margin, results (empty) — all 0 or empty because eval games = 0
- **best_model / latest_model / latest_checkpoint**: paths (staging at write time)

---

## 7. Commit manifest (iter 6 & 7)

### iter_006 commit_manifest.json
- iteration: 6  
- timestamp_utc: 2026-03-02T01:20:28.706362Z  
- accepted: true  
- best_model_iteration: 6  
- consecutive_rejections: 0  
- global_step: 1890  
- num_positions: 35,387  
- elo_ratings: {}  
- commit_method: "rename"  
- applied_settings: selfplay (games=900, temperature=1.0, policy_target_mode=visits, dirichlet_alpha=0.1, noise_weight=0.25, cpuct=1.45, simulations=192, q_value_weight=0.2, static_score_utility_weight=0.0, dynamic_score_utility_weight=0.0, parallel_leaves=64), training (lr=0.0016, q_value_weight=0.2, batch_size=1024, amp_dtype=bfloat16), replay (window_iterations=8, max_size=300000, newest_fraction=0.25, recency_window=0.0)

### iter_007 commit_manifest.json
- iteration: 7  
- timestamp_utc: 2026-03-02T01:52:50.101037Z  
- accepted: true  
- best_model_iteration: 7  
- consecutive_rejections: 0  
- global_step: 2420  
- num_positions: 35,924  
- elo_ratings: {}  
- commit_method: "rename"  
- applied_settings: (same structure as iter_006)

---

## 8. TensorBoard scalar tags (all that are written)

From `main.py` and trainer:

- **eval/**  
  elo_rating, win_rate_vs_mcts, score_margin_vs_mcts, game_length_vs_mcts, win_rate_as_p0_vs_mcts, win_rate_as_p1_vs_mcts, win_rate_vs_best, score_margin_vs_best, win_rate_as_p0_vs_best, win_rate_as_p1_vs_best  
- **selfplay/**  
  games_per_min, num_positions  
- **buffer/**  
  total_positions, num_iterations  
- **iter/**  
  (every key in train_metrics): total_loss, policy_loss, value_loss, score_loss, ownership_loss, policy_accuracy, policy_top5_accuracy, value_mse, grad_norm, policy_entropy, kl_divergence, ownership_accuracy, step_skip_rate  

---

## 9. Per-game / per-position metrics (not in metadata)

These exist only inside self-play game summaries or HDF5, not in metadata.jsonl:

- **Per game:** game_length, winner, final_score_diff, redundancy, unique_positions, avg_policy_entropy, avg_top1_prob, avg_num_legal, avg_root_q  
- **Per position (in HDF5):** states, action_masks, policies, values, score_margins, ownerships, stored_flip_types, slot_piece_ids (if store_canonical_only)

---

End of full metrics dump (iter 000 → 007).
