# tune_hyperparams.py — Parameter Inflation Verification

This document verifies that none of the 5 optimized hyperparameters **directly inflate** the optimization signal (score margin). "Direct inflation" means: the parameter is used in the formula that computes the metric, such that changing the param would change the metric even for identical game outcomes.

## Optimization Signal

| Signal | Purpose | Computation |
|--------|---------|-------------|
| **Objective** | What Optuna maximizes | `avg_model_score_margin` from `evaluate_final_strength()` |
| **Pruning** | Early stopping of bad trials | `policy_accuracy` from training |

---

## Data Flow: Score Margin (Objective)

```
evaluate_final_strength(config_path, trial_result, eval_games)
  → Evaluator(config).evaluate_vs_baseline(best_path, iter1_path, num_games)
  → For each game: _play_match() → terminal state
  → p0_score = compute_score_fast(state, 0)   # game engine
  → p1_score = compute_score_fast(state, 1)   # game engine
  → model_score_margin = model_score - baseline_score   # raw subtraction
  → avg_model_score_margin = mean(margins)
  → objective returns margin
```

**Critical:** The scores come from `compute_score_fast()` in `patchwork_engine.py`:
```python
return buttons - 2 * empty + bonus   # pure game rules, no config
```
No hyperparameter appears in this formula.

---

## Per-Parameter Verification

### 1. learning_rate
- **Where used:** Training optimizer (`src/training/`)
- **Used in eval?** No. Evaluation loads trained model and runs games. No LR in game loop or score computation.
- **Direct inflation:** ❌ None. LR only affects training dynamics, not the metric formula.

### 2. cpuct
- **Where used:** MCTS UCB exploration (`create_optimized_mcts` → MCTSConfig). Eval overrides with `eval_mcts.cpuct` (same for all trials).
- **Used in score computation?** No. Affects which moves MCTS chooses during games. Final scores come from `compute_score_fast(state)` — game rules only.
- **Direct inflation:** ❌ None. cpuct influences game outcomes indirectly (different moves → different scores) but is not in the margin formula.

### 3. score_scale
- **Where used:** `terminal_value_from_scores()` → `tanh(diff / score_scale)` for value targets (training labels + MCTS terminal backup)
- **Used in score computation?** No. Evaluation uses `compute_score_fast()` for scores. `evaluation.py` never references `score_scale`. Margin = raw `model_score - baseline_score`.
- **Direct inflation:** ❌ None. score_scale affects training targets and MCTS backup; it does not appear in the evaluation metric.

### 4. dirichlet_alpha
- **Where used:** MCTS root exploration noise (`create_optimized_mcts` → MCTSConfig.root_dirichlet_alpha)
- **Used in score computation?** No. Affects move selection during games. Scores from game engine only.
- **Direct inflation:** ❌ None.

### 5. q_value_weight
- **Where used:** Self-play data labeling (`selfplay_optimized.py`) — mixes MCTS root Q into value targets for training
- **Used in eval?** No. Evaluation runs MCTS games; q_value_weight is only for creating training data. MCTSConfig has no q_value_weight.
- **Used in score computation?** No.
- **Direct inflation:** ❌ None.

---

## Pruning Signal: policy_accuracy

```
policy_accuracy = (pred_actions == target_actions).float().mean()
```
- **Source:** `src/network/model.py` — compares network argmax to MCTS visit-count target
- **Params in formula?** None. It's a standard accuracy metric. Params affect *what* the targets are (through MCTS), but no param is multiplied/added into the accuracy value.
- **Direct inflation:** ❌ None.

---

## Config Used During Evaluation

`evaluate_final_strength()` loads `config_path` (trial's config) and passes it to `Evaluator(config)`. The Evaluator:
- Uses `eval_mcts.simulations` and `eval_mcts.cpuct` (overrides selfplay values; from base config, same across trials)
- Uses `selfplay.score_scale`, `selfplay.mcts.root_dirichlet_alpha` for MCTS (trial's sampled values)

Both model and baseline use the **same** trial config during eval. The metric is still raw game scores — the config only affects move selection, not how we compute scores from the terminal state.

---

## Conclusion

**All 5 parameters are safe.** None directly inflate the optimization signal. The score margin is computed purely from game-engine scores (`compute_score_fast`); no hyperparameter appears in that computation. Parameters affect model behavior (training + MCTS) and thus *indirectly* affect outcomes — which is the intended optimization target.
