# Full Metrics Reference — Patchwork AlphaZero

This document is the **single source of truth** for every metric logged per iteration. It includes the beat-humans metrics added for packing quality and search health (PW). Use it with **training_metrics_per_iteration.csv**, **iteration_XXX.json**, and **metadata.jsonl** for full understanding.

**Where metrics are written:** TensorBoard (`train/*`, `val/*`, `iter/*`, `selfplay/*`, `buffer/*`), committed `committed/iter_###/iteration_###.json` (summary dict), and `metadata.jsonl` (each line = one iteration). **build_training_csv.py** flattens iteration JSON into CSV and will include any key present in `train_metrics`, `selfplay_stats`, `eval_results`, and `applied_settings`.

---

## Design choice: no automated evaluation

**Evaluation is intentionally disabled in production configs.** In YAML we set:

- `evaluation.games_vs_best: 0`
- `evaluation.games_vs_pure_mcts: 0`

So **no** automated win-rate or Elo evaluation runs during training. This is a **design choice**: we **manually evaluate** (e.g. head-to-head games, A/B tests with `tools/ab_test_*`, or the GUI). The pipeline still writes `eval_results` when baselines exist; with 0 games, eval blocks are effectively no-ops and no eval_* series are produced. All strength signals come from **selfplay stats** and the **beat-humans metrics** below.

---

## 1) Train metrics (epoch-aggregated per iteration)

**Source:** `src/training/trainer.py` — aggregated over all training steps in the iteration (epoch loop). Returned as `train_metrics` to `main.py` and stored in iteration summary + metadata.

| Key | Definition | Notes |
|-----|------------|--------|
| `total_loss` | Weighted sum of policy + value + score + ownership (and any other heads) | Coarse progress; diagnose via component losses. |
| `policy_loss` | Cross-entropy of target π vs model policy p | Policy–MCTS alignment. |
| `value_loss` | MSE between predicted value and target value | Value head fit. |
| `score_loss` | Distributional CE over 201-bin score head (soft Gaussian targets from tanh margin) | Score head; weight 0.1. |
| `ownership_loss` | BCE on 2-channel ownership head (P0 filled, P1 filled) | Auxiliary; weight 0.15. |
| `policy_accuracy` | Top-1 accuracy vs target argmax | Match to MCTS top move. |
| `policy_top5_accuracy` | Target argmax in model top-5 | Ranking quality. |
| `value_mse` | Same as value loss (MSE) | Direct value signal. |
| `grad_norm` | L2 norm of gradients (before clipping) | Stability; clipped at max_grad_norm 1.0. |
| `policy_entropy` | Shannon entropy of model policy (masked) | Sharpness / collapse detector. |
| `kl_divergence` | KL(target ‖ policy) per batch, averaged | Alignment pressure. |
| `target_entropy` | Entropy of target π | From MCTS visit distribution. |
| `policy_cross_entropy` | CE(π, p); should ≈ H(π) + KL | Decomposition check. |
| `step_skip_rate` | Fraction of steps skipped (e.g. AMP overflow) | Target 0. |
| `ownership_accuracy` | Binary accuracy (threshold 0.5) on ownership head | Over valid cells only (masked). |
| `ownership_filled_fraction_mean` | Mean filled fraction (ownership-related) | Diagnostic. |
| `ownership_accuracy_all_filled_baseline` | Baseline when all predicted filled | Diagnostic. |
| `ownership_empty_recall` | Recall on empty cells | Ownership diagnostic. |
| `ownership_empty_precision` | Precision on empty cells | Ownership diagnostic. |
| `ownership_balanced_accuracy` | Balanced accuracy (ownership) | Ownership diagnostic. |
| `ownership_mae_empty_count` | MAE of predicted vs true empty count | Ownership diagnostic. |
| `approx_identity_check` | Mean \|H(π)+KL−CE\| over samples | Math consistency; should be tiny. |

**TensorBoard:** Step-level scalars under `train/*` (e.g. `train/total_loss`, `train/policy_loss`) and `val/*`; iteration-level **epoch-averaged** values are not re-logged under `train/` but are in the iteration summary. The **iter/** whitelist (see below) logs a subset to TensorBoard as `iter/<key>`.

---

## 2) Validation metrics (epoch-aggregated per iteration)

**Source:** `src/training/trainer.py` — `validate()` over the val split. Same structure as train (total_loss, policy_loss, value_loss, score_loss, ownership_loss, policy_accuracy, policy_top5_accuracy, policy_entropy, value_mse). Used for early stopping / best model only when enabled; not all val keys are necessarily in the iteration JSON — check trainer return value. Val is a **replay skew detector**, not out-of-distribution generalization.

---

## 3) Iter metrics (TensorBoard whitelist)

**Source:** `train_metrics` from the trainer. Only the following keys are logged to TensorBoard as `iter/<key>` (in `main.py`):

- `kl_divergence`
- `policy_entropy`
- `target_entropy`
- `policy_cross_entropy`
- `value_mse`
- `step_skip_rate`
- `approx_identity_check`
- `ownership_accuracy`
- `ownership_filled_fraction_mean`
- `ownership_accuracy_all_filled_baseline`
- `ownership_empty_recall`
- `ownership_empty_precision`
- `ownership_balanced_accuracy`
- `ownership_mae_empty_count`

All of `train_metrics` is still stored in the iteration summary and metadata; the whitelist only controls what gets a dedicated `iter/*` scalar in TB.

---

## 4) Selfplay metrics (per iteration)

**Source:** `src/training/selfplay_optimized_integration.py` — `_compute_stats(summaries, generation_time)` aggregates per-game summaries from workers (full game dict or shard metadata). Result is `selfplay_stats` in the iteration summary and metadata.

### 4.1 Core selfplay

| Key | Definition |
|-----|------------|
| `num_games` | Number of games generated this iteration. |
| `num_positions` | Total positions (states) generated. |
| `avg_game_length` | Mean moves/plies per game. |
| `p0_wins` | Wins for player 0. |
| `p1_wins` | Wins for player 1. |
| `generation_time` | Wall time for selfplay (seconds). |
| `games_per_minute` | num_games / (generation_time / 60). |
| `avg_policy_entropy` | Mean root policy entropy (search) over moves. |
| `avg_top1_prob` | Mean probability on top-1 action. |
| `avg_num_legal` | Mean number of legal actions at root. |
| `avg_redundancy` | 1 − (unique positions / moves); collapse signal. |
| `unique_positions` | Count of unique (hashed) positions. |
| `avg_root_q` | Mean root Q value from MCTS. |

### 4.2 Beat-humans: packing / quilt quality

Computed at **terminal state** for both players; definitions in `src/utils/packing_metrics.py` (same as `tools/ab_test_*`). 4-neighbor BFS for components; isolated 1×1 = empty cell with zero empty neighbors.

| Key | Definition |
|-----|------------|
| `selfplay_avg_final_empty_squares_mean` | Mean empty squares over all players and games. |
| `selfplay_avg_final_empty_components_mean` | Mean number of connected empty components (4-neighbor BFS). |
| `selfplay_avg_final_isolated_1x1_holes_mean` | Mean isolated 1×1 holes (zero empty neighbors). |
| `selfplay_p50_final_empty_squares_mean` | Median of per-game mean (across P0/P1) empty squares. |
| `selfplay_p90_final_empty_squares_mean` | 90th percentile of per-game mean empty squares (tail risk). |
| `selfplay_p50_final_empty_components_mean` | Median of per-game mean empty components. |
| `selfplay_p90_final_empty_components_mean` | 90th percentile (tail fragmentation). |
| `selfplay_p50_final_isolated_1x1_holes_mean` | Median of per-game mean isolated holes. |
| `selfplay_p90_final_isolated_1x1_holes_mean` | 90th percentile (catastrophe rate). |
| `selfplay_avg_final_empty_squares_abs_diff` | Mean over games of \|P0 empties − P1 empties\|. |
| `selfplay_avg_final_empty_components_abs_diff` | Mean over games of \|P0 components − P1 components\|. |
| `selfplay_avg_final_isolated_1x1_holes_abs_diff` | Mean over games of \|P0 isolated − P1 isolated\|. |

**Interpretation:** Fewer empties, fewer components, fewer isolated holes ⇒ better packing. p90 = worst 10% of games; improving p90 = fewer catastrophic quilts. abs_diff = asymmetry (systematic bias between players).

### 4.3 Beat-humans: search health (PW)

Collected **per model move** when MCTS runs: root legal count and PW-limited expanded child count (`src/mcts/alphazero_mcts_optimized.py`: `get_root_legal_count()`, `get_root_expanded_count()`).

| Key | Definition |
|-----|------------|
| `selfplay_avg_root_legal_count` | Mean legal actions at root over all model moves. |
| `selfplay_avg_root_expanded_count` | Mean children expanded at root (PW cap). |
| `selfplay_avg_root_expanded_ratio` | Mean of (expanded / max(legal, 1)) per move. |
| `selfplay_p90_root_legal_count` | 90th percentile of root legal count. |
| `selfplay_p90_root_expanded_ratio` | 90th percentile of expanded ratio. |

**Interpretation:** expanded_ratio = search breadth dial. Too high ⇒ too wide/shallow; too low ⇒ over-pruning. PW is enabled in config (see CODE_CONTEXT.md).

---

## 5) Buffer metrics

**Source:** `main.py` from `self.replay_buffer`.

| Key | Definition |
|-----|------------|
| `total_positions` | Positions stored in replay (capped by max_size). |
| `num_iterations` | Number of iterations represented in the buffer. |

Logged to TensorBoard as `buffer/total_positions` and `buffer/num_iterations`.

---

## 6) Evaluation metrics (when evaluation is enabled)

With `games_vs_best: 0` and `games_vs_pure_mcts: 0`, no eval games are run and **eval_*** fields are typically absent or empty. When eval is enabled in other configs, typical keys include win_rate, avg_model_score_margin, games_played, and optionally sprt (accept/reject, llr). See evaluation config and `main.py` for exact structure.

---

## 7) Applied settings (flattened config)

**Source:** `main.py` — flattened snapshot of selfplay/training/replay/adaptive_games (and any other applied) settings per iteration. Keys are prefixed (e.g. `selfplay_*`, `training_*`, `replay_*`). Used to correlate metric changes with schedule (LR, cpuct, games, dynamic_score_utility_weight, PW params, etc.).

---

## 8) Progressive widening (PW) — config summary

Current production config (e.g. `config_continue_from_iter70.yaml` / `config_best.yaml`) implements **progressive widening** in MCTS:

- `selfplay.mcts.progressive_widening.enabled: true`
- `k_root: 64` — max root children (e.g. top-K BUY by PUCT).
- `k0: 32`, `k_sqrt_coef: 8` — expansion formula (PW limits how many children are expanded).
- `always_include_pass: true`, `always_include_patch: true`

So at root we expand at most **k_root** children (and pass/patch are always included). The beat-humans metrics `selfplay_avg_root_expanded_count` and `selfplay_avg_root_expanded_ratio` directly reflect this: they show how much of the legal action space is actually considered and help tune PW (e.g. k_root, k_sqrt_coef) for depth vs breadth.

---

## 9) CSV and JSON layout

- **iteration_XXX.json:** Contains `iteration`, `selfplay_stats`, `train_metrics`, `eval_results`, `applied_settings`, `replay_buffer_positions`, `replay_buffer_iterations`, `accepted`, `iteration_time_s`, etc. Every key in `selfplay_stats` (including all beat-humans keys) and every key in `train_metrics` is present when produced.
- **metadata.jsonl:** One JSON object per line; same structure with `selfplay`, `train`, etc. (with optional truncation of large lists).
- **build_training_csv.py:** Flattens `train_metrics` → `train_<key>`, `selfplay_stats` → `selfplay_<key>` (e.g. `selfplay_avg_final_empty_squares_mean`), `eval_results` → eval_*, `applied_settings` → flattened with prefix. No fixed column list; all keys in the JSON are included. So the CSV will automatically gain columns for the new beat-humans metrics when those keys exist in committed iteration JSON.

---

**End.**
