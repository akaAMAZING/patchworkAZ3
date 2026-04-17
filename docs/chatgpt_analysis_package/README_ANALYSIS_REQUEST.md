# Patchwork AlphaZero — Data Package for AI Analysis

This folder is a **complete, maximum data package** for an AI (e.g. ChatGPT) to understand and analyze our training. **Omit nothing.** Use the CSVs, config, and code context as the single source of truth.

**No automated evaluation:** We set `evaluation.games_vs_best: 0` and `evaluation.games_vs_pure_mcts: 0` by design. **Evaluation is manual** (head-to-head games, A/B tests with `tools/ab_test_*`, or the GUI). Strength is judged via selfplay stats and the beat-humans metrics (packing quality + search health). See **FULL_METRICS.md** for the full metric set.

---

## ⚠️ Stall concern: score loss, ownership loss, ownership accuracy

We suspect **score loss** and/or **ownership loss / ownership accuracy** may be **stalling** (e.g. score_loss flat, ownership_accuracy stuck around ~85–86%). Please:

- **Plot and interpret** `train_score_loss`, `train_ownership_loss`, and **`train_ownership_accuracy`** (and optionally `train_value_mse`) over iterations in **training_metrics_per_iteration.csv**.
- **Reason** about causes using **CODE_CONTEXT.md** (how score targets are built, how ownership loss/accuracy are computed, and where masking applies).
- Suggest concrete changes (e.g. weights, target sigma, head capacity, or data validity) if you see a plateau.

All other metrics (policy loss, KL, grad_norm, selfplay stats, etc.) should still be analyzed for overall training health.

---

## Files in this package

| File | Description |
|------|-------------|
| **FULL_METRICS.md** | **Full metrics reference:** every train, val, iter, selfplay, and buffer metric; **beat-humans** (packing: empty_squares, empty_components, isolated_1x1_holes, p50/p90, abs_diff; search: root_legal_count, root_expanded_count, root_expanded_ratio, p90). Design note: no automated eval (manual evaluation). PW config summary. CSV/JSON layout. Use this for complete understanding. |
| **config_best.yaml** | **Up-to-date** full config: hardware, data encoding, network, training (optimizer, LR, batch, AMP, **score_loss_weight: 0.1**, **ownership_loss_weight: 0.15**, EMA), selfplay (MCTS, **progressive_widening** enabled: k_root 64, k0 32, k_sqrt_coef 8, win-first, schedules), replay buffer, evaluation (games_vs_* = 0 by design), **all iteration schedules**. |
| **training_metrics_per_iteration.csv** | One row per iteration. **Every metric** from iteration JSON: train_* (total_loss, policy_loss, value_loss, score_loss, ownership_loss, policy_accuracy, value_mse, grad_norm, policy_entropy, kl_divergence, step_skip_rate, ownership_accuracy, etc.); selfplay_* (num_games, num_positions, avg_game_length, p0_wins, p1_wins, generation_time, games_per_minute, avg_policy_entropy, avg_top1_prob, avg_num_legal, avg_redundancy, unique_positions, avg_root_q, **plus all beat-humans keys** when present: selfplay_avg_final_empty_squares_mean, selfplay_avg_final_empty_components_mean, selfplay_avg_final_isolated_1x1_holes_mean, p50/p90 variants, *_abs_diff, selfplay_avg_root_legal_count, selfplay_avg_root_expanded_count, selfplay_avg_root_expanded_ratio, selfplay_p90_root_*); eval_* (when present); applied_*; iteration_time_s, replay_buffer_*, accepted. **build_training_csv.py** picks up all keys in the JSON. |
| **training_metadata_constants.csv** | Constants not per-iteration: EMA decay, score/ownership weights, score_utility_scale, score_target_sigma, max_grad_norm; **PW**: k_root, k0, k_sqrt_coef. Short notes on score/ownership stall. |
| **CODE_CONTEXT.md** | **Code snippets and optimizations:** score/ownership loss and accuracy, score target building, D4, FiLM, EMA, prefetch, win-first MCTS, **progressive widening (PW)**, **beat-humans metrics** (packing_metrics.py, selfplay terminal, integration _compute_stats), **no automated evaluation**. File map. |
| **INPUT_CHANNELS_MARKOV.md** | **Fully Markov input channels in detail:** 56 spatial (32 encoder + 24 legalTL), channel-by-channel layout, x_global (61), x_track (8×54), shop (33, 10). Source-of-truth constants and how the model uses them. |
| **README_ANALYSIS_REQUEST.md** | This file. |
| **build_training_csv.py** | Script to regenerate training_metrics_per_iteration.csv from committed `iteration_XXX.json` (no fixed iter limit; includes all keys in train_metrics and selfplay_stats, including beat-humans). |

---

## Input channels and network architecture

- **Input channels:** **56**. Data encoding: `gold_v2_32ch`, `expected_spatial_channels: 32`; 56 = 32 encoder + 24 legalTL. Full channel-by-channel semantics: **INPUT_CHANNELS_MARKOV.md**.
- **Architecture (from config):**
  - **Trunk:** 18 residual blocks, 128 channels, BatchNorm, ReLU, **Squeeze-and-Excitation (se_ratio: 0.0625)**.
  - **FiLM:** global (61), track (8×54 → 432), shop (128); `film_per_block: true`, `film_track_use_conv: true`, `film_global_inject_dim: 64` into value + pass-logit.
  - **Policy:** Factorized: pass (1) + patch (81) + buy (24×81) = **2026** actions.
  - **Value:** scalar (tanh). **Score:** 201-bin categorical (distributional). **Ownership:** 2 channels (P0 filled, P1 filled), BCE.
  - **Training:** AdamW, weight_decay 0.0002, max_grad_norm 1.0, AMP bfloat16, EMA 0.999. Loss weights: policy 1.0, value 1.0, **score 0.1**, **ownership 0.15**.

---

## Questions we want the AI to answer

1. **Is training progressing solidly?**  
   Using **training_metrics_per_iteration.csv** (iters 0–70): trends in total_loss, policy_loss, value_loss, **score_loss**, **ownership_loss**, policy_accuracy, **ownership_accuracy**, KL, grad_norm, selfplay stats. Are losses decreasing and accuracies improving? Any instability, overfitting, or collapse?

2. **Score and ownership stall:**  
   Plot **train_score_loss**, **train_ownership_loss**, and **train_ownership_accuracy** over iterations. Do they plateau? Using **CODE_CONTEXT.md**, suggest possible causes (target construction, weights, head capacity, valid_mask, or data) and concrete changes.

3. **Config and schedules:**  
   Using **config_best.yaml**: Are the iteration schedules and training/selfplay/replay settings reasonable for AlphaZero-style training? Any risks or tweaks (including for score/ownership)? Note: **progressive widening (PW)** is enabled (k_root 64, k0 32, k_sqrt_coef 8); evaluation is **manual** (no automated eval games).

4. **Beat-humans metrics (when present in CSV):**  
   Using **FULL_METRICS.md**: Interpret trends in packing (empty_squares, empty_components, isolated_1x1_holes; p50/p90; abs_diff) and search health (root_legal_count, root_expanded_ratio, p90). Are fragmentation and isolated holes improving? Is expanded_ratio stable? Any asymmetry (abs_diff) spike?

Base all answers on the provided CSVs, config, **FULL_METRICS.md**, and CODE_CONTEXT; do not assume external data. Prefer clear conclusions with short numeric evidence (e.g. loss/accuracy at iter X vs Y, or trend in score_loss / ownership_accuracy / beat-humans metrics).
