# Patchwork AlphaZero — Data Package for AI Analysis

This folder is a **complete, maximum data package** for an AI (e.g. ChatGPT) to understand and analyze our training. **Omit nothing.** Use the CSVs, config, and code context as the single source of truth.

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
| **config_best.yaml** | **Up-to-date** full config: hardware, data encoding, network, training (optimizer, LR, batch, AMP, **score_loss_weight: 0.1**, **ownership_loss_weight: 0.15**, EMA), selfplay (MCTS, win-first, schedules), replay buffer, evaluation, **all iteration schedules** (LR, temperature, mcts, parallel_leaves, dirichlet, dynamic_score_utility, q_value_weight, cpuct, noise_weight, games, window_iterations). |
| **training_metrics_per_iteration.csv** | **Iterations 0–70** (71 rows). One row per iteration. **Every metric:** **train_score_loss**, **train_ownership_loss**, **train_ownership_accuracy**, train_policy_loss, train_value_loss, train_total_loss, train_policy_accuracy, train_policy_top5_accuracy, train_value_mse, train_grad_norm, train_policy_entropy, train_kl_divergence, train_step_skip_rate; selfplay_* (num_games, num_positions, avg_game_length, p0_wins, p1_wins, generation_time, games_per_minute, avg_policy_entropy, avg_top1_prob, avg_num_legal, avg_redundancy, unique_positions, avg_root_q); eval_* (when present); applied_* (full flatten of selfplay/training/replay/adaptive_games); iteration_time_s, replay_buffer_*, accepted. |
| **training_metadata_constants.csv** | Constants not per-iteration: **EMA decay** (0.999), **score_loss_weight** (0.1), **ownership_loss_weight** (0.15), **score_utility_scale** (30.0), **score_target_sigma** (1.5), max_grad_norm. Includes short notes on score/ownership stall. |
| **CODE_CONTEXT.md** | **Code snippets and optimizations:** score/ownership loss and accuracy (model.get_loss, OwnershipHead), score target building (value_targets tanh, trainer make_gaussian_score_targets), D4 augmentation, FiLM, EMA, prefetch, win-first MCTS, AMP. File map for where to look. Use this to interpret metrics and suggest code-level fixes. |
| **INPUT_CHANNELS_MARKOV.md** | **Fully Markov input channels in detail:** 56 spatial (32 encoder + 24 legalTL), channel-by-channel layout (occupancy, coords, frontier, valid_7x7, slot×orient shapes, legalTL 3×8), x_global (61), x_track (8×54), shop (33, 10). Source-of-truth constants and how the model uses them. |
| **README_ANALYSIS_REQUEST.md** | This file. |
| **build_training_csv.py** | Script to regenerate training_metrics_per_iteration.csv from committed `iteration_XXX.json` (no fixed iter limit). |

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
   Using **config_best.yaml**: Are the iteration schedules and training/selfplay/replay settings reasonable for AlphaZero-style training? Any risks or tweaks (including for score/ownership)?

Base all answers on the provided CSVs, config, and CODE_CONTEXT; do not assume external data. Prefer clear conclusions with short numeric evidence (e.g. loss/accuracy at iter X vs Y, or trend in score_loss / ownership_accuracy).
