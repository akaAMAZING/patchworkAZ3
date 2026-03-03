# Patchwork AlphaZero — Data Package for AI Analysis

This folder contains a complete data package for an AI (e.g. ChatGPT) to analyze our training progress. **Omit nothing.** Use the CSVs and config as the single source of truth; extract and reason over every metric.

---

## Files in this package

| File | Description |
|------|-------------|
| **training_metrics_per_iteration.csv** | **Iterations 0–45** (46 rows). One row per iteration. **Every metric** — no omissions. **Train:** policy_loss, value_loss, score_loss, ownership_loss, total_loss, policy_accuracy, policy_top5_accuracy, value_mse, **grad_norm**, **policy_entropy**, **kl_divergence**, ownership_accuracy, step_skip_rate. **Selfplay:** num_games, num_positions, avg_game_length, p0_wins, p1_wins, generation_time, games_per_minute, avg_policy_entropy, avg_top1_prob, avg_num_legal, avg_redundancy, unique_positions, avg_root_q. **Eval:** all scalar fields from `vs_previous_best` and `vs_pure_mcts` that appear in the iteration JSONs (may be zero / unused for recent iterations). **Applied settings:** every nested key (selfplay, training, replay, adaptive_games). Plus iteration_time_s, replay_buffer_*, accepted. |
| **training_metadata_constants.csv** | Config constants not stored per-iteration: **EMA decay** (0.999), loss weights, max_grad_norm. EMA is applied every step; no per-iteration EMA scalar. |
| **config_best.yaml** | Full training/config: hardware, data encoding, network, training (optimizer, LR, batch, AMP, loss weights, EMA, schedules), selfplay (MCTS, win-first, schedules), replay buffer, evaluation, iteration schedules (LR, temperature, mcts, parallel_leaves, dirichlet, dynamic_score_utility, q_value_weight, cpuct, noise_weight, games, window_iterations). Omit nothing when referring to “our config”. |
| **README_ANALYSIS_REQUEST.md** | This file. |

---

## Input channels and network architecture (for the AI)

- **Input channels:** **56**. (Config: `network.input_channels: 56`. Data encoding: `gold_v2_32ch` with `expected_spatial_channels: 32`; the 56 is the encoder output channels fed into the trunk.)
- **Architecture (from config):**
  - **Trunk:** ResNet-style: **18 residual blocks**, **128 channels**, BatchNorm, ReLU, **Squeeze-and-Excitation (se_ratio: 0.0625)**.
  - **FiLM conditioning:** Enabled; multimodal: global (61-dim), track (8×54 → 432-dim), shop (128-dim). `film_hidden: 256`, `film_per_block: true`, `film_track_use_conv: true`, `film_global_inject_dim: 64` (injected into value + pass-logit heads).
  - **Policy head:** Factorized conv head: 1×1 conv → 48 ch → pass (1) + patch (81) + buy (24×81 = 1944) → **2026** actions; global inject 64-dim into pass and spatial biases.
  - **Value head:** 48 ch, hidden 512 → scalar (tanh).
  - **Score head:** Categorical over 201 bins (e.g. [-100, +100]).
  - **Ownership head:** 48 ch → ownership logits.
  - **Training:** AdamW, weight_decay 0.0002, max_grad_norm 1.0, AMP bfloat16, EMA decay 0.999. Loss weights: policy 1.0, value 1.0, score 0.1, ownership 0.15.

---

## Questions we want the AI to answer

1. **Is training progressing solidly and properly?**  
   Using **training_metrics_per_iteration.csv** (and config): trends in total_loss, policy_loss, value_loss, score_loss, policy_accuracy, policy_top5_accuracy, **KL divergence**, **grad_norm**, selfplay stats (games/min, avg_game_length, avg_root_q, entropy). Are losses decreasing and accuracy improving over iterations 0–45? Any signs of instability, overfitting, or collapse?

2. **Config and schedules:**  
   Using **config_best.yaml**: Are the iteration schedules (LR, temperature, MCTS sims, cpuct, games, window) and the training/selfplay/replay settings reasonable for AlphaZero-style training? Any obvious risks or suggested tweaks?

Please base all answers on the provided CSVs and config; do not assume external data. Prefer clear, concise conclusions and, where relevant, short numeric evidence (e.g. loss/accuracy at iter X vs iter Y, or changes in KL / grad_norm / entropy over time).
