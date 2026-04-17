# Patchwork AlphaZero: Comprehensive Training & Implementation Paper

This document describes **every** implementation detail of the Patchwork AI training pipeline, keyed to `configs/config_best.yaml` and the codebase. It covers: config → code wiring, FiLM conditioning, fully Markov input channels (with each channel’s purpose), the fact that we **do not gate** and **auto-promote every iteration**, and a brief note on disabled config features at the end.

---

## 1. Overview

- **Goal:** Maximum playing strength over 200–400 iterations (production target).
- **Strategy:** No gating. Every model is promoted immediately. “Trust the process.”
- **Hardware target:** RTX 3080 (10GB), Ryzen 5900X, 16GB RAM.
- **Encoding:** `gold_v2_32ch` — 32 spatial channels from encoder; 24 legalTL channels computed on GPU → 56-channel trunk input. Multimodal: global (61), track (8×54), shop (33×10).

---

## 2. Config → Implementation (by section)

### 2.1 Top-level

| Config | Purpose | Implementation |
|--------|---------|----------------|
| `seed: 42` | Reproducibility | Used in main, trainer (per-iter seed), evaluation, selfplay, league. |
| `deterministic: false` | When true, disables cudnn benchmark | `src/network/gpu_inference_server.py`: `torch.backends.cudnn.benchmark = True` when false. |

### 2.2 `hardware`

```python
# src/training/main.py
if self.config["hardware"]["device"] == "cuda" and torch.cuda.is_available():
    self.device = torch.device("cuda")
# src/training/trainer.py — DataLoader
pin_memory=hw.get("pin_memory", ...),
persistent_workers=... if num_workers > 0 else False,
# prefetch_batches: overlap next(data_iter) with GPU (background thread)
```

- **device:** cuda (with fallback).
- **pin_memory:** true for faster transfer to GPU.
- **persistent_workers:** true when num_workers > 0 (trainer uses 0 workers for dataset).
- **prefetch_batches:** 2 — overlapping data loading with compute.

### 2.3 `data` (encoding / compatibility)

```yaml
data:
  allow_legacy_state_channels: false
  encoding_version: gold_v2_32ch
  expected_spatial_channels: 32
  expected_global_dim: 61
  expected_track_shape: [8, 54]
  expected_shop_shapes: [33, 10]
```

- **allow_legacy_state_channels:** When false, channel mismatch raises; when true, pad/truncate (`src/training/replay_buffer.py`).
- **encoding_version:** Must match `gold_v2_32ch`; HDF5/replay validated against it (`src/network/gold_v2_constants.ENCODING_VERSION`).
- **expected_*:** Replay buffer and dataset expect these shapes; encoder produces them (`GoldV2StateEncoder`).

### 2.4 `network`

- **input_channels: 56** — Trunk input: 32 (encoder) + 24 (GPU legalTL). Model and replay use 56 (`model.py`, `replay_buffer.py`).
- **num_res_blocks: 18** — Number of residual blocks in trunk (`model.py`).
- **channels: 128** — Conv channels in trunk and most heads.
- **policy_channels / policy_hidden:** Used by legacy policy head; with `use_factorized_policy_head: true` the **StructuredConvPolicyHead** uses fixed 48 hidden and 24 buy channels.
- **use_factorized_policy_head: true** — StructuredConvPolicyHead (pass + patch map + buy maps, 2026 actions).
- **value_channels / value_hidden:** 48 and 512 for value head.
- **max_actions: 2026** — PASS(1) + PATCH(81) + BUY(1944).
- **dropout: 0.0** — No dropout in trunk (optional Dropout2d when > 0).
- **weight_decay: 0.0002** — AdamW weight decay.
- **use_batch_norm: true** — BN in conv_input, res blocks, policy/value/ownership heads.
- **activation: relu** — ReLU in trunk and heads.
- **se_ratio: 0.0625** — Squeeze-and-Excitation in each residual block.
- **ownership_channels: 48** — Auxiliary ownership head (KataGo-style, dual-board: 2×9×9).

**FiLM (Feature-wise Linear Modulation):**

```yaml
use_film: true
film_hidden: 256
film_global_dim: 61
film_track_dim: 432   # 8*54 flat when not using conv
film_shop_dim: 128
film_per_block: true
film_track_use_conv: true
film_global_inject_dim: 64
```

- **use_film: true** — FiLM conditions the residual trunk from global + track + shop.
- **film_global_dim / film_track_dim / film_shop_dim:** Input dims to FiLM; track is 8×54 (432) if flat; shop encoder outputs `film_shop_dim`.
- **film_per_block: true** — Each of the 18 res blocks gets its own (γ, β) from FiLM: `film_out` is reshaped to `(B, num_res_blocks, 2, channels)` and applied per block in `ResidualBlock(x, gamma, beta)`.
- **film_track_use_conv: true** — Track (8, 54) is processed by Conv1d (8→32→32) then pooled to `film_hidden` instead of a flat MLP.
- **film_global_inject_dim: 64** — A projected global vector is concatenated into the value head and pass-logit (and optional buy/patch global bias). Implemented as `global_to_heads(x_global)` and optionally `trunk_to_heads(trunk.mean(2,3))` added to `g_heads`.

Code (FiLM application in trunk):

```python
# src/network/model.py — _trunk_forward
if self.use_film and self.film_mlp is not None:
    g_enc = self.film_global_mlp(x_global)
    if self.film_track_conv is not None:
        t_out = self.film_track_conv(x_track)       # (B, 32, 54)
        t_enc = self.film_track_pool(t_out.mean(dim=2))  # (B, film_hidden)
    else:
        t_enc = self.film_track_mlp(x_track.flatten(1))
    s_enc = self.shop_encoder(shop_ids, shop_feats)
    g = torch.cat([g_enc, t_enc, s_enc], dim=1)
    film_out = self.film_mlp(g)
    film = film_out.view(film_out.shape[0], self.num_res_blocks, 2, self.channels)
    gamma_list = [film[:, i, 0, :] for i in range(self.num_res_blocks)]
    beta_list = [film[:, i, 1, :] for i in range(self.num_res_blocks)]
# ...
for i, block in enumerate(self.res_blocks):
    x = block(x, gamma_list[i], beta_list[i])
```

ResidualBlock with FiLM:

```python
# src/network/model.py — ResidualBlock.forward
out = self.bn1(self.conv1(x))
if gamma is not None and beta is not None:
    out = out * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
out = F.relu(out, inplace=True)
# ... conv2, se, residual
```

### 2.5 Fully Markov input channels (gold_v2_32ch)

Encoder output is **32 spatial channels** (no history). LegalTL (24 ch) is computed on GPU from board occupancy and shop; trunk sees **56 channels** (32 + 24).

**Spatial channels 0–31 (encoder):**

| Channel(s) | Name / source | Purpose |
|------------|----------------|---------|
| 0 | cur_occ | Current player’s board occupancy (9×9 binary from occ words). |
| 1 | opp_occ | Opponent’s board occupancy. |
| 2 | coord_row | Row coordinate normalised (0..8). |
| 3 | coord_col | Column coordinate normalised. |
| 4 | cur_frontier | Frontier (empty cells adjacent to current player’s pieces). |
| 5 | opp_frontier | Frontier for opponent. |
| 6 | valid_7x7_cur | Valid 7×7 placement region (current). |
| 7 | valid_7x7_opp | Valid 7×7 placement region (opponent). |
| 8–31 | slot{s}_orient{o}_shape | For each of 3 slots and 8 orientations, a 9×9 shape mask for that piece/orient. Index = 8 + slot*8 + orient. |

Source (encoder):

```python
# src/network/encoder.py — GoldV2StateEncoder.encode_into
x_spatial[0] = StateEncoder._decode_occ_words(c_occ0, c_occ1, c_occ2)
x_spatial[1] = StateEncoder._decode_occ_words(o_occ0, o_occ1, o_occ2)
x_spatial[2], x_spatial[3] = self._coord_row, self._coord_col
x_spatial[4] = _frontier_plane(c_occ0, c_occ1, c_occ2)
x_spatial[5] = _frontier_plane(o_occ0, o_occ1, o_occ2)
x_spatial[6] = _valid_7x7_empty(c_occ0, c_occ1, c_occ2)
x_spatial[7] = _valid_7x7_empty(o_occ0, o_occ1, o_occ2)
# 8–31: slot×orient shape masks from _slot_orient_shape_masks
```

**Channels 32–55 (GPU):** Legal TL placement for the 3 visible slots × 8 orientations (24 planes), computed by `DeterministicLegalityModule` from board_free and visibility/affordability.

**Global vector (61):** From `_encode_scalars_shop_jit` in encoder:

- 0–2: c_pos, o_pos, (c_pos - o_pos) / track_len  
- 3–5: log1p(buttons) norm, tanh(button diff)  
- 6–8: income norms, income diff  
- 9–10: cur/opp filled fraction (popcount/81)  
- 11: bonus owner (-1/0/1 → 0, 0.5, 1)  
- 12–13: 7×7 completion distance cur/opp  
- 14–21: pending, remaining patches, patch markers, tie_player  
- 22–27: pass action features (steps, income crossed, patches crossed, etc.)  
- 28–60: per-slot (3×11) buy features: affordability, cost, time, income, area, buy_inc, buy_pat, etc.

**Track (8×54):** One-hot style; indices 0–1: current/opponent position; 2: button marks; 3: patch marks (eligibility); 4: pass landing; 5–7: buy landing for slots 0–2.

**Shop:** `shop_ids` (33), `shop_feats` (33×10): circle index, cost, time, income, area, n_orient, cur/opp afford; 8–9 filled on GPU (num_legal normalised).

### 2.6 Training

- **optimizer: adamw** — AdamW with weight_decay.
- **learning_rate: 0.0016** — Base; actual peak LR comes from `iteration.lr_schedule` (see below).
- **lr_schedule: cosine_warmup_per_phase** — Phases from `iteration.lr_schedule`; warmup once per phase; no restart each iteration (`trainer.py`).
- **warmup_steps: 20** — Used only by non–phase schedule (e.g. plain cosine_warmup).
- **d4_augmentation: dynamic** — D4 applied per sample in dataset; `store_canonical_only` in selfplay.
- **d4_on_gpu: false** — CPU D4 (numpy) in dataset; GPU path optional.
- **resume_optimizer_state / resume_scheduler_state / resume_scaler_state / resume_ema_state: true** — Load from committed checkpoint each iter so training is continuous across iterations.
- **resume_from_committed_state: true** — Only load optimizer/scheduler/EMA from committed (never staging).
- **min_lr:** Used in cosine schedule (~50:1 ratio with base).
- **batch_size: 1024**, **use_amp: true**, **amp_dtype: bfloat16**.
- **policy_loss_weight / value_loss_weight / score_loss_weight / ownership_loss_weight:** 1.0, 1.0, 0.1, 0.15. Total loss = sum of weighted losses.
- **epochs_per_iteration: 3**, **val_split: 0.08**, **val_frequency: 1000**.
- **checkpoint_frequency: 10000**, **keep_last_n_checkpoints: 200**, **save_best: false**.
- **early_stopping: enabled: false**.
- **max_grad_norm: 1.0** — Gradient clipping.
- **allow_tf32 / matmul_precision:** TF32 and high precision for matmul.
- **ema:** enabled, decay 0.999, used for selfplay and eval, saved in checkpoint.

### 2.7 Self-play

- **api_url / inference_backend: null** — Queue-based GPU server (eval_client), no HTTP API.
- **num_workers: 12** — Self-play workers.
- **max_game_length: 200**, **games_per_iteration: 900** — Overridden by `iteration.games_schedule` and adaptive_games.
- **store_canonical_only: true** — One state per move; D4 at train time.
- **augmentation: d4** — D4 in selfplay store (canonical); training applies D4 dynamically.
- **mcts.simulations:** From `iteration.mcts_schedule` (e.g. 192 → 384).
- **mcts.parallel_leaves:** From `parallel_leaves_schedule`.
- **static_score_utility_weight: 0.0**, **dynamic_score_utility_weight: 0.12** — KataGo-style; dynamic from schedule.
- **score_utility_scale: 30.0** — Must match `value_targets._SCORE_NORMALISE_DIVISOR`.
- **progressive_widening:** enabled; k_root, k0, k_sqrt_coef; always_include_pass / always_include_patch.
- **packing_ordering:** enabled; alpha, use_log_prior, weights (adj_edges, iso_hole_penalty, etc.) for ranking BUY actions.
- **cpuct / temperature / root_dirichlet_alpha / root_noise_weight / virtual_loss / fpu_reduction** — From config and iteration schedules.
- **win_first:** enabled; value_delta_min/max, value_delta_win_start/win_full; tiebreak score_then_visits; filter_before_sampling.
- **policy_target_mode: visits** — Targets = normalised visit counts.
- **q_value_weight:** From config and `q_value_weight_schedule`.
- **stream_to_disk / hdf5_compression / write_buffer_positions** — HDF5 writing.
- **bootstrap.use_pure_mcts: true** — Iter 0: pure MCTS only; no seed checkpoint for optimizer shape match.
- **opponent_mix: enabled: false** — No packer mix in training.

### 2.8 Replay buffer

- **max_size: 300000**, **min_size: 4000** — Subsampling when over max.
- **window_iterations:** From `iteration.window_iterations_schedule` (e.g. 8→16).
- **newest_fraction: 0.25** — Recency bias when subsampling.
- **prioritized: false** — Uniform sampling (within window and newest bias).
- **storage_format: hdf5**, **compression: lzf**.

### 2.9 Evaluation

- **games_vs_best: 0**, **games_vs_pure_mcts: 0** — So **no evaluation games** are run. Together with gating logic this implies **every iteration is auto-accepted** (see below).
- **lock_eval_cpuct_to_selfplay: true**, **paired_eval: true**, **eval_progress_interval: 10**.
- **sprt.enabled: false**, **micro_gate.enabled: false**, **elo.enabled: false**.

### 2.10 Iteration schedules

All schedules use step interpolation: for a given `iteration` the value is the one at the last schedule point with `iteration <= current`.

- **lr_schedule** — Peak LR per phase (e.g. 0.0016 → 0.0012 → 0.0008 → 0.0004 → 0.00025).
- **temperature_schedule** — Exploration temperature (e.g. 1.0 → 0.20).
- **mcts_schedule** — Simulations (192 → 384).
- **parallel_leaves_schedule** — Parallel leaves (64 → 32).
- **dirichlet_alpha_schedule** — Root Dirichlet alpha.
- **dynamic_score_utility_weight_schedule** — Score utility weight.
- **q_value_weight_schedule** — Q blend weight.
- **cpuct_schedule** — PUCT constant.
- **noise_weight_schedule** — Root noise weight.
- **games_schedule** — Games per iteration (sims×games ≈ constant).
- **window_iterations_schedule** — Replay window size.
- **adaptive_games:** When enabled, adjusts games_this_iter from last K committed avg_game_length and target_pos_iter; anti-thrash via max_step_change.

### 2.11 Inference

- **batch_size: 512**, **max_batch_wait_ms: 3**, **use_amp / amp_dtype**, **torch_compile: false**, **allow_tf32: true**.

### 2.12 Paths

- **checkpoints_dir, logs_dir, run_root, run_id** — Used throughout (main, run_layout, replay, logging).

---

## 3. No gating / auto-promote every iteration

We **do not gate**. Every iteration the new model is **accepted and promoted** to best.

- **evaluation.games_vs_best** and **evaluation.games_vs_pure_mcts** are **0**, so `num_eval_games <= 0`.
- In `_should_accept_model` (main.py): when evaluation is disabled, the code **auto-accepts**:

```python
# src/training/main.py — _should_accept_model
num_eval_games = int(
    eval_cfg.get("games_per_eval")
    or max(eval_cfg.get("games_vs_pure_mcts", 0), eval_cfg.get("games_vs_best", 0))
    or 0
)
if num_eval_games <= 0:
    logger.debug("[GATE] Evaluation disabled (games_per_eval <= 0): auto-accepting.")
    return True
```

- So every iteration (after 0) is accepted; iteration 0 is also always accepted (bootstrap). There is no win-rate threshold check, no SPRT, no micro-gate, and no league gating in our config (league.enabled: false). **Result: every iteration is promoted.**

---

## 4. Value and score targets

- **Value:** Binary terminal value from `value_targets.terminal_value_from_scores`: +1 / -1 / 0 (to_move perspective). Stored in replay as `values`.
- **Score:** Tanh-normalised margin: `tanh((to_move_score - opp_score) / 30.0)` in [-1, 1]. Stored as `score_margins`; training builds 201-bin soft labels via `make_gaussian_score_targets` (trainer.py) and uses cross-entropy for the score head.

---

## 5. Disabled or inactive features (brief)

- **save_best: false** — Best model by val_loss is not saved; we always promote latest.
- **early_stopping: enabled: false** — No early stop on val loss.
- **profile_training_steps** — Commented out; no step-level profiling.
- **evaluation.games_vs_best / games_vs_pure_mcts: 0** — No eval games → auto-accept (already described).
- **sprt / micro_gate / elo** — Disabled.
- **league.enabled: false** — No league pool or league gating.
- **opponent_mix.enabled: false** — No games vs packer.
- **playout_cap_randomization.enabled: false** — No cap randomization.
- **wandb.enabled: false** — No Weights & Biases logging.
- **inference.torch_compile: false** — No torch.compile for inference.

These are present in config so the feature set is visible and can be enabled for experiments (eval games, gating, league, opponent mix, etc.).

---

## 6. File and data flow (high level)

1. **Self-play:** Workers run MCTS with current best model (EMA), write HDF5 shards (states 32ch, x_global, x_track, shop_ids, shop_feats, policies, values, score_margins, ownerships) to staging.
2. **Merge:** Shards merged into one HDF5 per iteration; replay buffer adds iteration and merges with window; if total positions > max_size, subsample (with newest_fraction bias).
3. **Training:** Dataset loads from merged HDF5; D4 applied dynamically; GPU legality turns 32ch→56ch and fills shop 8–9; FiLM uses global/track/shop; loss = policy + value + score + ownership; optimizer/scheduler/EMA resumed from committed checkpoint.
4. **Commit:** Staging iter dir moved to committed; run_state and replay_state updated; metadata appended to logs/metadata.jsonl; TensorBoard events copied to logs/tensorboard. No gating → best_model_path and latest both point to the new checkpoint.

---

## 7. Complete list of metrics (TensorBoard + CSV)

All metrics are written to **TensorBoard** (when `logging.tensorboard.enabled: true`) and/or to **logs/metadata.jsonl** and **runs/.../committed/iter_*/iteration_*.json**. A single **CSV** with every metric for iterations 0–192 is produced by `tools/build_full_metrics_csv.py` → **docs/full_metrics_0_192.csv**. Each column is documented in **docs/METRICS_GLOSSARY.md** with source file and how it is calculated.

### 7.1 TensorBoard tags (canonical)

- **train/** (per-step, global_step): `learning_rate`, `total_loss`, `policy_loss`, `value_loss`, `grad_norm`, `policy_entropy`, `policy_accuracy`.
- **val/** (per val_frequency steps): `total_loss`, `policy_loss`, `value_loss`, `policy_entropy`, `policy_accuracy`.
- **iter/** (per iteration): `kl_divergence`, `policy_entropy`, `target_entropy`, `policy_cross_entropy`, `value_mse`, `step_skip_rate`, `approx_identity_check`, `ownership_accuracy`, `ownership_filled_fraction_mean`, `ownership_accuracy_all_filled_baseline`, `ownership_empty_recall`, `ownership_empty_precision`, `ownership_balanced_accuracy`, `ownership_mae_empty_count`.
- **selfplay/** (per iteration): `games_per_min`, `num_positions`, `num_games`, `avg_game_length`, `unique_positions`, `avg_redundancy`, `avg_policy_entropy`, `avg_top1_prob`, `avg_num_legal`, `avg_root_q`, `generation_time`; plus beat-humans and packing keys (e.g. `selfplay_avg_final_empty_squares_mean`, `selfplay_avg_root_legal_count`, …), opponent_mix keys, packing_ordering keys.
- **buffer/** (per iteration): `total_positions`, `num_iterations`.

Code refs: **src/training/main.py** (iteration-level TB, ~1540–1592), **src/training/trainer.py** (train/val per step, ~1112–1129).

### 7.2 Non–TensorBoard metrics (in CSV only or in iteration_*.json)

- **Metadata top-level:** `iteration`, `timestamp_utc`, `config_hash`, `config_path`, `best_model_hash`, `accepted`, `global_step`, `iter_time_s`, `replay_positions`, `consecutive_rejections`, `best_model`, `eval_vs_best_wr`, `eval_vs_best_margin`, `eval_vs_mcts_wr`, `elo`.
- **train_*** (epoch averages): all keys returned by `network.get_loss()` and trainer epoch aggregation: `policy_loss`, `value_loss`, `score_loss`, `ownership_loss`, `total_loss`, `policy_accuracy`, `policy_top5_accuracy`, `value_mse`, `grad_norm`, `policy_entropy`, `target_entropy`, `policy_cross_entropy`, `ce_minus_policy_entropy`, `kl_divergence`, `approx_identity_check`, ownership_* metrics, `step_skip_rate`.
- **selfplay_*** (from selfplay_stats): core (num_games, num_positions, avg_game_length, p0_wins, p1_wins, generation_time, games_per_minute, avg_policy_entropy, avg_top1_prob, avg_num_legal, avg_redundancy, unique_positions, avg_root_q); packing/beat-humans (e.g. selfplay_avg_final_empty_squares_mean, p50/p90 variants, avg_root_legal_count, avg_root_expanded_ratio); opponent_mix (frac_games_vs_packer, nn_vs_packer_winrate); packing_ordering (avg_packing_ordering_enabled, avg_packing_score_top1).
- **eval_vs_previous_best_*** (when eval runs): model_wins, baseline_wins, ties, total_games, win_rate, avg_game_length, avg_score_diff, avg_model_score_margin, avg_win_margin, avg_loss_margin.
- **applied_*** (from applied_settings): selfplay (games, temperature, simulations, cpuct, dirichlet_alpha, noise_weight, parallel_leaves, policy_target_mode, q_value_weight, …); training (lr, batch_size, amp_dtype, q_value_weight); replay (window_iterations, max_size, newest_fraction, recency_window); adaptive_games (games_this_iter, scheduled_games, target_pos_iter, fill_mode, clamp_low, clamp_high, …).

### 7.3 Code snippets supporting the CSV (where each data point comes from)

- **Iteration / run metadata:** `src/training/main.py` — `_append_metadata()` writes one JSONL line per iteration with `train`, `selfplay`, and top-level fields; `iteration_*.json` is the full summary saved at commit (`_save_iteration_summary`).
- **Train metrics (losses and diagnostics):** `src/network/model.py` — `get_loss()` returns the dict of all losses and metrics (policy_loss, value_loss, score_loss, ownership_loss, total_loss, policy_accuracy, policy_top5_accuracy, value_mse, policy_entropy, target_entropy, kl_divergence, ce_minus_policy_entropy, approx_identity_check, ownership_*). Trainer aggregates these over an epoch and reports epoch averages; those are what get written to metadata and iteration_*.json.
- **Value/score targets:** `src/training/value_targets.py` — `terminal_value_from_scores()` (binary value); `value_and_score_from_scores()` (value + tanh margin). Replay stores `values` and `score_margins`; trainer builds 201-bin soft targets via `make_gaussian_score_targets()` in `src/training/trainer.py`.
- **Self-play stats:** `src/training/selfplay_optimized_integration.py` — `_compute_stats()` aggregates game summaries into num_games, num_positions, avg_game_length, p0_wins, p1_wins, generation_time, games_per_minute, avg_policy_entropy, avg_top1_prob, avg_num_legal, avg_redundancy, unique_positions, avg_root_q; packing stats from `aggregate_packing_over_games()` and root stats from `aggregate_root_over_moves()` in `src/utils/packing_metrics.py`.
- **Replay positions:** `src/training/replay_buffer.py` — `total_positions` and `num_iterations` (window size) after each iteration.
- **Applied settings:** `src/training/main.py` — Schedule lookups (e.g. `_get_games_for_iteration`, `_get_window_iterations_for_iteration`) and adaptive_games logic fill `applied_settings` in the iteration summary.

Running `python tools/build_full_metrics_csv.py` produces **docs/full_metrics_0_192.csv** (193 rows × all columns) and **docs/METRICS_GLOSSARY.md** (one row per column with description and code ref). Another AI can interpret the CSV unambiguously using the glossary and this paper.

---

This paper is the single comprehensive reference for how Patchwork AI training is implemented and how it ties together.
