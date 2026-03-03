# TensorBoard Optimization — Exhaustive Inventory

**Scope:** Active codebase under `src/`. Single canonical logging spec; no conditional TB modes.

---

## PART A — Canonical TensorBoard Spec (Current Output)

All TB writes use **scalar** only (`add_scalar`). No histograms, text, images, figures, or embeddings. When `logging.tensorboard.enabled` is false, the writer is `_NoOpSummaryWriter()` and all writes are no-ops.

---

### Deduped list of tag names (alphabetical)

- `buffer/num_iterations`
- `buffer/total_positions`
- `iter/approx_identity_check`
- `iter/kl_divergence`
- `iter/policy_cross_entropy`
- `iter/policy_entropy`
- `iter/step_skip_rate`
- `iter/target_entropy`
- `iter/value_mse`
- `selfplay/avg_game_length`
- `selfplay/avg_num_legal`
- `selfplay/avg_policy_entropy`
- `selfplay/avg_redundancy`
- `selfplay/avg_root_q`
- `selfplay/avg_top1_prob`
- `selfplay/games_per_min`
- `selfplay/generation_time`
- `selfplay/num_games`
- `selfplay/num_positions`
- `selfplay/unique_positions`
- `train/grad_norm`
- `train/learning_rate`
- `train/policy_accuracy`
- `train/policy_entropy`
- `train/policy_loss`
- `train/total_loss`
- `train/value_loss`
- `val/policy_accuracy`
- `val/policy_entropy`
- `val/policy_loss`
- `val/total_loss`
- `val/value_loss`

**Not logged to TensorBoard:** No `eval/*` or `league/*` tags. Eval and league results are still computed and logged to the run logger / metadata; they are simply not written to TB.

---

### Per-tag details (type, file:line, cadence, description)

| Tag | Type | File:Line | Cadence | Description |
|-----|------|-----------|---------|-------------|
| `buffer/num_iterations` | scalar | `src/training/main.py` | per iteration | Number of iteration-chunks in the replay buffer. |
| `buffer/total_positions` | scalar | `src/training/main.py` | per iteration | Total positions in the replay buffer. |
| `iter/approx_identity_check` | scalar | `src/training/main.py` | per iteration | CE = H + KL identity check error (whitelisted). |
| `iter/kl_divergence` | scalar | `src/training/main.py` | per iteration | Epoch-avg KL(MCTS target \|\| network policy). |
| `iter/policy_cross_entropy` | scalar | `src/training/main.py` | per iteration | Epoch-avg policy CE. |
| `iter/policy_entropy` | scalar | `src/training/main.py` | per iteration | Epoch-avg policy entropy. |
| `iter/step_skip_rate` | scalar | `src/training/main.py` | per iteration | Fraction of steps skipped (e.g. AMP scale backoff); only if present in train_metrics. |
| `iter/target_entropy` | scalar | `src/training/main.py` | per iteration | Epoch-avg target (MCTS) policy entropy. |
| `iter/value_mse` | scalar | `src/training/main.py` | per iteration | Epoch-avg value MSE. |
| `selfplay/avg_game_length` | scalar | `src/training/main.py` | per iteration | Mean game length (moves) this iteration. |
| `selfplay/avg_num_legal` | scalar | `src/training/main.py` | per iteration | Mean number of legal actions per position. |
| `selfplay/avg_policy_entropy` | scalar | `src/training/main.py` | per iteration | Mean policy entropy over self-play positions (collapse canary). |
| `selfplay/avg_redundancy` | scalar | `src/training/main.py` | per iteration | Position redundancy (0 = unique; Oracle Part 2). |
| `selfplay/avg_root_q` | scalar | `src/training/main.py` | per iteration | Mean root Q from MCTS in self-play. |
| `selfplay/avg_top1_prob` | scalar | `src/training/main.py` | per iteration | Mean top-1 action probability (sharpness). |
| `selfplay/games_per_min` | scalar | `src/training/main.py` | per iteration | Self-play throughput: games per minute. |
| `selfplay/generation_time` | scalar | `src/training/main.py` | per iteration | Wall-clock time (seconds) for self-play generation. |
| `selfplay/num_games` | scalar | `src/training/main.py` | per iteration | Number of games generated this iteration. |
| `selfplay/num_positions` | scalar | `src/training/main.py` | per iteration | Total positions generated this iteration. |
| `selfplay/unique_positions` | scalar | `src/training/main.py` | per iteration | Sum of unique positions over games. |
| `train/grad_norm` | scalar | `src/training/trainer.py` | every 10 steps | Gradient norm (after clip). |
| `train/learning_rate` | scalar | `src/training/trainer.py` | every 10 steps | Current scheduler learning rate. |
| `train/policy_accuracy` | scalar | `src/training/trainer.py` | every 10 steps | Policy top-1 accuracy. |
| `train/policy_entropy` | scalar | `src/training/trainer.py` | every 10 steps | Mean policy entropy over batch. |
| `train/policy_loss` | scalar | `src/training/trainer.py` | every 10 steps | Policy cross-entropy loss. |
| `train/total_loss` | scalar | `src/training/trainer.py` | every 10 steps | Total weighted loss. |
| `train/value_loss` | scalar | `src/training/trainer.py` | every 10 steps | Value head MSE loss. |
| `val/policy_accuracy` | scalar | `src/training/trainer.py` | every `val_frequency` steps | Validation policy top-1 accuracy. |
| `val/policy_entropy` | scalar | `src/training/trainer.py` | every `val_frequency` steps | Validation policy entropy. |
| `val/policy_loss` | scalar | `src/training/trainer.py` | every `val_frequency` steps | Validation policy loss. |
| `val/total_loss` | scalar | `src/training/trainer.py` | every `val_frequency` steps | Validation total loss. |
| `val/value_loss` | scalar | `src/training/trainer.py` | every `val_frequency` steps | Validation value loss. |

**iter/ whitelist:** Only the following keys from `train_metrics` are written to TB: `kl_divergence`, `policy_entropy`, `target_entropy`, `policy_cross_entropy`, `value_mse`, `step_skip_rate`, `approx_identity_check`. Each is written only if present in `train_metrics`.

---

## PART B — Candidate Metrics (Not in Canonical TB)

The following are computed or available elsewhere (logger, metadata, eval_results, league diag) but are **not** written to TensorBoard under the current spec. Grouped by subsystem. For each: **suggested name**, **where available**, **caveats**.

---

### Training

| Suggested name | Where computed / available | Caveats |
|----------------|---------------------------|---------|
| `train/value_mse` | Same as value_loss; in `metrics` from `get_loss`, `src/network/model.py:961`. | Redundant with `train/value_loss` unless you want a separate name. |
| `train/target_entropy` | `get_loss` metrics, `src/network/model.py:982`. | Already in `epoch_metrics` → `iter/target_entropy`. |
| `train/kl_divergence` | `get_loss` metrics, `src/network/model.py:985`. | Already in `epoch_metrics` → `iter/kl_divergence`. |
| `train/ce_minus_policy_entropy` | `get_loss` metrics, `src/network/model.py:984`. | Already in `epoch_metrics` → `iter/ce_minus_policy_entropy`. |
| `train/approx_identity_check` | `get_loss` metrics, `src/network/model.py:986`. | Already in `epoch_metrics` → `iter/approx_identity_check`. |
| `train/ownership_accuracy` | `get_loss` metrics, `src/network/model.py:987`. | Already in `epoch_metrics` → `iter/ownership_accuracy`. |
| `train/policy_cross_entropy` | `get_loss` metrics, `src/network/model.py:983`. | Already in `epoch_metrics` → `iter/policy_cross_entropy`. |
| `train/learning_rate` | Already logged every 10 steps; also available at any step via `self.scheduler.get_last_lr()[0]`, `src/training/trainer.py`. | — |
| `train/grad_norm` | Already logged every 10 steps; `grad_norm` from `clip_grad_norm_`, `src/training/trainer.py:1042,1049,1053`. | — |
| `train/step_skip_rate` | Epoch-level only (in `train_metrics` as `step_skip_rate`), `src/training/trainer.py:1143–1144`. | Step-level skip events not currently aggregated for TB. |
| `train/amp_scale` | `self.scaler.get_scale()` before/after step when AMP+scaler used, `src/training/trainer.py:1039,1045`. | Only when `use_amp` and scaler is used (float16); bf16 has no scaler. |
| `train/step_skipped` | Boolean per step: `step_skipped = self.scaler.get_scale() < scale_before`, `src/training/trainer.py:1045`. | Could log as 0/1 scalar or aggregate; currently only epoch `step_skip_rate` flows to iter/. |
| `train/grad_clip_fraction` | Not computed; would require counting params whose grad was clipped vs `max_grad_norm`. | Would need extra logic in backward/step. |
| `train/per_head_grad_norm_value` | Not computed; grad norms per head would require `retain_grad()` or separate backward. | Expensive; would need instrumentation. |
| `train/per_head_grad_norm_policy` | Same as above for policy head. | Expensive. |
| `train/per_head_grad_norm_score` | Same as above for score head. | Expensive. |
| `train/score_entropy` | Not in current `get_loss`; score head has logits → could compute entropy over 201 bins. | Would need to add in model or trainer. |
| `train/score_pred_mean_std` | Not in current `get_loss`; could add from score_logits. | Would need to add in model. |
| `train/ratio_score_to_value_loss` | Could compute from existing metrics: score_loss / value_loss. | Easy from current metrics. |
| `train/ratio_score_to_policy_loss` | Same: score_loss / policy_loss. | Easy from current metrics. |

---

### Selfplay

| Suggested name | Where computed / available | Caveats |
|----------------|---------------------------|---------|
| `selfplay/num_games` | `selfplay_stats.get("num_games")`, set in `selfplay_optimized_integration.py:881` (len(summaries)). | Available; not currently logged to TB. |
| `selfplay/avg_game_length` | `selfplay_optimized_integration.py:883` — `float(np.mean(game_lengths))`. | Available in stats dict. |
| `selfplay/p0_wins` | `selfplay_optimized_integration.py:884` — `winners.count(0)`. | Available. |
| `selfplay/p1_wins` | `selfplay_optimized_integration.py:885` — `winners.count(1)`. | Available. |
| `selfplay/win_rate_p0` | Derived: `p0_wins / num_games` when model is P0 (need to know which side is “model”; in selfplay both are same model). | Interpretation: balance of first-move advantage in self-play. |
| `selfplay/generation_time` | `selfplay_optimized_integration.py:861` — wall-clock time for generation. | Available in aggregation. |
| `selfplay/avg_policy_entropy` | `selfplay_optimized_integration.py:888` — from per-game `avg_policy_entropy` in summaries; `selfplay_optimized.py:859,608`. | Policy collapse canary; computed in selfplay, not logged to TB. |
| `selfplay/avg_top1_prob` | `selfplay_optimized_integration.py:889`; per-game in `selfplay_optimized.py:860`. | Policy sharpness; not logged to TB. |
| `selfplay/avg_num_legal` | `selfplay_optimized_integration.py:890`; per-game in `selfplay_optimized.py:861`. | Legal move count; not logged to TB. |
| `selfplay/avg_redundancy` | `selfplay_optimized_integration.py:892` — position redundancy (Oracle Part 2). | 0 = fully unique; not logged to TB. |
| `selfplay/unique_positions` | `selfplay_optimized_integration.py:893`. | Sum of unique positions over games; not logged to TB. |
| `selfplay/avg_root_q` | `selfplay_optimized_integration.py:895`; per-game in `selfplay_optimized.py:866,615`. | Mean root Q from MCTS; not logged to TB. |
| `selfplay/resign_rate` | Not present in current selfplay stats. | Would require counting resign events if added to game loop. |
| `selfplay/draw_rate` | Not present; Patchwork may not have draws. | Depends on game rules. |
| `selfplay/avg_margin_distribution` | Not aggregated; per-game has `final_score_diff` in summaries. | Could aggregate from game summaries. |

---

### MCTS

| Suggested name | Where computed / available | Caveats |
|----------------|---------------------------|---------|
| `mcts/root_q` | Returned from `search()` as `root_q`, `src/mcts/alphazero_mcts_optimized.py:608–613`; per-move in selfplay, aggregated as `avg_root_q` in game summary. | Available in selfplay aggregation; not at eval level unless passed through. |
| `mcts/root_visit_entropy` | Not computed; would need entropy of root visit distribution over actions. | Easy from `visit_counts` dict at root. |
| `mcts/root_visit_count_total` | Sum of `root._visit_count` or visit_counts values at root. | Available inside MCTS after search. |
| `mcts/effective_branching_factor` | Not computed; could derive from visit distribution (e.g. exp(entropy)). | Would need to add in MCTS or selfplay. |
| `mcts/root_value_estimate` | `root.score_estimate` / value at root before search, `src/mcts/alphazero_mcts_optimized.py:581`. | Internal to MCTS; would need to expose. |
| `mcts/search_time_per_move` | `search_time` returned from `search()`, `src/mcts/alphazero_mcts_optimized.py:614`. | Could aggregate in selfplay/eval. |
| `mcts/root_score_estimate` | `_search_root_score`, `src/mcts/alphazero_mcts_optimized.py:581`. | KataGo dual-head; point space. |

---

### Eval

| Suggested name | Where computed / available | Caveats |
|----------------|---------------------------|---------|
| `eval/model_wins` | `_analyze_results` in `evaluation.py:709`; in stats dict. | Available in eval_results["vs_pure_mcts"] / ["vs_previous_best"]. |
| `eval/baseline_wins` | `evaluation.py:710`. | Available. |
| `eval/total_games` | `evaluation.py:711`; also in `sprt.games_played` when SPRT used. | Available. |
| `eval/avg_score_diff` | `evaluation.py:717`. | In stats; not currently logged. |
| `eval/avg_win_margin` | `evaluation.py:722` — mean score margin when model wins. | In stats; not logged. |
| `eval/avg_loss_margin` | `evaluation.py:723` — mean score margin when model loses. | In stats; not logged. |
| `eval/sprt_accept` | `evaluation.py:613` — inside `stats["sprt"]` when SPRT used. | Conditional on SPRT. |
| `eval/sprt_reject` | `evaluation.py:614`. | Conditional on SPRT. |
| `eval/sprt_llr` | `evaluation.py:617`. | Conditional on SPRT. |
| `eval/sprt_games_played` | `evaluation.py:624`. | Conditional on SPRT. |
| `eval/ties` | `evaluation.py:734` (currently always 0). | Available. |

---

### Replay buffer

| Suggested name | Where computed / available | Caveats |
|----------------|---------------------------|---------|
| `buffer/total_positions` | Already logged; `replay_buffer.total_positions`, `src/training/replay_buffer.py:173–174`. | — |
| `buffer/num_iterations` | Already logged; `replay_buffer.num_iterations`, `src/training/replay_buffer.py:177–178`. | — |
| `buffer/num_entries` | Same as num_iterations (len of _entries). | Redundant. |
| `buffer/oldest_iteration` | `min(it for it, _, _ in self._entries)` when _entries non-empty. | Not currently exposed; easy to add. |
| `buffer/newest_iteration` | `max(it for it, _, _ in self._entries)`. | Not currently exposed. |
| `buffer/positions_per_iteration` | List of `n` from `(it, p, n)` in `_entries`. | Would need to log as multiple tags or histogram. |
| `buffer/newest_fraction_realized` | When subsampling with newest_fraction, the actual fraction of samples from “newest” window. | Computed inside get_training_data; not returned. |
| `buffer/age_distribution` | Distribution of iteration indices in sampled batch. | Would require instrumentation of sampling. |
| `buffer/unique_positions_estimate` | Not computed; would require hashing positions. | Expensive. |
| `buffer/source_iterations` | `out.attrs["source_iterations"]` in merge, `src/training/replay_buffer.py:548`. | Available at write time, not on ReplayBuffer API. |

---

### League

| Suggested name | Where computed / available | Caveats |
|----------------|---------------------------|---------|
| `league/vs_best_wr` | Already logged; `gate_result.vs_best_wr`. | — |
| `league/suite_mean_wr` | Already logged when suite_winrates. | — |
| `league/suite_worst_wr` | Already logged when suite_winrates. | — |
| `league/cycles` | Already logged; `diag["cycles_in_200_triples"]`, `src/training/league.py:795`. | — |
| `league/exploitability` | Already logged; `diag["exploitability_proxy"]`, `src/training/league.py:796`. | — |
| `league/pool_size` | Already logged; `diag["pool_size"]`, `src/training/league.py:775`. | — |
| `league/best_id` | `get_diagnostics` returns `best_id`, `src/training/league.py:775`. | String; could log as hash or skip. |
| `league/candidate_mean_wr` | `diag["candidate_mean_wr"]` when candidate_id, `src/training/league.py:783`. | Only when league gate runs with candidate. |
| `league/candidate_worst_wr` | `diag["candidate_worst_wr"]`, `src/training/league.py:785`. | Same. |
| `league/candidate_vs_best_wr` | `diag["candidate_vs_best_wr"]`, `src/training/league.py:790`. | Same. |
| `league/anchor_count` | `diag["anchor_count"]`, `src/training/league.py:777`. | Available. |

---

### Perf / timing

| Suggested name | Where computed / available | Caveats |
|----------------|---------------------------|---------|
| `perf/step_h2d_ms` | `trainer.py:1096–1105` — only when `profile_training_steps > 0` and step < profile_steps; logged to logger, not TB. | Conditional on config. |
| `perf/step_d4_ms` | Same block. | Conditional. |
| `perf/step_fwd_ms` | Same block. | Conditional. |
| `perf/step_bwd_ms` | Same block. | Conditional. |
| `perf/step_ema_ms` | Same block. | Conditional. |
| `perf/step_other_ms` | Same block. | Conditional. |
| `perf/step_total_ms` | Same block. | Conditional. |
| `perf/iter_time_sec` | `iter_time = time.time() - iter_start`, `src/training/main.py:1527`; in summary dict, not TB. | Available in main loop. |
| `perf/dataloader_time` | Not currently broken out; could add in train_epoch. | Would need timing around batch_iter. |
| `perf/gpu_util` | Not present. | Would require nvidia-smi or py3nvml. |

---

### Value / score calibration (training)

| Suggested name | Where computed / available | Caveats |
|----------------|---------------------------|---------|
| `calibration/value_mae` | Not computed; could add MAE between value and target_value in get_loss. | Would add in model. |
| `calibration/score_mae` | Not in current get_loss; score head is distributional (201 bins). | Could add mean absolute error to bin centers. |
| `calibration/score_entropy` | Not in get_loss; could add entropy of predicted score distribution. | Would add in model. |
| `calibration/score_pred_mean` | Mean of predicted score distribution (over bins). | Would add in model. |
| `calibration/score_target_mean` | Mean of target score distribution. | Would add in model. |

---

### Action mask / legal stats (training or selfplay)

| Suggested name | Where computed / available | Caveats |
|----------------|---------------------------|---------|
| `legal/num_legal_per_sample` | In training, `action_masks` is (B, A); sum over dim=1 gives legal counts. Not aggregated. | Easy in trainer or dataset. |
| `legal/frac_legal` | Mean over batch of (num_legal / A). | Easy. |
| `legal/mask_zero_count` | Count of zero mask entries; “illegal mass”. | Easy from batch. |
| `legal/avg_num_legal` | In selfplay, `avg_num_legal` in game summary and aggregation, `selfplay_optimized.py:861`, `selfplay_optimized_integration.py:890`. | Available in selfplay; not in training unless added. |

---

### Policy distribution stats (training)

| Suggested name | Where computed / available | Caveats |
|----------------|---------------------------|---------|
| `policy/topk_mass` | Not computed; could add sum of top-k softmax probs (e.g. k=1,5,10). | Would add in get_loss or trainer. |
| `policy/entropy` | Already as `policy_entropy` in get_loss and train. | — |
| `policy/effective_branching` | exp(entropy); could add in trainer from policy_entropy. | Trivial. |
| `policy/max_prob` | max over actions of softmax; “sharpness”. | Would add in get_loss. |
| `policy/fraction_zero_pi` | Fraction of samples with zero probability on target action. | In checkme trainer as TB tag; not in src. |

---

### Summary

- **PART A:** All current TB output is **scalar**; 40+ distinct tag names (including dynamic `iter/<k>`). Cadence: every 10 steps (train/*), every `val_frequency` (val/*), per iteration (eval/*, selfplay/*, buffer/*, iter/*, league/*). Conditional flags: `logging.tensorboard.enabled`, league enabled + iteration > 0, suite_winrates for suite WRs.
- **PART B:** Candidates are grouped by training, selfplay, MCTS, eval, replay, league, perf, calibration, action-mask, and policy-distribution. Many are already available in dicts (train_metrics, selfplay_stats, eval_results, league diag) but not written to TB; a few would need new computation or instrumentation.
