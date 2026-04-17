# Full Metrics Glossary

Every column in `full_metrics_0_192.csv` with source and how it is calculated.
TensorBoard tags map to these: `train/*` → train_*, `selfplay/*` → selfplay_*, `iter/*` → train_*, `buffer/*` → replay_*.

| Column | Description | Source | Code / Formula |
|--------|-------------|--------|----------------|
| `iteration` | Zero-based iteration index (0..192). | logs/metadata.jsonl | entry['iteration'] |
| `timestamp_utc` | UTC timestamp when iteration was committed. | src/training/main.py | _append_metadata |
| `config_hash` | Hash of config used for this run. | src/training/main.py | _config_hash |
| `config_path` | Path to YAML config file. | configs/config_best.yaml | paths |
| `best_model_hash` | File hash of best model checkpoint. | src/training/main.py | _file_hash(best_model_path) |
| `accepted` | True if model was accepted (promoted). With eval disabled, always True. | src/training/main.py | _original_gate / _should_accept_model |
| `global_step` | Cumulative training step count across all iterations. | src/training/main.py | train_iteration() return |
| `iter_time_s` | Wall-clock time in seconds for this iteration (selfplay + train + commit). | src/training/main.py | time.time() - iter_start |
| `iteration_time_s` | (See code or TensorBoard tag.) | — | — |
| `replay_positions` | Total positions in replay buffer after this iteration. | src/training/main.py | self.replay_buffer.total_positions |
| `replay_buffer_iterations` | Number of iterations currently in replay window. | src/training/replay_buffer.py | len(self._entries) |
| `consecutive_rejections` | Count of consecutive gate rejections (0 when no gating). | src/training/main.py | self.consecutive_rejections |
| `best_model` | Path to best model checkpoint. | src/training/main.py | self.best_model_path |
| `eval_vs_best_wr` | Win rate vs previous best (when eval games run). | src/training/main.py | eval_results['vs_previous_best']['win_rate'] |
| `eval_vs_best_margin` | Avg score margin vs previous best. | src/training/evaluation.py | avg_model_score_margin |
| `eval_vs_mcts_wr` | Win rate vs pure MCTS (when eval games run). | src/training/evaluation.py | evaluate_vs_baseline(..., 'pure_mcts') |
| `elo` | ELO rating (when evaluation.elo.enabled). | tools/elo_system.py | get_rating(player_id) |
| `applied_adaptive_games_anti_thrash_prev` | (See code or TensorBoard tag.) | — | — |
| `applied_adaptive_games_avg_len_est` | (See code or TensorBoard tag.) | — | — |
| `applied_adaptive_games_clamp_high` | (See code or TensorBoard tag.) | — | — |
| `applied_adaptive_games_clamp_low` | (See code or TensorBoard tag.) | — | — |
| `applied_adaptive_games_fill_mode` | (See code or TensorBoard tag.) | — | — |
| `applied_adaptive_games_games_needed` | (See code or TensorBoard tag.) | — | — |
| `applied_adaptive_games_games_this_iter` | (See code or TensorBoard tag.) | — | — |
| `applied_adaptive_games_max_size` | (See code or TensorBoard tag.) | — | — |
| `applied_adaptive_games_scheduled_games` | (See code or TensorBoard tag.) | — | — |
| `applied_adaptive_games_scheduled_window_iters` | (See code or TensorBoard tag.) | — | — |
| `applied_adaptive_games_target_pos_iter` | (See code or TensorBoard tag.) | — | — |
| `applied_replay_max_size` | (See code or TensorBoard tag.) | — | — |
| `applied_replay_newest_fraction` | (See code or TensorBoard tag.) | — | — |
| `applied_replay_recency_window` | (See code or TensorBoard tag.) | — | — |
| `applied_replay_window_iterations` | Replay window size. | src/training/main.py | window_iterations_schedule |
| `applied_selfplay_cpuct` | PUCT constant used. | src/training/main.py | cpuct_schedule |
| `applied_selfplay_dirichlet_alpha` | (See code or TensorBoard tag.) | — | — |
| `applied_selfplay_dynamic_score_utility_weight` | (See code or TensorBoard tag.) | — | — |
| `applied_selfplay_games` | Games actually run (after adaptive_games). | src/training/main.py | applied_settings.selfplay.games |
| `applied_selfplay_noise_weight` | (See code or TensorBoard tag.) | — | — |
| `applied_selfplay_parallel_leaves` | (See code or TensorBoard tag.) | — | — |
| `applied_selfplay_policy_target_mode` | (See code or TensorBoard tag.) | — | — |
| `applied_selfplay_q_value_weight` | (See code or TensorBoard tag.) | — | — |
| `applied_selfplay_simulations` | MCTS simulations used. | src/training/main.py | mcts_schedule |
| `applied_selfplay_static_score_utility_weight` | (See code or TensorBoard tag.) | — | — |
| `applied_selfplay_temperature` | Temperature used. | src/training/main.py | schedule lookup |
| `applied_training_amp_dtype` | (See code or TensorBoard tag.) | — | — |
| `applied_training_batch_size` | (See code or TensorBoard tag.) | — | — |
| `applied_training_lr` | Learning rate used. | src/training/main.py | lr_schedule |
| `applied_training_q_value_weight` | (See code or TensorBoard tag.) | — | — |
| `eval_vs_previous_best_avg_game_length` | (See code or TensorBoard tag.) | — | — |
| `eval_vs_previous_best_avg_loss_margin` | (See code or TensorBoard tag.) | — | — |
| `eval_vs_previous_best_avg_model_score_margin` | (See code or TensorBoard tag.) | — | — |
| `eval_vs_previous_best_avg_score_diff` | (See code or TensorBoard tag.) | — | — |
| `eval_vs_previous_best_avg_win_margin` | (See code or TensorBoard tag.) | — | — |
| `eval_vs_previous_best_baseline_wins` | (See code or TensorBoard tag.) | — | — |
| `eval_vs_previous_best_model_wins` | (See code or TensorBoard tag.) | — | — |
| `eval_vs_previous_best_ties` | (See code or TensorBoard tag.) | — | — |
| `eval_vs_previous_best_total_games` | (See code or TensorBoard tag.) | — | — |
| `eval_vs_previous_best_win_rate` | (See code or TensorBoard tag.) | — | — |
| `selfplay_avg_final_empty_components_abs_diff` | (See code or TensorBoard tag.) | — | — |
| `selfplay_avg_final_empty_components_mean` | Mean connected empty components. | src/utils/packing_metrics.py | aggregate_packing_over_games |
| `selfplay_avg_final_empty_components_mean_vs_packer` | (See code or TensorBoard tag.) | — | — |
| `selfplay_avg_final_empty_squares_abs_diff` | (See code or TensorBoard tag.) | — | — |
| `selfplay_avg_final_empty_squares_mean` | Mean empty squares at game end (packing quality). | src/utils/packing_metrics.py | aggregate_packing_over_games |
| `selfplay_avg_final_isolated_1x1_holes_abs_diff` | (See code or TensorBoard tag.) | — | — |
| `selfplay_avg_final_isolated_1x1_holes_mean` | Mean isolated 1x1 holes. | src/utils/packing_metrics.py | aggregate_packing_over_games |
| `selfplay_avg_final_isolated_1x1_holes_mean_vs_packer` | (See code or TensorBoard tag.) | — | — |
| `selfplay_avg_game_length` | Mean game length in moves. | src/training/selfplay_optimized_integration.py | np.mean(game_lengths) |
| `selfplay_avg_num_legal` | Mean number of legal actions at root. | src/training/selfplay_optimized_integration.py | np.mean(num_legals) |
| `selfplay_avg_packing_ordering_enabled` | (See code or TensorBoard tag.) | — | — |
| `selfplay_avg_packing_score_top1` | (See code or TensorBoard tag.) | — | — |
| `selfplay_avg_policy_entropy` | Mean policy entropy over root nodes. | src/training/selfplay_optimized_integration.py | np.mean(entropies) |
| `selfplay_avg_redundancy` | Position redundancy (0=unique). | src/training/selfplay_optimized_integration.py | np.mean(redundancies) |
| `selfplay_avg_root_expanded_count` | Mean root expanded nodes. | src/utils/packing_metrics.py | aggregate_root_over_moves |
| `selfplay_avg_root_expanded_ratio` | expanded/legal ratio. | src/utils/packing_metrics.py | aggregate_root_over_moves |
| `selfplay_avg_root_legal_count` | Mean root legal action count. | src/utils/packing_metrics.py | aggregate_root_over_moves |
| `selfplay_avg_root_q` | Mean root Q value. | src/training/selfplay_optimized_integration.py | np.mean(root_qs) |
| `selfplay_avg_top1_prob` | Mean top-1 action probability. | src/training/selfplay_optimized_integration.py | np.mean(top1_probs) |
| `selfplay_frac_games_vs_packer` | Fraction of games vs packer (opponent_mix). | src/training/selfplay_optimized_integration.py | len(vs_packer_games)/len(summaries) |
| `selfplay_games_per_minute` | num_games / (generation_time/60). | src/training/selfplay_optimized_integration.py | _compute_stats |
| `selfplay_generation_time` | Wall time for self-play (seconds). | src/training/selfplay_optimized_integration.py | generation_time |
| `selfplay_nn_vs_packer_winrate` | Win rate in vs-packer games. | src/training/selfplay_optimized_integration.py | _compute_stats |
| `selfplay_num_games` | Number of self-play games this iteration. | src/training/selfplay_optimized_integration.py | _compute_stats: len(summaries) |
| `selfplay_num_positions` | Total positions collected. | src/training/selfplay_optimized_integration.py | sum(s.get('num_positions',0) for s in summaries) |
| `selfplay_p0_wins` | Games won by player 0. | src/training/selfplay_optimized_integration.py | winners.count(0) |
| `selfplay_p1_wins` | Games won by player 1. | src/training/selfplay_optimized_integration.py | winners.count(1) |
| `selfplay_p50_final_empty_components_mean` | (See code or TensorBoard tag.) | — | — |
| `selfplay_p50_final_empty_squares_mean` | Median empty squares. | src/utils/packing_metrics.py | aggregate_packing_over_games |
| `selfplay_p50_final_isolated_1x1_holes_mean` | (See code or TensorBoard tag.) | — | — |
| `selfplay_p90_final_empty_components_mean` | (See code or TensorBoard tag.) | — | — |
| `selfplay_p90_final_empty_squares_mean` | 90th percentile empty squares. | src/utils/packing_metrics.py | aggregate_packing_over_games |
| `selfplay_p90_final_isolated_1x1_holes_mean` | (See code or TensorBoard tag.) | — | — |
| `selfplay_p90_root_expanded_ratio` | (See code or TensorBoard tag.) | — | — |
| `selfplay_p90_root_legal_count` | (See code or TensorBoard tag.) | — | — |
| `selfplay_selfplay_avg_final_empty_components_abs_diff` | (See code or TensorBoard tag.) | — | — |
| `selfplay_selfplay_avg_final_empty_components_mean` | (See code or TensorBoard tag.) | — | — |
| `selfplay_selfplay_avg_final_empty_components_mean_vs_packer` | (See code or TensorBoard tag.) | — | — |
| `selfplay_selfplay_avg_final_empty_squares_abs_diff` | (See code or TensorBoard tag.) | — | — |
| `selfplay_selfplay_avg_final_empty_squares_mean` | (See code or TensorBoard tag.) | — | — |
| `selfplay_selfplay_avg_final_isolated_1x1_holes_abs_diff` | (See code or TensorBoard tag.) | — | — |
| `selfplay_selfplay_avg_final_isolated_1x1_holes_mean` | (See code or TensorBoard tag.) | — | — |
| `selfplay_selfplay_avg_final_isolated_1x1_holes_mean_vs_packer` | (See code or TensorBoard tag.) | — | — |
| `selfplay_selfplay_avg_packing_ordering_enabled` | (See code or TensorBoard tag.) | — | — |
| `selfplay_selfplay_avg_packing_score_top1` | (See code or TensorBoard tag.) | — | — |
| `selfplay_selfplay_avg_root_expanded_count` | (See code or TensorBoard tag.) | — | — |
| `selfplay_selfplay_avg_root_expanded_ratio` | (See code or TensorBoard tag.) | — | — |
| `selfplay_selfplay_avg_root_legal_count` | (See code or TensorBoard tag.) | — | — |
| `selfplay_selfplay_frac_games_vs_packer` | (See code or TensorBoard tag.) | — | — |
| `selfplay_selfplay_nn_vs_packer_winrate` | (See code or TensorBoard tag.) | — | — |
| `selfplay_selfplay_p50_final_empty_components_mean` | (See code or TensorBoard tag.) | — | — |
| `selfplay_selfplay_p50_final_empty_squares_mean` | (See code or TensorBoard tag.) | — | — |
| `selfplay_selfplay_p50_final_isolated_1x1_holes_mean` | (See code or TensorBoard tag.) | — | — |
| `selfplay_selfplay_p90_final_empty_components_mean` | (See code or TensorBoard tag.) | — | — |
| `selfplay_selfplay_p90_final_empty_squares_mean` | (See code or TensorBoard tag.) | — | — |
| `selfplay_selfplay_p90_final_isolated_1x1_holes_mean` | (See code or TensorBoard tag.) | — | — |
| `selfplay_selfplay_p90_root_expanded_ratio` | (See code or TensorBoard tag.) | — | — |
| `selfplay_selfplay_p90_root_legal_count` | (See code or TensorBoard tag.) | — | — |
| `selfplay_unique_positions` | Unique position count. | src/training/selfplay_optimized_integration.py | sum(s.get('unique_positions',0) for s in summaries) |
| `train_approx_identity_check` | |H(pi)+KL(pi||p)-CE| mean; should be ~0. | src/network/model.py | approx_identity_error |
| `train_ce_minus_policy_entropy` | CE - H(pi); diagnostic. | src/network/model.py | policy_cross_entropy - policy_entropy |
| `train_grad_norm` | Global gradient norm (before clip). | src/training/trainer.py | torch.nn.utils.clip_grad_norm_; grad_norm.item() |
| `train_kl_divergence` | KL(MCTS_target || network_policy). | src/network/model.py | (target_policy_norm * (target_log - log_probs)).sum(dim=-1).mean() |
| `train_ownership_accuracy` | Binary accuracy of ownership head (threshold 0.5). | src/network/model.py | (ownership_pred == target_ownership).float().mean() |
| `train_ownership_accuracy_all_filled_baseline` | Baseline acc if predicting all filled. | src/network/model.py | ownership_filled_fraction_mean |
| `train_ownership_balanced_accuracy` | 0.5*(empty_recall + filled_recall). | src/network/model.py | get_loss ownership_balanced_accuracy |
| `train_ownership_empty_precision` | Precision for empty cells. | src/network/model.py | get_loss ownership_empty_precision |
| `train_ownership_empty_recall` | Recall for empty cells (target=0). | src/network/model.py | get_loss ownership_empty_recall |
| `train_ownership_filled_fraction_mean` | Mean of target ownership (filled cells). | src/network/model.py | target_ownership.mean() |
| `train_ownership_loss` | BCE with logits for ownership head (2×9×9). | src/network/model.py | get_loss: F.binary_cross_entropy_with_logits(ownership_logits, target_ownership) |
| `train_ownership_mae_empty_count` | MAE of predicted vs true empty count per sample. | src/network/model.py | get_loss ownership_mae_empty_count |
| `train_policy_accuracy` | Fraction of samples where argmax(policy) == argmax(target). | src/network/model.py | get_loss: (pred_actions == target_actions).float().mean() |
| `train_policy_cross_entropy` | Same as policy_loss (CE). | src/network/model.py | policy_loss |
| `train_policy_entropy` | Mean entropy of network policy over batch. | src/network/model.py | -(policy_probs * log_probs).sum(dim=-1).mean() |
| `train_policy_loss` | Cross-entropy loss between target policy and network policy. | src/network/model.py | get_loss: -(target_policy_norm * log_probs).sum(dim=-1).mean() |
| `train_policy_top5_accuracy` | Fraction where target action in top-5 predicted. | src/network/model.py | get_loss: (top5_pred == target_actions.unsqueeze(1)).any(dim=1).float().mean() |
| `train_score_loss` | Cross-entropy of 201-bin score head vs soft target from score_margins. | src/network/model.py | get_loss: -(tgt * logp).sum(dim=-1).mean() |
| `train_step_skip_rate` | Fraction of steps skipped (e.g. NaN). | src/training/trainer.py | steps_skipped / num_batches |
| `train_target_entropy` | Mean entropy of MCTS target policy. | src/network/model.py | -(target_policy_norm * target_log).sum(dim=-1).mean() |
| `train_total_loss` | policy_weight*policy_loss + value_weight*value_loss + score_weight*score_loss + ownership_weight*ownership_loss. | src/network/model.py | get_loss |
| `train_value_loss` | MSE between value head and target value. | src/network/model.py | get_loss: F.mse_loss(value, target_value) |
| `train_value_mse` | Same as train_value_loss (MSE). | src/network/model.py | value_loss.item() |
