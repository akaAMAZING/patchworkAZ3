# Patchwork AZ Run Metrics Export

All metrics from runs iter000 through iter024. Source: logs/metadata.jsonl, logs/tensorboard/, runs/patchwork_production/committed/.

## Files

- **iter_metrics.csv**: One row per iteration (0–24). Columns:
  - `iteration`, `timestamp_utc`, `config_hash`, `config_path`, `best_model_hash`
  - `accepted`, `global_step`, `iter_time_s`, `replay_positions`, `consecutive_rejections`
  - `best_model`, `eval_vs_best_wr`, `eval_vs_best_margin`, `eval_vs_mcts_wr`
  - `train_*`: policy_loss, value_loss, score_loss, ownership_loss, total_loss, policy_accuracy, policy_top5_accuracy, value_mse, grad_norm, policy_entropy, kl_divergence, ownership_accuracy, step_skip_rate
  - `selfplay_*`: num_games, num_positions, avg_game_length, p0_wins, p1_wins, generation_time, games_per_minute, avg_policy_entropy, avg_top1_prob, avg_num_legal, avg_redundancy, unique_positions, avg_root_q

- **tensorboard_scalars.csv**: TensorBoard scalar events. Columns: `step_or_iteration`, `tag`, `value`.
  - Tags include: train/* (per-step), val/*, iter/*, selfplay/*, buffer/*, eval/*

- **training_epochs.csv**: Per-epoch training metrics from training.log (iters 12–24). Columns: iteration, epoch, epoch_time_s, loss, pol_loss, val_loss, own_loss, own_acc_pct, pol_acc_pct, top5_pct, val_mse, grad.

- **run_state.json**, **elo_state.json**, **environment.json**: Run context (hardware, config, ELO state).

## Source

- Metadata: `logs/metadata.jsonl`
- TensorBoard: `logs/tensorboard/`
- Training logs: `runs/patchwork_production/committed/iter_*/training.log`
