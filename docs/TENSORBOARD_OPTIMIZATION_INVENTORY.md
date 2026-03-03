# TensorBoard — Canonical Spec

Single logging configuration. No conditional modes; all tags below are written when TensorBoard is enabled (`logging.tensorboard.enabled: true`). All writes are **scalars** (`add_scalar`); no histograms, text, images, figures, or embeddings.

---

## Deduped tag list (alphabetical)

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
- `train/policy_top5_accuracy`
- `train/total_loss`
- `train/value_loss`
- `val/policy_accuracy`
- `val/policy_top5_accuracy`
- `val/policy_entropy`
- `val/policy_loss`
- `val/total_loss`
- `val/value_loss`

---

## Per-tag reference

| Tag | File:Line | Cadence | Description |
|-----|-----------|---------|-------------|
| **buffer/** | | | |
| `buffer/num_iterations` | `src/training/main.py` | per iteration | Number of iteration-chunks in the replay buffer. |
| `buffer/total_positions` | `src/training/main.py` | per iteration | Total positions in the replay buffer. |
| **iter/** | | | |
| `iter/approx_identity_check` | `src/training/main.py` | per iteration | CE = H + KL identity check error (epoch avg). |
| `iter/kl_divergence` | `src/training/main.py` | per iteration | KL(MCTS target \|\| network policy) (epoch avg). |
| `iter/policy_cross_entropy` | `src/training/main.py` | per iteration | Policy CE (epoch avg). |
| `iter/policy_entropy` | `src/training/main.py` | per iteration | Policy entropy (epoch avg). |
| `iter/step_skip_rate` | `src/training/main.py` | per iteration | Fraction of training steps skipped (e.g. AMP scale backoff). |
| `iter/target_entropy` | `src/training/main.py` | per iteration | Target policy entropy (epoch avg). |
| `iter/value_mse` | `src/training/main.py` | per iteration | Value MSE (epoch avg). |
| **selfplay/** | | | |
| `selfplay/avg_game_length` | `src/training/main.py` | per iteration | Mean game length (moves) in self-play. |
| `selfplay/avg_num_legal` | `src/training/main.py` | per iteration | Mean number of legal actions per position. |
| `selfplay/avg_policy_entropy` | `src/training/main.py` | per iteration | Mean policy entropy over self-play positions. |
| `selfplay/avg_redundancy` | `src/training/main.py` | per iteration | Position redundancy (0 = unique). |
| `selfplay/avg_root_q` | `src/training/main.py` | per iteration | Mean MCTS root Q-value. |
| `selfplay/avg_top1_prob` | `src/training/main.py` | per iteration | Mean top-1 policy probability. |
| `selfplay/games_per_min` | `src/training/main.py` | per iteration | Self-play throughput (games per minute). |
| `selfplay/generation_time` | `src/training/main.py` | per iteration | Wall-clock time (seconds) for self-play generation. |
| `selfplay/num_games` | `src/training/main.py` | per iteration | Number of games generated. |
| `selfplay/num_positions` | `src/training/main.py` | per iteration | Total positions generated. |
| `selfplay/unique_positions` | `src/training/main.py` | per iteration | Unique positions (after dedup) in self-play. |
| **train/** | | | |
| `train/grad_norm` | `src/training/trainer.py` | every 10 steps | Gradient norm (after clip). |
| `train/learning_rate` | `src/training/trainer.py` | every 10 steps | Scheduler learning rate. |
| `train/policy_accuracy` | `src/training/trainer.py` | every 10 steps | Policy top-1 accuracy. |
| `train/policy_entropy` | `src/training/trainer.py` | every 10 steps | Policy entropy over batch. |
| `train/policy_loss` | `src/training/trainer.py` | every 10 steps | Policy cross-entropy loss. |
| `train/policy_top5_accuracy` | `src/training/trainer.py` | every 10 steps | Policy top-5 accuracy. |
| `train/total_loss` | `src/training/trainer.py` | every 10 steps | Total weighted loss. |
| `train/value_loss` | `src/training/trainer.py` | every 10 steps | Value MSE loss. |
| **val/** | | | |
| `val/policy_accuracy` | `src/training/trainer.py` | every `val_frequency` steps | Validation policy top-1 accuracy. |
| `val/policy_top5_accuracy` | `src/training/trainer.py` | every `val_frequency` steps | Validation policy top-5 accuracy. |
| `val/policy_entropy` | `src/training/trainer.py` | every `val_frequency` steps | Validation policy entropy. |
| `val/policy_loss` | `src/training/trainer.py` | every `val_frequency` steps | Validation policy loss. |
| `val/total_loss` | `src/training/trainer.py` | every `val_frequency` steps | Validation total loss. |
| `val/value_loss` | `src/training/trainer.py` | every `val_frequency` steps | Validation value loss. |

---

## Not logged (removed from TensorBoard)

- **eval/** — All eval tags (elo_rating, win_rate_vs_mcts, score_margin_*, game_length_*, win_rate_as_p0/p1_*) have been removed.
- **league/** — All league tags (vs_best_wr, suite_mean_wr, suite_worst_wr, cycles, exploitability, pool_size) have been removed.
- **train/** — ownership_loss, score_loss are no longer logged.
- **val/** — score_loss, ownership_loss are no longer logged. (policy_top5_accuracy is logged.)
- **iter/** — Only the whitelist above is logged; other train_metrics keys (e.g. grad_norm, total_loss, policy_loss, …) are not written as iter/*.

Evaluation and league logic still run and write to logs/CSV/diagnostics; only TensorBoard logging for those subsystems was removed.
