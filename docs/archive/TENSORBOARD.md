# TensorBoard Metric Reference — Patchwork AlphaZero

Quick-reference for all 14 tracked metrics. Open during training runs to diagnose issues early.

---

## Training Metrics (logged every 10 steps)

| Metric | What It Means | Healthy Range | Red Flag |
|---|---|---|---|
| **train/total_loss** | Sum of policy + value + ownership losses | 3→1 over first 20 iters, then slow decline | Flat or rising after iter 5 |
| **train/policy_loss** | Cross-entropy between network policy and MCTS visit targets | 4→2 early, then gradual decline | Sudden spike or plateau above 3.0 |
| **train/value_loss** | MSE between predicted and actual game outcome | 0.8→0.3 over training | Stuck above 0.6 after iter 10 |
| **train/policy_accuracy** | Top-1 match: network's best move = MCTS best move | 15%→40%+ over training | Drops below 20% or stalls |
| **train/policy_entropy** | Entropy of network's policy output during training | 2.0→4.0 (healthy exploration) | Below 1.0 = collapsing; above 6.0 = not learning |
| **train/grad_norm** | L2 norm of gradients (clipped to 1.0) | 0.3–1.0 | Consistently pinned at 1.0 (clipping too hard) or < 0.01 (dead) |
| **train/learning_rate** | Current LR from cosine-warmup schedule | Ramps up during warmup, decays to min_lr | N/A (deterministic schedule) |

## Validation Metrics (logged every val_frequency steps)

| Metric | What It Means | Healthy Range | Red Flag |
|---|---|---|---|
| **val/total_loss** | Same as train but on held-out data | Tracks train/total_loss closely | val >> train = overfitting |
| **val/policy_loss** | Validation policy cross-entropy | Tracks train/policy_loss | Gap > 0.5 from train = overfitting |
| **val/value_loss** | Validation value MSE | Tracks train/value_loss | Gap > 0.15 from train = overfitting |

## Self-Play Health (logged per iteration)

| Metric | What It Means | Healthy Range | Red Flag |
|---|---|---|---|
| **selfplay/policy_entropy** | Avg entropy of MCTS visit distributions during self-play | 1.5–3.5 | Below 1.0 = MCTS collapsing to single moves |
| **selfplay/top1_prob** | Avg probability of most-visited move in MCTS | 0.3–0.7 | Above 0.85 = policy collapse starting |
| **selfplay/game_length** | Average moves per game | 40–120 (game dependent) | Sudden drop (< 30) or spike (> 150) |

## Evaluation (logged per iteration)

| Metric | What It Means | Healthy Range | Red Flag |
|---|---|---|---|
| **eval/win_rate_vs_best** | Win rate of new model vs previous best (40 games) | 45–60% (noisy with 40 games, ±8% SE) | Below 35% for 3+ consecutive iters |
| **eval/score_margin** | Avg point margin when winning vs previous best | Positive and growing | Consistently negative |

---

## What To Watch For

### Policy Collapse (the #1 killer)
- **selfplay/policy_entropy** dropping below 1.0
- **selfplay/top1_prob** rising above 0.85
- **train/policy_entropy** dropping below 1.0
- If you see this: the model is becoming too confident, MCTS stops exploring, targets get peakier, death spiral

### Rock-Paper-Scissors Cycling
- **eval/win_rate_vs_best** stays ~50% but strength isn't growing
- Not detectable from TensorBoard alone — run `python tools/eval_latest_vs_oldest.py` every 10-20 iters
- Check: does iter 20 beat iter 1? If not, cycling may be occurring

### Overfitting
- **val/total_loss** diverges upward while **train/total_loss** keeps dropping
- Gap between val and train losses growing over iterations
- Mitigated by: replay buffer window (8 iters), data augmentation, fresh optimizer each iter

### Healthy Training Looks Like
1. Losses drop quickly in first 5-10 iters, then slow steady decline
2. Policy entropy stays in 2.0-4.0 range (model explores but has preferences)
3. Win rate vs previous best fluctuates around 50% (expected — each model is only slightly better)
4. Score margin slowly trends positive over many iterations
5. Game lengths stay in a reasonable range without sudden changes
