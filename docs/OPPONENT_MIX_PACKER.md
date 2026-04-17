# Opponent mix: vs packer (training self-play)

A small fraction of training self-play games can be played **NN vs packer** (greedy packing heuristic) instead of NN vs NN. This adds opponent diversity with minimal compute and minimal risk, to help reduce fragmentation tail (components / isolated holes) and move toward beating humans.

## What it does

- **Config:** `selfplay.opponent_mix` (default: **off**).
- When **enabled**, each game is chosen to be “vs packer” with probability **vs_packer_prob** (constant or ramped by iteration).
- In a vs-packer game:
  - One side is the **packer** (greedy: min empty components, min empty squares, max buttons); the other is the **NN + MCTS**.
  - **Packer moves are not stored** in the replay buffer. Only **NN moves** are stored (positions where the NN chose the move). So replay is still NN-only; we do **not** store packer moves.
  - Terminal outcome (winner, scores) is used as usual to fill value/score/ownership targets for the stored NN positions.
- Which side is packer: **alternate_sides: true** → alternate by game index (P0 packer in even-index games, P1 in odd); **false** → random.

## Recommended settings

- **vs_packer_prob:** **0.15** (10–15% of games). Keeps replay almost unchanged; small compute overhead.
- **Ramp:** Prefer ramping from 0 → 0.15 over ~10 iters to avoid shocking the distribution, e.g.:
  - **ramp.enabled: true**
  - **start_iter: 88**, **end_iter: 97**
  - **start_prob: 0.00**, **end_prob: 0.15**
- **store_only_nn_moves: true** — only NN moves go to replay; packer moves are never stored.
- **alternate_sides: true** — balance which side is packer.

## Example config (enable with ramp)

```yaml
selfplay:
  opponent_mix:
    enabled: true
    vs_packer_prob: 0.15
    alternate_sides: true
    store_only_nn_moves: true
    ramp:
      enabled: true
      start_iter: 88
      end_iter: 97
      start_prob: 0.00
      end_prob: 0.15
```

To use a **constant** probability (no ramp), set **ramp.enabled: false**; then **vs_packer_prob** is used every iteration.

## Logging (TensorBoard / iteration summary)

- **selfplay_frac_games_vs_packer** — fraction of games this iteration that were vs packer.
- **selfplay_nn_vs_packer_winrate** — NN side win rate in vs-packer games only.
- **selfplay_avg_final_empty_components_mean_vs_packer** — mean empty components (both players) in vs-packer games only.
- **selfplay_avg_final_isolated_1x1_holes_mean_vs_packer** — mean isolated 1×1 holes in vs-packer games only.

## Note on LR / epochs

We are **not** increasing LR when adding opponent mix. If you increase **epochs_per_iteration** (e.g. 2 → 3), keep LR unchanged. If KL spikes after that, consider reducing LR slightly rather than increasing it.

## Packing-ordering rollout

During rollout of the **packing placement-ordering heuristic** (BUY rank for Progressive Widening), we keep **opponent_mix disabled** for training. This reduces distribution-shift noise and avoids keeping KL elevated while we evaluate the effect of the new heuristic. Packer remains available for **evaluation** (e.g. `tools/ab_test_widening_iter069.py --opponent packer` or `--opponent mixed`).
