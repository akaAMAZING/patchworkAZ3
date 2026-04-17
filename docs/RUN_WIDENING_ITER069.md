# Progressive Widening + Margin Tie-Break (from iter_069)

## 1. Run A/B test (no training)

Compare baseline MCTS vs new MCTS (same checkpoint, same sims):

```bash
# Default: 200 games, 208 sims, checkpoint = runs/<run_id>/committed/iter_069/iteration_069.pt
python tools/ab_test_widening_iter069.py

# Custom checkpoint and games
python tools/ab_test_widening_iter069.py --checkpoint path/to/iteration_069.pt --games 200 --sims 208 --seed 42
```

Opponent: previous-best (iter_068) if present, else pure-MCTS 64 sims. Report includes win rate A vs opponent, win rate B vs opponent, avg final empty squares, avg game length, avg root legal count and avg root expanded count (to confirm widening).

---

## 2. Start training from iter_069 with new run_id

- **New run_id**: `patchwork_production_widening` (keeps results separate).
- **Weights**: Continue from iter_069 (no restart from scratch).
- **Replay**: Reset replay buffer or use a small window for the first 2 iterations so new search targets dominate quickly.
- **Optimizer**: Optionally reset optimizer state (recommended when changing target distribution); if kept, use lower LR for 1–2 iters.

### Commands

1. **One-time: set run_id and optional replay reset**

   Edit `configs/config_best.yaml` (or override via CLI):

   - `paths.run_id: patchwork_production_widening`
   - Optional: for first 2 iters use a smaller replay window (e.g. reduce `replay_buffer.window_iterations` for iter 69–70 only via schedule, or clear replay state and let buffer refill).

2. **Start training from iter_069**

   ```bash
   python -m src.training.main --config configs/config_best.yaml --start-iteration 69 --resume runs/patchwork_production/committed/iter_069/iteration_069.pt --run-id patchwork_production_widening
   ```

   Use `--resume` for the iter_069 checkpoint and `--run-id patchwork_production_widening` so outputs go to `runs/patchwork_production_widening/`. If your trainer expects a “resume from checkpoint” path, point it to the iter_069 committed checkpoint (e.g. `runs/patchwork_production/committed/iter_069/iteration_069.pt`) as the initial weights. The exact flag depends on your main.py (e.g. `--resume` or loading from `run_state.json` last committed).

3. **Optional: reset optimizer for first iter of new run**

   In code or config, for the first iteration after switching to the new run_id, set `resume_optimizer_state: false` (or equivalent) so optimizer state is not loaded; optionally use a slightly lower LR for that iteration.

---

## 3. Config diff (config_best.yaml)

- **selfplay.mcts**
  - `dynamic_score_utility_weight`: `0.3` → `0.12`
  - **New** `progressive_widening`:
    - `enabled: true`
    - `k_root: 64`
    - `k0: 32`
    - `k_sqrt_coef: 16`
    - `always_include_pass: true`
    - `always_include_patch: true`
  - **win_first**
    - `gate_dsu_enabled`: `true` → `false`

- **iteration.dynamic_score_utility_weight_schedule**
  - Cap at `0.12` by iter 40 and hold (e.g. entries at iter 0, 10, 25, 40, 500 with values 0, 0.06, 0.10, 0.12, 0.12).

- **paths.run_id** (for continuation)
  - Set to `patchwork_production_widening` when starting the new run from iter_069.
