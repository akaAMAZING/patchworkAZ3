# 201-bin + WIN_FIRST Validation Report

## Summary

Final validation of the 201-bin + WIN_FIRST integration is implemented and runnable via:

```bash
python tools/validate_201bin_winfirst.py --config configs/config_best.yaml --checkpoint <path>
```

With optional: `--benchmark-minutes 5`, `--ab-games 200`, `--skip-benchmark`, `--skip-ab`.

---

## 1) Checkpoint compatibility smoke (critical)

**Implemented:**

- Load checkpoint (scalar-head or 201-bin) into the 201-bin model.
- **Logs:** Capture `src.network.model` WARNINGs; report whether `score_head` weights were dropped/reinitialized (expected for scalar-head checkpoints).
- **Forward:** Run on CPU with correct multimodal inputs (32ch state + x_global, x_track, shop_ids, shop_feats) for gold_v2; assert `score_logits` shape `(B, 201)` and no NaNs.
- **GPU server + eval_client:** Subprocess server (picklable top-level `_run_gpu_server_standalone` for Windows); send 3 gold_v2 requests (or legacy if not use_film); assert responses are 5-tuples unpacking to `(priors_legal, value, mean_points, score_utility)` with:
  - `value` in [-1, 1]
  - `mean_points` in ~[-100, 100]
  - `score_utility` in ~[-1, 1]

**Result (with 201-bin checkpoint):** PASS — forward_ok=True, server_ok=True. With a scalar-head checkpoint you will see `score_head_skipped=True` in the report.

---

## 2) Self-play speed benchmark

**Implemented:**

- Uses `SelfPlayGenerator.generate(iteration=1, ...)` with config override so iteration 1 runs a small number of games (`max(5, benchmark_minutes*2)`).
- Reports: `games`, `games_per_min`, `eval_req_per_sec` (estimated), `avg_batch_size`.
- GPU utilization / avg eval latency: not instrumented (N/A in table); can be added via server-side metrics if needed.

**Run:** Use `--benchmark-minutes 5` (default) and do not use `--skip-benchmark` to populate the metrics table.

---

## 3) Behavioral A/B (WIN_FIRST on vs off)

**Implemented:**

- Same checkpoint; **A:** WIN_FIRST enabled (config default), **B:** WIN_FIRST disabled (`win_first.enabled=false`).
- Uses `PatchworkEvaluator.evaluate_vs_baseline(..., baseline_type="pure_mcts", num_games=ab_games)` for each.
- Metrics: **win rate**, **avg final margin** (`avg_model_score_margin`), **avg margin (losses only)** (`avg_loss_margin`), **worst-5% loss tail** (5th percentile of loss margins).

**Run:** Use `--ab-games 200` (or 400) and do not use `--skip-ab`.

---

## 4) WIN_FIRST root selection debug (1 game)

**Implemented:**

- **MCTS:** `WinFirstConfig.debug_log_one_game` (parsed from `win_first.debug_log_one_game` in config). When True, the first time `_select_action_win_first` is called we log:
  - `best_Qv`, `delta`, `candidate_count`
  - Top-3 candidates by `(Qv, Qs_points, Nv)`.
- **Validation script:** Sets `win_first.debug_log_one_game: true`, runs 1 game via `init_optimized_worker` + `_WORKER.play_game(0, 0, 42)` on CPU, captures `src.mcts.alphazero_mcts_optimized` logger. **Phase 4 passes** only if the captured log contains `"WIN_FIRST debug"`.

**Manual verification:** If the harness does not capture the log (logger propagation / timing), set in config:

```yaml
selfplay:
  mcts:
    win_first:
      debug_log_one_game: true
```

Run one self-play game and check stdout/logs for `[WIN_FIRST debug] best_Qv=...`.

**Bug fix:** `_compute_terminal_score_utility` previously used `sat = (2.0 / math.pi) * math.atan` (float * builtin), causing a TypeError. Fixed to a proper `def sat(x): return (2.0 / math.pi) * math.atan(x)`.

---

## 5) Go/no-go

- **GO** if: (1) checkpoint smoke passes, (2) no material speed regression (benchmark passes or skipped), (3) A/B shows WIN_FIRST does not regress (or improves), (4) WIN_FIRST debug log found (or manually verified).
- **NO-GO** if any of (1)–(4) fail; the script prints which phase failed and the recommendation.

---

## Metrics table (example)

| Phase              | Passed | Details |
|--------------------|--------|--------|
| 1 Checkpoint smoke | PASS   | score_head_skipped=False, forward_ok=True, server_ok=True |
| 2 Self-play benchmark | PASS | games=10, games/min=2.0, eval req/s=… |
| 3a WIN_FIRST ON    | -      | WR=55% avg_margin=2.1 worst_5%_loss=-12 |
| 3b WIN_FIRST OFF   | -      | WR=52% avg_margin=1.8 worst_5%_loss=-15 |
| 4 WIN_FIRST debug  | PASS   | one-game log |

---

## Files touched

- **`src/mcts/alphazero_mcts_optimized.py`:** `WinFirstConfig.debug_log_one_game`, parser, `_win_first_debug_logged`, one-time debug log in `_select_action_win_first` (including early-return path); fix `_compute_terminal_score_utility` sat definition.
- **`tools/validate_201bin_winfirst.py`:** New script for (1)–(5); `_run_gpu_server_standalone` for Windows spawn; CPU forward and gold_v2 payload for server test; benchmark via `generate(iteration=1)`; A/B via evaluator; phase 4 via worker play_game + log capture.

---

## Recommendation

- With **201-bin checkpoint** and current code: (1) passes; (2) and (3) require a real run with `--benchmark-minutes 5` and `--ab-games 200` (no skip).
- For **production:** Run full validation with the **current best scalar-head checkpoint** (e.g. `checkpoints/best_model.pt` or latest from `runs/.../committed/`). Confirm (1) shows `score_head_skipped=True`, (2) games/min in line with baseline, (3) WIN_FIRST on is not worse than off. Then **GO** to resume training.
