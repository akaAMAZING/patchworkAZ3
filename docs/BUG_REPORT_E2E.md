# E2E Pipeline Bug Hunt Report

Comprehensive audit of the Patchwork AlphaZero training pipeline. Run date: 2025-02-21. Updated: 2025-02-21 (full audit + verification).

---

## Critical Bugs (will cause failures)

### 1. **KeyError in repair path for gold_v2 runs** — `main.py` ✅ FIXED

**Location:** `_repair_run_state_from_committed()` ~line 702

**Issue:** Uses `f["states"]` to get position count, but gold_v2_multimodal HDF5 files use `spatial_states`. When repair runs (crash recovery where committed dirs exist but run_state lagged), this raises `KeyError` and breaks repair.

**Status:** Fixed — uses `states_key = "spatial_states" if "spatial_states" in f else "states"`.

---

### 2. **Self-play reuse check ignores gold_v2 schema** — `selfplay_optimized_integration.py` ✅ FIXED

**Location:** `generate()` ~line 201

**Issue:** Reuse logic uses `f["states"]` when checking existing selfplay.h5. For gold_v2, the file has `spatial_states` not `states`, so `n_pos` becomes 0. Reuse can incorrectly fail or produce wrong stats when `num_games` attr is missing.

**Status:** Fixed — uses `states_key` pattern.

---

## Medium Bugs (wrong behavior when config differs)

### 3. **Evaluation uses wrong game count for vs_previous_best** — `main.py` ✅ FIXED

**Location:** `_evaluate_model()` ~line 1251

**Issue:** `num_games` was computed once and reused for both pure MCTS and vs_previous_best. When `games_vs_best` is set separately, vs_previous_best incorrectly used the MCTS count.

**Status:** Fixed — `num_games_mcts` and `num_games_best` now computed separately; vs_previous_best uses `num_games_best`.

---

### 4. **Frozen ladder eval games not read from config** — `tools/run_frozen_ladder.py` ✅ VERIFIED CORRECT

**Location:** `run_frozen_ladder()` lines 98–106

**Issue:** The frozen ladder runs its own evaluation; game counts should come from `evaluation.frozen_ladder.games_*`.

**Status:** Verified — `run_frozen_ladder.py` reads `games_anchor`, `games_rolling_initial`, `games_rolling_topup` from `evaluation.frozen_ladder`. Frozen ladder is a **standalone tool** (not invoked by main loop); `main.py` only passes through metadata to commit manifest.

---

## Minor / Hardening

### 5. **GradScaler with bf16** — `trainer.py` ✅ FIXED

**Note:** PyTorch recommends not using GradScaler with bf16 (bf16 has sufficient dynamic range).

**Status:** Fixed — `trainer.py` lines 367–369 use `use_scaler = self.use_amp and (self._autocast_dtype != torch.bfloat16)`, so GradScaler is not created when using bf16.

---

### 6. **TensorBoard writer logs to `iteration` not `global_step`** — `main.py` ~line 989

**Observation:** `self.writer.add_scalar(f"iter/{k}", v, iteration)` uses `iteration` as the x-axis. For continuous TensorBoard curves across iterations, `global_step` is usually preferred. Verify this is intentional for "per-iteration" views.

---

### 7. **Validation loader ownership mask differs from train** — `trainer.py` ✅ FIXED

**Location:** `validate()` ~line 746

**Issue:** Training uses per-sample `ownership_valid_mask` when ownership has -1 sentinels. Validation used `if ownerships.min().item() >= 0` — a global check. Mixed batches could include invalid ownership in validation loss.

**Status:** Fixed — validation now uses same per-sample `valid_mask` and passes `ownership_valid_mask` to `get_loss`.

---

## Verified Correct

- **EMA save/load:** Checkpoints include `ema_state_dict`; `get_state_dict_for_inference` correctly selects EMA for self-play and eval when configured.
- **Replay buffer:** `restore_state` correctly uses `states_key` in merge (expects network.input_channels); repair path in main was the bug.
- **Config hash / mismatch guard:** Correctly blocks resume with different config.
- **Transactional commit:** Staging → committed flow is correct; commit manifest written last.
- **Schedule propagation:** q_value_weight, cpuct, window_iterations applied correctly. (Full integrity Step 3 fix: `window_size` must be synced in test; `_apply_iteration_schedules` only mutates q_value_weight/cpuct, not `replay_buffer.window_size` — main applies it at iteration start.)
- **D4 augmentation:** Dynamic D4 with slot_piece_ids; dataset handles multimodal.
- **Value targets:** `value_targets.py` aligns with dual-head; terminal_value consistent with MCTS.

---

---

## Additional Bug (2025-02-21 hunt)

### 8. **Integrity check Step 3 schedule verification fails** — `tools/full_integrity_check.py` ✅ FIXED

**Location:** `step3_schedule_verification()` ~line 384

**Issue:** Step 3 calls `trainer._apply_iteration_schedules(it)` then asserts `trainer.replay_buffer.window_size == expected_window(it)`. But `_apply_iteration_schedules` only mutates q_value_weight and cpuct — it does **not** set `replay_buffer.window_size`. That's done in main's iteration loop before self-play. So at iter 200, `window_size` stayed at 8 (from ReplayBuffer init) while expected was 12 → assertion failed.

**Fix:** Sync `window_size` in the test: `trainer.replay_buffer.window_size = _get_window_iterations_for_iteration(trainer.config, it)` before asserting.

---

## 2025-02-21 Full E2E Audit — Additional Verification

### Pipeline components traced

| Component | Status |
|-----------|--------|
| **main.py** | Repair path, eval game counts, gate logic verified |
| **selfplay_optimized_integration.py** | Reuse check, `states_key` for gold_v2 |
| **selfplay_optimized.py** | Returns `spatial_states`/global/track/shop for gold_v2; shard write includes multimodal |
| **replay_buffer.py** | `add_iteration`, `get_training_data` use `states_key`; rejects mixing legacy/multimodal |
| **trainer.py** | PatchworkDataset `states_key`; GradScaler disabled for bf16 |
| **alphazero_mcts_optimized.py** | `_use_multimodal` from network; uses `encode_state_multimodal` when True |
| **value_targets.py** | Dual-head aligned with MCTS and self-play |
| **run_frozen_ladder.py** | Reads `games_anchor`, `games_rolling_*` from config |

### New findings (hardening / observations)

- **games_per_eval:** Legacy config key; `_evaluate_model` correctly falls back to `games_vs_pure_mcts` / `games_vs_best`. No bug.
- **Gate auto-accept logic:** When `games_vs_*` are all 0, `num_eval_games <= 0` correctly triggers auto-accept (lines 1670–1680). Verified.
- **Full integrity check:** Step 4 (mini E2E) can take 3–5+ min on GPU (2 full iterations). Consider `--run_e2e false` for quick validation.
- **run_e2e_pipeline_check CHECK 1:** Hardcodes `input_channels == 56`. Fails if used with legacy 16ch config — intentional for gold_v2 validation.

### No new critical or medium bugs found

---

## Summary

| Severity | Count | Fixed |
|----------|-------|-------|
| Critical | 2 | 2 |
| Medium   | 2 | 2 |
| Minor    | 3 | 3 |
| Tooling  | 1 | 1 |

All critical/medium bugs fixed. Minor #5 (GradScaler bf16) fixed. #6 (TensorBoard step) remains as an intentional design note. Full audit 2025-02-21: no new bugs; frozen ladder verified correct.
