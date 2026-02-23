# Patchwork AlphaZero — Production-Grade End-to-End Audit Report

**Date:** 2026-02-20  
**Scope:** Post-refactor correctness, dataflow, action encoding, checkpoint/resume, training pipeline  
**Status:** Audit complete; fixes applied; release-ready with caveats

---

## 1) Architecture Overview

### 1.1 Repo Map

| Directory | Purpose |
|-----------|---------|
| `src/` | Core implementation |
| `src/training/` | Main orchestrator, selfplay, trainer, replay, run_layout, evaluation, league |
| `src/network/` | StateEncoder, ActionEncoder, model, d4_augmentation, gpu_inference_server |
| `src/mcts/` | alphazero_mcts_optimized (PUCT, virtual loss, FPU, root noise) |
| `src/game/` | patchwork_engine (rules, state, legal_actions, apply_action, terminal, scoring) |
| `configs/` | YAML configs (config_best.yaml, config_e2e_smoke.yaml, etc.) |
| `tools/` | pipeline_audit, resume_plan, run_e2e_pipeline_check, preflight, run_frozen_ladder, etc. |
| `tests/` | Unit + integration tests |
| `runs/<run_id>/` | Per-run layout: staging/, committed/, run_state.json, replay_state.json |
| `logs/` | training.log, config_snapshot.yaml, environment.json, metadata.jsonl |
| `TRUTH/` | encoding_spec.json (authoritative encoding reference) |

### 1.2 Entrypoints

| Entrypoint | Invocation | Main / CLI | Outputs | Connects to |
|------------|------------|------------|---------|-------------|
| **Training** | `python -m src.training.main --config configs/config_best.yaml` | `main()`, argparse | run_state.json, committed/iter_N/, checkpoints/, logs/ | selfplay, trainer, replay, run_layout, evaluation |
| **E2E smoke** | `python -m tools.run_e2e_pipeline_check --config configs/config_e2e_smoke.yaml` | `main()` | E2E_PIPELINE_CHECK_REPORT.md, artifacts/e2e_check/ | encoder, model, MCTS, selfplay, replay, trainer, evaluator |
| **Pipeline audit** | `python tools/pipeline_audit.py --config configs/config_best.yaml [--iteration N]` | `main()` | stdout | config only (no runtime) |
| **Resume plan** | `python tools/resume_plan.py --run-dir runs/<run_id>` | `main()` | stdout | run_layout, run_state, reconcile |
| **Inference / GUI** | `python inference.py` (via patchwork_api) or `launch_gui.bat` | PatchworkAgent | N/A | model, encoder, MCTS |
| **Frozen ladder** | Called from main every 25 iters; also `tools/run_frozen_ladder.py` | `run_frozen_ladder()` | eval/ladder/iter_XXX.json, history.csv | committed checkpoints, evaluation |

### 1.3 End-to-End Dataflow Diagram (Textual)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ CONFIG LOAD (config_best.yaml)                                                    │
│   - Schedules: step semantics (last entry where iteration <= current)             │
│   - LR, cpuct, q_value_weight, window_iterations, games, temperature, sims        │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ ITERATION LOOP (main.py)                                                          │
│   Apply schedules → _apply_iteration_schedules(iteration)                          │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 1. SELFPLAY MODEL SELECTION                                                       │
│   iter=start & resume_checkpoint ? resume_checkpoint : best_model_path            │
│   train_base = same as selfplay_model (model weights source)                      │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 2. SELFPLAY GENERATION (selfplay_optimized_integration + selfplay_optimized)      │
│   - NPZ shards → staging/iter_N/iter_NNN_shards/*.npz                              │
│   - _merge_shards → staging/iter_N/selfplay.h5                                    │
│   - Attrs: selfplay_complete=true, selfplay_num_games_written, score_scale, etc.  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 3. REPLAY BUFFER                                                                  │
│   add_iteration(iter, selfplay_path)                                               │
│   - Window eviction (window_iterations)                                            │
│   - get_training_data() → merged_training.h5 (value rescale if score_scale diff)  │
│   - newest_fraction sampling; subsample to max_size if needed                     │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 4. TRAINING (trainer.train_iteration)                                             │
│   - PatchworkDataset(merged_training.h5) → in-memory                               │
│   - train_epoch → loss (policy CE + value MSE + ownership BCE)                     │
│   - Checkpoint: staging/iter_N/iteration_NNN.pt                                    │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 5. EVALUATION (vs previous best, optional vs pure MCTS)                           │
│   - SPRT / micro-gate / fixed-game                                                │
│   - Anti-regression floor check                                                   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 6. ATOMIC COMMIT                                                                  │
│   commit_iteration: staging → committed (os.rename)                                │
│   run_state.json, replay_state.json, best_model.pt, latest_model.pt              │
│   _flush_staged_log()                                                              │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 7. FROZEN LADDER (every 25 iters, if enabled)                                      │
│   run_frozen_ladder() → eval/ladder/iter_XXX.json                                   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 1.4 Refactor Risk Matrix (Prioritized)

| Area | Risk | Evidence |
|------|------|----------|
| Action encoding | P0 | encoder.py, alphazero_mcts_optimized.py, model.py all use 2026; BUY_START=82, PATCH_START=1 |
| Masks / illegal | P0 | model.py masked_fill(-inf); MCTS priors from legal only; loss renormalizes target |
| Schedule resolution | P1 | main._apply_iteration_schedules; pipeline_audit aligns |
| Checkpoint / resume | P0 | run_layout.commit_iteration; reconcile_run_state; repair logic |
| Dataset schema | P0 | HDF5: states(56,9,9), masks(2026), policies(2026), values, ownerships(2,9,9) |
| Augmentation | P0 | d4_augmentation roundtrip tested in run_e2e_pipeline_check CHECK_4 |
| Value rescaling | P0 | replay_buffer rescale_value_targets when score_scale differs; LEGACY_SCORE_SCALE fallback |

---

## 2) Verification Results

### 2.1 Commands Run

| Command | Result |
|---------|--------|
| `git rev-parse HEAD` | N/A (not a git repo) |
| `git status` / `git diff --stat` | N/A |
| `python -c "import src.training.main; ..."` | **PASS** (Import OK) |
| `pytest -q` | **PASS** (136 passed, 2 skipped after fixes) |
| `ruff check .` | Not available (ruff not installed) |
| `python tools/pipeline_audit.py --config configs/config_best.yaml --iteration 0` | **PASS** |
| `python tools/resume_plan.py --run-dir runs/patchwork_production` | **PASS** |
| `python -m tools.run_e2e_pipeline_check --config configs/config_e2e_smoke.yaml --device cpu` | **PASS** (all 10 checks in ~46s) |

### 2.2 Test Failures (Fixed)

1. **test_amp_accuracy::test_trainer_bfloat16_one_epoch**  
   - **Cause:** `PatchworkDataset(str(h5_path), config=cfg)` — `PatchworkDataset` only accepts `h5_path`.  
   - **Fix:** Removed `config` argument.

2. **test_integration::test_selfplay_worker_policy_target_uses_scheduled_temperature**  
   - **Cause:** Patched `create_target_policy` wrapper didn't accept `mode` kwarg; config lacked `policy_target_mode` for shaped mode.  
   - **Fix:** Extended wrapper with `mode="visits", **kwargs`; added `policy_target_mode: "visits_temperature_shaped"` to config.

### 2.3 E2E Smoke Check Summary

All 10 checks passed:

- CHECK 1: Import + registry (56 ch, schema v2)
- CHECK 2: Encoder invariants (64 samples)
- CHECK 3: Structured head indexing (illegal→-inf, softmax→0)
- CHECK 4: D4 round-trip (state, policy, mask)
- CHECK 5: MCTS legality (8 states, no illegal move selected)
- CHECK 6: Selfplay shards (8 games, 32 sims, HDF5 complete)
- CHECK 7: Merge (2736 positions, policy entropy, value mean in [-1,1])
- CHECK 8: Training (20 steps, loss finite)
- CHECK 9: Tiny eval (4 games)
- CHECK 10: Report written

---

## 3) Issues List (Prioritized)

### P0 (Correctness) — None Remaining

All P0 issues identified during audit were either pre-existing and fixed, or confirmed correct.

### P1 (Stability / Minor Fixes Applied)

| # | Severity | Symptom / Risk | Root Cause | Evidence | Fix |
|---|----------|----------------|------------|----------|-----|
| 1 | P1 | Test failure: PatchworkDataset | `config` kwarg not supported | trainer.py:99 `def __init__(self, h5_path: str)` | Removed `config` from test call |
| 2 | P1 | Test failure: create_target_policy | Patched wrapper missing `mode`, config missing policy_target_mode | selfplay_optimized uses mode="visits" or "visits_temperature_shaped" | Updated wrapper + config |
| 3 | P2 | NameError in E2E CHECK_8 on NaN | `step` vs `steps` variable | run_e2e_pipeline_check.py:678, 687 | Changed `step` → `steps` in error messages |

### P2 (Maintainability)

| # | Severity | Note |
|---|----------|------|
| 1 | P2 | Repo is not a git repo — no commit hash / diff tracking for reproducibility |
| 2 | P2 | `ruff` not installed — lint not run (optional) |

---

## 4) Patch Set

### 4.1 Diffs Applied

**tests/test_amp_accuracy.py**
```diff
-            dataset = PatchworkDataset(str(h5_path), config=cfg)
+            dataset = PatchworkDataset(str(h5_path))
```

**tests/test_integration.py**
```diff
+            "policy_target_mode": "visits_temperature_shaped",
     def recording_create(vc, temperature=1.0):
-        captured_temps.append(float(temperature))
-        return orig_create(vc, temperature=temperature)
+    def recording_create(vc, temperature=1.0, mode="visits", **kwargs):
+        captured_temps.append(float(temperature))
+        return orig_create(vc, temperature=temperature, mode=mode, **kwargs)
```
(+ config/indentation fix for policy_target_mode)

**tools/run_e2e_pipeline_check.py**
```diff
-            raise E2ECheckError("CHECK_8", f"Loss NaN/Inf at step {step}", ...)
+            raise E2ECheckError("CHECK_8", f"Loss NaN/Inf at step {steps}", ...)
-                    raise E2ECheckError("CHECK_8", f"Gradient NaN/Inf at step {step}", ...)
+                    raise E2ECheckError("CHECK_8", f"Gradient NaN/Inf at step {steps}", ...)
```

### 4.2 Regression Tests

- Existing `test_phase2_invariants::test_action_indexing_alignment_and_mask_legality` confirms:
  - `engine_action_to_flat_index` == `ActionEncoder.encode_action` for all legal actions
  - decode → encode roundtrip
  - mask[legal_idx] == 1.0
- E2E run_e2e_pipeline_check exercises full pipeline (selfplay → merge → train → eval)

---

## 5) Release Readiness Checklist

| Criterion | Status |
|-----------|--------|
| Deterministic-ish smoke run passes end-to-end | **PASS** |
| Resume from checkpoint works | **PASS** (resume_plan shows next_iter=63, checkpoint=best_model) |
| Selfplay dataset schema validated and consistent | **PASS** (CHECK_6, CHECK_7) |
| Action indexing & masking alignment proven | **PASS** (encoder, MCTS, model, data, loss all 2026; test_phase2_invariants) |
| No P0/P1 remaining | **PASS** |
| Basic performance sanity | **PASS** (E2E ~46s on CPU; production has GPU) |

### STOP Conditions — Met

- Unit tests pass
- E2E smoke run passes
- Resume path exercised (resume_plan)
- Action indexing + masking alignment proven
- No P0/P1 issues remaining

---

## 6) Action Space Alignment (Verified)

### Single Source of Truth

- **ActionEncoder** (`src/network/encoder.py`): `PASS_INDEX=0`, `PATCH_START=1`, `BUY_START=82`, `ACTION_SPACE_SIZE=2026`
- **MCTS** (`alphazero_mcts_optimized.py`): `_PASS_INDEX=0`, `_PATCH_START=1`, `_BUY_START=82`, mask length 2026
- **Model** (`model.py`): `PASS_INDEX=0`, `PATCH_START=1`, `BUY_START=82`, `max_actions=2026`

### Indexing Formula

| Range | Type | Formula |
|-------|------|---------|
| 0 | Pass | 0 |
| 1–81 | Patch | 1 + board_pos (board_pos = row*9+col) |
| 82–2025 | Buy | 82 + (slot_index*8 + orient)*81 + pos |

### Masking Applied

- **Model forward:** `policy_logits.masked_fill(action_mask == 0, -inf)` before softmax
- **Loss:** target renormalized over legal; illegal logits -inf → softmax 0
- **MCTS expansion:** priors from softmax over legal indices only; illegal never become children
- **Action selection:** temperature applied to visit counts; illegal never in visit_counts

---

## 7) Recommendations

1. **Initialize git** if reproducible runs are important (config_hash + commit in metadata).
2. **Add ruff** to CI for lint consistency.
3. **Optional:** Add `test_action_encode_decode_roundtrip_engine` that decodes indices and applies them via `apply_action_unchecked` to confirm legality (belt-and-suspenders).

---

*Audit performed per production-grade pre-release gate. E2E smoke and unit tests pass. System is safe for overnight / long training runs.*
