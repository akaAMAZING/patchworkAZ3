# Pre-Flight Bug Report — 480h Training Readiness

**Scope:** Manual codebase audit before long training. No tests run.

---

## CRITICAL — Fix Before Long Run

### 1. Checkpoint save: no error handling or cleanup
**File:** `src/training/trainer.py:45-52`  
**Risk:** If `torch.save()` fails (disk full, I/O error), we leave a partial `.tmp` file. A subsequent successful save could then `os.replace()` a corrupted temp over the good checkpoint. No cleanup on failure.  
**Status:** FIXED — Added try/except, cleanup .tmp on failure, re-raise.

### 2. Checkpoint load: no error handling
**File:** `src/training/trainer.py:852-873`  
**Risk:** `torch.load()` can raise (corrupted file, wrong format, missing keys). Uncaught exception → crash on resume.  
**Status:** FIXED — Wrapped load in try/except, log + re-raise with clear message.

### 3. Replay buffer window_size can be 0
**File:** `src/training/replay_buffer.py:48`  
**Risk:** Config `window_iterations: 0` → `window_size=0`. `add_iteration` evicts everything we add. `get_training_data` would raise "Replay buffer is empty" or we train on empty data.  
**Status:** FIXED — Clamp `window_size >= 1` at init and in `_get_window_iterations_for_iteration`.

### 4. Empty training dataset not validated
**File:** `src/training/trainer.py:1005-1010`  
**Risk:** If dataset has 0 samples (e.g. bad path, empty HDF5), we create DataLoader with 0 batches. Training runs 0 steps, saves checkpoint. Silent no-op.  
**Status:** FIXED — Early check: `if len(dataset) == 0: raise RuntimeError(...)`.

---

## ALREADY SAFE (Verified)

- **LR division by zero** (`trainer.py:521`): Already guarded with `if base_lr > 0 else 0.0`.
- **Schedule IndexError** (`main.py:116`): Already has `i + 1 < len(entries)` check.
- **best_model_path None on iter 0**: Intentional — self-play uses pure MCTS bootstrap when `network_path is None`.
- **total_loss NaN**: Model (`model.py:553-561`) raises on `torch.isnan(total_loss)`.
- **config_best.yaml**: Has valid `paths`, `window_iterations: 8`, no zero values.

---

## MEDIUM — Consider for Future

- **HDF5 merge on disk full**: `get_training_data` writes large files. If disk fills mid-write, we get an exception and leave a `.tmp` file. Consider disk-space check before large merge.
- **Checkpoint cleanup during resume**: `_cleanup_checkpoints` deletes old files. Single-process: safe. Multi-process: could delete a file another process is loading. Not applicable for typical single-GPU run.
- **TensorBoard flush**: `writer.close()` may not flush all events on crash. Consider explicit `writer.flush()` before critical checkpoints.

---

## Summary

- 4 critical bugs identified and fixed.
- Your `config_best.yaml` is valid for production.
- Safe to proceed with 480h training after applying the fixes in this session.
