# Complete Changelog: Self-Play Speedups, Resume, and Bug Fixes

**Single source of truth.** Copy sections into an older non-regressing codebase to apply speed improvements, improved resume, and bug fixes (e.g. valid_7x7 D4). No references to other files—everything is inlined below.

---

# PART 1 — SELF-PLAY SPEEDUPS

## 1.1 GPU score utility (main throughput fix)

**Goal:** Restore self-play from ~20 games/min to ≥30 games/min by eliminating 201-dim score logits over IPC.

**Protocol change:**

| | Old | New |
|---|-----|-----|
| **Response** | `(rid, priors_legal, value_scalar, score_logits_201)` or tanh score | `(rid, priors_legal, value_scalar, mean_points_scalar, score_utility_scalar)` |
| **Request** | (no score field) | Add `score_center_points: float` (center for dynamic score utility, node’s to_move perspective) |

- **Error/fallback response:** `(rid, fallback_priors, np.float32(0.0), np.float32(0.0), np.float32(0.0))`.
- **Legacy:** If parsing legacy request tuples, support 6th element = `score_center_points` (default 0.0).

**Score utility math (KataGo-style):**

- Bins: integers [score_min, score_max] inclusive (default -100..100) ⇒ 201 bins.
- `sat(x) = (2/π) * atan(x)`.
- `dynamic_util(center) = E_s[ sat((s - center)/scale) ]`, `static_util = E_s[ sat(s/scale) ]`.
- `score_utility = dynamic_w * dynamic_util + static_w * static_util`.
- Center in node’s to_move perspective: `center = root_score` if node.to_move == root.to_move else `center = -root_score`. Sent per request as `score_center_points`.

**GPU inference server:**

- Request: parse `score_center_points` per request (legacy: 6th element if len>=6 else 0.0).
- After forward: keep score_logits on GPU. Compute:
  - `p = softmax(score_logits)`
  - `mean_points = (p * bins).sum(dim=-1)` in POINTS
  - `sat = (2/π)*atan`; dynamic_util from `(bins - center)/scale` per sample; optional static_util
  - `score_utility = dynamic_w * dynamic_util + static_w * static_util`
- Return 5-tuple `(rid, priors_legal, value, mean_points, score_utility)`. Do not send score logits to CPU.
- Constants once from config: `_score_min`, `_score_max`, `_score_scale` (= score_utility_point_scale), `_static_w`, `_dynamic_w`, `_score_bins_t` on device.

**Eval client:**

- Stash type: `Dict[int, Tuple[np.ndarray, float, float, float]]` (priors, value, mean_points, score_utility).
- All submit methods (legacy, multimodal, shm): add `score_center_points: float = 0.0` and include in request/payload.
- `receive()` returns 4 values: `(priors_legal, value, mean_points, score_utility)`.
- `evaluate` / `evaluate_multimodal`: accept and pass `score_center_points`; return 4-tuple.

**MCTS:**

- GPU path: submit with `score_center_points` (root_score or -root_score per node). On receive: set `node.score_estimate = mean_points`; leaf utility = value + score_utility (no local score computation).
- Local path (no GPU server): keep computing mean_points and score_utility from logits locally; still pass `score_center_points` into evaluate for API consistency.
- Config: `score_min`, `score_max`, `score_utility_point_scale` (e.g. 30), `static_score_utility_weight` (0), `dynamic_score_utility_weight` (e.g. 0.3). No `score_utility_scale`.

**Config:** `inference.torch_compile: false`, `inference.max_batch_wait_ms: 2` or `3` for stable batching.

---

## 1.2 Encoder: Numba JIT for scalars/shop

**Idea:** Encode x_global, x_track, shop_ids, shop_feats in a Numba JIT function so the hot scalar/shop path is ~35% faster than pure NumPy.

- Add Numba: `@njit(cache=True, fastmath=True, nogil=True)` (or pure-Python fallback if numba not installed).
- New function `_encode_scalars_shop_jit(slot_pids, shop_pids, n_shop, n_circle, c_pos, c_buttons, ... x_global, x_track, shop_ids, shop_feats, track_length, track_len, log1p_200, log1p_20)` that fills:
  - x_global: position/resource features (pos, buttons, income, fill fractions, etc.)
  - x_track: track positions and button/patches crossed (use small JIT helpers: `_popcount32`, `_jit_clamp_pos`, `_jit_buttons_crossed`, `_jit_patches_crossed`)
  - shop_ids, shop_feats: from circle/slot state
- In `encode_into`: after filling spatial ch 0–7 and 8–31 with NumPy, call `_encode_scalars_shop_jit(...)` with pre-extracted slot_pids and shop_pids (no dict access inside JIT). Pass prebuilt LUTs as flat arrays (e.g. `_piece_cost_lut`, `_buttons_after_i64`, etc.) so JIT sees typed arrays.
- Channels 0–7 and 8–31 remain NumPy (spatial planes and slot×orient masks).

---

## 1.3 SharedMemory (SHM) IPC for self-play

**Idea:** Workers encode directly into shared-memory slots; GPU server reads from SHM. Removes large pickle payloads over queues.

- **WorkerSharedBuffer:** Per-worker buffer with `n_slots = parallel_leaves` (e.g. 32). Each slot holds: spatial (32,9,9), global, track, shop arrays sized for one state. Pre-allocate as multiprocessing.shared_memory.SharedMemory; create views (e.g. numpy arrays) for encoder to write into.
- **Encoder:** Provide `encode_into(state, to_move, x_spatial, x_global, x_track, shop_ids, shop_feats)` that writes in-place into provided buffers (zero buffers first; SHM may be stale).
- **GPU server:** On init, accept `worker_shm_names` or equivalent; open the same SHM segments read-only. When processing a batch, copy from SHM into device tensors (or memory-map). Request payload identifies slot index and worker id so server knows which SHM region to read.
- **Eval client submit_shm:** Send (worker_id, slot_index, score_center_points) instead of full state tensor; server reads state from SHM slot.
- **Integration:** When creating self-play workers, create one WorkerSharedBuffer per worker; pass shm name/slot layout to server; workers call `encode_into` into their slot before submitting.

---

## 1.4 Worker thread limits

- In self-play worker init (before any torch/engine work): `torch.set_num_threads(max(1, n_threads))` and `torch.set_num_interop_threads(max(1, n_threads))` from config `selfplay.mcts.num_threads` (default 1). Reduces oversubscription when many workers run.
- Optionally at worker process entrypoint: `torch.set_num_threads(1)` so each worker doesn’t spawn many threads.

---

## 1.5 Training pipeline: batched dataset access and prefetch

**Problem:** Default DataLoader calls `dataset[i]` for every sample in a batch, causing hundreds of calls per step; with in-memory or heavy HDF5 this can be ~10s per batch.

- **BatchIndexSampler:** Sampler that yields *lists* of indices, one list per batch (e.g. `[i, i+1, ..., i+batch_size-1]`). Shuffle the index list with a fixed seed (e.g. `seed + epoch`), then yield slices of size `batch_size`.
- **_BatchIterableDataset:** Wraps dataset + BatchIndexSampler. In `__iter__`, for each index list from the sampler, yield `dataset[indices]` (one call per batch). Dataset must support `__getitem__(self, indices)` returning a batch dict/tuple (batched indexing).
- Use this wrapper as the DataLoader’s dataset so each step gets one batch from a single `dataset[indices]` call.
- **Prefetch:** Optional `_prefetch_generator(loader, prefetch_batches=2)`: background thread that calls `next(loader)` and puts batches in a queue; main thread consumes. Overlaps CPU (next batch) with GPU (current batch). Only when `prefetch_batches > 0` and device is CUDA.

**HDF5 / in-memory:** If dataset is in-memory, load full HDF5 (or merged replay) at init into RAM so training does not do per-sample HDF5 random access (which with compression can be 5+ s/batch). Support `dataset[indices]` returning a dict with stacked tensors for the batch.

---

## 1.6 Config summary (speed)

- `inference.torch_compile: false`
- `inference.max_batch_wait_ms: 2` or `3`
- `selfplay.mcts.num_threads: 1` (or small)
- `hardware.prefetch_batches: 2` for training prefetch

---

# PART 2 — RESUME IMPROVEMENTS

## 2.1 Run layout and atomic iterations

- **Directories:** `runs/<run_id>/staging/iter_<N>/` (work-in-progress for iteration N), `runs/<run_id>/committed/iter_<N>/` (finalized after commit). `run_state.json` at run root is the authoritative state; update only at commit.
- **Commit:** After self-play + training + eval for iteration N, write a commit manifest (e.g. `commit_manifest.json`) into staging/iter_N, then atomically move `staging/iter_N` → `committed/iter_N` (e.g. `shutil.move` on same filesystem; if different filesystems, copy+fsync+delete). Write manifest into committed dir after move. Update `run_state.json` with `last_committed_iteration`, `best_model_path`, `latest_checkpoint`, `global_step`, etc., using atomic write (.tmp + replace).
- **Deterministic run root:** `get_run_root(config, cli_run_id, cli_run_dir)` so same config + same CLI args ⇒ same directory ⇒ same run. Optionally: if user did not pass `--run-id` and there is exactly one subdir under `runs/` that has `run_state.json`, use that (single-run resume without explicit run-id).

---

## 2.2 Discard partial staging on startup

- **Always** at train startup (before auto-resume logic): compute `last_comm = max_committed_iteration(run_root)` (scan `committed/` for highest N with `commit_manifest.json`). Call `cleanup_staging(run_root, last_comm)`: delete every `staging/iter_*` that does **not** have a corresponding committed marker. Partial iterations are always discarded so each run restarts the iteration from self-play.
- **Reason:** If you only discard staging when `run_state.json` exists, a run that never wrote run_state (e.g. crash before first commit) would leave staging/iter_0 behind and the next start could reuse partial data. Discarding on every startup avoids that.

---

## 2.3 Auto-resume from run_state

- **When:** `iteration.auto_resume` is true (default), and `start_iteration == 0`, and `run_state_path` exists. If user passed explicit `--resume <path>`, use that and do not override with auto-resume.
- **Read run_state:** Load JSON. `last_committed_iteration = state["last_committed_iteration"]` or legacy: `next_iteration - 1`.
- **Reconcile with filesystem:** `reconcile_run_state(run_root, last_comm)` returns `(effective_last_comm, needs_repair)`. If `max_committed_iteration(run_root) > last_comm` (e.g. commit succeeded but run_state wasn’t updated), set `effective_last_comm = max_on_disk` and `needs_repair = True`.
- **Repair:** If needs_repair, restore replay buffer from disk (see below), then repair run_state: for each committed iter from old_last+1 to effective_last, update run_state fields (last_committed_iteration, best_model_path, latest_checkpoint, global_step, etc.) from commit manifests and replay state; rewrite `run_state.json`. Optionally repair Elo/league from manifests.
- **Set window_size:** Before restore/repair, set `replay_buffer.window_size` from config schedule for iteration `last_comm + 1`.
- **Resume outcome:** `start_iteration = last_comm + 1`, `resume_checkpoint = best_model_path or latest_model_path or latest_checkpoint`. If that path exists, log auto-resume banner and return `(start_iteration, resume_checkpoint)`.

---

## 2.4 Replay buffer restore

- **restore_state():** If `replay_state.json` (or equivalent) exists, load list of `{iteration, path, positions}`. For each entry, if `path` exists (and is under committed, not staging), add to in-memory entries. Sort by iteration. Enforce current `window_size` by evicting oldest entries. Replace internal _entries so idempotent. Skip paths that don’t exist (log staging paths as “staging discarded”).
- **When to call:** (1) During auto-resume after reconcile, if needs_repair. (2) At train() startup when `start_iteration > 0` (so after _try_auto_resume), call `replay_buffer.restore_state()` so the sliding window is restored for the resumed run.

---

## 2.5 Boundary-only resume (optimizer/EMA/scheduler/scaler)

**Default (warm restart):** Each iteration creates a new Trainer; optimizer, scheduler, scaler, and EMA are created fresh. Only model weights are loaded from previous checkpoint. This avoids LR schedule collapse (global_step >> per-iteration steps → LR stuck at min).

**Optional continuity:** Config `training.resume_from_committed_state: true` means: load optimizer, scheduler, scaler, and EMA from the **committed** checkpoint **only on process restart** (first iteration after auto_resume). Never load them on normal iteration-to-iteration in the same process.

**Guard:** `is_first_iteration_after_resume = (iteration == start_iteration and resume_checkpoint is not None)`.

**Trainer construction for this iteration:**

- `optimizer_state_checkpoint = previous_checkpoint` (same as model source; never from staging).
- `load_committed_state = resume_from_committed and optimizer_state_ckpt is not None and is_first_iteration_after_resume`.
- If `load_committed_state`: set `force_resume_optimizer_state`, `force_resume_scheduler_state`, `force_resume_scaler_state`, `force_resume_ema` to True (override config). Otherwise leave them None so config default (false) applies.
- Trainer __init__: when `force_resume_*` is True, call `_try_load_optimizer_state(checkpoint_path, resume_opt=True, resume_sched=True, resume_scaler=True, resume_ema=True, model_source=previous_checkpoint)` after creating optimizer/scheduler/scaler/EMA. Load from checkpoint: optimizer_state_dict, scheduler_state_dict, scaler_state_dict, ema_state_dict (if present). On success, set `global_step` from checkpoint. Log clearly: [OPT_RESUME] and [BOUNDARY-RESUME] with is_first_iteration_after_resume and load_committed_state.

**Safety asserts (boundary resume):** When `is_first_iteration_after_resume and previous_checkpoint`, resolve path and assert: checkpoint is under `checkpoints/` or `committed/`, and **not** under `staging/`. Boundary resume must never use a staging checkpoint.

---

## 2.6 Checkpoint and monotonicity logging

- Log at iteration start: `[CHECKPOINT] selfplay iter N using weights checkpoint: <path>`. Assert path exists.
- Before training: `[CHECKPOINT] training iter N starting from checkpoint: <path>`. Assert path exists.
- After commit: log committing iteration N → path (e.g. committed/iter_N).
- This makes it easy to verify monotonic progression and that self-play/training never accidentally use an older or staging checkpoint.

---

## 2.7 Run state and commit manifest

- **run_state.json** should include at least: last_committed_iteration, best_model_path, best_iteration, latest_model_path, latest_checkpoint, global_step, consecutive_rejections, config_hash, run_id, seed, timestamp_utc.
- **commit_manifest.json** per committed iter: e.g. iteration, paths to checkpoint and selfplay HDF5, num_positions, optional elo_ratings, commit_method (rename/copy).

---

# PART 3 — BUG FIXES

## 3.1 Valid_7x7 D4 recompute (7x7 fix)

**Bug:** valid_7x7 is a *per-region* feature: each of the 3×3 top-left cells (0,0)..(2,2) is 1 iff the 7×7 region with that top-left is empty. Under D4 symmetry, you transform the board (rotation/reflection). If you only spatially transform the valid_7x7 plane like a normal image, the result is wrong: the meaning of “region (r,c)” changes with the transform. So after any non-identity D4 transform, valid_7x7 channels must be **recomputed** from the transformed occupancy planes.

**Convention:** Channels 0–1 = current/opponent occupancy (9×9, 1=filled, 0=empty). Channels 6–7 = valid_7x7 for current and opponent: shape (9,9) with 1 only at (top, left) in {0,1,2}×{0,1,2}, where the 7×7 with top-left (top, left) is empty.

**Recompute (single sample):**

```python
def _recompute_valid_7x7_from_plane(occ_plane: np.ndarray) -> np.ndarray:
    """occ_plane: (9,9) float, 1=filled 0=empty. Returns (9,9) valid_7x7."""
    out = np.zeros((9, 9), dtype=np.float32)
    for top in range(3):
        for left in range(3):
            if occ_plane[top:top+7, left:left+7].sum() < 0.5:
                out[top, left] = 1.0
    return out
```

**Single-sample D4 transform (state shape e.g. (32,9,9)):**

- Transform channels 0–7 with the same spatial transform (e.g. permute rows/cols).
- If transform index `ti != 0`: after transforming ch 0–7, **recompute** ch 6 and ch 7 from the **transformed** ch 0 and ch 1: `out[6] = _recompute_valid_7x7_from_plane(out[0])`, `out[7] = _recompute_valid_7x7_from_plane(out[1])`. Do not use the spatially transformed old ch 6/7.
- Channels 8–31: apply spatial transform and orient/slot permutations as before.

**Batch D4 transform (states shape (B, 32, 9, 9)):**

- Transform all ch 0–7 with the same spatial transform for the batch.
- If `ti != 0`: set `out[:, 6] = 0`, `out[:, 7] = 0`. Then for each (top, left) in (0,1,2)×(0,1,2):  
  `block_cur = out[:, 0, top:top+7, left:left+7].reshape(B, -1)`  
  `out[:, 6, top, left] = (block_cur.sum(axis=1) < 0.5).astype(np.float32)`  
  and similarly for ch 7 from `out[:, 1, ...]`.
- Then do ch 8–31 with spatial + orient permutes.

**GPU D4 (if you have a GPU augmentation path):**

- After transforming ch 0–7 on GPU, zero ch 6–7 and recompute: for each (top, left), `block_cur = out[:, 0, top:top+7, left:left+7].reshape(B, -1)`; `out[:, 6, top, left] = (block_cur.sum(dim=1) < 0.5).float()`; same for ch 7 from `out[:, 1, ...]`. Doing this even for identity (ti=0) avoids host sync on ti.

**Tests:** Add tests that (1) compute a reference valid_7x7 from an occupancy plane (empty/full and one mixed), and (2) for each D4 transform, transform the board, recompute ch 6–7 from transformed board, and compare to the reference applied to the transformed occupancy. This catches using the wrong plane or forgetting to recompute.

---

## 3.2 Score margin and replay validation

- **Replay buffer / HDF5:** When loading score_margins, validate: finite; `max(abs(score_margins)) <= 120` (or your range); if you migrated from tanh scalar, require integer-like: e.g. `mean(|sc - round(sc)|) <= 1e-3`. On failure, raise with a clear message to wipe staging/committed and regenerate self-play.
- **Shard validation:** When writing or validating self-play shards that have scores, require score_margins finite and within range; if config has score_loss_weight > 0 but data has no scores, fail early.

---

## 3.3 Checkpoint load: incompatible score head

- If the model has a 201-dim score head and the checkpoint has a 1-dim (tanh) score head (or vice versa), do not blindly load: drop the incompatible score head weights, log a warning, and reinit the score head. This avoids silent wrong behavior when moving between old and new score formats.

---

# PART 4 — IMPLEMENTATION CHECKLIST

Use this to port into an older codebase.

**Speed:**  
- [ ] GPU server returns (mean_points, score_utility) and accepts score_center_points; no score logits over IPC.  
- [ ] Eval client: 4-tuple response; score_center_points in all submit paths.  
- [ ] MCTS: GPU path uses scalars only; set node.score_estimate = mean_points; utility = value + score_utility.  
- [ ] Encoder: Numba JIT for x_global/x_track/shop; encode_into in-place for SHM.  
- [ ] SHM: WorkerSharedBuffer per worker; server reads from SHM; workers encode_into slot then submit.  
- [ ] Worker: torch.set_num_threads(1) (or config).  
- [ ] Training: BatchIndexSampler + _BatchIterableDataset; dataset[indices] batched; optional prefetch.  
- [ ] Config: torch_compile false, max_batch_wait_ms 2–3.

**Resume:**  
- [ ] Run layout: staging/iter_N, committed/iter_N, run_state.json; atomic commit (move or copy+fsync).  
- [ ] Startup: always cleanup_staging after max_committed_iteration; then _try_auto_resume.  
- [ ] reconcile_run_state; repair run_state and replay if filesystem ahead of run_state.  
- [ ] Replay: restore_state() from disk when start_iteration > 0 and in repair path; enforce window_size.  
- [ ] Boundary resume: resume_from_committed_state + is_first_iteration_after_resume; force_resume_* only when load_committed_state; assert checkpoint under committed/ or checkpoints/, not staging.  
- [ ] [CHECKPOINT] and [BOUNDARY-RESUME] logs; assert checkpoint paths exist.

**Bugs:**  
- [ ] D4: after non-identity transform, recompute valid_7x7 (ch 6–7) from transformed occupancy planes; single-sample and batch and GPU paths.  
- [ ] Score margin validation on load and in shards; reject non-integer/tanh if using integer margins.  
- [ ] Load checkpoint: drop incompatible score head and reinit with warning.

---

*End of changelog. All content above is self-contained.*
