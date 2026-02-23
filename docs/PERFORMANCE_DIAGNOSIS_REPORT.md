# Patchwork AlphaZero Training Performance Diagnosis Report

**Date:** 2026-02-22  
**Hardware:** RTX 3080 10GB, Ryzen 5900X (12c/24t), 16GB RAM  
**Config:** batch_size=1024, epochs_per_iteration=2, bf16 AMP, AdamW

**UPDATE:** GPU D4 patch applied — D4 moved from CPU DataLoader to GPU after H2D.
See "Results table (before/after)" for dramatic throughput improvement.

---

## 1) Findings (with timings + profiler summary)

### Micro-timing instrumentation (per-step averages over ~35–70 steps)

| Phase | Mean (ms) | Std (ms) |
|-------|-----------|----------|
| **(i) next(data_iter)** [CPU aug/collation] | **2828** | 1459 |
| (ii) H2D transfer | 2.9 | 2.6 |
| (iii) forward+backward+step | 381 | 139 |
| (iv) logging/sync (.item etc) | 0.08 | 0.05 |
| **TOTAL per step** | **3268** | 1538 |

- **Throughput:** 0.31 steps/sec ≈ **317 positions/sec**
- **GPU time / CPU overhead ratio:** 381 / 2828 ≈ **0.13**
- **Conclusion:** GPU compute is much smaller than CPU overhead → **data/sync bound**.

### torch.profiler summary (CPU + CUDA)

- Trace exported to `tools/profile_trace.json`.
- **Top CPU ops by total time:**
  - `_SingleProcessDataLoaderIter.__next__` (DataLoader + Dataset `__getitem__`): **≈2.9 s per step**
  - This includes: `apply_d4_augment_batch`, NumPy indexing, `torch.from_numpy`.
- **CUDA gaps:** Large idle gaps between step boundaries; GPU waits on `next(data_iter)`.
- **Synchronizations:** `.item()` on `grad_norm` every step; `get_loss` returns Python floats (no extra sync). Main sync is implicit: CPU blocks on `next(loader)`.

### Model compute vs overhead

```
GPU_time / CPU_time ≈ 381 / 2828 ≈ 0.13
```

GPU is underutilized because each step is dominated by CPU augmentation.

---

## 2) Root causes (ranked)

| Rank | Bottleneck | Evidence | Impact |
|------|------------|----------|--------|
| **1** | **D4 augmentation in `__getitem__`** | `next(data_iter)` ~2.8 s; D4 (`apply_d4_augment_batch`) runs inside `PatchworkDataset.__getitem__` for dynamic augmentation. | **≈87% of step time** |
| **2** | **Serial data / compute pipeline** | With `num_workers=0`, `next(loader)` blocks the main thread; no overlap with GPU. | Prefetch adds overlap but worker cannot keep up when load time (≈2.8 s) >> compute (≈0.4 s). |
| **3** | **Per-batch NumPy allocation** | `apply_d4_augment_batch` allocates `out_states = np.empty_like(states)` etc. | Contributes to CPU cost. |
| **4** | `grad_norm.item()` per step | Implicit CUDA sync every step. | Small (≈0.08 ms) vs data loading; secondary. |

### Why it “used to be faster”

- **D4 moved into `__getitem__`:** With `d4_augmentation: dynamic`, D4 is applied per batch in the Dataset instead of at storage time (`store`). Each batch now runs `apply_d4_augment_batch`, which adds ~2.5+ s per batch.
- **Larger, multimodal data:** 56-channel spatial states, global/track/shop inputs increase indexing and transform cost.
- **In-memory dataset + num_workers=0:** Comment says workers add pickle overhead; in-memory + single process means no parallelism on data loading.

---

## 3) Patch (code changes)

### A) Background prefetch (`src/training/trainer.py`)

- Added `_prefetch_generator(loader, prefetch_batches=2)` to overlap `next(data_iter)` with GPU compute.
- Enabled only when `hardware.prefetch_batches > 0` and `device.type == "cuda"`.
- **Config:** `configs/config_best.yaml` → `hardware.prefetch_batches: 2`

### B) `validate()` fix (`src/training/trainer.py`)

- `validate()` referenced `own_valid_mask` without defining it.
- Fixed by mirroring train logic: compute `valid_mask`, then `own_valid_mask = valid_mask.to(device)` when ownership is used.

### C) Profiling tool (`tools/profile_training.py`)

- Micro-timing for: `next(data_iter)`, H2D, forward+backward, logging/sync.
- Optional `torch.profiler` with trace export.
- `--prefetch` flag for before/after validation.

---

## 4) Results (before/after)

| Metric | Baseline (no prefetch) | With prefetch |
|--------|------------------------|---------------|
| next(data_iter) mean | 2874 ms | 2178 ms |
| forward+backward mean | 355 ms | 1018 ms* |
| steps/sec | 0.31 | 0.31 |
| positions/sec | 317 | 320 |

\*Forward/backward variance differs run-to-run; prefetch changes timing of when GPU is measured.

**Conclusion:** Prefetch reduces measured `next()` time when batches are ready, but **data loading (≈2.8 s) far exceeds compute (≈0.4 s)**, so a single prefetch worker cannot keep the GPU fed. Throughput stays ~0.3 steps/sec. The prefetch overlay is still beneficial when compute and load times are closer.

---

## 5) Results table (before/after) — GPU D4 patch

| Metric | Before (CPU D4 in __getitem__) | After (GPU D4 after H2D) |
|--------|--------------------------------|---------------------------|
| next(data_iter) | **2828 ms** | **17 ms** |
| H2D transfer | 2.9 ms | 2.4 ms |
| forward+backward+step | 381 ms | 168 ms |
| Total per step | **3268 ms** | **187 ms** |
| steps/sec | **0.31** | **5.34** |
| positions/sec | **317** | **5464** |
| GPU/CPU ratio | 0.13 (data-bound) | 9.97 (compute-bound) |

---

## 6) Next steps (optional further improvements)

1. ~~**Move D4 to GPU (highest impact)**~~ **DONE**
   - Implement D4 transforms as PyTorch ops (e.g., `torch.rot90`, `torch.flip`, index permutations).  
   - Apply after `.to(device)` in the training loop instead of in `__getitem__`.  
   - Removes ~2.5 s CPU cost per batch and allows overlap with the next batch load.

2. **num_workers > 0 with shared/preloaded data**  
   - Experiment with `num_workers=4–8` so multiple workers prepare batches in parallel.  
   - Consider shared memory / memory-mapped arrays to avoid full dataset copy per worker (e.g., on Linux).

3. **Convert to torch at init (Option 2)**  
   - Store tensors in RAM at init instead of NumPy.  
   - Reduces `torch.from_numpy` per batch; D4 would need to be ported to tensors.

4. **Numba/Cython for D4**  
   - Compile hot paths in `apply_d4_augment_batch` and related functions to cut CPU time.

---

## How to re-run profiling

```bash
# Micro-timing only
python tools/profile_training.py --config configs/config_best.yaml --synthetic --steps 200

# With prefetch
python tools/profile_training.py --config configs/config_best.yaml --synthetic --steps 200 --prefetch

# Full profiler trace
python tools/profile_training.py --config configs/config_best.yaml --synthetic --steps 50 --profiler
# Trace written to tools/profile_trace.json (open in chrome://tracing)
```

---

## 7) GPU D4 robustness (post-patch hardening)

### Why this fix works
- **Before:** D4 ran in `Dataset.__getitem__` on CPU (NumPy). Each `next(loader)` blocked ~2.8 s.
- **After:** Dataset returns canonical data + `slot_piece_ids`; D4 runs on GPU after H2D transfer.
- **Effect:** `next(data_iter)` ~17 ms; GPU compute dominates; ~17× speedup.

### Config controls
| Key | Default | Description |
|-----|---------|-------------|
| `training.d4_on_gpu` | `false` | Apply D4 on GPU after H2D. Set `true` for production (CUDA). |
| `training.d4_gpu_deterministic_test_mode` | `false` | Use fixed seed for D4 transform indices (reproducibility). For tests only. |

### Fallback behavior
- **CUDA unavailable or `d4_on_gpu: false`:** D4 is applied on CPU (in Dataset when `d4_on_gpu=false`, or in trainer loop when device is CPU).
- **No `slot_piece_ids` in data:** D4 is skipped (dataset may have pre-augmented).

### How to profile
```bash
# Synthetic (in-memory)
python tools/profile_training.py --config configs/config_best.yaml --synthetic --steps 200

# Real data (iter000)
python tools/profile_training.py --config configs/config_best.yaml --data path/to/merged_training.h5 --steps 200
```

### Known caveats
- `apply_d4_augment_batch_gpu` requires CUDA; falls back to CPU path when `device.type != "cuda"`.
- Batch structure is now dict-based (`batch["states"]`, etc.); use `batch_to_dict()` for tuple compat.
- One `.cpu().numpy()` per batch to decode unique keys; no `.item()` in hot loop (see `tools/check_d4_no_sync.py`).

---

## 8) D4 LUT cache (versioned, self-healing)

### Cache layout
- **Versioned filenames:** `d4_buy_lut_pc33_v2.npy`, `d4_legal_lut_pc33_v2.npy`
- **Metadata:** `d4_lut_meta_pc33_v2.json` (pc_max, lut_version, created_at, shape, git)
- **Old unversioned caches** (`d4_buy_lut.npy`, `d4_legal_lut.npy`) are **ignored**. Do not manually delete; use the clear tool.

### Constants (d4_constants.py)
- **PC_MAX=33:** Supports piece IDs 1–32 and empty. `COMPACT_SIZE = 8 × 33³ = 287496`
- **Encoding:** `p_enc = (slot_id + 1).clamp(0, 33)`; `p0c = (p0_enc - 1).clamp(0, PC_MAX-1)`
- **Clamping:** `p0/p1/p2` use `clamp(-1, PC_MAX-1)` everywhere

### Clear cache
```bash
python tools/clear_d4_cache.py           # Dry-run (list only)
python tools/clear_d4_cache.py --yes    # Delete old unversioned caches
python tools/clear_d4_cache.py --yes --all   # Delete all D4 caches
```

### Force rebuild
Delete the versioned files or bump `LUT_VERSION` in `src/network/d4_lut_cache.py`.

### Profiling commands
```bash
# Synthetic (gold_v2 shapes)
python tools/profile_training.py --config configs/config_best.yaml --synthetic --steps 200

# Real data (merged_training.h5 or iter000)
python tools/profile_training.py --config configs/config_best.yaml --data path/to/merged_training.h5 --steps 200

# Export profiler trace
python tools/profile_training.py --config configs/config_best.yaml --data path/to/merged_training.h5 --steps 50 --profiler
# Trace: tools/profile_trace.json
```

### No-sync CI check
```bash
python tools/check_d4_no_sync.py
```
