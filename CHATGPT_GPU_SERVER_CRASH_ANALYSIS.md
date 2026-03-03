# Comprehensive Analysis Package: GPU Server Process Crash (Exit 3221226505)

**Purpose:** Give this document (and optionally the referenced code snippets) to ChatGPT or another analyst. You can paste this entire markdown file into ChatGPT; if helpful, also paste the full contents of the three key files from both codebases (paths in §7) so it can diff or analyze line-by-line. The **BROKEN** codebase crashes when starting the GPU inference server subprocess on Windows. The **WORKING** reference implementation (`checkme/`) has run 36+ iterations with the same config on the same machine without this crash. Your task: identify every material difference, hypothesize root cause(s), and propose a concrete fix so the crash does not happen.

---

## 1. The Error

### 1.1 Observed failure

- **When:** During self-play phase, at the start of an iteration (e.g. iter 5 or 6). Step: `[1/3] SELF-PLAY (generator: iter005)`.
- **What:** The **GPU inference server** is started as a separate process via `multiprocessing.Process(target=run_gpu_inference_server, ...)`. The parent waits on a `ready_q.get(timeout=...)` for the child to put `"ready"`. The child **never** puts `"ready"`; it **exits** before doing so.
- **Exit code:** `3221226505` (decimal). On Windows this is `0xC0000409` = **STATUS_STACK_BUFFER_OVERRUN** — a native (non-Python) crash, typically in a DLL or driver (e.g. CUDA/cuDNN, or the C runtime), not a Python exception.

### 1.2 Parent-side traceback (BROKEN codebase)

```text
Training failed: GPU server process exited before signalling ready (exit code: 3221226505). Check for CUDA/GPU errors, out-of-memory, or missing checkpoint. Run with a single worker to avoid GPU server (selfplay.num_workers=1) as a workaround.

Traceback (most recent call last):
  File "...\src\training\selfplay_optimized_integration.py", line 186, in _start_gpu_server
    status = ready_q.get(timeout=gpu_ready_timeout)
  File "...\lib\multiprocessing\queues.py", line 114, in get
    raise Empty
_queue.Empty

During handling of the above exception:
  ...
  File "...\src\training\main.py", line 1500, in _generate_selfplay_data
    return self.selfplay_generator.generate(iteration, network_path, output_dir=output_dir)
  File "...\src\training\selfplay_optimized_integration.py", line 295, in _start_gpu_server
    raise RuntimeError(msg)
RuntimeError: GPU server process exited before signalling ready (exit code: 3221226505). ...
```

So: the child process is **spawned**, then it exits with 0xC0000409 **before** it calls `ready_q.put("ready")`. That implies the crash happens during:

- `run_gpu_inference_server()` entry (imports, signal setup), and/or  
- `GPUInferenceServer(config, checkpoint_path, ..., worker_shm_names=...)` (model load, CUDA init, **opening SharedMemory by name**), and/or  
- `server._warmup_inference()` (first GPU forward pass / cuDNN benchmark).

No Python exception is raised in the child (otherwise the child would put `"error:..."` on `ready_q`); the process is terminated by the OS/native code.

---

## 2. Environment and Setup (Same for Both)

- **OS:** Windows 10 (or 11).  
- **Python:** 3.10.  
- **Multiprocessing:** `mp.get_context("spawn")` is used (Windows default); the GPU server runs in a **spawned** child process.  
- **Config (same YAML):** e.g. `config_best.yaml` — `selfplay.num_workers: 12`, `selfplay.mcts.parallel_leaves: 64`, and an `iteration.parallel_leaves_schedule` that can set per-iteration `parallel_leaves` to 64, 48, or 32.  
- **Hardware:** Single GPU (e.g. RTX 3080), CUDA available.  
- **Observation:** The **WORKING** reference (`checkme/`) has completed **36+ iterations** in one sitting with this same config on the same machine. So the crash is **not** a generic “Windows + CUDA + spawn” impossibility; something in the **BROKEN** implementation differs and triggers the crash.

---

## 3. File-by-File and Code Differences

Below, **WORKING** = reference implementation in `checkme/` (no crash). **BROKEN** = current project (crash with exit 3221226505).

---

### 3.1 `selfplay_optimized_integration.py`

#### 3.1.1 Imports and class state

| Aspect | WORKING (checkme) | BROKEN (current) |
|--------|-------------------|------------------|
| Imports | No `import queue`. No `from src.training.replay_buffer import ...`. | `import queue`. Has `from src.training.replay_buffer import _validate_score_margins, SCORE_MARGIN_MAX_ABS`. |
| Instance attr | Has `self._shm_parallel_leaves: Optional[int] = None`. | No `_shm_parallel_leaves`. |

#### 3.1.2 `_start_gpu_server` signature and SHM sizing

**WORKING:**

```python
def _start_gpu_server(self, network_path: str, num_workers: int, iteration_config: Optional[dict] = None):
    """... iteration_config: Schedule-applied config; if provided, parallel_leaves is taken from it."""
    # ...
    cfg = iteration_config if iteration_config is not None else self.config
    parallel_leaves = int(cfg.get("selfplay", {}).get("mcts", {}).get("parallel_leaves", 32))
    self._worker_shm_bufs = {}
    self._worker_shm_names = {}
    for wid in range(num_workers):
        try:
            buf = WorkerSharedBuffer(n_slots=parallel_leaves, worker_id=wid, create=True)
            # ...
        except Exception as e:
            # ... clear and break
    if self._worker_shm_names:
        self._shm_parallel_leaves = parallel_leaves
    else:
        self._shm_parallel_leaves = None
```

- Uses **current iteration’s** `parallel_leaves` only (from `iteration_config` or base config).  
- **n_slots** = that single value (e.g. 32, 48, or 64 for this iteration).  
- Tracks it in `_shm_parallel_leaves`.

**BROKEN:**

```python
def _start_gpu_server(self, network_path: str, num_workers: int):
    """Start GPU inference server process with per-worker response queues."""
    # ...
    parallel_leaves_base = int(self.config.get("selfplay", {}).get("mcts", {}).get("parallel_leaves", 32))
    pl_schedule = self.config.get("iteration", {}).get("parallel_leaves_schedule", [])
    if pl_schedule:
        max_scheduled = max(int(e.get("parallel_leaves", parallel_leaves_base)) for e in pl_schedule)
        n_slots = max(parallel_leaves_base, max_scheduled)
    else:
        n_slots = parallel_leaves_base
    self._worker_shm_bufs = {}
    self._worker_shm_names = {}
    for wid in range(num_workers):
        try:
            buf = WorkerSharedBuffer(n_slots=n_slots, worker_id=wid, create=True)
            # ...
```

- **No** `iteration_config` parameter.  
- **n_slots** = **max over the whole schedule** (e.g. max(64, 64, 48, 32) = 64 for every iteration).  
- So BROKEN creates **larger or equal** SHM per worker (e.g. always 64 slots vs WORKING’s 32/48/64 depending on iteration).  
- No `_shm_parallel_leaves`.

**Possible impact:** Larger SHM allocation in the parent, or different sizing when the child attaches to the same SHM by name. On Windows, spawn + SharedMemory sizing/alignment could interact badly with driver/DLL behavior and contribute to a stack buffer overrun.

#### 3.1.3 Call site in `generate()`

**WORKING:**

```python
iteration_config = self._apply_iteration_schedules(iteration)
# ...
if use_gpu_server:
    self._start_gpu_server(network_path, num_workers, iteration_config)
    self.use_gpu_server = True
    iter_pl = int(iteration_config.get("selfplay", {}).get("mcts", {}).get("parallel_leaves", 32))
    if self._shm_parallel_leaves is not None and self._shm_parallel_leaves != iter_pl:
        logger.error(...)
        raise RuntimeError("parallel_leaves mismatch: SHM=... vs iteration_config=...")
```

**BROKEN:**

```python
iteration_config = self._apply_iteration_schedules(iteration)
# ...
if use_gpu_server:
    self._start_gpu_server(network_path, num_workers)   # no iteration_config
    self.use_gpu_server = True
```

So BROKEN never passes `iteration_config` and never does the SHM/iteration mismatch check.

#### 3.1.4 Ready-wait and exception handling

**WORKING:**

- Timeout: **fixed 120.0** seconds.  
- Single `except Exception as e:` — does not special-case `queue.Empty`.  
- On timeout or child exit: if process still alive, terminate and join; then `raise RuntimeError(f"GPU server failed to start: {e}")`.

**BROKEN:**

- Timeout: from config `selfplay.gpu_server_ready_timeout_s` (default **180**).  
- Explicit `except queue.Empty:` with different messages for “still alive” (timeout) vs “process dead” (exit code 3221226505 message).  
- Then `except Exception as e:` for other errors.

So BROKEN has **longer** default wait and **more specific** error reporting; unlikely to be the cause of the crash, but noted for completeness.

#### 3.1.5 `_stop_gpu_server`

**WORKING:** Clears `self._shm_parallel_leaves = None` after clearing SHM buffers.  
**BROKEN:** Does not touch `_shm_parallel_leaves` (doesn’t have it).

---

### 3.2 `gpu_inference_server.py`

#### 3.2.1 GPUInferenceServer.__init__ — score utility and MCTS config

**WORKING:** Reads from `mcts_cfg`:

- `score_min`, `score_max` (defaults -100, 100)  
- `score_utility_point_scale` → `_score_scale`  
- `static_score_utility_weight`, `dynamic_score_utility_weight` → `_static_w`, `_dynamic_w`  
- Builds `_score_bins_t` from `_score_min` to `_score_max + 1`.  
- Has `_fallback_count`, `_total_requests`.

**BROKEN:** Uses `ValueHead.SCORE_MIN` / `ValueHead.SCORE_MAX` for defaults and different config keys:

- `score_min`, `score_max` (with ValueHead defaults)  
- `score_utility_scale` → `_score_utility_scale`  
- No separate `_static_w` / `_dynamic_w` in the same way (different score-utility handling elsewhere in serve loop).  
- No `_fallback_count` / `_total_requests` in the same form.

Init order is the same: config → model load → CUDA/cuDNN settings → encoding → score bins → **then open worker SHM**. So the **order** in which SHM is opened (after model load) is the same; only the **size** of the SHM (dictated by parent’s n_slots) and possibly **number/names** of segments differ.

#### 3.2.2 run_gpu_inference_server entry point

**WORKING:**

```python
try:
    logger.debug("Initializing GPU inference server (device=%s)...", device)
    sys.stdout.flush()
    server = GPUInferenceServer(...)
    logger.debug("GPU server initialized successfully, running warmup...")
    sys.stdout.flush()
    server._warmup_inference()
    if ready_q is not None:
        ready_q.put("ready")
    server.serve(...)
except Exception as e:
    logger.error(f"GPU server failed: {e}", exc_info=True)
    sys.stdout.flush()
    if ready_q is not None:
        try:
            ready_q.put(f"error:{str(e)}")
        except:
            pass
    raise
```

- Puts `"ready"` **after** warmup.  
- On **any** Python exception: puts `"error:..."` then re-raises.  
- No `finally`; no `error_status` variable.

**BROKEN:**

```python
error_status = None
try:
    logger.debug("Initializing GPU inference server (device=%s)...", device)
    sys.stdout.flush()
    server = GPUInferenceServer(...)
    logger.debug("GPU server initialized successfully, running warmup...")
    sys.stdout.flush()
    server._warmup_inference()
    if ready_q is not None:
        ready_q.put("ready")
    server.serve(...)
except Exception as e:
    logger.error(...)
    sys.stdout.flush()
    error_status = f"error:{str(e)}"
    raise
finally:
    if ready_q is not None and error_status is not None:
        try:
            ready_q.put(error_status)
        except Exception:
            pass
```

- Same order: init → warmup → put `"ready"`.  
- Uses `finally` to send `error_status` so parent always gets a message on Python exception.  
- For a **native** crash (0xC0000409), neither implementation can run the `except` or `finally`; the process just dies. So this difference does not explain the crash, but it’s a behavioral difference.

---

### 3.3 `shared_state_buffer.py` (WorkerSharedBuffer)

#### 3.3.1 When attaching to existing SHM (`create=False`)

**WORKING:**

```python
self._shm = SharedMemory(create=False, name=name)
if self._shm.size < self.SLOT_BYTES:
    raise ValueError(f"SHM buffer too small: size={self._shm.size} < SLOT_BYTES={self.SLOT_BYTES}")
derived = self._shm.size // self.SLOT_BYTES
if derived < 1:
    raise ValueError(f"SHM buffer has no full slots: derived={derived}")
self.n_slots = n_slots if n_slots is not None else derived
```

- Validates `_shm.size >= SLOT_BYTES` and `derived >= 1`.  
- Documents that on Windows the OS may round size up to a page boundary, so `size % SLOT_BYTES` can be non-zero and only full slots are used.

**BROKEN:**

```python
self._shm = SharedMemory(create=False, name=name)
# Derive n_slots from actual SHM size (caller may pass n_slots for verification)
derived = self._shm.size // self.SLOT_BYTES
self.n_slots = n_slots if n_slots is not None else derived
```

- **No** check that `_shm.size >= SLOT_BYTES` or `derived >= 1`.  
- If Windows returns a size smaller than one slot, or rounded in an unexpected way, `derived` could be 0 and `n_slots` could become 0, leading to later access (e.g. `_base(slot)` or views) that could corrupt stack or trigger undefined behavior in numpy/C.

So: **WORKING** is defensive on attach; **BROKEN** is not.** This could matter in the child process when it opens SHM by name (child runs in a different process, possibly with different view of the same segment).

---

## 4. Summary Table of Differences

| Item | WORKING (checkme) | BROKEN (current) |
|------|-------------------|-------------------|
| SHM slot count | Current iteration’s `parallel_leaves` only (from `iteration_config`) | Max over `parallel_leaves_schedule` (and base) |
| `_start_gpu_server` args | `(network_path, num_workers, iteration_config=None)` | `(network_path, num_workers)` |
| `generate()` calls | `_start_gpu_server(network_path, num_workers, iteration_config)` | `_start_gpu_server(network_path, num_workers)` |
| `_shm_parallel_leaves` | Set when SHM created; cleared in _stop; used in mismatch check | Not present |
| Mismatch check | After starting GPU server, raise if SHM slots != iteration parallel_leaves | None |
| Ready timeout | Fixed 120s | Config `gpu_server_ready_timeout_s` (default 180) |
| ready_q exception handling | Single `except Exception` | Separate `except queue.Empty` then `except Exception` |
| run_gpu_inference_server on error | Put `"error:..."` in except block | Set `error_status`, put in `finally` |
| WorkerSharedBuffer attach | Validates size >= SLOT_BYTES and derived >= 1 | No validation; can set n_slots=0 |
| GPU server score/MCTS config | Different keys and defaults (score_utility_point_scale, etc.) | ValueHead.SCORE_*, score_utility_scale |

---

## 5. Config Relevant to SHM and Workers

From `config_best.yaml` (same file used for both):

```yaml
selfplay:
  num_workers: 12
  mcts:
    parallel_leaves: 64
  # ...
iteration:
  parallel_leaves_schedule:
    - start_iter: 0
      parallel_leaves: 64
    - start_iter: 50
      parallel_leaves: 48
    - start_iter: 150
      parallel_leaves: 32
```

So for early iters (e.g. 5): schedule gives **64**; for later iters 48 or 32.  
- **WORKING:** Creates SHM with 64 slots at iter 5 (same as schedule).  
- **BROKEN:** Creates SHM with max(64,64,48,32)=**64** slots every time. So total SHM size is the same at iter 5; the difference is that BROKEN **never** uses 32 or 48 for SHM size (always 64). So at iter 150+, WORKING would create smaller SHM (32 slots); BROKEN would still create 64. That could mean different allocation patterns or attach behavior on Windows for different runs.

---

## 6. What to Do With This

1. **Compare every difference** above against the actual code paths that run in the **child** process (run_gpu_inference_server → GPUInferenceServer.__init__ → _warmup_inference). Focus on:  
   - **SHM:** size passed from parent (n_slots), and the child’s attach logic (WorkerSharedBuffer create=False).  
   - **Order of operations:** model load vs CUDA init vs SHM attach.  
   - **Any** other code path or dependency that could trigger a stack buffer overrun (e.g. numpy view over invalid size, or C extension).

2. **Hypothesize** why exit code **0xC0000409** appears in BROKEN but not in WORKING. Possibilities to consider (non-exclusive):  
   - Larger or differently sized SHM in BROKEN causing a bad attach or use in the child.  
   - Missing validation in BROKEN’s WorkerSharedBuffer when attaching (e.g. n_slots=0 or derived=0).  
   - Different code paths in GPU server init (score bins, config keys) leading to different allocations or stack usage.  
   - A combination of the above.

3. **Propose a concrete fix** (patch or list of edits) so that:  
   - The BROKEN codebase behaves like WORKING with respect to SHM sizing and iteration_config (per-iteration parallel_leaves, mismatch check).  
   - WorkerSharedBuffer attach path is made safe (validate size and derived slots like WORKING).  
   - Any other change you identify as necessary so the GPU server child process no longer crashes with 0xC0000409.

4. If you can, **suggest a minimal repro** (e.g. “run only _start_gpu_server with iteration_config set to X and num_workers=12”) or a **diagnostic** (e.g. “log SHM size and n_slots in the child before opening”) to confirm the fix.

---

## 7. File Paths (for reference)

- **BROKEN (current project):**
  - `src/training/selfplay_optimized_integration.py`
  - `src/network/gpu_inference_server.py`
  - `src/mcts/shared_state_buffer.py`
- **WORKING (reference):**
  - `checkme/src/training/selfplay_optimized_integration.py`
  - `checkme/src/network/gpu_inference_server.py`
  - `checkme/src/mcts/shared_state_buffer.py`

End of analysis package.
