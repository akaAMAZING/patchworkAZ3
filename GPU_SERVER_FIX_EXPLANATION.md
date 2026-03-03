# GPU Server Crash Fix (0xC0000409 STATUS_STACK_BUFFER_OVERRUN)

## Why this prevents stack buffer overrun

1. **SHM attach validation (WorkerSharedBuffer)**  
   When the child process or a worker attaches to shared memory with `create=False`, it used to derive `n_slots = size // SLOT_BYTES` with no checks. On Windows, segment size can be page-rounded or wrong, giving `derived == 0` or smaller than the parent’s slot count. Code then built numpy views with `_base(slot) = slot * SLOT_BYTES` and passed them to C/numpy; with `n_slots == 0` or a slot index beyond the real size, that can write past the buffer and trigger **STATUS_STACK_BUFFER_OVERRUN** (0xC0000409) in the runtime.  
   The fix: require `size >= SLOT_BYTES`, `derived >= 1`, and (when provided) `derived >= expected_n_slots`, and raise a clear `ValueError` otherwise. That turns a native crash into a Python exception and guarantees we never create views over an undersized segment.

2. **Explicit expected slot count everywhere**  
   Parent, GPU server, and workers now all use the same **expected_n_slots** (max over `parallel_leaves_schedule`, or config `parallel_leaves`). The parent creates SHM with that size; the server and workers attach with `expected_n_slots` and the new validation. So there is no silent mismatch: either the segment has at least that many slots (and we’re safe) or we get a Python error.

3. **Config consistency**  
   The GPU server process is started with the same `self.config` used to compute `_get_expected_n_slots()` in the parent; `expected_n_slots` is passed explicitly in kwargs so the child does not re-derive it from a different config path.

4. **Ready/error handshake**  
   The child puts `"ready"` only after SHM attach, model load, and warmup. On any exception it puts `("error", repr(e), traceback.format_exc())` so the parent can raise a `RuntimeError` that includes the child’s traceback. That makes initialization failures (including SHM validation) visible as Python errors instead of a silent native exit.

5. **Cleanup on startup failure**  
   If the parent never sees `"ready"` (timeout or error), a `finally` block destroys all SHM segments and clears the buffers dict so the next run does not attach to stale segments.

---

## Files changed

| File | Changes |
|------|--------|
| **src/mcts/shared_state_buffer.py** | Added `expected_n_slots` to `__init__`. When `create=False`: validate `size >= SLOT_BYTES`, `derived >= 1`, and (if `expected_n_slots` is set) `derived >= expected_n_slots`; raise `ValueError` with name, size, SLOT_BYTES, derived, expected. |
| **src/network/gpu_inference_server.py** | `GPUInferenceServer.__init__` takes `expected_n_slots` and passes it when opening each WorkerSharedBuffer; logs after each attach (wid, name, size, SLOT_BYTES, derived, expected_n_slots). `run_gpu_inference_server` takes `expected_n_slots` in kwargs, passes it to the server; on exception puts `("error", repr(e), traceback.format_exc())` on `ready_q` (no longer uses a string `"error:..."` or `finally`). |
| **src/training/selfplay_optimized_integration.py** | Added `_get_expected_n_slots()` (max over schedule or config). `_start_gpu_server`: create SHM with `expected_n_slots`, pass it to the child in kwargs; wait for `ready` or tuple `("error", msg, tb)` and raise with child traceback; on failure run `finally` to destroy SHM and clear dicts; set `self._expected_n_slots` only on success; clear it in `_stop_gpu_server`. Pool initargs extended with `self._expected_n_slots` for both `_run_game_batch` and `_generate_parallel`. |
| **src/training/selfplay_optimized.py** | `init_optimized_worker` takes `expected_n_slots` and passes it to `WorkerSharedBuffer(..., expected_n_slots=expected_n_slots)` when attaching. |

---

## Applying the patch

```bash
git apply GPU_SERVER_FIX.patch
```

Or apply the changes by hand from the diff.

---

## Quick stability test

After applying:

- Run with **12 workers** for **15+ iterations** (e.g. `python -m src.training.main --config configs/config_best.yaml`), including multiple process restarts (stop and start again).
- Confirm: no early GPU server exit with 0xC0000409.
- If there is an SHM mismatch (e.g. wrong size or name), you should get a **Python `ValueError` or `RuntimeError`** with a readable message and (for child errors) the child traceback, instead of a native crash.

---

## Optional log line

The GPU server already logs after each SHM attach:

```text
[GPU Server] SHM attach wid=%d name=%s size=%d SLOT_BYTES=%d derived=%d expected_n_slots=%s
```

Use this to confirm sizes and slot counts match across runs.
