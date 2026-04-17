# Windows CUDA Startup Audit

**Context:** Parent times out waiting 180s for GPU server ready. Faulting module: `nvcuda64.dll`, exception `0xc0000409` (STATUS_STACK_BUFFER_OVERRUN). Crash is in the spawned GPU child before it signals ready.

---

## 1. Step-by-step startup timeline (main → ready_q.put)

| Step | Location | Action |
|------|----------|--------|
| 1 | `src/training/main.py` | `if __name__ == "__main__": main()` — no `freeze_support()` |
| 2 | `main.py` | `AlphaZeroTrainer(...)` then `trainer.train(...)` |
| 3 | `main.py` | `train()` → `_generate_selfplay_data(iteration, network_path, ...)` |
| 4 | `main.py` | `_generate_selfplay_data` → `self.selfplay_generator.generate(iteration, network_path, ...)` |
| 5 | `src/training/selfplay_optimized_integration.py` | `generate()`: computes `num_workers`, applies schedules |
| 6 | `selfplay_optimized_integration.py:321` | `use_gpu_server = self._should_use_gpu_server(network_path)` |
| 7 | `selfplay_optimized_integration.py:129` | **Parent** calls `torch.cuda.is_available()` (CUDA touch in parent) |
| 8 | `selfplay_optimized_integration.py:330` | `_start_gpu_server(network_path, num_workers)` |
| 9 | `selfplay_optimized_integration.py:151-157` | Create SHM buffers, `ctx = mp.get_context("spawn")`, create `req_q`, `resp_qs`, `stop_evt`, `ready_q` |
| 10 | `selfplay_optimized_integration.py:185-195` | `ctx.Process(target=run_gpu_inference_server, args=(..., ready_q), ...)`, `gpu_process.start()` |
| 11 | **Child process** | New interpreter; imports needed for `run_gpu_inference_server` |
| 12 | **Child** | `src/network/gpu_inference_server.py` loaded → **module-level `import torch`** (line 36) |
| 13 | **Child** | `run_gpu_inference_server()` body: signal ignore, logging, then `server = GPUInferenceServer(...)` |
| 14 | **Child** | `GPUInferenceServer.__init__`: `torch.device(device)`, `_apply_determinism_if_requested` (cudnn), `create_network(config).to(device)`, `torch.load(..., map_location=self.device)`, load checkpoint, `model.eval()`, backends (cudnn.benchmark etc.), optional `torch.compile`, `_score_bins_t` on device, WorkerSharedBuffer attach |
| 15 | **Child** | `server._warmup_inference()`: dummy forward, `torch.cuda.synchronize()` |
| 16 | **Child** | `ready_q.put("ready")` |
| 17 | **Parent** | `ready_q.get(timeout=gpu_ready_timeout)` — if child crashes before step 16, parent blocks until timeout (180s default) |

---

## 2. Table: Every CUDA touch

| File | Line(s) | Function / scope | Parent vs child | Risk on Windows |
|------|---------|-------------------|-----------------|------------------|
| `src/training/selfplay_optimized_integration.py` | 27 | Module top level | **Parent** (on import) | **Risk:** Parent imports `torch` when any code path loads integration. |
| `src/training/selfplay_optimized_integration.py` | 129 | `_should_use_gpu_server()` | **Parent** | **Risk:** Parent calls `torch.cuda.is_available()` before spawn; loads CUDA driver in parent. |
| `src/network/gpu_inference_server.py` | 36 | Module top level | **Child** (when child imports this module) | Safe: only in child process. |
| `src/network/gpu_inference_server.py` | 51-59 | `_apply_determinism_if_requested()` | Child (`GPUInferenceServer.__init__`) | Safe. |
| `src/network/gpu_inference_server.py` | 67 | `torch.device(device)` | Child | First device use in child; can trigger driver init. |
| `src/network/gpu_inference_server.py` | 87-90 | `create_network(...).to(device)`, `torch.load(..., map_location=self.device)` | Child | **High:** First large GPU alloc and kernel path; common crash point. |
| `src/network/gpu_inference_server.py` | 96-101 | `torch.backends.cuda/cudnn` | Child | Can touch driver. |
| `src/network/gpu_inference_server.py` | 103-108 | `torch.compile(...)` | Child | Heavy; can trigger extra CUDA init. |
| `src/network/gpu_inference_server.py` | 125-127 | `torch.arange(..., device=self.device)` | Child | GPU alloc. |
| `src/network/gpu_inference_server.py` | 170-194 | `_warmup_inference()` | Child | **High:** First real forward + `torch.cuda.synchronize()`; cuDNN benchmark. |
| `src/training/selfplay_optimized.py` | 33 | Module top level | Parent & pool workers | Parent imports this; workers (when backend=server) use CPU only; still `import torch` in parent. |
| `src/training/main.py` | 142 | Module top level | Parent | Parent imports `torch` for trainer/eval. |
| `src/training/main.py` | 824-827 | `torch.cuda.get_device_name(0)` etc. | Parent | Parent touches CUDA when building trainer device. |

---

## 3. Does the parent touch CUDA before spawn?

**Answer: Yes.**

**Proof:**

- `generate()` is called from `_generate_selfplay_data()` (parent).
- Before `_start_gpu_server()` is called, `_should_use_gpu_server(network_path)` is invoked at `selfplay_optimized_integration.py:321`.
- `_should_use_gpu_server()` at line 129 calls `torch.cuda.is_available()`.
- So the parent executes a CUDA API call before spawning the GPU server process.
- Additionally, the parent has already imported `torch` (integration line 27, and main.py line 142). When `main` creates `AlphaZeroTrainer`, device setup at `main.py:824-827` uses `torch.cuda.get_device_name(0)` and related calls if `device == "cuda"`, so the parent may touch CUDA even earlier (during trainer init, before first `generate()`).

---

## 4. Can import-time side effects touch CUDA?

**Answer: Yes (indirectly).**

**Proof:**

- **`import torch`:** At import time PyTorch typically does not call into the CUDA driver; the first `torch.cuda.*` or tensor-on-CUDA use usually triggers driver load. So plain `import torch` in a module is low risk at import, but any later use in that process touches CUDA.
- **Modules that import torch at top level and are loaded by the parent before spawn:**  
  `src/training/selfplay_optimized_integration.py` (line 27), `src/training/main.py` (line 142), `src/training/selfplay_optimized.py` (line 33). So the parent process has torch loaded before spawn.
- **Child:** When the child runs `run_gpu_inference_server`, it imports `gpu_inference_server`, which has `import torch` at line 36. That import itself does not run CUDA code; the first CUDA touch in the child is inside `GPUInferenceServer.__init__` (e.g. `torch.device(device)` or later `.to(self.device)` / `torch.load(..., map_location=self.device)`).
- **Conclusion:** No proven import-time *execution* of CUDA code (e.g. no module-level `torch.cuda.is_available()`). But import-time *loading* of `torch` in both parent and child is present; the parent then explicitly calls `torch.cuda.is_available()` before spawn.

---

## 5. Why the parent only sees a timeout

**Mechanism:**

1. Parent starts the GPU server process and then blocks on `ready_q.get(timeout=gpu_ready_timeout)` (180s default) in `selfplay_optimized_integration.py:204`.
2. There is no separate thread or `multiprocessing` primitive that notifies the parent when the child process exits. The parent only learns something from the child if the child puts a message on `ready_q` (`"ready"` or `("error", ...)`).
3. If the child crashes with a **native exception** (e.g. in `nvcuda64.dll`, exception `0xc0000409`), the process is terminated by the OS. The Python `except` block in `run_gpu_inference_server` (lines 498-506) never runs, so the child never calls `ready_q.put(("error", ...))`.
4. Therefore the parent never receives any item on `ready_q` and remains blocked in `ready_q.get(timeout=180)` until the timeout expires.
5. After timeout, the code checks `self.gpu_process.is_alive()` and, if the child is dead, terminates/joins and raises with a message that can include `exitcode` (lines 205-216). So the parent can eventually report "timeout" and optionally "Child exit code: ...", but only after the full 180s, and there is no explicit "child crashed before ready" path that short-circuits the wait.

---

## 6. Top 5 likely triggers for the nvcuda64.dll crash

1. **First GPU memory allocation or first kernel launch in the child (model load / .to(device))**  
   **File:** `src/network/gpu_inference_server.py`  
   **Lines:** 87-90 (`create_network(config).to(self.device)`, `torch.load(checkpoint_path, map_location=self.device)`, `load_model_checkpoint`).  
   **Reason:** Driver and runtime are stressed here; stack buffer overrun in the driver can surface during alloc or first kernel. **Confidence: High.**

2. **cuDNN benchmark / first forward in warmup**  
   **File:** `src/network/gpu_inference_server.py`  
   **Lines:** 182-194 (`_warmup_inference`: forward pass, `torch.cuda.synchronize()`).  
   **Reason:** With `cudnn.benchmark = True`, first conv runs can trigger heavy driver/cuDNN paths. **Confidence: High.**

3. **torch.device("cuda") or early backend settings**  
   **File:** `src/network/gpu_inference_server.py`  
   **Lines:** 67, 96-101 (`torch.device(device)`, `torch.backends.cuda.matmul.allow_tf32`, `torch.backends.cudnn.benchmark`).  
   **Reason:** First backend/cuda context setup can load and initialize the driver. **Confidence: Medium.**

4. **Parent having already initialized CUDA**  
   **File:** `src/training/selfplay_optimized_integration.py` line 129; `main.py` device setup.  
   **Reason:** On some Windows/driver combinations, parent loading the driver before spawn could affect or conflict with child init. **Confidence: Low–Medium.**

5. **torch.compile (if enabled)**  
   **File:** `src/network/gpu_inference_server.py` lines 103-108.  
   **Reason:** Extra compilation and kernel paths could hit driver bugs. **Confidence: Low** (only if enabled in config).

---

## 7. Conclusion

- **Top suspect file:** `src/network/gpu_inference_server.py`
- **Top suspect line range:** **86-90** (model creation, `torch.load(..., map_location=self.device)`, load into model) and **152-194** (warmup forward and `torch.cuda.synchronize()`).
- **Confidence:** **High** that the crash occurs in the child during one of: first GPU alloc (model/checkpoint load), first backend/cudnn setup, or first inference (warmup). **Uncertain** whether parent’s prior CUDA touch contributes; recommended to defer parent CUDA use and add instrumentation to confirm.

---

## 8. Windows multiprocessing notes

- **`if __name__ == "__main__"`:** Used in `main.py:2585`; entry is guarded.
- **`freeze_support()`:** Not called in `main.py`. Recommended for frozen executables on Windows; for normal Python runs optional but safe to add.
- **Spawn:** `ctx = mp.get_context("spawn")` is used in `selfplay_optimized_integration.py` for the GPU process and for pools (lines 154, 385, 442); correct for Windows.
- **Daemon:** GPU process is created with `daemon=False` (line 194); appropriate so the child can outlive the parent if needed.
- **Queues/pipes:** Queues and Event are created from the same `ctx` and passed to the child; usage is consistent with spawn.

---

## 9. Error handling gap

- Parent only waits on `ready_q.get(timeout=...)`. There is no **concurrent check** that the child process has exited (e.g. `gpu_process.join(timeout=0)` or `is_alive()` in a loop or thread) before the 180s expire. So when the child dies immediately, the parent still waits the full timeout. Adding a "poll child liveness while waiting" path would allow a message like "child exited with code X before signalling ready" without waiting 180s.

---

## 10. Minimal patch summary (applied)

- **Parent no longer touches CUDA before spawn:** `_should_use_gpu_server()` now uses config only (`hardware.device == "cuda"` and `num_workers > 1`); removed `torch.cuda.is_available()` and removed `import torch` from `selfplay_optimized_integration.py`.
- **Child startup breadcrumbs:** In `gpu_inference_server.py`, `_breadcrumb()` emits lines like `[GPU Server] BREADCRUMB: N_name`. Order: 1_child_entry, 2_imports_complete, 4_first_cuda_query_begin, 5_first_cuda_query_complete, 6_model_load_begin, 7_model_load_complete, 8_warmup_begin, 9_warmup_complete, 10_ready_signal_sent. Optional log file via env `GPU_SERVER_BREADCRUMB_LOG`.
- **Parent detects child death before timeout:** While waiting on `ready_q`, parent polls `gpu_process.is_alive()` every 1s; if child exits, raises immediately with "GPU server child process exited before signalling ready" and exit code (no 180s wait).
- **Explicit error message:** When child dies before ready, parent now says "child process exited before signalling ready" and suggests checking breadcrumb log / stdout.
- **Windows multiprocessing:** `main.py` now calls `multiprocessing.freeze_support()` under `if __name__ == "__main__"`.
- **Repro helper:** `scripts/windows_cuda_probe.py` — optional `--cuda-query` and `--model-load CONFIG CHECKPOINT`, logs progress to stdout and optional `--log FILE`, exit nonzero on failure.

---

## 11. Commands to validate on Windows

```batch
cd c:\Users\Shanks\Desktop\Codes\patchworkaz - Copy - v2

python scripts/windows_cuda_probe.py
python scripts/windows_cuda_probe.py --cuda-query
python scripts/windows_cuda_probe.py --cuda-query --model-load configs/config_best.yaml runs\YOUR_RUN\committed\iter_001\best_model_iter001.pt
python scripts/windows_cuda_probe.py --log probe.log --cuda-query --model-load configs/config_best.yaml PATH_TO_ANY_CHECKPOINT.pt

set GPU_SERVER_BREADCRUMB_LOG=gpu_breadcrumb.log
python -m src.training.main --config configs/config_best.yaml --iterations 1
```

---

## Repeated-cycle failure analysis

**New fact:** Training runs successfully for 5–6 cycles, then the GPU child crashes in `nvcuda64.dll`. This section focuses on **repeated lifecycle** bugs, not first-start only.

---

### A. Is the GPU server long-lived or recreated?

**Answer: Recreated every cycle.**

**Evidence:**

- `src/training/main.py`: The iteration loop is `for iteration in range(start_iteration, total_iterations):` (line 1268). Each iteration calls `_generate_selfplay_data(iteration, selfplay_model, ...)` (line 1392), which calls `self.selfplay_generator.generate(...)`.
- `src/training/selfplay_optimized_integration.py`: In `generate()` (lines 348–351), when `use_gpu_server` is true we call `_start_gpu_server(network_path, num_workers)` at the **start** of the call. In the `finally` block (lines 388–391), we call `_stop_gpu_server()` and set `self.use_gpu_server = False`. So **every** call to `generate()` (once per iteration) **starts a new GPU server process** and **stops it** in `finally` before returning.
- There is no long-lived GPU server; each iteration gets a **new** `Process`, **new** queues, and **new** SHM buffers.

**Where is old-process cleanup done?**

- **Stopping the child:** In `_stop_gpu_server()` (lines 267–287): we call `self.stop_evt.set()`, then `self.gpu_process.join(timeout=5)`, then if still alive `terminate()` and `join(timeout=2)`. So the **process** is joined/terminated.
- **SHM:** Same method destroys all `_worker_shm_bufs` and clears `_worker_shm_bufs` / `_worker_shm_names` (lines 280–286).
- **Queues and Event:** Not closed or reset. `self.req_q`, `self.resp_qs`, `self.stop_evt` are **never** closed. Next cycle `_start_gpu_server` **overwrites** them with new objects (lines 156–159). The previous cycle’s queues are then only referenced by the dead `self.gpu_process` (which held them as args) until we overwrite `self.gpu_process` next cycle. So **old queues are never explicitly closed** — possible handle/pipe leak on Windows across cycles.
- **Process reference:** After `_stop_gpu_server` we never set `self.gpu_process = None`. Next cycle we overwrite it with the new `Process`, so the old one becomes garbage, but we do not explicitly clear it.

**If reused (N/A):** The server is not reused; models/buffers are not “replaced” in a long-lived process. Each cycle loads a **new** child that does a full init (model load + warmup) then exits when we call `_stop_gpu_server`.

---

### B. Per-cycle GPU / server actions (table)

| File | Line(s) | Function | Action | Repeats every cycle? | Possible leak/risk |
|------|---------|----------|--------|------------------------|--------------------|
| `selfplay_optimized_integration.py` | 156–159 | `_start_gpu_server` | Create new `req_q`, `resp_qs`, `stop_evt`, `ready_q` | Yes | Old queues never closed; pipe/handle leak |
| `selfplay_optimized_integration.py` | 161–174 | `_start_gpu_server` | Create new SHM buffers, `_worker_shm_bufs`, `_worker_shm_names` | Yes | Cleaned on failure in `finally`; on success cleaned in `_stop_gpu_server` |
| `selfplay_optimized_integration.py` | 189–199 | `_start_gpu_server` | `ctx.Process(...)`, `gpu_process.start()` | Yes | Old `gpu_process` reference not set to `None` after stop |
| `selfplay_optimized_integration.py` | 267–276 | `_stop_gpu_server` | `stop_evt.set()`, `join(5)`, optional `terminate()`, `join(2)` | Yes | Child exits; no `req_q`/`resp_qs`/`stop_evt` close |
| `selfplay_optimized_integration.py` | 280–286 | `_stop_gpu_server` | Destroy SHM, clear dicts, `_expected_n_slots = None` | Yes | SHM cleanup correct; `gpu_process` not set to `None` |
| `gpu_inference_server.py` | 107–110 | `GPUInferenceServer.__init__` | `torch.load(..., map_location=self.device)`, `load_model_checkpoint`, `model.eval()` | Yes (in new child) | Each child loads once; no accumulation inside child |
| `gpu_inference_server.py` | 152–199 | `_warmup_inference` | Dummy forward, `torch.cuda.synchronize()` | Yes (once per child) | Heavy cuDNN path once per cycle in new process |
| `main.py` | 1635–1638 | `train()` end-of-iteration | `torch.cuda.synchronize()`, `torch.cuda.empty_cache()`, `time.sleep(3)` | Yes | Intended to reduce driver race; parent keeps CUDA context |
| `main.py` | 1792–1798 | `_train_network` return | `gc.collect()`, `SetProcessWorkingSetSize` (Windows) | Yes | Frees CPU memory before next self-play; no CUDA release |
| `selfplay_optimized_integration.py` | 386–391 | `_generate_parallel` | New `ctx.Pool(...)`, `pool.close()`, `pool.join()` | Yes | Pool closed/joined; standard; no explicit queue close on pool |

---

### C. Missing cleanup

- **Child process:** Joined (and optionally terminated) in `_stop_gpu_server`. **No missing join** for the GPU server process.
- **Queues not closed:** After the child exits, `self.req_q`, `self.resp_qs` (and the local `ready_q`) are **never** closed. On Windows, `multiprocessing.Queue` uses pipes; not calling `close()` (and optionally `cancel_join_thread()`) can leave pipe handles open in the parent. Over 5–6 cycles this can accumulate and contribute to resource pressure or driver issues.
- **Event:** `self.stop_evt` is not “closed” (Event has no close in standard API); it is just overwritten next cycle. No explicit leak from Event itself.
- **CUDA in parent:** Parent (trainer) keeps a CUDA context for the whole run. We do not “release” the context between cycles; we only `empty_cache()` and `sleep(3)` (main.py 1635–1638). So parent and child **never** use CUDA at the same time (child is stopped before training), but the **parent context stays active** while we repeatedly create/destroy **child** contexts. That pattern (one long-lived parent context + repeated short-lived child contexts) may stress the driver after several cycles.
- **Old checkpoint/model references:** In the **child**, the model and checkpoint are local to the process; when the process exits they are freed. In the **parent**, we do not hold references to the child’s model. No obvious accumulation of model refs across cycles.
- **Repeated warmup:** Each new child runs **one** warmup in `run_gpu_inference_server` before `ready_q.put("ready")`. So warmup is once per child (once per cycle). Not “repeated warmup” in the same process; the only repetition is “new process → full init → warmup” every cycle.
- **Worker pool:** A **new** pool is created each `_generate_parallel` (lines 417–426). We call `pool.close()` and `pool.join()` (507, 525). We do not call the pool’s internal queue `close()`; relying on pool lifecycle is standard. Lower priority than closing the **GPU server** queues.

---

### D. Trainer-side CUDA vs GPU server child (overlap?)

**Conclusion: No overlap.** Trainer (parent) does **not** use CUDA while the GPU server child is running.

**Evidence:**

- **Order of operations** in `train()` (main.py 1268–1638): For each iteration we do:
  1. `_generate_selfplay_data(...)` → `generate()` → **starts GPU server**, runs `_generate_parallel` (games), then in **finally** calls `_stop_gpu_server()` and returns. So the **child is stopped before** `_generate_selfplay_data` returns.
  2. `_train_network(...)` — trainer uses `self.device` (CUDA) for training.
  3. Evaluation (evaluator uses CUDA).
  4. End-of-iteration: `torch.cuda.synchronize()`, `empty_cache()`, `sleep(3)`.

So the **only** time the GPU server child is alive is during step 1. Steps 2–4 happen **after** the child has been stopped. Therefore trainer and GPU server **never** use CUDA concurrently. The comment at main.py 1630–1633 (“driver teardown races with a new context init”) refers to the **sequence**: previous child’s context is torn down when that child exits; then parent uses CUDA (train/eval); then we sync/empty_cache/sleep; then **next** iteration we spawn a **new** child (new context init). So the risk is **teardown of child context vs init of next child’s context** (or parent still holding context), not parent and child at the same time.

---

### E. Top 5 “fails after 5–6 cycles” hypotheses (ranked)

1. **Unclosed queues in parent → pipe/handle accumulation (Windows)**  
   **Evidence:** `selfplay_optimized_integration.py` lines 156–159 create new `Queue`/`Event` every `_start_gpu_server`. `_stop_gpu_server` (267–286) never calls `close()` (or `cancel_join_thread()`) on `req_q` or `resp_qs`. Old queues are only dropped when we overwrite `self.req_q`/`self.resp_qs` next cycle. On Windows, unclosed Queue pipes can leak handles; after 5–6 cycles the system or driver may misbehave. **Confidence: High.**

2. **Parent CUDA context long-lived + repeated child context create/destroy**  
   **Evidence:** main.py 1635–1638: comment explicitly states that without sync/empty_cache/sleep(3), “driver teardown races with a new context init” and “nvcuda64.dll can crash (STATUS_STACK_BUFFER_OVERRUN, 0xc0000409)”. So the driver is already known to be sensitive to this pattern. Parent never releases its CUDA context; each cycle we destroy one child context and create another. After several cycles the driver may hit a bug or internal leak. **Confidence: High.**

3. **`gpu_process` and queue references retained across cycles**  
   **Evidence:** After `_stop_gpu_server` we do not set `self.gpu_process = None` (selfplay_optimized_integration.py 267–286). The old `Process` object holds the old queues in its `args`. So cleanup of those queues is deferred to GC. If GC does not run promptly, we temporarily hold multiple Process + Queue sets. **Confidence: Medium.**

4. **Insufficient delay between child exit and next child start**  
   **Evidence:** We sync/empty_cache/sleep(3) at **end** of iteration (after train/eval). The **next** iteration then starts the new child. So the delay is “train + eval + 3s”. If train+eval is short, the 3s might be enough; if the driver needs more time to fully release the previous child’s resources, 3s might not be enough on some systems. **Confidence: Medium.**

5. **SHM or pool worker cleanup race**  
   **Evidence:** We destroy SHM in `_stop_gpu_server` after joining the GPU process; pool is closed/joined in `_generate_parallel`. The GPU server child also attaches to SHM; when the process exits the OS detaches. No evidence of a leak here, but a rare race (e.g. handle reuse) could theoretically contribute. **Confidence: Low.**

---

### F. Smallest patch set for lifecycle safety (summary)

- **Guaranteed child cleanup:** Keep current join/terminate in `_stop_gpu_server`. Add: after join, set `self.gpu_process = None`.
- **Queue/resource cleanup:** In `_stop_gpu_server`, after the process has exited and SHM is destroyed, call `close()` on `self.req_q` and each queue in `self.resp_qs`; then call `cancel_join_thread()` on each to avoid blocking on the feeder thread (Windows). Set `self.req_q = None`, `self.resp_qs = None`, `self.stop_evt = None` so we do not reuse closed objects.
- **Explicit logging of process start PID / exit code per cycle:** In `_start_gpu_server`, log at info level the PID when we start (e.g. “GPU server process started PID=12345”). In `_stop_gpu_server`, after join, log the exit code (e.g. “GPU server stopped PID=12345 exitcode=0”).
- **Per-cycle memory logging (if torch used):** In main.py, after the existing `torch.cuda.synchronize()` / `empty_cache()` / `sleep(3)` block, optionally log `torch.cuda.memory_allocated(0)` and `torch.cuda.memory_reserved(0)` when `torch.cuda.is_available()` so we can spot parent-side growth across cycles.
- **Optional `torch.cuda.empty_cache()`:** Already present at main.py 1636; keep it. No need to add it in the integration module (parent no longer uses torch there).
- **Repeated warmup:** Each child does one warmup; no change. We do not add “skip warmup” for a long-lived server because the server is not long-lived.

Implementing the patch and helper script next.

---

## Parent-child CUDA lifecycle interaction audit

**Focus:** Could the parent’s CUDA usage or context lifetime interact badly with repeated child startup/shutdown on Windows?

---

### 1. Parent CUDA timeline

| File | Line(s) | Function | Touch | First vs repeated | Keeps CUDA alive across cycles? |
|------|---------|----------|-------|-------------------|----------------------------------|
| `src/training/main.py` | 824-827 | `AlphaZeroTrainer.__init__` | `torch.cuda.is_available()`, `torch.device("cuda")`, `torch.cuda.get_device_name(0)`, `torch.cuda.get_device_properties(0).total_memory` | **First** | Yes — creates/initializes parent CUDA context; no teardown anywhere in process lifetime. |
| `src/training/main.py` | 974-996 | `_snapshot_run_metadata` | `torch.cuda.is_available()`, `torch.cuda.device_count()`, `torch.cuda.get_device_name(0)`, `torch.backends.cudnn.version()` / `is_available()` | Repeated (once at init) | Context already alive. |
| `src/training/main.py` | 1634-1646 | `train()` end-of-iteration | `torch.cuda.synchronize()`, `torch.cuda.empty_cache()`, `torch.cuda.memory_allocated(0)`, `torch.cuda.memory_reserved(0)` | Repeated (every cycle) | Context stays alive; empty_cache does not destroy context. |
| `src/training/main.py` | 1787 | `_train_network` | Passes `self.device` to `train_iteration()` | Repeated (every cycle) | Trainer uses parent device (CUDA); context already alive. |
| `src/training/trainer.py` | 171-172, 185-186 | checkpoint save/load | `torch.cuda.get_rng_state_all()`, `torch.cuda.set_rng_state_all()` | Repeated when saving/loading checkpoint | Context already alive. |
| `src/training/evaluation.py` | 353, 356, 387, 390, 442-443, 482, 485, 497, 500, 636-637 | `Evaluator.evaluate_vs_baseline`, `_load_models`, etc. | `torch.load(..., map_location=self.device)`, `model.to(self.device)`, `torch.cuda.empty_cache()` | Repeated (every eval) | Parent context already alive; eval loads models to parent device. |

**When does the parent first create a CUDA context?**  
In `AlphaZeroTrainer.__init__` at **main.py 824-827**, when `hardware.device == "cuda"` and `torch.cuda.is_available()`. The first of those calls that actually touches the driver (typically `torch.device("cuda")` or `get_device_name(0)`) initializes the context.

**Does the parent context remain alive across all training cycles?**  
**Yes.** There is no code path that calls any “destroy context” or “reset CUDA” API. The parent process runs until exit; the context is created once in `__init__` and never torn down.

**Does the parent keep any model, tensor, stream, or allocator state alive between cycles?**  
- **Trainer:** `train_iteration()` in trainer.py receives `device` and builds/loads a model per call; the model is local to that call. After `_train_network()` returns, the trainer’s model can be GC’d; no persistent GPU model is stored on the trainer object across iterations.  
- **Evaluator:** Holds `self.device` (a torch.device). It does **not** hold a persistent model; it loads models per eval and uses them in the eval method. So no long-lived GPU model in the evaluator across cycles.  
- **Streams/allocator:** PyTorch does not expose explicit “release context” in normal use; the allocator and default stream remain associated with the process. So effectively **yes**: the parent keeps the CUDA context (and its default stream/allocator state) alive between cycles.

---

### 2. Child CUDA timeline

| File | Line(s) | Function | Touch | First vs repeated |
|------|---------|----------|-------|-------------------|
| `src/network/gpu_inference_server.py` | 36 | Module load | `import torch` | First (no driver touch until use). |
| `src/network/gpu_inference_server.py` | 84-85 | `GPUInferenceServer.__init__` | `_breadcrumb("4_first_cuda_query_begin")`, `torch.device(device)` | **First** child CUDA touch (creates child context). |
| `src/network/gpu_inference_server.py` | 99 | `_apply_determinism_if_requested` | `torch.backends.cudnn.deterministic`, `benchmark = False` | First (same init). |
| `src/network/gpu_inference_server.py` | 106-109 | `GPUInferenceServer.__init__` | `create_network(...).to(self.device)`, `torch.load(..., map_location=self.device)`, `load_model_checkpoint`, `model.eval()` | First (model load to GPU). |
| `src/network/gpu_inference_server.py` | 115-119 | `GPUInferenceServer.__init__` | `torch.backends.cuda.matmul.allow_tf32`, `cudnn.allow_tf32`, `cudnn.benchmark` | First. |
| `src/network/gpu_inference_server.py` | 124-127 | `GPUInferenceServer.__init__` | `torch.arange(..., device=self.device)` | First. |
| `src/network/gpu_inference_server.py` | 182-213 | `_warmup_inference` | Dummy forward, `torch.cuda.synchronize()` | First (warmup once per child). |
| `src/network/gpu_inference_server.py` | serve() loop | Various | Tensors to device, inference | Repeated until stop_evt. |

**When does the child first touch CUDA?**  
In the child process, when `run_gpu_inference_server` runs and constructs `GPUInferenceServer`: the first real touch is **gpu_inference_server.py line 84** `torch.device(device)` (with `device="cuda"`), which initializes the child’s CUDA context.

**Does each cycle create a fresh child CUDA context?**  
**Yes.** Each cycle starts a **new** process via `ctx.Process(target=run_gpu_inference_server, ...)`. That process has its own address space and its own PyTorch CUDA context. When the child exits (after `_stop_gpu_server` join/terminate), the process and its context are destroyed. The next cycle spawns a new process, which creates a **new** context. So: one new process per cycle ⇒ one new child CUDA context per cycle.

**Does child teardown explicitly destroy GPU resources?**  
No. The code does not call `torch.cuda.empty_cache()` or any “close context” in the child before process exit. Teardown is **process exit only**: the OS tears down the process and the driver releases the child’s context. So we rely on process exit for full child GPU teardown.

---

### 3. Cycle timeline (one full iteration)

Order of operations for one iteration (e.g. iteration N), with CUDA activity marked:

1. **Parent:** Iteration loop already running; parent CUDA context **already initialized** (from `__init__`).
2. **Parent:** `_generate_selfplay_data(iteration, selfplay_model, ...)` called (main.py 1392).
3. **Parent:** `generate()` → `_start_gpu_server(network_path, num_workers)` (selfplay_optimized_integration.py 350). Parent does **not** use CUDA during this call (no torch in that module).
4. **Child:** Process starts, runs `run_gpu_inference_server`; **child first CUDA touch** in `GPUInferenceServer.__init__` (gpu_inference_server.py 84, 106-109, 115-127, then warmup 182-213). **Child CUDA active.**
5. **Child:** Serves inference requests until `stop_evt` is set. **Child CUDA active.**
6. **Parent:** `_stop_gpu_server()`: `stop_evt.set()`, child exits. **Child process and child CUDA context destroyed.**
7. **Parent:** `_train_network(...)` (main.py 1472): `train_iteration(..., self.device, ...)`. **Parent CUDA active** (training on GPU).
8. **Parent:** `_evaluate_model(...)` (main.py 1512): Evaluator loads models to `self.device`, runs eval. **Parent CUDA active.**
9. **Parent:** End-of-iteration (main.py 1634-1646): `torch.cuda.synchronize()`, `empty_cache()`, `sleep(3)`, memory log. **Parent CUDA still initialized** (context not destroyed).
10. **Next iteration:** Back to step 2; when `use_gpu_server`, step 3 starts a **new** child (new process, **new** child CUDA context). Parent context has **never** been torn down.

**CUDA active in parent:** From first touch in `__init__` (main.py 824-827) until process exit; and actively used during steps 7–9 each cycle.  
**CUDA active in child:** From first touch in `GPUInferenceServer.__init__` (step 4) until process exit (step 6).  
**Overlap:** Parent and child **do not** use CUDA at the same time. The child is stopped (step 6) before parent training (step 7).  
**Before next child start:** When the next iteration starts the next child (step 3), the parent’s CUDA context is **still initialized** and has just been used (train + eval + sync/empty_cache/sleep). There is no “parent context torn down” phase.

---

### 4. Does the parent keep a CUDA context alive while children are repeatedly recreated?

**Answer: Yes.**

**Evidence:**

- Parent creates the context in `AlphaZeroTrainer.__init__` at **main.py 824-827** (when `hardware.device == "cuda"` and `torch.cuda.is_available()`): `self.device = torch.device("cuda")`, `torch.cuda.get_device_name(0)`, `get_device_properties(0)`.
- There is no call in the codebase to `torch.cuda.reset_peak_memory_stats`, `torch.cuda.ipc_collect`, or any API that destroys or “releases” the PyTorch CUDA context in the parent. The parent process runs until `train()` returns and the process exits.
- So from the first iteration through the last, the **parent’s CUDA context is always initialized**.
- Each cycle spawns a **new** child process (selfplay_optimized_integration.py 189-198); that child creates its **own** context (gpu_inference_server.py 84, 106-109, etc.). When the child exits, only the **child’s** context is destroyed (by process exit). The parent’s context is unchanged and still active when the **next** child is started.

So: **one long-lived parent CUDA context + repeated creation and destruction of child processes (each with a fresh child CUDA context)** is the exact pattern in the code.

---

### 5. Top 5 lifecycle-interaction hypotheses (ranked)

1. **Long-lived parent context + repeated child context create/destroy stresses the Windows driver**  
   **Evidence:** main.py 1630-1633 comment: “nvcuda64.dll can crash (STATUS_STACK_BUFFER_OVERRUN, 0xc0000409) if the driver teardown races with a new context init”. Parent never tears down; each cycle we destroy one child context (process exit) and create another (new process). So the driver repeatedly sees “one process with an existing context” and “another process creating then destroying a context”. **Rank: 1.**

2. **Parent CUDA context created too early (before any self-play)**  
   **Evidence:** main.py 824-827 runs in `__init__`, before `train()` or any `generate()`. So the parent context exists before the **first** child is ever started. If the driver or OS is sensitive to “context already exists in process A when process B (child) first initializes CUDA”, this ordering is the one we have. **Rank: 2.**

3. **No explicit child-side cleanup before exit**  
   **Evidence:** Child does not call `torch.cuda.empty_cache()` or synchronize before process exit (gpu_inference_server.py has no teardown). Relying on process exit alone may leave the driver in a state that interacts badly with the still-alive parent context. **Rank: 3.**

4. **Parent allocator state and sleep(3) may be insufficient**  
   **Evidence:** main.py 1635-1638: we call `synchronize()`, `empty_cache()`, then `sleep(3)`. This does not release the parent’s context; it only frees cached allocator memory and waits. If the driver needs “no active CUDA in the parent” before the next child init, this does not provide it. **Rank: 4.**

5. **Metadata/snapshot CUDA calls keep context “warm” at init**  
   **Evidence:** main.py 856 calls `_snapshot_run_metadata()`, which (974-996) calls `torch.cuda.is_available()`, `device_count()`, `get_device_name(0)` (989-991). So immediately after first context creation we touch CUDA again for logging. Minor; unlikely primary cause but contributes to “parent context heavily used from the start”. **Rank: 5.**

---

### 6. Best next patch (reduce parent–child lifecycle risk)

**Recommendation: Add instrumentation first, then one minimal behavioral change.**

- **Instrumentation (do first):**  
  - Log **parent PID** once at trainer init (or at start of `train()`).  
  - Log **child PID** when GPU server starts (already at info: “GPU server process started PID=…”).  
  - Log **iteration** when starting and stopping the GPU server.  
  - Log **whether parent CUDA is initialized** (`torch.cuda.is_initialized()` if available) **immediately before** each `_start_gpu_server` call.  
  - Log **parent allocated/reserved memory** (e.g. `torch.cuda.memory_allocated(0)`, `memory_reserved(0)`) **before** each child start and **after** each child stop (after `_stop_gpu_server` returns).  
  - Child first touch is already covered by existing breadcrumbs (e.g. “4_first_cuda_query_begin”); child exit code already logged in `_stop_gpu_server`.  

- **Smallest high-value behavioral change to test:**  
  **Defer parent CUDA initialization until after the first self-play phase (or until first train).**  
  - Today: parent touches CUDA in `AlphaZeroTrainer.__init__` (main.py 824-827) and in `_snapshot_run_metadata` (989-991), so the context exists before any child.  
  - Change: do **not** call `torch.device("cuda")` / `get_device_name` / `get_device_properties` in `__init__` when `device == "cuda"`. Instead, set `self.device = None` and resolve it to `torch.device("cuda")` (or `"cpu"`) on **first use** (e.g. at the start of `_train_network` or the first time `self.device` is needed for eval). Ensure `_snapshot_run_metadata` and any other early use of `self.device` run only after device is resolved, or guard them so they don’t touch CUDA until after the first `generate()` completes.  
  - Effect: the **first** GPU server child would start and stop **before** the parent has ever created a CUDA context. Subsequent cycles would still have “parent context alive before next child”, but the first cycle would test “child-only CUDA” and we could see if the crash moves to a later cycle or disappears for the first few.  
  - Why highest-value next test: it directly tests the “parent context exists before child” hypothesis with minimal change (lazy device init) and is easy to revert.

---

### 7. Instrumentation checklist (exact logs added)

- [x] Parent PID: in `[LIFECYCLE] first parent CUDA touch parent_pid=…` (main.py __init__) and in `before_child_start` / `after_child_stop` (main.py train loop).  
- [x] Child PID: `[LIFECYCLE] GPU server process started … child_pid=…` (selfplay_optimized_integration.py _start_gpu_server).  
- [x] Iteration: in `[LIFECYCLE] starting GPU server for iteration %s`, `before_child_start iter=…`, `after_child_stop iter=…`, and in start/stop logs.  
- [x] First parent CUDA touch: `[LIFECYCLE] first parent CUDA touch parent_pid=…` (main.py 824-830).  
- [x] Child first CUDA touch: breadcrumb “4_first_cuda_query_begin” (gpu_inference_server.py).  
- [x] Before each child start: `[LIFECYCLE] before_child_start … cuda_initialized=… alloc_mb=… reserved_mb=…` (main.py before _generate_selfplay_data).  
- [x] After each child stop: `[LIFECYCLE] after_child_stop … alloc_mb=… reserved_mb=…` (main.py after _generate_selfplay_data returns).  
- [x] Child exit code: `[LIFECYCLE] GPU server stopped child_pid=… exitcode=…` (selfplay_optimized_integration.py _stop_gpu_server).
