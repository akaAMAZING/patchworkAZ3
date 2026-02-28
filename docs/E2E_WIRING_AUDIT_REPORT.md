# Patchwork AlphaZero — End-to-End Wiring Audit Report

**Date:** 2025-02-28  
**Scope:** Schedule coverage, selfplay config provenance, replay buffer/window, atomic resume, data/model shapes, smoke test.

---

## 1) Schedule coverage audit

All `iteration.*_schedule` keys in `configs/config_best.yaml` are applied. Mapping:

| YAML key | Where applied | File:function |
|----------|----------------|---------------|
| `iteration.lr_schedule` | Peak LR set before each iteration; scheduler uses it | `main.py`: train loop sets `self.config["training"]["learning_rate"] = iter_lr` (via `_get_lr_phase_info`). `trainer.py`:`_create_scheduler` (cosine_warmup_per_phase) reads `config.get("iteration", {}).get("lr_schedule", [])` and `config["training"]["learning_rate"]`. |
| `iteration.temperature_schedule` | Selfplay MCTS temperature | `selfplay_optimized_integration.py`:`_apply_iteration_schedules` → `config["selfplay"]["mcts"]["temperature"]`. |
| `iteration.mcts_schedule` | Selfplay MCTS simulations | Same; `config["selfplay"]["mcts"]["simulations"]`. Overridden for iter 0 by `bootstrap.mcts_simulations` when `bootstrap.use_pure_mcts` is true. |
| `iteration.parallel_leaves_schedule` | MCTS parallel leaves | Same; `config["selfplay"]["mcts"]["parallel_leaves"]`. |
| `iteration.dirichlet_alpha_schedule` | Root Dirichlet alpha | Same; `config["selfplay"]["mcts"]["root_dirichlet_alpha"]`. |
| `iteration.dynamic_score_utility_weight_schedule` | KataGo dynamic score utility | `main.py`:`_apply_q_value_weight_and_cpuct_schedules` → `config["selfplay"]["mcts"]["dynamic_score_utility_weight"]`. Shared config; selfplay deepcopy receives it. |
| `iteration.q_value_weight_schedule` | Q-value weight for policy target | Same; `config["selfplay"]["q_value_weight"]`. |
| `iteration.cpuct_schedule` | MCTS cpuct | Same; `config["selfplay"]["mcts"]["cpuct"]` (and `evaluation.eval_mcts.cpuct` when `lock_eval_cpuct_to_selfplay`). |
| `iteration.noise_weight_schedule` | Root noise weight | `selfplay_optimized_integration.py`:`_apply_iteration_schedules` → `config["selfplay"]["mcts"]["root_noise_weight"]`. |
| `iteration.games_schedule` | Number of games per iteration | `selfplay_optimized_integration.py`:`_get_num_games` (and `main.py`:`_get_num_games_for_iteration` for provenance). Iter 0 uses `selfplay.bootstrap.games`. |
| `iteration.window_iterations_schedule` | Replay buffer window size | `main.py`:`_get_window_iterations_for_iteration`; assigned to `self.replay_buffer.window_size` at iteration start and in `_try_auto_resume` / repair. |

**Confirmed:** No schedule present in YAML is unapplied. All 11 schedules affect runtime (selfplay and/or training).

---

## 2) Selfplay config provenance

- **Deepcopy:** `selfplay_optimized_integration.py`:`_apply_iteration_schedules` does `config = copy.deepcopy(self.config)` and returns that copy.
- **Schedules on the copy:** The returned `iteration_config` has `temperature`, `simulations`, `root_dirichlet_alpha`, `root_noise_weight`, `parallel_leaves` applied (and for iter 0, `simulations` overridden by `bootstrap.mcts_simulations` when `use_pure_mcts`).
- **q_value_weight / cpuct / dynamic_score_utility_weight:** Applied in `main.py`:`_apply_iteration_schedules` → `_apply_q_value_weight_and_cpuct_schedules(self.config, iteration)`, which **mutates** the shared `self.config`. The selfplay generator holds the same `self.config` reference; when it then calls `_apply_iteration_schedules(iteration)`, the deepcopy is of the **already-mutated** config, so the copy includes q_value_weight, cpuct, and dynamic_score_utility_weight.
- **Passed to workers:** `_generate_parallel(..., iteration_config)` passes `iteration_config` (the schedule-applied copy) as the second argument to the pool initializer (`initargs=(network_path, iteration_config, ...)`). `selfplay_optimized_integration.py`:`_init_worker_ignore_sigint` → `init_optimized_worker(*args)` → `OptimizedSelfPlayWorker(network_path, config, ...)`. So workers and MCTS receive the schedule-applied config.
- **Bootstrap (iteration 0):** Uses the same scheduled config copy; `_get_num_games(0)` returns `selfplay.bootstrap.games`, and inside `_apply_iteration_schedules(0)` the bootstrap block sets `config["selfplay"]["mcts"]["simulations"] = bootstrap_sims` when `use_pure_mcts` is true. So iter 0 gets bootstrap games + bootstrap sims on top of the same schedule logic (temperature, alpha, noise, etc. from schedule at iter 0).

---

## 3) Replay buffer / window_iterations_schedule correctness

- **Window application:** `main.py` sets `self.replay_buffer.window_size = _get_window_iterations_for_iteration(self.config, iteration)` at the start of each iteration (train loop) and in `_try_auto_resume` (for `last_comm + 1`) and in `_repair_run_state_from_committed` (per repaired iteration via `_get_window_iterations_for_iteration(self.config, i)`).
- **Sampling pool:** `replay_buffer.py` keeps `_entries: List[Tuple[int, str, int]]` (iteration, path, num_positions). Eviction: `add_iteration` and `finalize_iteration_for_commit` both `while len(self._entries) > self.window_size: self._entries.pop(0)`. So only the last `window_size` iterations are retained; the schedule is applied at iteration boundaries so the correct window is used for eviction.
- **max_size cap:** `get_training_data` uses `target_total = min(total, self.max_size)`; subsampling (uniform or newest_fraction) produces at most `max_size` positions. No leak beyond `max_size`.
- **newest_fraction:** When `newest_fraction > 0` and league recency is used, `_recency_window` defines the “newest” portion of the buffer by position count; allocation is `newest_take = min(int(target_total * self.newest_fraction), n_newest_total)` and `rest_take = target_total - newest_take`. No off-by-one identified; remainder is distributed by fractional allocation.

---

## 4) Atomic resume invariants

- **Staging discarded on restart:** `main.py`:`train()` starts with `cleanup_staging(self.run_root, last_comm, config=self.config)` (and `last_comm = max_committed_iteration(self.run_root)`). So partial staging is always removed before any resume logic.
- **train_base only from committed:** On each iteration, `train_base` is set to `resume_checkpoint` only when `iteration == start_iteration and resume_checkpoint`; otherwise `train_base = self.best_model_path`. `resume_checkpoint` comes from `_try_auto_resume`, which reads `run_state.json` and returns `self.best_model_path or self.latest_model_path or self.latest_checkpoint` — all of which are only ever set in `_commit_iteration_impl` to paths under `committed/` or `checkpoints/` (e.g. `latest_dest = self._atomic_copy_checkpoint(src_ckpt, "latest_model.pt")`, `best_model_path = str(best_dest)` or `str(src_ckpt)`). So train_base never points at staging.
- **Explicit guard:** `main.py` (train loop): if `resume_from_committed_state` and `train_base` contains `"staging"`, `raise ValueError(...)`. So no code path should pass a staging path when resume_from_committed_state is true.
- **Optimizer/scheduler/scaler/EMA:** Loaded in `trainer.py`:`_try_load_optimizer_state` from the checkpoint path passed as `train_base` (same as `previous_checkpoint`). That path is only ever committed or checkpoints/; staging is never used for resume.
- **Phase-boundary scheduler skip:** `trainer.py`:`_try_load_optimizer_state` uses `_is_phase_boundary()`; when true, it does not load `scheduler_state_dict`, so the new phase gets a fresh warmup. Confirmed in code.

---

## 5) Data/model shape invariants

- **Runtime tensor channels:** Dataset yields states; for gold_v2 the encoder produces 32 spatial channels on disk (`expected_spatial_channels: 32`). The model’s `PatchworkNet.forward` receives state from the dataset; in `_apply_gpu_legality` (`model.py`), if `n_ch == 32` it does `state = torch.cat([state, legalTL_24], dim=1)` → 56 channels. So the trunk always sees 56 channels when `input_channels==56` and data is 32ch; the 32→56 expansion is done inside the model, and `network.input_channels` (56) matches the trunk input.
- **encoding_version:** Config `data.encoding_version: gold_v2_32ch`; replay buffer and dataset use `encoding_version` and `expected_spatial_channels` for merge/validation. Model expects 56ch at trunk after legality concat.
- **max_actions:** Config `network.max_actions: 2026`. Model policy head output is 2026 (`model.py`: structured head `logits = torch.cat([pass_vec, patch_vec, buy_vec], dim=1)` → (B, 2026); `create_network` uses `max_actions=int(net_config.get("max_actions", ...))`). Encoder/action space is 2026 (`gold_v2_constants.MAX_ACTIONS`, `encoder.ACTION_SPACE_SIZE`). HDF5 and replay buffer use shape 2026 for masks/policies. Consistent.

---

## 6) Smoke test

- **Harness:** `tools/smoke_e2e.py`. Config: `configs/config_smoke_e2e.yaml`.
- **Usage:** `python -m tools.smoke_e2e` (or `--config configs/config_smoke_e2e.yaml`). Set `PYTHONUTF8=1` and `PYTHONIOENCODING=utf-8` on Windows if needed for Unicode output.
- **Flow:**
  - Run 1: `--iterations 1` — completes iteration 0 and commits.
  - Run 2: `--iterations 2` with `--allow-config-mismatch` — auto-resumes, runs iteration 1 and commits (config hash differs due to max_iterations override).
- **Asserts:**
  - `last_committed_iteration` is 0 after run 1 and 1 after run 2.
  - From commit manifests: `applied_settings` for iter 0 vs iter 1 differ (games, simulations, temperature, cpuct).
  - Training LR for iter 1 equals scheduled `iter_lr` (0.0005 in smoke config).
- Phase-boundary scheduler skip (scheduler not loaded at new phase) is implemented in `trainer.py`:`_try_load_optimizer_state` via `_is_phase_boundary()`; the smoke test does not assert it directly but the LR assertion confirms the phase LR is applied.

---

## Checklist (confirmed)

- [x] Every `iteration.*_schedule` in YAML is applied in code and affects runtime.
- [x] Selfplay workers and MCTS receive a single schedule-applied config copy (deepcopy in selfplay integration, with main having mutated q_value_weight/cpuct/dynamic_score_utility_weight first).
- [x] Bootstrap (iter 0) uses the same scheduled config with bootstrap games/sims overrides.
- [x] `window_iterations` from schedule is applied at iteration boundaries and used for eviction; buffer is capped at `max_size`; newest_fraction logic is consistent.
- [x] On restart: staging is discarded; train_base is only committed/checkpoints; optimizer/scheduler/scaler/EMA load only from that checkpoint; phase-boundary scheduler skip is in place.
- [x] Runtime tensor has 56 channels at trunk (32ch encoder + 24 legal in model); encoding_version and expected_spatial_channels respected; max_actions 2026 matches policy head and targets.

---

## Remaining risks

1. **Windows path handling:** `_is_committed_checkpoint_path` uses `Path(path).resolve().parts`; ensure drive-letter and path separators never produce a false `"staging"` substring in a committed path.
2. **Repair path:** If `_repair_run_state_from_committed` runs, it sets `replay_buffer.window_size` per repaired iteration; the order of operations (window_size set before `finalize_iteration_for_commit`) is correct, but any future change to repair order could affect eviction.
3. **League recency:** When `league.enabled` is true, `newest_fraction` and `_recency_window` override; verify in production that recency_newest_frac/window are intended vs non-league newest_fraction.
