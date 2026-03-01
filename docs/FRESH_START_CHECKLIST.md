# FRESH START CHECKLIST — 201-bin + WIN_FIRST

Use this after wiping all artifacts (runs/, replay buffer, checkpoints staging, etc.) to start from iter000. Ensures the code and config do not accidentally resume old state.

---

## 1) Config correctness (`configs/config_best.yaml`)

### Values safe for iter000 fresh run

| Key | Current | For fresh start | Notes |
|-----|---------|----------------|--------|
| `training.resume_optimizer_state` | `true` | **Keep true** | Only used when `train_base` is set (committed/ or checkpoints/). For iter0 from scratch `train_base` is None → no load. |
| `training.resume_scheduler_state` | `true` | **Keep true** | Same as above. |
| `training.resume_scaler_state` | `true` | **Keep true** | Same. |
| `training.resume_ema_state` | `true` | **Keep true** | Same. |
| `training.resume_from_committed_state` | `true` | **Keep true** | Ensures we never load optimizer/EMA from staging; only from committed/ or checkpoints/. |
| **`iteration.auto_resume`** | `true` | **Set to `false`** | **Recommended for fresh start.** When `false`, the trainer never reads `run_state.json`. If you wipe runs/ but leave a stale `run_state.json` elsewhere, or later recreate the run dir, `auto_resume: true` would try to restore `best_model_path`/`latest_model_path` from state; if those paths no longer exist, iter0 would hit "Self-play checkpoint missing" or "Training checkpoint missing". |

### Minimal config change for clean iter000

In `configs/config_best.yaml`, under `iteration:` set:

```yaml
iteration:
  max_iterations: 500
  auto_resume: false   # was true — set false for fresh start
```

No other resume-related keys need changing. After the first run commits iter000, you can set `auto_resume: true` again for normal resumable runs if you like.

---

## 2) Starting checkpoint behavior (iter000)

### Where initial weights come from when no prior iteration exists

- **Entry point:** `main.py` → `train(start_iteration=0, resume_checkpoint=None)`. Then `_try_auto_resume(0, None)`.
- **When `auto_resume` is false (or `run_state.json` is missing):** `_try_auto_resume` returns `(0, None)`. So `start_iteration=0`, `resume_checkpoint=None`; `self.best_model_path` stays `None`.
- **For iteration 0:**
  - `selfplay_model` = `resume_checkpoint` if (iteration == start_iteration and resume_checkpoint) else `self.best_model_path` → **None**.
  - `train_base` = same → **None**.

### Does iter000 start from random init or from a checkpoint?

- **No `--resume` and no run_state (fresh wipe):** iter000 starts from **random init**.
  - Self-play: `network_path=None` → **bootstrap (pure MCTS)**. No GPU server; `SelfPlayGenerator.generate(0, None, ...)` uses `bootstrap.use_pure_mcts` and `bootstrap.mcts_simulations` (config: 192 sims, 900 games).
  - Training: `train_base=None` → trainer builds a new model and trains from **scratch** (no checkpoint load). Optimizer/scheduler/scaler/EMA are not loaded.
- **With `--resume path/to/checkpoint`:** iter000 uses that checkpoint for both self-play and training.
  - Self-play: `selfplay_model = resume_checkpoint` → GPU server (or local eval) with that checkpoint.
  - Training: `train_base = resume_checkpoint` → load weights from checkpoint. If path is under `checkpoints/` or `committed/`, optimizer/scheduler/EMA are also loaded (when resume_* are true).

### Starting checkpoint path after a wipe

- **Random init (no checkpoint):** No path required. Wipe everything; run with `auto_resume: false`. No need for any file under `checkpoints/` or `runs/`.
- **Start from a seed checkpoint (e.g. old scalar-head best):**
  - Copy your seed checkpoint into the repo (e.g. `checkpoints/seed_model.pt`) **after** the wipe, or use a path that will exist at run time.
  - Run:  
    `python -m src.training.main --config configs/config_best.yaml --resume checkpoints/seed_model.pt`  
  - Do **not** rely on `run_state.json` or anything under `runs/` for that path; it must be a path you control (e.g. `checkpoints/` or an absolute path).

### 201-bin score head when starting from scalar-head checkpoint

- The model is created with 201-bin `ValueHead` (see `network/model.py`: `SCORE_MIN=-100`, `SCORE_MAX=100`, `score_head` is `nn.Linear(..., SCORE_BINS)`).
- Loading a scalar-head checkpoint: `load_model_checkpoint` / `get_state_dict_for_inference` in `src/network/model.py` handle incompatible `value_head.score_head` (old `score_head.0.weight/bias` vs new `score_head.weight/bias`). Incompatible keys are skipped and the 201-bin score head is **randomly initialized**; other weights load as usual. This is what validation’s Phase 1 reports as `score_head_skipped=True`.

---

## 3) GPU inference plumbing

Verified end-to-end:

| Component | Expectation | Status |
|-----------|-------------|--------|
| **Server response** | 5-tuple `(rid, priors_legal, value, mean_points, score_utility)` | **OK** — `gpu_inference_server.py` puts `(int(rid), priors_legal, v, mean_pts, su)` on the queue (line ~424). |
| **eval_client.receive()** | Unpacks to 4 values: `(priors_legal, value, mean_points, score_utility)` | **OK** — `gpu_eval_client.py` `receive()` strips `rid` and returns the other four. |
| **MCTS usage** | `u = value + score_utility`; score in points; root center = `mean_points`; DSUW gating | **OK** — `alphazero_mcts_optimized.py` uses `mean_points` as root score estimate, receives `score_utility` from server, applies gate scaling to effective static/dynamic weights per request. |

Server computes `mean_points` and `score_utility` from the 201-bin distribution on GPU; no logits over IPC. Scale and bins match `ValueHead.SCORE_MIN/MAX` and config `score_utility_scale` (30.0).

---

## 4) Training target plumbing

| Item | Expectation | Status |
|------|-------------|--------|
| **Replay score_margins** | Tanh-normalised in [-1, 1] | **OK** — `value_targets.value_and_score_from_scores()` uses `tanh(raw_margin / 30.0)`; self-play writes these into `score_margins` dataset. |
| **Points target** | `round(score_utility_scale * atanh(clamp(m, -0.999999, 0.999999)))` | **OK** — `trainer.py` `build_score_bin_labels_from_tanh_margins()` clamps and uses `score_utility_scale * atanh(m)` then rounds and clamps to bin range. |
| **Bins** | `SCORE_MIN`..`SCORE_MAX` (201 bins) | **OK** — `ValueHead.SCORE_MIN=-100`, `SCORE_MAX=100`; trainer uses `getattr(ValueHead, "SCORE_MIN/MAX", ±100)` and builds labels in that range. |
| **Score loss** | CE against `score_logits` | **OK** — Trainer builds soft labels from tanh margins and computes CE vs model’s score logits. |
| **Scales** | `score_utility_scale` and divisor 30.0 consistent | **OK** — `value_targets._SCORE_NORMALISE_DIVISOR = 30.0`; `selfplay.mcts.score_utility_scale: 30.0`; trainer default `score_utility_scale` 30.0. Optional: set `training.score_utility_scale: 30.0` in config for explicitness. |

---

## 5) Preflight and smoke

### Preflight

```bash
python tools/preflight.py --config configs/config_best.yaml
```

- **[1/6] Hardware & Environment** — PASS  
- **[2/6] Config Validation** — PASS  
- **[3/6] VRAM Estimation** — Informational  
- **[4/6] Core Correctness (sanity checks)** — PASS  
- **[5/6] Invariance Tests (D4 + shop order)** — PASS  
- **[6/6] Pipeline Smoke Test** — Can fail on Windows with:  
  `'charmap' codec can't encode characters in position 0-99`  
  This is a **Windows console encoding** issue (non-ASCII in logs), not a training or fresh-start logic bug. Preflight still validates config, hardware, and sanity; the smoke run itself (2 iters, bootstrap + train) is equivalent to the validation script’s flow.

### Minimal self-play smoke (GPU server + MCTS)

To confirm GPU server + one MCTS search without full preflight smoke:

```bash
python tools/validate_201bin_winfirst.py --config configs/config_best.yaml --checkpoint checkpoints/latest_model.pt --skip-benchmark --skip-ab --ab-games 0
```

- Runs Phase 1 (checkpoint smoke: load, forward, GPU server + eval_client).  
- Then Phase 4 (one WIN_FIRST debug game).  
- Requires an existing checkpoint (e.g. `latest_model.pt`). If you wiped checkpoints, run with a seed checkpoint or use preflight’s tiny config (which uses bootstrap for iter0 and does not need a checkpoint).

Alternatively, the **shortest existing smoke** that starts the GPU server and runs MCTS is the validation script above with a valid checkpoint, or preflight’s smoke (which uses CPU and bootstrap for iter0 when no checkpoint is used).

---

## DELIVERABLE SUMMARY

| Topic | Result |
|-------|--------|
| **Exact config for iter000 fresh run** | Set `iteration.auto_resume: false`. Leave all `training.resume_*` and `resume_from_committed_state` as in config_best; they only apply when `train_base` is set. |
| **Starting checkpoint** | **Optional.** Not required: iter0 can be bootstrap (pure MCTS) + train from scratch. If you want to start from a seed (e.g. scalar-head best): copy checkpoint to a stable path (e.g. `checkpoints/seed_model.pt`) after wipe and run with `--resume checkpoints/seed_model.pt`. |
| **How iter0 is chosen** | With no run_state and no `--resume`: `selfplay_model` and `train_base` are both `None` → bootstrap + from-scratch training. With `--resume X`: both use `X`. |
| **Code paths that break after wipe** | If `run_state.json` still exists and points to paths under a wiped `runs/` or removed checkpoints, and `auto_resume: true`, then `_try_auto_resume` can set `best_model_path` to a missing file → assert "Self-play checkpoint missing" or "Training checkpoint missing". **Fix:** set `auto_resume: false` for fresh start, or ensure run_state is removed with the wipe (e.g. wipe entire `runs/<run_id>/` including `run_state.json`). |
| **Preflight** | Steps 1–5 PASS. Step 6 (pipeline smoke) may fail on Windows with a console encoding error; not a logic failure. |
| **201-bin + scalar-head** | Loading a scalar-head checkpoint into the 201-bin model skips incompatible score_head keys and reinitializes the score head; other weights load. Validated by `validate_201bin_winfirst` Phase 1. |

---

## Recommended commands for a fresh iter000 run

1. Wipe artifacts (runs/, replay buffer, staging; optionally clear or replace checkpoints).
2. Set in config: `iteration.auto_resume: false`.
3. **Option A — from scratch:**  
   `python -m src.training.main --config configs/config_best.yaml`
4. **Option B — from seed checkpoint:**  
   `python -m src.training.main --config configs/config_best.yaml --resume checkpoints/seed_model.pt`  
   (ensure `checkpoints/seed_model.pt` exists after the wipe.)

After the first commit (iter000), you can set `auto_resume: true` again for normal resumable training.
