# Continue training from iter70 with same run_id (TensorBoard continuous)

Resume from the committed iter70 checkpoint and produce **iter71, iter72, ...** under the **same run_id** (`patchwork_production`) so TensorBoard stays continuous. No overwrites of iter_070 or earlier.

## Config: PW ON, DSU/gate baseline

Use **config_continue_from_iter70.yaml**, which has:

- **Progressive Widening (PW):** ON  
  - `progressive_widening.enabled: true`  
  - `k_root: 64`, `k0: 32`, `k_sqrt_coef: 8`  
  - `always_include_pass` / `always_include_patch: true`
- **DSU/gate baseline (no gate-off experiment):**
  - `win_first.gate_dsu_enabled: true`
  - `dynamic_score_utility_weight: 0.30` (and schedule capped at 0.30)

This is the “PW-only” setup; it does **not** use gate off + 0.12.

## Safe resume behavior

- Training writes **only** to `iter_071`, `iter_072`, ... and **never** overwrites `iter_070` or earlier.
- You must pass **`--start-iteration 71`** and **`--resume`** pointing to the iter70 checkpoint (e.g. `.../committed/iter_070/iteration_070.pt`). If you use `--start-iteration 70`, the loop would run iteration 70 again; use 71 to start from the next iteration.
- If `committed/iter_071` (or whatever start you chose) **already exists**, the run **fails by default** to avoid overwriting. Use **`--force`** to advance to the next free iteration instead.

## Exact command (iter71+ with same run_id)

Replace `FULL_PATH_TO_ITER_070_CHECKPOINT` with your actual path (e.g.  
`C:\...\patchworkaz - Copy - v2\runs\patchwork_production\committed\iter_070\iteration_070.pt`).

**Windows (CMD):**

```cmd
python -m src.training.main --config configs/config_continue_from_iter70.yaml ^
  --start-iteration 71 ^
  --resume "FULL_PATH_TO_ITER_070_CHECKPOINT" ^
  --run-id patchwork_production
```

**Windows (PowerShell):**

```powershell
python -m src.training.main --config configs/config_continue_from_iter70.yaml `
  --start-iteration 71 `
  --resume "FULL_PATH_TO_ITER_070_CHECKPOINT" `
  --run-id patchwork_production
```

**Linux/macOS:**

```bash
python -m src.training.main --config configs/config_continue_from_iter70.yaml \
  --start-iteration 71 \
  --resume "FULL_PATH_TO_ITER_070_CHECKPOINT" \
  --run-id patchwork_production
```

## Optional: flush replay and reset optimizer on resume

To avoid old (pre-PW) targets dominating after resume:

- **`--flush-replay-on-resume`**  
  Clears the replay buffer state for this run **before** self-play for the first resumed iteration (iter71). New data will be from the current PW setup.
- **`--reset-optimizer-on-resume`**  
  Loads **model weights only** from the checkpoint; optimizer, scheduler, scaler, and **EMA are reinitialized** (EMA is **not** loaded from the checkpoint).

Recommended when changing search/targets: use **both** if you flush replay.

**Example with both flags (PowerShell):**

```powershell
python -m src.training.main --config configs/config_continue_from_iter70.yaml `
  --start-iteration 71 `
  --resume "FULL_PATH_TO_ITER_070_CHECKPOINT" `
  --run-id patchwork_production `
  --flush-replay-on-resume `
  --reset-optimizer-on-resume
```

**Example without flags (keep replay and optimizer state):**

```powershell
python -m src.training.main --config configs/config_continue_from_iter70.yaml `
  --start-iteration 71 `
  --resume "FULL_PATH_TO_ITER_070_CHECKPOINT" `
  --run-id patchwork_production
```

## If the start iteration is already committed: `--force`

If you re-run and `committed/iter_071` already exists, the process will **fail** unless you use **`--force`**, which advances the start iteration to the next free one (e.g. 72).

```powershell
python -m src.training.main --config configs/config_continue_from_iter70.yaml `
  --start-iteration 71 `
  --resume "FULL_PATH_TO_ITER_070_CHECKPOINT" `
  --run-id patchwork_production `
  --force
```

## Confirmation

- Outputs go only to **iter_071, iter_072, ...** under `runs/patchwork_production/`.
- **iter_070** and earlier in `committed/` are **never overwritten** by this flow.
- TensorBoard for `patchwork_production` continues from the same run directory.
