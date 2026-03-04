# GUI “Best AI” verification (iter69 + 3000 sims)

## What was wrong before (no assumptions)

- **Model:** The launcher and GUI defaulted to `checkpoints/latest_model.pt`. That file is only updated when the **training** script commits an iteration (it’s a copy of the **last committed** iter). So:
  - If you never ran training after iter69, `latest_model.pt` could still be from an older iter.
  - If you ran training and the last committed was iter69, then `latest_model.pt` was effectively iter69 — but that was not explicit or guaranteed.
- **Iter69:** Your stated best model is **iter69 in committed**. To be sure that exact checkpoint is used, the path must point explicitly to:
  `runs\patchwork_production\committed\iter_069\iteration_069.pt`
- **Simulations:** Defaults were **800** everywhere. You asked if **3000** is high enough: for strength, more sims = stronger (slower). 3000 is a good balance; the GUI allows up to 20000.

## What is used now (verified)

| Component | What it uses |
|-----------|----------------|
| **launch_gui.bat** | Model: `runs\patchwork_production\committed\iter_069\iteration_069.pt` if that file exists; otherwise `checkpoints\latest_model.pt`. Sim count: **3000**. |
| **GUI default nn path** | `...\runs\patchwork_production\committed\iter_069\iteration_069.pt` (full path in `App.jsx`). |
| **GUI default sims** | **3000** (stored in `pw_nn_sims`; first-time or cleared storage gets 3000). |
| **API default sims** | **3000** (`_NN_SIMULATIONS` and `--simulations` default in `patchwork_api.py`). |
| **Config** | `configs/config_best.yaml` (architecture must match the checkpoint). |

So:

1. **Is iter69 what is getting used?**  
   **Yes**, as long as:
   - You run the batch script and the file `runs\patchwork_production\committed\iter_069\iteration_069.pt` exists (batch uses it; API starts with it).
   - In the GUI you either don’t click “Load” (so the API keeps the model it started with from the batch) or you click “Load” with the default path (iter69). The default path in the GUI is now iter69, and a one-time migration updates old “latest_model.pt” path to iter69 in localStorage.

2. **Is this the strongest possible implementation in the GUI?**  
   - Same game state and rules as the training engine (state → `/legal`, `/solve_nn`, `/apply`).
   - Same config (`config_best.yaml`) and same MCTS code path; no weaker fallback when NN is loaded.
   - Temperature 0 for determinism; sim count is whatever you set (default 3000, max 20000 in GUI).

3. **Is 3000 sims high enough?**  
   - 3000 is a solid default for strength vs speed. For maximum strength (slower moves), increase “NN simulations” in the GUI up to 20000.

## How to confirm on your machine

1. **Which model the API has:**  
   After starting the API (via the batch or by hand), open:
   `http://localhost:8000/nn/status`  
   Check `model_path` — it should be either the iter_069 path or the path you passed with `--model`.

2. **Which model the GUI will load:**  
   In the GUI, open the NN / settings panel and look at “Model path”. It should show the iter69 path (or whatever you set). When you click “Load”, that path is sent to `/nn/load`.

3. **Sim count:**  
   - Batch: see the batch script (`--simulations 3000`).  
   - API default: see `patchwork_api.py` (`_NN_SIMULATIONS = 3000` and `--simulations` default).  
   - GUI: see “NN simulations” in the UI (default 3000; sent in `/solve_nn` and `/nn/load`).

## If iter_069 is missing

If `runs\patchwork_production\committed\iter_069\iteration_069.pt` does not exist (e.g. different machine or run layout):

- The **batch** script falls back to `checkpoints\latest_model.pt` and prints that it’s using the fallback.
- You can copy the iter69 checkpoint into the repo, e.g.:
  - `runs\patchwork_production\committed\iter_069\iteration_069.pt`, or
  - `checkpoints\latest_model.pt` (then the batch will use it as fallback).
- Or run the API manually with the path to your best checkpoint and set the GUI path to that file.
