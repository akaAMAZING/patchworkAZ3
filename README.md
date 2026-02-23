# Patchwork AlphaZero

An AlphaZero-style training system for the board game **Patchwork**, built with PyTorch.

The system plays itself thousands of games, learns from the results, and iteratively produces stronger and stronger neural networks — the same approach used by DeepMind's AlphaZero for Chess, Go, and Shogi.

---

## Quick Cheatsheet

```bash
# Preflight (optional, validates config + hardware)
python tools/preflight.py --config configs/config_best.yaml

# Run training
python -m src.training.main --config configs/config_best.yaml

# Monitor progress (separate terminal)
tensorboard --logdir logs/tensorboard
# → open http://localhost:6006

# Eval: compare two checkpoints (uses standard Elo; 64.5% WR ≈ 104 gap)
python tools/eval_latest_vs_oldest.py --config configs/config_best.yaml --model_a 0 --model_b 24 --games 20
# Model indices from discover_checkpoints; use --no-gui for CLI, omit for GUI

# Training analysis & handoff
python output_to_ChatGPT.py -o handoff_prompt.md                    # Metrics summary for ChatGPT handoff
python output_to_ChatGPT.py --include-context existing.md -o out.md # Prepend existing context
python tools/deep_training_analysis.py -o report.md                 # Full engineer-grade report (red flags, trends, all data)
python tools/deep_training_analysis.py --run-validation -o report.md # + validation sanity checks
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU       | NVIDIA with 6GB+ VRAM | RTX 3080 (10GB) |
| CPU       | 6+ cores | 12+ cores (Ryzen 5900X) |
| RAM       | 12 GB | 16 GB |
| Disk      | 20 GB free | 50+ GB free |
| OS        | Windows 10/11, Linux | Windows 10/11 |

---

## Installation

### 1. Clone and enter the project

```bash
cd C:\Users\Shanks\Desktop\Codes\patchworkaz
```

### 2. Create a virtual environment (if not already done)

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac
```

### 3. Install dependencies

**With GPU (recommended):**
```bash
pip install -r requirements-cuda.txt
```

> **Note:** CPU-only runs are possible with `requirements-cuda.txt` by setting `hardware.device: cpu` in the config, but training will be slow.

---

## Project Structure

```
patchworkaz/
├── .gitignore
├── README.md
├── requirements-cuda.txt           # GPU/CUDA dependencies (PyTorch cu121)
├── reset_training.bat               # Clean runs, checkpoints, logs, data (fresh start)
├── launch_gui.bat                  # One-click: API (iter59) + GUI + browser
│
├── configs/                        # Training configurations
│   └── config_best.yaml            #   Production config (18-block 128ch, FiLM, iteration schedules)
│
├── src/                            # Core source code
│   ├── game/
│   │   └── patchwork_engine.py     #   Numba-optimized Patchwork game engine
│   ├── mcts/
│   │   ├── alphazero_mcts_optimized.py   # AlphaZero MCTS (UCB, Dirichlet, FPU)
│   │   └── gpu_eval_client.py      #   Client for GPU inference server
│   ├── network/
│   │   ├── model.py                #   ResNet + SE blocks + FiLM + policy/value/ownership heads
│   │   ├── encoder.py              #   16-channel state encoding, D4 augmentation
│   │   └── gpu_inference_server.py #   Batched GPU inference for self-play workers
│   └── training/
│       ├── main.py                 #   Main orchestrator (entry point)
│       ├── trainer.py              #   Training loop (AdamW, cosine warmup, bf16 AMP)
│       ├── run_layout.py           #   Transactional staging/commit, run lock, resume logic
│       ├── selfplay_optimized.py   #   Self-play game generation worker
│       ├── selfplay_optimized_integration.py  # Multi-worker coordinator
│       ├── evaluation.py           #   SPRT gating, Elo/Glicko-2, baselines
│       ├── league.py               #   League training (PFSP, payoff matrix, optional)
│       ├── replay_buffer.py        #   Sliding-window HDF5 replay buffer
│       └── value_targets.py        #   Terminal value computation (score_tanh)
│
├── tools/                          # Utility scripts (run from repo root)
│   ├── preflight.py                #   Pre-run validation + smoke test
│   ├── deep_preflight.py           #   End-to-end pipeline smoke test (encoder, net, train, ckpt, selfplay→HDF5)
│   ├── full_integrity_check.py     #   Bulletproof one-command pipeline validation (Steps 0–5)
│   ├── sanity_check.py             #   Action encoding & rule correctness checks
│   ├── validation_sanity_checks.py #   Val-set class balance, baselines, per-class F1, ECE
│   ├── benchmark.py                #   Throughput benchmarks (encoder, MCTS, inference)
│   ├── eval_latest_vs_oldest.py    #   Compare checkpoints across iterations
│   ├── elo_system.py               #   Glicko-2 rating system (built-in + optional glicko2 pkg)
│   ├── tune_hyperparams.py         #   Optuna 2D hyperparameter tuning (cpuct, q_value_weight)
│   ├── tune_analysis_robust.py     #   Robust parameter recommendation from Optuna trials
│   ├── analyze_optuna.py           #   Optuna study analysis, importance, plots
│   └── TUNE_PARAM_VERIFICATION.md  #   Verification that tuned params do not inflate metrics
│
├── scripts/                        # Automation scripts
│   └── run_overnight.ps1           #   PowerShell launcher (preflight + train)
│
├── GUI/                            # Play Patchwork in browser
│   ├── patchwork_api.py            #   FastAPI backend (game logic + optional NN agent)
│   ├── src/App.jsx                 #   React frontend (Vite)
│   └── package.json                #   React, Vite deps
│
└── tests/                          # Automated tests
    ├── conftest.py                 #   Pytest fixtures
    ├── test_deep_preflight_smoke.py #   Deep preflight (Steps A–D, E with env var)
    ├── test_full_integrity_check_smoke.py
    ├── test_integration.py
    ├── test_schedule.py            #   Iteration schedule propagation
    └── test_amp_accuracy.py        #   bf16 AMP numerical checks
```

### Runtime directories (created automatically, gitignored)

```
runs/<run_id>/                      # Transactional run root (run_id from config or config hash)
  staging/iter_<N>/                 # Per-iteration work-in-progress (discarded if interrupted)
  committed/iter_<N>/               # Finalized iteration outputs
    selfplay.h5                     # Self-play data
    merged_training.h5              # Merged replay buffer (when used)
    iteration_NNN.pt                # Trained checkpoint
    iteration_NNN.json              # Summary
    commit_manifest.json            # Commit metadata
  run_state.json                    # Authoritative resume state (updated only at commit)
  replay_state.json                 # Replay buffer entries (committed paths only)
  elo_state.json                   # Elo/Glicko-2 ratings

checkpoints/best_model.pt           # Gate-promoted model (used for self-play)
checkpoints/latest_model.pt         # Most recent model (crash recovery)
checkpoints/iteration_XXX.pt        # Per-iteration snapshots
data/validation/                   # Validation split
logs/tensorboard/                   # TensorBoard metrics
logs/metadata.jsonl                 # Per-iteration metrics (append-only)
logs/training.log                   # Training log (staged, flushed on commit)

tuning_2d/                          # Optuna hyperparameter tuning (tune_hyperparams.py)
  optuna_study.db                   # SQLite study storage
  trial_NNNN/                       # Per-trial outputs

elo_ratings.json                    # Glicko-2 ratings (when eval.elo.enabled)
bench_results.json                  # Benchmark output (optional)
```

---

## How to Run

The main config is `configs/config_best.yaml` (18 blocks, 128ch network, FiLM conditioning, iteration schedules for LR/cpuct/q_value_weight/MCTS sims).

### Step 1: Run preflight checks

```bash
python tools/preflight.py --config configs/config_best.yaml
```

This validates:
1. Hardware (GPU, RAM, disk space)
2. Config correctness
3. VRAM estimation (will it fit?)
4. Core game engine sanity checks
5. Full pipeline smoke test (2 tiny iterations on CPU)

Add `--skip-smoke` to skip the slow smoke test if you've already verified:
```bash
python tools/preflight.py --config configs/config_best.yaml --skip-smoke
```

### Deep preflight (end-to-end pipeline smoke test)

After upgrades (bf16 AMP, FiLM support, 56-channel gold_v2 encoder), run the deterministic deep preflight to validate the full pipeline:

```bash
python tools/deep_preflight.py --config configs/config_best.yaml --device auto --tmp_dir /tmp/patchwork_smoke
```

Runs on CPU by default; uses CUDA+AMP if available. Add `--skip-e` to skip the self-play→HDF5→dataset step (faster).

| Step | Description |
|------|-------------|
| A | Encoder correctness (shape, channels 14/15 constants, pos/button reconstruction, flip invariance) |
| B | Network forward + AMP bf16 sanity |
| C | One training step (get_loss, backward, grad clip, optimizer) |
| D | Checkpoint save/load compatibility |
| E | Minimal self-play → HDF5 merge → PatchworkDataset load |

Pytest: run `pytest tests/test_deep_preflight_smoke.py -v` (Steps A–D only by default). Set `RUN_DEEP_PREFLIGHT_E2E=1` to include Step E.

### Full integrity check (bulletproof pipeline validation)

After changes to bf16 AMP, FiLM, network size, state channels (14→16), or schedule logic (cpuct, q_value_weight), run the full integrity check to validate the entire pipeline:

```bash
python tools/full_integrity_check.py --config configs/config_best.yaml --device auto --tmp_dir /tmp/patchwork_integrity --run_slow false --run_e2e true
```

**CRITICAL SAFETY:** All artifacts go under `--tmp_dir`. Never overwrites production checkpoints or logs.

| Step | Description |
|------|-------------|
| S1 | Compile-all: syntax validation via `python -m compileall` on src/, tools/, tests/ |
| S2 | Import sweep: attempt to import all modules under src/ (fail on import errors unless in allowlist) |
| 0 | Environment + config sanity (device, AMP dtype, input_channels=16, FiLM indices) |
| 1 | Unit tests: `pytest -m "not slow"`; if `--run_slow true` also `pytest -m "slow"` |
| 2 | Deep preflight (Steps A–D; Step E if `--run_e2e true`) |
| 3 | Schedule propagation verification (q_value_weight, cpuct, lock_eval_cpuct) |
| 4 | Mini E2E smoke (2 iterations, safe paths) — only if `--run_e2e true` |
| 5 | Checkpoint cycle (load, forward, verify finite) — only if `--run_e2e true` |

Output: PASS/FAIL checklist, timings, and JSON report at `<tmp_dir>/integrity_report.json`. Exit 0 on success.

**Pytest:** `RUN_FULL_INTEGRITY=1 pytest -q` runs the integrity smoke test (Steps 0–3 only, no E2E by default).

### Step 2: Start training (first run)

```bash
python -m src.training.main --config configs/config_best.yaml
```

For unambiguous "same command resumes" (recommended for overnight runs), use an explicit run identifier:

```bash
python -m src.training.main --config configs/config_best.yaml --run-id patchwork_production
# or a full path:
python -m src.training.main --config configs/config_best.yaml --run-dir runs/patchwork_production
```

Same `--run-id` or `--run-dir` every time = same run = seamless resume.

**Run lock**: A `.lock` file in the run directory prevents two processes from using the same run. If you see "Run directory … is locked by process N", another training run is active—exit it or use a different `--run-dir`.

This will:
1. Generate bootstrap self-play data using pure MCTS (one-time, ~5-10 min)
2. Train the initial neural network
3. Evaluate it against pure MCTS baseline
4. Repeat: self-play -> train -> evaluate -> promote

**Let it run overnight.** Press `Ctrl+C` to stop gracefully at any time.

### Step 3: Monitor progress

In a separate terminal:
```bash
tensorboard --logdir logs/tensorboard
```
Then open `http://localhost:6006` in your browser.

Key metrics to watch:
- `iter/total_loss`: should decrease over time
- `iter/policy_accuracy`: should increase (20% -> 40%+ is good)
- `eval/win_rate_vs_mcts`: should climb toward 90%+
- `eval/win_rate_vs_best`: should hover around 50% (healthy self-improvement)

---

## How to Resume / Continue Training

### Transactional iteration commits (safe resume)

Training uses **transactional commits**: each iteration is committed atomically only after self-play, training, evaluation, and gating complete. If the process stops mid-iteration (crash, Ctrl+C, power loss), the next launch **discards partial staging** and restarts that iteration from step 1.

**Resume is always at an iteration boundary** — never mid-iteration.

### Automatic resume (recommended)

Just run the **exact same command** again:

```bash
python -m src.training.main --config configs/config_best.yaml
```

The system reads `runs/<run_id>/run_state.json` and automatically:
- Discards any `staging/iter_*` directories (partial iterations)
- Starts at `last_committed_iteration + 1`
- Loads the correct checkpoint (`checkpoints/best_model.pt`)
- Restores replay buffer from committed iterations only
- Continues the rejection counter and Elo ratings

Log messages you may see:
- `Found partial staging for iter007; discarding and restarting iter007 from self-play.`
- `Committed iter007 successfully; last_committed_iteration updated.`
- `Reconciling run_state: filesystem has iter007 committed but run_state had 6; will repair.` (crash recovery)

**This is the intended workflow.** Run the same command every night and it continues from where it left off.

### Manual resume (if auto-resume fails)

Check `logs/training.log` for the last completed iteration number, then:

```bash
python -m src.training.main --config configs/config_best.yaml --start-iteration 15 --resume checkpoints/best_model.pt
```

Replace `15` with `last_completed_iteration + 1`.

### What if I need to start fresh?

Delete the runtime directories and start over:

```powershell
# PowerShell — or use reset_training.bat (Windows, with confirmation)
Remove-Item -Recurse -Force runs, checkpoints, logs, data -ErrorAction SilentlyContinue
python -m src.training.main --config configs/config_best.yaml
```

> **You MUST start fresh if you:** changed the network architecture (channels, blocks, heads) or the state encoding.

---

## Training Timeline (Approximate)

Using `config_best.yaml` on RTX 3080:

| Iterations | Strength level |
|------------|----------------|
| 0–20       | Beats random/pure MCTS easily |
| 20–80      | Strong intermediate, optimizes placement |
| 80–200     | Expert level, understands economy & timing |
| 200–400    | Superhuman territory, diminishing returns |

The config targets up to 400 iterations with scheduled increases in MCTS simulations (192→512) and learning rate decay phases.

---

## Utility Scripts

All scripts run from the repo root.

### Sanity checks
```bash
python tools/sanity_check.py              # Full checks (~2 min)
python tools/sanity_check.py --quick      # Quick checks (~15 sec)
```
Validates action encoding round-trips, legality masks, terminal state consistency, and A-vs-A fairness.

### Performance benchmarks
```bash
python tools/benchmark.py --config configs/config_best.yaml
python tools/benchmark.py --config configs/config_best.yaml --quick
```
Measures throughput of state encoding, legal action encoding, pure MCTS, network inference, and shard merging.

### Compare checkpoints
```bash
python tools/eval_latest_vs_oldest.py --config configs/config_best.yaml --ckpt_dir checkpoints --window 5 --games 20
```
Plays the latest iteration checkpoint against the oldest in a sliding window to verify learning progress. Does not write anything to disk.

### Champion vs field evaluation
```bash
# Iter 24 vs {1, 6, 12, 18} with 192 sims, 50 games per opponent
python tools/eval_latest_vs_oldest.py --config configs/config_best.yaml --champion-vs-field --main 24 --opponents 1,6,12,18 --games-per-opponent 50 --override_sims 192
```
Or use the GUI: check **Champion vs field**, select 1 main model, select 1+ opponents (Ctrl+click), set games per opponent and sims. Reports win rates with 95% CIs and standard Elo gap per matchup.

### Analyze per-iteration metrics
```bash
# Per-iteration distributions: legal count, policy perplexity, visit entropy (Iter 0–24)
python tools/analyze_iteration_metrics.py --config configs/config_best.yaml --run-dir runs/patchwork_production --iters 0-24

# Value target stats + calibration (bucket predicted value into deciles)
python tools/analyze_iteration_metrics.py --config configs/config_best.yaml --run-dir runs/patchwork_production --iters 0-24 --value-stats --calibration
```
Requires `matplotlib` for plots.

### Meta-analysis (full-depth Patchwork analytics)

arXiv-style metrics and visualizations from replay data:

```bash
# Full analysis: first-player advantage, piece tier list & EV when buyable,
# pass/buy propensity, patch placement heatmap, value by phase, value calibration
python tools/meta_analysis.py --config configs/config_best.yaml --run-dir runs/patchwork_production

# Specific iteration range, limit positions for speed
python tools/meta_analysis.py --config configs/config_best.yaml --run-dir runs/patchwork_production --iters 40-80 --max-positions 50000

# Include ownership heatmap (model inference; slower)
python tools/meta_analysis.py --config configs/config_best.yaml --run-dir runs/patchwork_production --ownership
```

Outputs: `logs/meta_analysis/` — JSON report, CSVs (first_player_advantage, piece_ranking, piece_value), and matplotlib figures (first-player curve, piece tier list, piece value/EV when buyable, patch heatmap, value-by-phase, calibration, ownership). Requires `matplotlib` for plots.

### Validation sanity checks (class balance & baselines)
```bash
# Class balance (ownership: empty vs filled), baseline accuracies, per-class precision/recall/F1
python tools/validation_sanity_checks.py --config configs/config_best.yaml

# With checkpoint: adds model metrics (macro-F1, ECE calibration)
python tools/validation_sanity_checks.py --config configs/config_best.yaml --checkpoint runs/patchwork_production/committed/iter_025/iteration_025.pt

# From specific HDF5 file
python tools/validation_sanity_checks.py --config configs/config_best.yaml --data path/to/selfplay.h5
```
Worth running once to interpret metrics: if "empty" is ~80%+, then 84% accuracy is not surprising; baseline "always predict empty" shows how much room there is to improve.

### Per-25-iters test suite
```bash
# At iter 25, 50, 75, ...: run analysis + save to logs/analysis_iterN/
python tools/run_per_25_tests.py --config configs/config_best.yaml --run-dir runs/patchwork_production --last-iter 24 --out-dir logs/analysis_iter25
```
See `TESTS_PER_25_ITERS.md` for full test checklist and champion-vs-field eval.

### Hyperparameter tuning (Optuna)
```bash
python tools/tune_hyperparams.py --config configs/config_best.yaml --n-trials 15 --max-iters 12 --eval-games 360
```
Tunes `cpuct` and `q_value_weight` via Optuna. Uses SQLite for resume; `--n-trials` is total target.

```bash
python tools/analyze_optuna.py --storage sqlite:///tuning_2d/optuna_study.db --plot
python tools/tune_analysis_robust.py
```
Analyze trial results and get robust parameter recommendations.

### Overnight launcher (PowerShell)
```powershell
.\scripts\run_overnight.ps1                                    # Default: config_best.yaml
.\scripts\run_overnight.ps1 -SkipPreflight                      # Skip preflight checks
.\scripts\run_overnight.ps1 -BenchmarkOnly                     # Just run benchmarks
```
Runs preflight checks, then starts training with timestamped log files.

### Run tests
```bash
pytest tests/ -v
pytest tests/test_deep_preflight_smoke.py -v -m "not slow"   # Deep preflight (Steps A–D)
# Include Step E (self-play pipeline): set RUN_DEEP_PREFLIGHT_E2E=1 before pytest
# Full integrity check: RUN_FULL_INTEGRITY=1 pytest -q
```

---

## GUI — Play Patchwork in Browser

A React + FastAPI web app lets you play Patchwork against humans or an AlphaZero NN agent.

**Requirements:** `pip install fastapi uvicorn` (add to venv if not already installed), and `npm` for the frontend.

**One-click launcher (Windows):**
```batch
launch_gui.bat
```
Opens API (iter59 by default) and GUI in separate windows, then opens your browser. Edit `ITER=59` at the top of `launch_gui.bat` to play vs a different iteration.

**Manual start:**
```bash
# Terminal 1: Start API server (game logic)
python GUI/patchwork_api.py --host 127.0.0.1 --port 8000

# With trained AlphaZero model (e.g. iter59)
python GUI/patchwork_api.py --model runs/patchwork_production/committed/iter_059/iteration_059.pt --config configs/config_best.yaml --simulations 800

# Terminal 2: Build and serve frontend
cd GUI
npm install && npm run build && npm run preview
# Or for dev: npm run dev
```

Open `http://localhost:4173` (preview) or `http://localhost:5173` (dev) to play.

---

## Key Design Decisions

- **No gating (always promote latest):** Following KataGo/Leela Chess Zero findings — 16-game evaluations are too noisy for reliable gating decisions. Small regressions self-correct in the next iteration. Evaluation still runs for monitoring.
- **Fresh optimizer per iteration (warm restart):** KataGo-aligned. Prevents optimizer state from becoming stale as the data distribution shifts.
- **SPRT-based evaluation:** Statistically rigorous early stopping for eval games. Saves compute when the outcome is clear. Configurable; can be disabled.
- **Score-margin value targets:** Network learns `tanh(score_diff / scale)` rather than binary win/loss, giving richer gradient signal.
- **Playout Cap Randomization:** KataGo technique — randomly reduces MCTS simulations for some moves (optional via `playout_cap_randomization`).
- **D4 augmentation:** Horizontal and vertical board flips applied during self-play to 4x the training data.
- **Ownership head:** KataGo-style auxiliary head that predicts which player owns each board cell, improving spatial understanding.
- **FiLM conditioning:** Optional Feature-wise Linear Modulation (gamma/beta) per residual block, conditioned on scalar input channels (position, buttons, etc.).
- **Glicko-2 Elo:** Sophisticated rating system with rating deviation and volatility; built-in implementation plus optional `glicko2` package.
- **League training (optional):** PFSP opponent sampling, payoff matrix, anchor suite gating. Disabled by default; enable via `league.enabled`.

---

## Troubleshooting

### "CUDA out of memory"
- **During training:** Reduce `training.batch_size` (try 512 or 256)
- **During self-play:** Reduce `inference.batch_size` (try 256 or 128)
- Reduce `network.channels` or `network.num_res_blocks`
- Ensure no other GPU processes are running

### Training seems stuck (loss not decreasing)
- Check TensorBoard for gradients (`grad_norm` should be reasonable, not 0 or exploding)
- Verify self-play is generating diverse games (`game_length` should vary)
- Consider restarting fresh if you changed config significantly

### "Size mismatch" errors when resuming
- You changed the network architecture but tried to resume from an old checkpoint
- Solution: start fresh (delete `data/`, `checkpoints/`, `logs/`)

### Self-play is very slow / low GPU utilization
- Increase `selfplay.num_workers` (e.g. 12–14 on 12-core CPU)
- Increase `inference.batch_size` (e.g. 256–384) and `inference.max_batch_wait_ms` (e.g. 15–18)
- Set `selfplay.mcts.parallel_leaves` to `simulations // 2` (max) for fuller GPU batches
- Check that GPU is being used for inference (`nvidia-smi`)

### Auto-resume not working
- Check that `runs/<run_id>/run_state.json` exists and is valid JSON
- Check that the checkpoint path in `run_state.json` points to an existing file
- Fall back to manual resume: `--start-iteration N --resume checkpoints/best_model.pt`
