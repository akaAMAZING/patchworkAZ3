# Patchwork AlphaZero Pipeline Audit

End-to-end implementation-accurate description of the training + self-play pipeline. This document reflects the **actual current implementation**, not intended design.

### Audit Script (prevent drift)

```bash
python tools/pipeline_audit.py --config configs/config_best.yaml
python tools/pipeline_audit.py --config configs/config_best.yaml --iteration 25
```

Prints: AMP dtype + scaler, EMA enabled + weights_for_actor, policy_target_mode + action_selection_temp, optimizer/scheduler resume, lr_schedule. (Dual-Head: score_scale N/A — value=win/loss/tie, score_margin=raw.)

### Validation Tools

- **Preflight**: `python tools/preflight.py --config configs/config_best.yaml` — hardware, config, smoke test
- **Deep preflight**: `python tools/deep_preflight.py --config configs/config_best.yaml` — encoder, net, train, ckpt, selfplay→HDF5
- **Full integrity**: `python tools/full_integrity_check.py --config configs/config_best.yaml --run_e2e true` — compile, import, unit tests, schedule propagation, mini E2E

---

## 0) Repo + Run Identity

### Git Commit Hash / Uncommitted Diffs

**Git**: Run `git rev-parse HEAD` and `git status` / `git diff` to obtain. If repo is not initialized, no commit hash or uncommitted diff available.

### Exact Training Launch Command

```bash
python -m src.training.main --config configs/config_best.yaml
```

With explicit run identity for deterministic resume:

```bash
python -m src.training.main --config configs/config_best.yaml --run-id patchwork_production
# or
python -m src.training.main --config configs/config_best.yaml --run-dir runs/patchwork_production
```

### Full Resolved Config (After Overrides and Schedule Interpolation)

Config is loaded from `configs/config_best.yaml`. **Schedules use step semantics**: for each schedule, select the last entry with `entry.iteration <= current_iteration`. No linear interpolation between steps.

**Key resolved values at iteration 0** (config_best.yaml):

```yaml
# Core
seed: 42
deterministic: false

# Hardware
hardware.device: cuda
hardware.pin_memory: true
hardware.persistent_workers: true

# Network (18 ResBlocks, 128 channels)
network.input_channels: 56
network.num_res_blocks: 18
network.channels: 128
network.policy_channels: 48
network.policy_hidden: 512
network.use_factorized_policy_head: true    # StructuredConvPolicyHead (config_best)
network.value_channels: 48
network.value_hidden: 512
network.max_actions: 2026
network.use_film: true
network.film_global_dim: 61  # gold_v2 FiLM uses global features

# Training
training.learning_rate: 0.0016   # Base; iteration.lr_schedule overrides at 80, 200
training.lr_schedule: cosine_warmup_per_phase
training.batch_size: 1024
training.use_amp: true
training.amp_dtype: bfloat16
training.max_grad_norm: 1.0
training.allow_tf32: true
training.matmul_precision: high
training.epochs_per_iteration: 2
training.val_split: 0.08
training.val_frequency: 200
training.score_loss_weight: 0.02
training.ownership_loss_weight: 0.15
training.ema.enabled: true
training.ema.decay: 0.999
training.ema.use_for_selfplay: true
training.ema.use_for_eval: true

# Self-play (KataGo Dual-Head: value=win/loss/tie, score_margin=raw integer)
selfplay.games_per_iteration: 120   # iter 0: bootstrap.games overrides
selfplay.bootstrap.games: 900
selfplay.mcts.simulations: 192       # Schedule: 192→224@80→256@140→288@200→320@300→352@400
selfplay.mcts.cpuct: 1.47           # Schedule: 1.47→1.52@30→1.58@80→1.66@140→1.75@200
selfplay.mcts.temperature: 1.0      # Schedule: 1.0→0.8@30→0.5@80→0.3@150→0.25@300
selfplay.mcts.score_utility_weight: 0.02   # MCTS: utility = value + w * score_margin
selfplay.q_value_weight: 0.25       # Schedule: 0.25→0.35@50→0.45@100→0.50@220
selfplay.policy_target_mode: visits
selfplay.store_canonical_only: true # 1 pos/move in buffer; D4 applied at train time (dynamic)

# Replay
replay_buffer.max_size: 300000      # 56ch ~30KB/pos; 500k would ~15GB (OOM on 16GB RAM)
replay_buffer.min_size: 4000        # Below this: use current iter only (no merge)
replay_buffer.window_iterations: 8   # Schedule: 8→8@80→11@140→12@200→14@300→15@400
replay_buffer.newest_fraction: 0.25

# Evaluation (config_best: disabled)
evaluation.games_vs_best: 0
evaluation.games_vs_pure_mcts: 0   # → auto-accept every model
evaluation.frozen_ladder.simulations: 256
evaluation.frozen_ladder.temperature: 0
evaluation.frozen_ladder.root_noise_weight: 0
```

**LR schedule (step semantics)**:

| iteration | lr |
|-----------|-----|
| 0 | 0.0016 |
| 80 | 0.0008 |
| 200 | 0.0004 |

**MCTS simulations schedule**:

| iteration | sims |
|-----------|-----|
| 0 | 192 |
| 80 | 224 |
| 140 | 256 |
| 200 | 288 |
| 300 | 320 |
| 400 | 352 |

**Games schedule** (canonical-only: 1 pos/move ⇒ games × ~42 ≈ positions/iter):

| iteration | games | ~positions/iter |
|-----------|-------|-----------------|
| 0 | 900 | ~37.8k |
| 80 | 900 | ~37.8k |
| 140 | 650 | ~27.3k |
| 200 | 600 | ~25.2k |
| 300 | 510 | ~21.4k |
| 400 | 480 | ~20.2k |

**q_value_weight schedule**:

| iteration | q_value_weight |
|-----------|----------------|
| 0 | 0.25 |
| 50 | 0.35 |
| 100 | 0.45 |
| 220 | 0.50 |

**cpuct schedule**:

| iteration | cpuct |
|-----------|-------|
| 0 | 1.47 |
| 30 | 1.52 |
| 80 | 1.58 |
| 140 | 1.66 |
| 200 | 1.75 |

### Hardware

- **Recommended**: RTX 3080 (10GB VRAM), Ryzen 5900X (12 cores / 24 threads), 16GB RAM
- **Device selection**: `config["hardware"]["device"]` (default `cuda`); falls back to CPU if CUDA unavailable
- **Storage**: In-memory HDF5 load (~9GB for 300K positions @ 56ch); LZF compression on disk; SSD recommended for shard I/O

### PyTorch + CUDA

- **requirements-cuda.txt**: `torch>=2.0.0` with `--index-url https://download.pytorch.org/whl/cu121`
- CUDA 12.1 wheel; actual versions from installed packages

### Distributed Training

**None.** Single-GPU only. No DDP, FSDP, gradient sync, or multi-node. Training uses `DataLoader` with `num_workers=0` and `pin_memory=True`.

### Mixed Precision / Determinism

- **AMP**: Trainer honors `training.amp_dtype`: bfloat16 → `autocast(dtype=torch.bfloat16)` (no GradScaler); float16 → `autocast(dtype=torch.float16)` with GradScaler. MCTS/GPU inference reads `inference.amp_dtype`. Startup log: `[TRAINING AMP] use_amp=... autocast_dtype=... scaler_enabled=...`
- **TF32**: `training.allow_tf32` or `inference.allow_tf32` controls matmul/cudnn TF32 in both trainer and inference.
- **cudnn.benchmark**: `False` in MCTS (`alphazero_mcts_optimized.py`); `True` in GPU inference server when `deterministic=false`.
- **Determinism**: `config.deterministic: false` by default. When true, `cudnn.deterministic=True`, `cudnn.benchmark=False`. `torch.use_deterministic_algorithms(True)` is **disabled** (CUBLAS compatibility).

### torch.compile

- **Config**: `inference.torch_compile: false` (config_best.yaml)
- **Usage**: Only in `gpu_inference_server.py`; when enabled, `torch.compile(model, mode="reduce-overhead")`. Training does **not** use compile.

---

## 1) High-Level Pipeline Timeline (One Iteration)

### Step-by-Step Timeline

| Step | Description | Inputs | Outputs | "Done" Marker |
|------|-------------|--------|---------|----------------|
| **0** | Apply iteration schedules | `iteration`, `config` | `config` modified in-place (LR, cpuct, q_value_weight, window_iterations, etc.) | N/A |
| **1** | Select checkpoints for self-play | `best_model_path` (or `resume_checkpoint` on first iter after resume) | Model path for self-play | `selfplay_model` set |
| **2** | Generate self-play data | Model path, iteration, `staging/iter_N/` | NPZ shards → merged `staging/iter_N/selfplay.h5` | HDF5 with `selfplay_complete=true`, `selfplay_num_games_written >= expected`. Iter 0: `bootstrap.games` (900); iter≥1: `games_schedule`. |
| **3** | Merge shards into HDF5 | Shard dir `iter_NNN_shards/*.npz` | `selfplay.h5` in staging | `_merge_shards()` completes |
| **4** | Register + merge replay buffer | `replay_buffer.add_iteration(iter, staging/selfplay.h5)`; if `total_positions >= min_size`: `get_training_data()` → `merged_training.h5` | In-memory entries; merged HDF5 in staging | `effective_data_path` = merged path or single-iter path |
| **5** | Replay eviction / path swap | `finalize_iteration_for_commit()` at **commit** time | Staging path → committed path; oldest iter beyond `window_iterations` evicted | `len(_entries) <= window_size`; replay_state has committed paths only |
| **6** | Training loop | `effective_data_path`, `train_base` checkpoint | `staging/iter_N/iteration_NNN.pt` | `train_iteration()` returns |
| **7** | Validation loop | Val split (8%), every 200 steps | Val metrics logged to TensorBoard | Inline in `train_epoch` |
| **8** | Checkpoint / commit atomicity | Staging dir | `commit_iteration()`: move staging→committed, update `run_state.json` | `committed/iter_N/commit_manifest.json` exists |
| **9** | Frozen ladder (optional, manual) | `python tools/run_frozen_ladder.py --run-dir runs/<id> --iteration N` | `eval/ladder/iter_XXX.json` | Standalone tool; not invoked by main loop |

### File Paths

- **Staging**: `runs/<run_id>/staging/iter_<N>/`
- **Committed**: `runs/<run_id>/committed/iter_<N>/`
- **Run state**: `runs/<run_id>/run_state.json`
- **Replay state**: `runs/<run_id>/replay_state.json` (main passes `run_root / "replay_state.json"`)
- **Checkpoints**: `checkpoints/best_model.pt`, `checkpoints/latest_model.pt`, `checkpoints/best_model_iterNNN.pt`, `committed/iter_N/iteration_NNN.pt`
- **Self-play HDF5**: `committed/iter_N/selfplay.h5`
- **Merged training**: `staging/iter_N/merged_training.h5` (written by `replay_buffer.get_training_data()`)

### Resume Behavior

- **What loads**: `run_state.json` (last_committed_iteration, best_model_path, latest_model_path, global_step, consecutive_rejections), `replay_state.json`, Elo ratings, league state.
- **What doesn't**: Optimizer/scheduler state (fresh per iteration).
- **Reconcile**: If `committed/` has iters not in `run_state`, `_repair_run_state_from_committed()` replays commits and updates state.
- **Cleanup**: `cleanup_staging()` **always discards** partial staging. Uncommitted iterations restart from self-play (no preservation of complete selfplay).

---

## 2) Game + Action Space Implementation

### Action Encoding (2026 actions)

From `src/network/encoder.py` and `src/mcts/alphazero_mcts_optimized.py`:

| Index range | Action type | Encoding |
|-------------|-------------|----------|
| 0 | Pass | `PASS_INDEX = 0` |
| 1–81 | Patch placement | `PATCH_START + board_pos` (board_pos = row*9 + col) |
| 82–2025 | Buy | `BUY_START + (slot_index * 8 + orient) * 81 + pos` |

- **PATCH_START** = 1
- **BUY_START** = 82
- **NUM_SLOTS** = 3, **NUM_ORIENTS** = 8
- **ACTION_SPACE_SIZE** = 82 + 3*8*81 = 2026

Engine action → index:

- `AT_PASS` → 0
- `AT_PATCH` → 1 + action[1] (board_pos)
- `AT_BUY` → 82 + (offset-1)*8*81 + orient*81 + top*9+left

### Illegal Move Masking

- **Where**: Network logits **after** `policy_head` forward.
- **How**: `policy_logits.masked_fill(action_mask == 0, float("-inf"))` in `model.py` (both `forward()` and `get_loss()`).
- **Effect**: Illegal actions get `-inf` before softmax; softmax output is 0. No renormalization; `log_softmax` clamps to `min=-100`.
- **MCTS expansion**: Priors come from `softmax(logits)` over **legal indices only**; illegal actions never appear as children.

### Symmetries / Augmentations

Config: `selfplay.augmentation`: `"none"` | `"flip"` | `"d4"`. Code default (when key missing): `"flip"`. **config_best**: `d4`.

| Mode | Variants | Transforms |
|------|----------|------------|
| `none` | 1 | identity only |
| `flip` | 4 | id, v, h, vh (same as legacy `augment_flips`) |
| `d4` | 8 | id, r90, r180, r270, m, m_r90, m_r180, m_r270 |

- **Spatial channels** 0–27 (boards, coords, slot×orient). Channels 28–60 are scalars and **never** transformed.
- **Action remapping**: Piece-id-aware orientation and position remapping via `d4_augmentation.transform_action_vector`.
- **Ownership**: `apply_ownership_transform` matches state spatial transform.
- **Backward compat**: `augment_flips: true` (no `augmentation` key) → `flip`.

### Canonical-Only Storage (store_canonical_only: true)

**We do NOT store 8× D4 variants in the buffer.** Each move stores **1 canonical position** only. D4 is applied **dynamically at train time** in `PatchworkDataset.__getitem__` via `training.d4_augmentation: dynamic` — a random transform per sample on each access. This yields:

- **Positions per game**: ~42 (avg game length)
- **Positions per iteration**: 900 games × ~42 ≈ **37,800** (vs ~302k if we stored 8× pre-augmented)
- **Buffer fill**: ~38k pos/iter at iter 0; window 8 iter ≈ 302k total positions
- **slot_piece_ids**: Stored per position so D4 can apply correct piece-aware transforms at train time

### Terminal Scoring

- **Score**: `score = buttons - 2*empty + bonus` (bonus=7 if 7×7 tile owned, else 0).
- **Winner**: `get_winner_fast()`: if `p0_score > p1_score` → 0, else if `p1_score > p0_score` → 1, else `1 - TIE_PLAYER`.
- **Tie handling**: Deterministic tie-breaker via `TIE_PLAYER` (first player to have passed).
- **Value target (KataGo Dual-Head)**: `value = +1` (win), `-1` (loss), `0` (tie) from `to_move` perspective. `score_margin = (current_player_score - opponent_score)` as raw integer. No score_scale; value head learns win-probability, score head learns margin.

---

## 3) State Encoding (56 Channels — gold_v2)

From `encode_state_multimodal()` / `GoldV2StateEncoder` in `src/network/encoder.py`:

**Spatial (flipped/rotated by augmentation):**
| Channel | Shape | Semantics |
|---------|-------|-----------|
| 0 | (9,9) | current_board_occupancy |
| 1 | (9,9) | opponent_board_occupancy |
| 2 | (9,9) | coord_row_norm (row/(BOARD_SIZE-1)) |
| 3 | (9,9) | coord_col_norm (col/(BOARD_SIZE-1)) |
| 4-11 | (9,9) | slot0_orient0..7_shape |
| 12-19 | (9,9) | slot1_orient0..7_shape |
| 20-27 | (9,9) | slot2_orient0..7_shape |

**Scalar (full-plane, never spatially transformed):**
| Channel | Semantics |
|---------|------------|
| 28 | current_income_norm (income/12) |
| 29 | opponent_income_norm |
| 30 | position_diff_norm ((c_pos-o_pos)/53) |
| 31 | button_diff_norm |
| 32 | bonus_7x7_status (0/0.5/1) |
| 33 | pending_patches_norm |
| 34 | absolute_time_norm (c_pos/53) |
| 35 | absolute_buttons_norm |
| 36-39 | dist_to_next_income/patch (current, opponent) |
| 40-42 | pass_steps, pass_income_marks, pass_patches |
| 43-51 | slot0/1/2 cost_norm, time_norm, income_norm |
| 52-60 | slot0/1/2 can_afford, income_crossed_if_buy, patches_crossed_if_buy |

- **Current player**: State is always from the **player to move** perspective.
- **dtype**: `np.float32` for all channels.

---

## 4) Network Architecture

### Stem

- `conv_input`: Conv2d(56, 128, 3, padding=1), no bias if BatchNorm
- `bn_input`: BatchNorm2d(128)
- Activation: ReLU(inplace=True)

### Residual Blocks (18×)

- `ResidualBlock`: Conv(128,128,3,padding=1) → BN → [FiLM] → ReLU → Conv(128,128,3,padding=1) → BN → SE (if se_ratio>0) → ReLU + residual
- **SE**: SqueezeExcitation, `se_ratio=0.0625` → reduced = max(1, 128*0.0625) = 8 channels
- **Order**: conv1→bn1→[FiLM on conv1 output]→ReLU→conv2→bn2→SE→ReLU(residual + out)

### FiLM Application (see Section 5)

- Applied after `bn1(conv1(x))`, before ReLU: `out = out * (1 + gamma) + beta`

### Policy Head (Legacy: use_factorized_policy_head=false)

- Conv2d(128, 48, 1) → BN → ReLU
- Flatten → Linear(48*81, 2026)
- **Logits shape**: (B, 2026)
- **Mask**: `logits.masked_fill(action_mask==0, -inf)` before softmax
- **Init**: Final linear zero-init (fc.weight, fc.bias)

### Policy Head (Structured: use_factorized_policy_head=true)

- Conv2d(128, 48, 1) → BN → ReLU → shared features `p` (B, 48, 9, 9)
- **buy_map**: Conv2d(48, 24, 1) → (B, 24, 9, 9), channel = slot×8 + orient
- **patch_map**: Conv2d(48, 1, 1) → (B, 1, 9, 9)
- **pass**: Linear(mean(p), 1) → (B, 1)
- **Concat**: pass + flatten(patch_map) + flatten(buy_map) → (B, 2026)
- **Indexing**: 0=pass; 1..81=patch (pos=row×9+col); 82..2025=buy (82+(slot×8+orient)×81+pos)
- **Mask**: `logits.masked_fill(action_mask==0, -inf)` before softmax
- **Init**: buy_conv, patch_conv, pass_linear zero-init (neutral start)

### Value Head (KataGo Dual-Head)

- Conv2d(128, 48, 1) → BN → ReLU
- Linear(48*81, 512) → ReLU
- Linear(512, 1) → tanh → **value** (win-probability, range [-1, 1])
- **Score head**: Linear(512, 1) → raw scalar (no tanh) → **score_margin** (Current Player Score − Opponent Score)
- **Init**: Both fc2 and score_head zero-init (neutral start)

### Ownership Head (auxiliary)

- Conv2d(128, 48, 1) → BN → ReLU → Conv2d(48, 2, 1)
- Output (B, 2, 9, 9) logits; trained with BCE-with-logits
- **Weight**: 0.15

### Initialization

- Conv: Kaiming normal (fan_out, relu)
- Linear: Kaiming normal
- BN: weight=1, bias=0
- Value fc2, policy fc: zeros (neutral start)
- Structured policy: buy_conv, patch_conv, pass_linear zeros (neutral start)
- FiLM final layer: zeros (identity at init)

### Regularizers

- `dropout: 0.0` (no dropout)
- `weight_decay: 0.0002` (AdamW); no exclusions for norm/bias (PyTorch default)

---

## 5) FiLM Conditioning

### Global Inputs (33 channels — Full Clarity config)

From `film_input_plane_indices: [28, 29, 30, ..., 60]` (config_best.yaml):

- **Indices**: All scalar channels 28–60 (33 channels total)
- **Semantics**: current_income_norm, opponent_income_norm, position_diff, button_diff, bonus_status, pending_patches_norm, absolute_time_norm, absolute_buttons_norm, dist_to_next_income/patch (×4), pass_steps/income_marks/patches, per-slot cost/time/income norms, per-slot can_afford + buy_income/patch_crossed
- **Extraction**: `film_use_plane_mean: true` → `state[:, idx, :, :].mean(dim=(1,2))` per channel
- **Output**: FiLM MLP → (B, 2*128*18) = (B, 4608), reshaped to (B, 18, 2, 128) → gamma and beta per block

### FiLM MLP

- `Linear(33, 256)` → ReLU
- `Linear(256, 2*channels*num_res_blocks)` → output
- Final layer **zero-initialized** → γ=0, β=0 at init → identity

### Application

- **Where**: After `bn1(conv1(x))` in each ResBlock, before ReLU
- **Formula**: `out = out * (1 + gamma[:,:,None,None]) + beta[:,:,None,None]` (model.py line 61)
- **Per-block**: γ and β are (B, 128) per block
- **Clipping**: None
- **Identity example**: At init, FiLM output is 0 → `out * 1 + 0 = out`

---

## 6) Targets and Losses

### Policy Target

- **Source**: MCTS visit counts
- **policy_target_mode** (config `selfplay.policy_target_mode`): `"visits"` (default) or `"visits_temperature_shaped"` (legacy).
  - **visits**: `pi(a) = N(a) / sum(N)` — raw normalized visit counts. No temperature shaping. Decouples exploration (action selection) from supervised target.
  - **visits_temperature_shaped** (legacy): `pi ∝ count^(1/T)` via log-space; uses scheduled temperature. Same as pre-2025 behavior.
- **Action selection**: Uses `temperature_schedule` for early moves; `temperature_threshold` makes it greedy (T=0) after N moves. This affects only which move is **chosen**, not the stored target.
- **Implementation**: `ActionEncoder.create_target_policy(visit_counts, ..., mode=policy_target_mode)`.
- **Illegal actions**: Target is constructed from legal visits only; mask applied before storage. In loss, target is renormalized: `target / target.sum()`.
- **A/B experiment** (policy_target_mode): Run 25–50 iters with same seeds, sims, config except:
  - A: `policy_target_mode: visits_temperature_shaped` (legacy)
  - B: `policy_target_mode: visits` (recommended)
  Compare: ladder Elo at iter 25/50, selfplay policy_entropy/top1_prob, stability (fewer regressions). If B ≥ A with better stability, keep B as default.
- **Loss**: Cross-entropy: `-(target_norm * log_softmax(logits)).sum(dim=-1).mean()`. No label smoothing.

### Value Target (KataGo Dual-Head)

- **Value**: Binary `+1` (win), `-1` (loss), `0` (tie) from `to_move` perspective. No score_scale.
- **Score margin**: Raw integer `(current_player_score - opponent_score)` as separate target.
- **Q-value mixing** (value only): `value = (1 - q_value_weight) * z + q_value_weight * root_q` (root_q from MCTS). Score margin is never mixed.
- **Rescaling on merge**: N/A. Dual-Head uses no score_scale. Legacy HDF5 without `score_margins` is **rejected** when `score_loss_weight > 0` (replay_buffer raises RuntimeError).

### Loss Weights

- Policy: 1.0
- Value: 1.0
- Score: 0.02 (Huber loss, delta=5.0)
- Ownership: 0.15
- No dynamic schedules; fixed

### AMP (Section 6 — training)

- **Autocast**: Trainer uses `autocast(device_type="cuda", dtype=training.amp_dtype)`. bf16 → no GradScaler; float16 → GradScaler.

---

## 7) MCTS Implementation

### PUCT Formula

From `alphazero_mcts_optimized.py`:

```
effective_n = n + virtual_loss
q_value = (total_value - virtual_loss) / effective_n   if effective_n > 0
        = -fpu_reduction                                if parent node (root's children)
        = 0                                             if leaf
exploration = cpuct * prior * sqrt(n_total + 1) / (1 + n)
ucb = q_value + exploration
```

- **virtual_loss**: 3.0 (config)
- **fpu_reduction**: 0.25 (KataGo-style)
- **Q backup**: Values in [-1, 1] (win/loss); backup negates when player changes along path
- **Score utility**: MCTS uses `utility = value + score_utility_weight * score_margin` (config: 0.02) for non-terminal network evaluation

### Dirichlet Noise

- **Alpha**: From schedule (e.g. 0.10→0.08→0.05→0.03→0.015)
- **Weight (ε)**: From schedule (e.g. 0.25→0.22→0.18→0.12→0.08→0.05)
- **Mix**: `prior = (1 - ε) * prior + ε * dirichlet(alpha)`, then normalize
- **When**: Root only, after setting priors from network
- **RNG**: `_noise_rng = np.random.default_rng(seed)`; `set_noise_seed()` called per game

### Temperature (Action Selection)

- **Root only**: `visits^(1/T)` for sampling; T=0 → argmax
- **Schedule**: temperature_schedule (e.g. 1.0→0.8→0.5→0.3→0.25); `temperature_threshold` moves after which T=0

### "Always Promote"

- `win_rate_threshold: 0.0`, `max_consecutive_rejections: 1` → effectively always accept
- Iteration 0: always accept (bootstrap)
- **config_best**: `games_vs_best: 0`, `games_vs_pure_mcts: 0` → evaluation disabled; every model auto-accepted

### Resign / Draw / Early Termination

- **Resign**: None
- **Draw**: Handled by tie-breaker (TIE_PLAYER)
- **Early termination**: `max_game_length: 200`; game stops after 200 moves

---

## 8) Self-Play Data Format (HDF5)

### Per-Position Records

| Dataset | Shape | Dtype | Semantics |
|---------|-------|-------|-----------|
| states | (N, 56, 9, 9) | float32 | Encoded spatial state from to_move perspective (gold_v2) |
| action_masks | (N, 2026) | float32 | 0/1 legal mask |
| policies | (N, 2026) | float32 | Target policy (renormalized over legal) |
| values | (N,) | float32 | Value target (+1/-1/0 or Q-mixed) |
| score_margins | (N,) | float32 | Score margin (current_player_score − opponent_score) |
| ownerships | (N, 2, 9, 9) | float32 | Terminal ownership targets |
| slot_piece_ids | (N, 3) | int16 | Required when store_canonical_only; piece IDs for D4 at train time |

### Attributes

- `num_games`, `num_positions`
- `selfplay_complete`: True when done
- `selfplay_num_games_written`: actual games written (SELFPLAY_NUM_GAMES_ATTR)
- `selfplay_schema_version`: 2 (run_layout.SELFPLAY_SCHEMA_VERSION)
- `expected_channels`: 56 (SELFPLAY_EXPECTED_CHANNELS_ATTR; from config network.input_channels)
- `encoding_version`: "full_clarity_v1"
- `value_target_type`: "dual_head" (value=win/loss/tie, score_margin=raw integer)

### Compression

- `hdf5_compression: lzf`
- `hdf5_compression_level: 0` (ignored for LZF)

### Completion / Resume

- Resume checks `selfplay_complete` and `selfplay_num_games_written >= expected`. Legacy fallback: `num_games >= expected` or `n_pos >= expected*20`.
- **Cleanup always discards** uncommitted staging. `cleanup_staging()` removes any `staging/iter_*` without a committed marker. Complete selfplay in staging is **not** preserved; interrupted iterations always restart from self-play.

---

## 9) Replay Buffer Behavior

### max_size / min_size

- **max_size**: 300,000 **positions** (config_best; 56ch ~30KB/pos ⇒ ~9GB loaded; 500k would ~15GB OOM on 16GB RAM)
- **min_size**: 4,000 (config_best). When `total_positions < min_size`, training uses current iteration only (no merge).
- **Unit**: Positions (samples), not games or bytes
- **Positions per iteration**: With `store_canonical_only: true`, 1 pos/move ⇒ 900 games × ~42 ≈ 37.8k. No 8× D4 pre-storage; D4 applied at train time.

### Eviction

- **By iteration**: Keep last `window_iterations` iterations (schedule: 8→8@80→11@140→12@200→14@300→15@400)
- **By size**: If `total_positions > max_size`, subsample to `target_total = min(total, max_size)`
- **When**: Window eviction at `add_iteration()` and `finalize_iteration_for_commit()`; subsampling at `get_training_data()`

### Sampling Distribution

- **newest_fraction > 0**: `newest_fraction` of samples from newest `_recency_window` fraction of buffer; rest proportional to size
- **newest_fraction = 0**: Proportional by size
- **config_best**: `newest_fraction: 0.25`. When league disabled, `_recency_window=0` → fallback: newest = last iteration only

### Prioritization

- **prioritized: false** — no prioritization; uniform/proportional within groups

### Legacy HDF5 (Dual-Head)

- **Rejection**: When `training.score_loss_weight > 0`, replay buffer **refuses** to merge HDF5 files without `score_margins`. Legacy score_tanh-only data cannot be used. Wipe `runs/<run_id>/staging` and `committed` and restart to regenerate with Dual-Head schema.

---

## 10) Training Loop (PyTorch, No Composer)

### Batch Size

- **Global**: 1024 (config `training.batch_size`)
- **Per-GPU**: Same (single GPU)
- **Gradient accumulation**: None

### Optimizer

- AdamW: `lr`, `weight_decay=0.0002`, `betas=(0.9, 0.999)`, `eps=1e-8`
- No param groups for excluding norm/bias from weight decay

### LR Schedule

- **Type**: `cosine_warmup_per_phase` (phases from `iteration.lr_schedule`; warmup once per phase, no restart each iter)
- **base_lr**: From `iteration.lr_schedule` (0.0016 @0, 0.0008 @80, 0.0004 @200)
- **warmup_steps**: min(50, phase_total//10) per phase
- **min_lr**: 0.000032 (~50:1 ratio with base)
- **Step**: **Optimizer step** (batch), not iteration
- **Scope**: Per phase; cosine decay within each phase

### Gradient Clipping

- **max_grad_norm**: 1.0
- Applied after `unscale_` (if scaler) or before `optimizer.step()`

### D4 at Train Time (dynamic)

- **training.d4_augmentation: dynamic**: Dataset applies random D4 transform per sample in `__getitem__`. No pre-augmented storage; `slot_piece_ids` from HDF5 enable correct piece-aware transforms. Effective 8× augmentation without 8× buffer size.

### EMA / SWA / Checkpoint Averaging

- **EMA** (Exponential Moving Average): When `training.ema.enabled=true`, Polyak averaging after every optimizer step. Config: `decay`, `use_for_selfplay`, `use_for_eval`, `save_checkpoint`. Checkpoint `model_state_dict` = EMA when use_for_selfplay (production actor). `train/ema_l2_diff` logged every 100 steps.

### Mixed Precision

- **Autocast**: Trainer honors `training.amp_dtype`: bfloat16 → `autocast(dtype=torch.bfloat16)`, float16 → `autocast(dtype=torch.float16)`.
- **GradScaler**: Disabled when amp_dtype=bfloat16 (bf16 needs no loss scaling). Enabled for float16.
- **TF32**: `training.allow_tf32` (or `inference.allow_tf32`) controls `matmul.allow_tf32` and `cudnn.allow_tf32`.
- **Matmul precision**: `training.matmul_precision` or `inference.matmul_precision` if set.
- **Startup log**: `[TRAINING AMP] use_amp=... autocast_dtype=... scaler_enabled=... allow_tf32=...`
- **Fused optimizer**: Not explicitly used
- **DataLoader**: `num_workers=0`, `pin_memory` from `config["hardware"]["pin_memory"]` (true in config_best), `batch_size=None` (BatchIndexSampler). `persistent_workers` is only used when `num_workers > 0`, so with num_workers=0 it is effectively off. Config `num_data_workers` / `prefetch_factor` are **removed** — in-memory dataset intentionally uses single process (multiprocess would pickle-copy arrays and add latency).

---

## 11) Eval (Frozen Ladder)

**Note**: Frozen ladder is a **standalone tool** (`python tools/run_frozen_ladder.py --run-dir runs/<id> --iteration N`). It is **not** invoked by the main training loop. Run manually at iter 25, 50, 75, etc. for Elo ladder evaluation.

### Opponent Selection (run_frozen_ladder.py)

- **iter_1** (anchor_permanent): 100 games
- **iter_(N-25)** (anchor_previous): 100 games, if N≥26
- **Rolling**: Up to 3 from {N-50, N-75, N-100} that exist; 50 games base (`games_rolling_initial`); if WR ∈ [wr_ambiguous_lo, wr_ambiguous_hi] (default 0.45–0.55), +50 top-up (`games_rolling_topup`)

### Missing Checkpoint

- Skip opponent; log warning; record in `skipped_opponents`

### Seed / Color

- `build_eval_schedule()`: paired eval (seed, model_plays_first); 50/50 color balance per batch

### Pinned Settings

- `simulations`: from `frozen_ladder.simulations` (256 in config_best)
- `temperature`: 0
- `root_noise_weight`: 0
- `add_noise`: False in Evaluator

### Elo / Wilson CI

- **Elo**: `wr_to_elo_diff(wr) = 400 * log10(wr/(1-wr))`, WR clamped to [1e-6, 1-1e-6]
- **Wilson CI**: `wilson_ci(wins, n, z=1.96)` for 95% CI

---

## 12) Logging / Telemetry

### Metrics

| Metric | Where | Frequency | Source |
|--------|-------|-----------|--------|
| train/total_loss, policy_loss, value_loss, policy_accuracy, policy_entropy | train_epoch | Every 10 steps | Training batch |
| train/grad_norm | train_epoch | Every 10 steps | Training |
| train/learning_rate | train_epoch | Every 10 steps | Scheduler |
| val/* | train_epoch | Every 200 steps | Validation |
| eval/win_rate_vs_best, score_margin | main | Per iter | Eval vs best |
| selfplay/policy_entropy, top1_prob, game_length, redundancy | main | Per iter | Self-play stats |
| iter/* rollups | iteration_XXX.json | Per iter | Avg of epoch metrics (total_loss, policy_accuracy, etc.) |

### Definitions

- **policy_top1_acc**: `(pred_actions == target_actions).float().mean()` where pred = argmax(logits), target = argmax(target_policy)
- **policy_top5_acc**: Target in top-5 predicted actions
- **policy_entropy**: `-(probs * log(probs)).sum(dim=-1).mean()` over softmax(logits)
- **perplexity**: `exp(clip(entropy, 0, 50))` (from `tools/analyze_iteration_metrics.py`; policy entropy over softmax(logits))
- **redundancy**: `1 - unique_positions / move_count` (0 = all unique, 1 = all duplicates)
- **value calibration**: Not computed in main training loop; `tools/analyze_iteration_metrics.py --calibration` bins predictions into deciles, writes `calibration.csv`

### redundancy=0.00

- `redundancy = 1 - len(seen_state_hashes) / move_count` per game; averaged across games as `avg_redundancy`

---

## 13) Known Sharp Edges

### Config Read Only at Startup

- All config values (schedules, network, training, replay) are read at startup. Changes require restart.
- `--allow-config-mismatch` bypasses config hash check on resume.

### Silent Fallbacks / Ignored Config

- **Legacy score_scale**: OBSOLETE. Dual-Head uses no score_scale. Replay merge rejects legacy HDF5 without `score_margins` when `score_loss_weight > 0`.
- **Legacy**: `allow_legacy_state_channels: false` (default) — fail on non-56ch data; no padding.
- **Ownership**: Old HDF5 without ownerships gets `-1` sentinel; loss masked per-sample.
- **amp_dtype**: FIXED. Trainer honors `training.amp_dtype`; bf16 → no GradScaler.
- **num_data_workers**: REMOVED. In-memory dataset uses num_workers=0 by design; config knobs removed.
- **create_target_policy**: `temperature <= 0` raises `ValueError`. Selfplay clamps `policy_temp = max(self.temperature, 1e-6)` so this is never hit in practice.

### Potential Mismatches

- Policy target uses scheduled T (from temperature_schedule) only when `policy_target_mode: visits_temperature_shaped`; with `visits` mode, stored target = raw normalized counts.
- Value: Terminal uses `terminal_value_from_scores` (binary win/loss/tie); Q-mixing uses MCTS root Q for value only. Score margin is never mixed.

### Other

- **Staging always discarded**: On resume, `cleanup_staging()` removes all uncommitted staging. Interrupted iterations always restart from self-play.
- **Repair**: Crash after commit move but before run_state write → `reconcile_run_state` repairs from filesystem.
- **Graceful shutdown**: First Ctrl+C → stop after current iteration commits. Second Ctrl+C → exit immediately, unless commit is in progress (`_commit_in_progress` guard prevents corruption). `touch runs/<run_id>/STOP_AFTER_ITERATION` also triggers graceful stop; file is deleted when honored.
- **Minimal smoke test (seamless overnight)**: (1) Start run, let it reach end-of-iteration commit. (2) Ctrl+C once during late selfplay/training → must exit only after manifest exists. (3) Restart → `python tools/resume_plan.py --run-dir runs/<run_id>` shows next_iteration=last_committed+1; staging **always** deleted for partial iter (iteration restarts from self-play). (4) Same but double Ctrl+C during commit → must still exit only after manifest (covered by `test_double_sigint_during_commit_still_exits_after_manifest`).
- **Optimizer resume invariant**: When `resume_optimizer_state=true`, optimizer source MUST equal train_base (model weights). Logged as `[OPT_RESUME] source=X train_base=Y (must match)`. If mismatch, refuse load and use fresh optimizer.
- **League**: `league.enabled: false`; PFSP/anchors unused in config_best.

---

## Code Snippets (Key Formulas)

### Masking (model.py)

```python
policy_logits = policy_logits.masked_fill(action_mask == 0, float("-inf"))
```

### FiLM (ResidualBlock.forward)

```python
out = self.bn1(self.conv1(x))
if gamma is not None and beta is not None:
    out = out * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
out = F.relu(out, inplace=True)
```

### PUCT (MCTSNode.get_ucb, alphazero_mcts_optimized.py:262-272)

```python
effective_n = n + vl
if effective_n == 0:
    q_value = -fpu_reduction if self.parent else 0.0  # Root's children: FPU penalty for unexplored
else:
    q_value = (self._total_value[idx] - vl) / effective_n
exploration = cpuct * p * math.sqrt(self.n_total + 1) / (1.0 + n)
return q_value + exploration
```

### Policy Target (encoder.py ActionEncoder.create_target_policy)

**visits mode** (default): `pi(a) = N(a) / sum(N)` — raw normalized counts; temperature unused.

**visits_temperature_shaped mode** (legacy):

```python
# mode="visits_temperature_shaped" only; temperature must be > 0
logits = np.full(ACTION_SPACE_SIZE, -np.inf)
for action, count in visit_counts.items():
    if count <= 0:
        continue
    idx = self.encode_action(action)
    logits[idx] = math.log(float(count)) / temperature
valid = np.isfinite(logits)
# softmax over valid; policy /= policy.sum()
```

**Selfplay call** (selfplay_optimized.py): `create_target_policy(visit_counts, temperature=..., mode=policy_target_mode)`. For visits mode, temperature is ignored; for shaped mode, `policy_temp = max(self.temperature, 1e-6)`.

### Value Target (value_targets.py — Dual-Head)

```python
def terminal_value_from_scores(score0, score1, winner, to_move) -> float:
    if int(score0) == int(score1):
        return 0.0
    return 1.0 if int(winner) == int(to_move) else -1.0

def value_and_score_from_scores(score0, score1, winner, to_move) -> tuple[float, float]:
    value = terminal_value_from_scores(score0, score1, winner, to_move)
    score_margin = float(score0 - score1) if to_move == 0 else float(score1 - score0)
    return value, score_margin
```

### Replay Sampling (replay_buffer.py)

- Proportional: `raw = [n * target_total / total for each entry]`, round and distribute remainder.
- Newest bias: Split into newest/rest groups; allocate `newest_fraction * target` to newest, rest proportional.

### LR Schedule (trainer.py, cosine_warmup_per_phase)

```python
# Phases from iteration.lr_schedule; warmup once per phase
phase_total_steps = iters_in_phase * total_train_steps
warmup_steps = min(warmup_steps_cfg, max(1, phase_total_steps // 10))
min_lr_ratio = min_lr / iter_lr

def lr_lambda(step):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, phase_total_steps - warmup_steps)
    progress = min(progress, 1.0)
    cosine_decay = 0.5 * (1.0 + cos(pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
```
