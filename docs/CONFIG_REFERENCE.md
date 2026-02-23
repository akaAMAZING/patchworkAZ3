# Config Reference — Full Wiring Breakdown

Every parameter in `configs/config_best.yaml`, with location and code snippet showing where/how it is wired.

---

## Top-level

### `seed: 42`
**Status:** WIRED  
**Purpose:** Global random seed for reproducibility.

```python
# src/training/main.py (multiple)
"seed": int(self.config.get("seed", 42))
# src/training/trainer.py
seed=config.get("seed", 42) + iteration
# src/training/evaluation.py
base_seed = int(self.config.get("seed", 42))
# src/training/selfplay_optimized_integration.py
base_seed = int(self.config.get("seed", 42))
# src/training/league.py
self._rng = random.Random(config.get("seed", 42))
```

---

### `deterministic: false`
**Status:** WIRED  
**Purpose:** Enables torch deterministic algorithms when true (slower).

```python
# src/network/gpu_inference_server.py:39, 85
if not bool(config.get("deterministic", False)):
    torch.backends.cudnn.benchmark = True
```

---

## hardware

### `hardware.device: cuda`
**Status:** WIRED

```python
# src/training/main.py:412
if self.config["hardware"]["device"] == "cuda" and torch.cuda.is_available():
    self.device = torch.device("cuda")
# src/training/selfplay_optimized.py:279, 692
hw = (config.get("hardware", {}) or {}).get("device", "cpu")
```

---

### `hardware.pin_memory: true`
**Status:** WIRED

```python
# src/training/trainer.py:1007, 1016
pin_memory=hw.get("pin_memory", config["hardware"]["pin_memory"]),
```

---

### `hardware.persistent_workers: true`
**Status:** WIRED (effective only when `num_workers` > 0; trainer uses 0)

```python
# src/training/trainer.py:1008, 1017
persistent_workers=bool(hw.get("persistent_workers", False)) if num_workers > 0 else False,
```

---

## data

### `data.allow_legacy_state_channels: false`
**Status:** WIRED  
**Purpose:** When true, pad/truncate state channels on mismatch instead of raising.

```python
# src/training/replay_buffer.py:404-406
allow_legacy = bool(
    (self.config.get("data", {}) or {}).get("allow_legacy_state_channels", False)
)
if allow_legacy:
    # pad or truncate st_raw to expected_channels
```

---

## network

### `network.input_channels: 56`
**Status:** WIRED

```python
# src/network/model.py:557
input_channels=int(net_config.get("input_channels", DEFAULT_INPUT_CHANNELS)),
# src/training/replay_buffer.py:231
expected_channels = int(self.config.get("network", {}).get("input_channels", 25))
# src/training/selfplay_optimized.py:663, selfplay_optimized_integration.py:486
channels = int((config.get("network") or {}).get("input_channels", 16))
```

---

### `network.num_res_blocks: 18`
**Status:** WIRED

```python
# src/network/model.py:558
num_res_blocks=int(net_config["num_res_blocks"]),
```

---

### `network.channels: 128`
**Status:** WIRED

```python
# src/network/model.py:559
channels=int(net_config["channels"]),
```

---

### `network.policy_channels: 48`
**Status:** WIRED

```python
# src/network/model.py:560
policy_channels=int(net_config["policy_channels"]),
```

---

### `network.policy_hidden: 512`
**Status:** WIRED

```python
# src/network/model.py:561
policy_hidden=int(net_config.get("policy_hidden", 256)),
```

---

### `network.use_factorized_policy_head: true`
**Status:** WIRED

```python
# src/network/model.py:562
use_factorized_policy_head=bool(net_config.get("use_factorized_policy_head", True)),
```

---

### `network.value_channels: 48`
**Status:** WIRED

```python
# src/network/model.py:563
value_channels=int(net_config["value_channels"]),
```

---

### `network.value_hidden: 512`
**Status:** WIRED

```python
# src/network/model.py:564
value_hidden=int(net_config["value_hidden"]),
```

---

### `network.max_actions: 2026`
**Status:** WIRED

```python
# src/network/model.py:565
max_actions=int(net_config.get("max_actions", DEFAULT_MAX_ACTIONS)),
```

---

### `network.dropout: 0.0`
**Status:** WIRED

```python
# src/network/model.py:568
dropout=float(net_config["dropout"]),
```

---

### `network.weight_decay: 0.0002`
**Status:** WIRED

```python
# src/network/model.py:407, trainer.py
weight_decay = self.config["network"]["weight_decay"]
```

---

### `network.use_batch_norm: true`
**Status:** WIRED

```python
# src/network/model.py:566
use_batch_norm=bool(net_config["use_batch_norm"]),
```

---

### `network.activation: relu`
**Status:** NOT WIRED — model uses ReLU implicitly; no activation switch implemented.

---

### `network.se_ratio: 0.0625`
**Status:** WIRED

```python
# src/network/model.py:567
se_ratio=float(net_config["se_ratio"]),
```

---

### `network.ownership_channels: 48`
**Status:** WIRED

```python
# src/network/model.py:569
ownership_channels=int(net_config.get("ownership_channels", 0)),
```

---

### `network.use_film: true`
**Status:** WIRED

```python
# src/network/model.py:570
use_film=bool(net_config.get("use_film", False)),
```

---

### `network.film_hidden: 256`
**Status:** WIRED

```python
# src/network/model.py:571
film_hidden=int(net_config.get("film_hidden", 256)),
```

---

### `network.film_input_plane_indices: [28, ..., 60]`
**Status:** WIRED

```python
# src/network/model.py:551-552
film_indices = net_config.get("film_input_plane_indices", [])
film_input_plane_indices=film_indices,
```

---

### `network.film_use_plane_mean: true`
**Status:** WIRED

```python
# src/network/model.py:573
film_use_plane_mean=bool(net_config.get("film_use_plane_mean", True)),
```

---

### `network.film_per_block: true`
**Status:** WIRED

```python
# src/network/model.py:574
film_per_block=bool(net_config.get("film_per_block", True)),
```

---

## training

### `training.optimizer: adamw`
**Status:** WIRED

```python
# src/training/trainer.py:404-405
opt_name = config["optimizer"].lower()
# ... creates AdamW, Adam, or SGD
```

---

### `training.learning_rate: 0.0016`
**Status:** WIRED

```python
# src/training/trainer.py:415-416, 478-479, 526
lr = config["learning_rate"]
base_lr = config["learning_rate"]
iter_lr = config["learning_rate"]
```

---

### `training.lr_schedule: cosine_warmup_per_phase`
**Status:** WIRED

```python
# src/training/trainer.py:484
schedule_type = config["lr_schedule"]
if schedule_type == "cosine_warmup_per_phase":
    # uses iteration.lr_schedule for phase LRs
```

---

### `training.warmup_steps: 50`
**Status:** WIRED

```python
# src/training/trainer.py:487, 536
warmup_steps_cfg = config["warmup_steps"]
warmup_steps_cfg = config.get("warmup_steps", 200)
```

---

### `training.d4_augmentation: dynamic`
**Status:** WIRED

```python
# src/training/trainer.py:103
self.d4_dynamic = str(train_cfg.get("d4_augmentation", "store")).lower() == "dynamic"
```

---

### `training.resume_optimizer_state: false`
**Status:** WIRED

```python
# src/training/trainer.py:386
resume_opt = bool(train_config.get("resume_optimizer_state", False))
```

---

### `training.resume_scheduler_state: false`
**Status:** WIRED

```python
# src/training/trainer.py:387
resume_sched = bool(train_config.get("resume_scheduler_state", False))
```

---

### `training.resume_scaler_state: false`
**Status:** WIRED

```python
# src/training/trainer.py:388
resume_scaler = bool(train_config.get("resume_scaler_state", resume_opt))
```

---

### `training.min_lr: 0.000032`
**Status:** WIRED

```python
# src/training/trainer.py:479, 538
min_lr = config["min_lr"]
min_lr = config.get("min_lr", iter_lr / 50)
```

---

### `training.batch_size: 1024`
**Status:** WIRED

```python
# src/training/trainer.py:972
batch_size = int(config["training"]["batch_size"])
```

---

### `training.use_amp: true`
**Status:** WIRED

```python
# src/training/trainer.py:338
requested_amp = bool(train_config.get("use_amp", False))
```

---

### `training.amp_dtype: bfloat16`
**Status:** WIRED

```python
# src/training/trainer.py:343
amp_dtype_str = str(train_config.get("amp_dtype", "bfloat16")).lower()
self._autocast_dtype = torch.bfloat16 if amp_dtype_str == "bfloat16" else torch.float16
```

---

### `training.policy_loss_weight: 1.0`
**Status:** WIRED

```python
# src/training/trainer.py:346
self.policy_weight = train_config["policy_loss_weight"]
```

---

### `training.value_loss_weight: 1.0`
**Status:** WIRED

```python
# src/training/trainer.py:347
self.value_weight = train_config["value_loss_weight"]
```

---

### `training.score_loss_weight: 0.02`
**Status:** WIRED

```python
# src/training/trainer.py:348, replay_buffer.py:343
self.score_loss_weight = float(train_config.get("score_loss_weight", 0.02))
```

---

### `training.ownership_loss_weight: 0.15`
**Status:** WIRED

```python
# src/training/trainer.py:349
self.ownership_weight = float(train_config.get("ownership_loss_weight", 0.0))
```

---

### `training.epochs_per_iteration: 2`
**Status:** WIRED

```python
# src/training/trainer.py:973
num_epochs = config["training"]["epochs_per_iteration"]
```

---

### `training.val_split: 0.08`
**Status:** WIRED

```python
# src/training/trainer.py:966, 977
val_split=config["training"]["val_split"],
```

---

### `training.val_frequency: 200`
**Status:** WIRED

```python
# src/training/trainer.py:682
if val_loader is not None and self.global_step % self.config["training"]["val_frequency"] == 0:
```

---

### `training.checkpoint_frequency: 10000`
**Status:** WIRED

```python
# src/training/trainer.py:697
if self.global_step % self.config["training"]["checkpoint_frequency"] == 0:
```

---

### `training.keep_last_n_checkpoints: 200`
**Status:** WIRED

```python
# src/training/trainer.py:831
keep_n = self.config["training"]["keep_last_n_checkpoints"]
```

---

### `training.save_best: true`
**Status:** WIRED

```python
# src/training/main.py:632, 1810
save_best = bool((self.config.get("training", {}) or {}).get("save_best", True))
if accepted and ... and save_best:
    best_dest = self._atomic_copy_checkpoint(...)
```

---

### `training.best_metric: val_loss`
**Status:** NOT WIRED — only used conceptually with early_stopping.

---

### `training.early_stopping.enabled: false`
**Status:** NOT WIRED — early stopping not implemented.

---

### `training.early_stopping.patience: 5`
**Status:** NOT WIRED

---

### `training.early_stopping.min_delta: 0.001`
**Status:** NOT WIRED

---

### `training.max_grad_norm: 1.0`
**Status:** WIRED

```python
# src/training/trainer.py:350
self.max_grad_norm = train_config["max_grad_norm"]
```

---

### `training.allow_tf32: true`
**Status:** WIRED

```python
# src/training/trainer.py:326-328
if device.type == "cuda":
    allow_tf32 = bool(train_config.get("allow_tf32", True))
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
```

---

### `training.matmul_precision: high`
**Status:** WIRED

```python
# src/training/trainer.py:329-331
matmul_prec = str(train_config.get("matmul_precision", "high")).lower()
if matmul_prec in ("high", "highest", "medium"):
    torch.set_float32_matmul_precision(matmul_prec)
```

---

### `training.ema.enabled: true`
**Status:** WIRED

```python
# src/training/trainer.py:368
self.ema_enabled = bool(ema_cfg.get("enabled", False))
# src/network/model.py:684
if "ema" in config.get("training", {}):
```

---

### `training.ema.decay: 0.999`
**Status:** WIRED

```python
# src/training/trainer.py:360
self.ema_decay = float(ema_cfg.get("decay", 0.999))
```

---

### `training.ema.use_for_selfplay: true`
**Status:** WIRED

```python
# src/training/trainer.py:361
self.ema_use_for_selfplay = bool(ema_cfg.get("use_for_selfplay", True))
```

---

### `training.ema.use_for_eval: true`
**Status:** WIRED

```python
# src/training/trainer.py:362
self.ema_use_for_eval = bool(ema_cfg.get("use_for_eval", True))
```

---

### `training.ema.save_checkpoint: true`
**Status:** WIRED

```python
# src/network/model.py:685
ema_cfg = config["training"]["ema"]
# save_checkpoint controls whether EMA state is saved in checkpoint
```

---

## selfplay

### `selfplay.api_url: http://127.0.0.1:8000`
**Status:** NOT WIRED — workers use local GPU server queues, not HTTP.

---

### `selfplay.inference_backend: null`
**Status:** WIRED — null → use GPU server; other values select alternate backends.

```python
# src/training/selfplay_optimized.py:684, 790
backend = (config.get("selfplay", {}) or {}).get("inference_backend")
```

---

### `selfplay.api_timeout: 120`
**Status:** WIRED

```python
# src/training/selfplay_optimized.py:274
timeout_s=float(sp.get("api_timeout", 60.0)),
```

---

### `selfplay.api_retry_attempts: 3`
**Status:** WIRED

```python
# src/training/selfplay_optimized.py:275
retry_attempts=int(sp.get("api_retry_attempts", 3)),
```

---

### `selfplay.num_workers: 14`
**Status:** WIRED

```python
# src/training/selfplay_optimized_integration.py:86, 180, 293
num_workers = int(self.config.get("selfplay", {}).get("num_workers", 1))
```

---

### `selfplay.max_game_length: 200`
**Status:** WIRED

```python
# src/training/selfplay_optimized.py:237
self.max_game_length = int(sp.get("max_game_length", 200))
# src/training/evaluation.py:648
max_moves = int(self.config["selfplay"]["max_game_length"])
```

---

### `selfplay.games_per_iteration: 120`
**Status:** WIRED — base value; `iteration.games_schedule` overrides per iteration.

```python
# src/training/main.py:333, selfplay_optimized_integration.py:652
base = int(config.get("selfplay", {}).get("games_per_iteration", 400))
num_games = self.config["selfplay"]["games_per_iteration"]
```

---

### `selfplay.store_canonical_only: true`
**Status:** WIRED

```python
# src/training/selfplay_optimized.py:227
self.store_canonical_only = bool(sp.get("store_canonical_only", True))
```

---

### `selfplay.augmentation: d4`
**Status:** WIRED

```python
# src/training/selfplay_optimized.py:217-224
if "augmentation" in sp:
    self.augmentation = str(sp["augmentation"])
...
if self.augmentation not in ("none", "flip", "d4"):
    self.augmentation = "flip"
```

---

### `selfplay.mcts.simulations: 320`
**Status:** WIRED — base; `iteration.mcts_schedule` overrides; `bootstrap.mcts_simulations` for iter 0.

```python
# src/mcts/alphazero_mcts_optimized.py:822
simulations=int(mcts_cfg.get("simulations", 800)),
# src/training/selfplay_optimized_integration.py:679-682, 700-704
config["selfplay"]["mcts"]["simulations"] = entry["simulations"]
# bootstrap override for iter 0
```

---

### `selfplay.mcts.parallel_leaves: 24`
**Status:** WIRED

```python
# src/mcts/alphazero_mcts_optimized.py:823
parallel_leaves=int(mcts_cfg.get("parallel_leaves", 32)),
```

---

### `selfplay.mcts.score_utility_weight: 0.02`
**Status:** WIRED

```python
# src/mcts/alphazero_mcts_optimized.py:831
score_utility_weight=float(mcts_cfg.get("score_utility_weight", 0.02)),
```

---

### `selfplay.mcts.cpuct: 1.47`
**Status:** WIRED — base; `iteration.cpuct_schedule` overrides.

```python
# src/mcts/alphazero_mcts_optimized.py:824
cpuct=float(mcts_cfg.get("cpuct", 1.5)),
# src/training/main.py:350-354
cpuct = float(_step_schedule_lookup(cpuct_sched, iteration, "cpuct", cpuct_base))
config["selfplay"]["mcts"]["cpuct"] = cpuct
```

---

### `selfplay.mcts.temperature: 1.0`
**Status:** WIRED — base; `iteration.temperature_schedule` overrides.

```python
# src/mcts/alphazero_mcts_optimized.py:825
temperature=float(mcts_cfg.get("temperature", 1.0)),
# src/training/selfplay_optimized_integration.py:672-675
config["selfplay"]["mcts"]["temperature"] = entry["temperature"]
```

---

### `selfplay.mcts.temperature_threshold: 15`
**Status:** WIRED

```python
# src/mcts/alphazero_mcts_optimized.py:826
temperature_threshold=int(mcts_cfg.get("temperature_threshold", 15)),
```

---

### `selfplay.mcts.root_dirichlet_alpha: 0.10`
**Status:** WIRED — base; `iteration.dirichlet_alpha_schedule` overrides.

```python
# src/mcts/alphazero_mcts_optimized.py:827
root_dirichlet_alpha=float(mcts_cfg.get("root_dirichlet_alpha", 0.3)),
# src/training/selfplay_optimized_integration.py:686-689
config["selfplay"]["mcts"]["root_dirichlet_alpha"] = entry["alpha"]
```

---

### `selfplay.mcts.root_noise_weight: 0.25`
**Status:** WIRED — base; `iteration.noise_weight_schedule` overrides.

```python
# src/mcts/alphazero_mcts_optimized.py:828
root_noise_weight=float(mcts_cfg.get("root_noise_weight", 0.25)),
# src/training/selfplay_optimized_integration.py:693-696
config["selfplay"]["mcts"]["root_noise_weight"] = entry["weight"]
```

---

### `selfplay.mcts.virtual_loss: 3.0`
**Status:** WIRED

```python
# src/mcts/alphazero_mcts_optimized.py:829
virtual_loss=float(mcts_cfg.get("virtual_loss", 3.0)),
```

---

### `selfplay.mcts.num_threads: 1`
**Status:** NOT WIRED — MCTSConfig has no `num_threads`; MCTS uses `parallel_leaves`.

---

### `selfplay.mcts.fpu_reduction: 0.25`
**Status:** WIRED

```python
# src/mcts/alphazero_mcts_optimized.py:830
fpu_reduction=float(mcts_cfg.get("fpu_reduction", 0.25)),
```

---

### `selfplay.playout_cap_randomization.enabled: false`
**Status:** WIRED

```python
# src/training/selfplay_optimized.py:250
pcr = sp.get("playout_cap_randomization", {}) or {}
self.pcr_enabled = bool(pcr.get("enabled", False))
```

---

### `selfplay.playout_cap_randomization.cap_fraction: 0.00`
**Status:** WIRED

```python
# src/training/selfplay_optimized.py:252
self.pcr_fraction = float(pcr.get("cap_fraction", 0.25))
```

---

### `selfplay.playout_cap_randomization.fast_probability: 0.00`
**Status:** WIRED

```python
# src/training/selfplay_optimized.py:253
self.pcr_fast_prob = float(pcr.get("fast_probability", 0.0))
```

---

### `selfplay.bootstrap.use_pure_mcts: true`
**Status:** WIRED

```python
# src/training/selfplay_optimized_integration.py:700-701
if iteration == 0 and bootstrap_cfg.get("use_pure_mcts"):
```

---

### `selfplay.bootstrap.mcts_simulations: 192`
**Status:** WIRED

```python
# src/training/selfplay_optimized.py:228, selfplay_optimized_integration.py:702-704
self.bootstrap_sims = int((sp.get("bootstrap", {}) or {}).get("mcts_simulations", 32))
config["selfplay"]["mcts"]["simulations"] = bootstrap_sims
```

---

### `selfplay.bootstrap.games: 900`
**Status:** WIRED — used for iteration 0.

```python
# src/training/main.py:331, selfplay_optimized_integration.py:648
return int(config.get("selfplay", {}).get("bootstrap", {}).get("games", 200))
return self.config["selfplay"]["bootstrap"]["games"]
```

---

### `selfplay.stream_to_disk: true`
**Status:** NOT WIRED — return mode (dict vs shard) controlled elsewhere.

---

### `selfplay.hdf5_compression: lzf`
**Status:** WIRED

```python
# src/training/selfplay_optimized_integration.py:502-508
compression = sp_cfg.get("hdf5_compression", "lzf")
if compression in ("none", "", "null"):
    compression = None
```

---

### `selfplay.hdf5_compression_level: 0`
**Status:** WIRED

```python
# src/training/selfplay_optimized_integration.py:505
compression_level = int(sp_cfg.get("hdf5_compression_level", 4))
compression_opts = compression_level if compression == "gzip" else None
```

---

### `selfplay.write_buffer_positions: 8192`
**Status:** NOT WIRED

---

### `selfplay.policy_target_mode: visits`
**Status:** WIRED

```python
# src/training/selfplay_optimized.py:233-235
self.policy_target_mode = str(sp.get("policy_target_mode", "visits")).lower()
if self.policy_target_mode not in ("visits", "visits_temperature_shaped"):
    self.policy_target_mode = "visits"
```

---

### `selfplay.q_value_weight: 0.13`
**Status:** WIRED — base; `iteration.q_value_weight_schedule` overrides.

```python
# src/training/main.py:344-348
qvw_base = float(sp.get("q_value_weight", 0.0))
qvw = float(_step_schedule_lookup(qvw_sched, iteration, "q_value_weight", qvw_base))
config["selfplay"]["q_value_weight"] = qvw
# src/training/selfplay_optimized.py:246
self.q_value_weight = float(sp.get("q_value_weight", 0.0))
```

---

## replay_buffer

### `replay_buffer.max_size: 300000`
**Status:** WIRED

```python
# src/training/replay_buffer.py:46
self.max_size = int(rb_config.get("max_size", 500_000))
```

---

### `replay_buffer.min_size: 4000`
**Status:** WIRED

```python
# src/training/replay_buffer.py:47
self.min_size = int(rb_config.get("min_size", 8_000))
```

---

### `replay_buffer.window_iterations: 8`
**Status:** WIRED — base; `iteration.window_iterations_schedule` overrides.

```python
# src/training/replay_buffer.py:48
self.window_size = int(rb_config.get("window_iterations", 5))
# src/training/main.py:338, 819
self.replay_buffer.window_size = _get_window_iterations_for_iteration(self.config, iteration)
```

---

### `replay_buffer.newest_fraction: 0.25`
**Status:** WIRED

```python
# src/training/replay_buffer.py:49
self.newest_fraction = float(rb_config.get("newest_fraction", 0.0))
```

---

### `replay_buffer.prioritized: false`
**Status:** NOT WIRED — prioritized replay not implemented.

---

### `replay_buffer.storage_format: hdf5`
**Status:** NOT WIRED — only HDF5 supported.

---

### `replay_buffer.compression: lzf`
**Status:** WIRED

```python
# src/training/replay_buffer.py:347
compression = rb.get("compression", "lzf") or "lzf"
```

---

### `replay_buffer.compression_level: 0`
**Status:** WIRED

```python
# src/training/replay_buffer.py:348
comp_level = int(rb.get("compression_level", 0))
comp_opts = comp_level if compression == "gzip" else None
```

---

## evaluation

### `evaluation.lock_eval_cpuct_to_selfplay: true`
**Status:** WIRED

```python
# src/training/main.py:356-358
lock_eval = bool(eval_cfg.get("lock_eval_cpuct_to_selfplay", True))
if lock_eval and "eval_mcts" in eval_cfg:
    config["evaluation"]["eval_mcts"]["cpuct"] = cpuct
```

---

### `evaluation.games_vs_best: 0`
**Status:** WIRED

```python
# src/training/main.py:1618, 1587
num_games = int(eval_cfg.get("games_per_eval") or eval_cfg.get("games_vs_pure_mcts", 0))
# games_vs_best used for gate/games_per_eval fallback
```

---

### `evaluation.games_vs_pure_mcts: 0`
**Status:** WIRED

```python
# src/training/main.py:1196
num_games = int(eval_cfg.get("games_per_eval") or eval_cfg.get("games_vs_pure_mcts", 0))
```

---

### `evaluation.eval_progress_interval: 10`
**Status:** WIRED

```python
# src/training/evaluation.py:319-320
if progress_interval is None:
    progress_interval = int(self.config.get("evaluation", {}).get("eval_progress_interval", 10))
```

---

### `evaluation.paired_eval: true`
**Status:** WIRED

```python
# src/training/evaluation.py:396, 505
paired_eval = bool(self.config.get("evaluation", {}).get("paired_eval", True))
```

---

### `evaluation.skip_pure_mcts_after_iter: 0`
**Status:** WIRED

```python
# src/training/main.py:1200
skip_after = self.config["evaluation"].get("skip_pure_mcts_after_iter", None)
run_pure_mcts = int(iteration) <= int(skip_after) if skip_after is not None else True
```

---

### `evaluation.eval_baselines` (type, simulations)
**Status:** WIRED

```python
# src/training/evaluation.py:363-367
baselines = self.config.get("evaluation", {}).get("eval_baselines", []) or []
entry = next((b for b in baselines if b.get("type") == "pure_mcts"), None)
baseline_sims = int(entry.get("simulations", 200))
```

---

### `evaluation.win_rate_threshold: 0.0`
**Status:** WIRED

```python
# src/training/main.py:1600
win_rate_threshold = self.config["evaluation"]["win_rate_threshold"]
```

---

### `evaluation.anti_regression_floor: 0.0`
**Status:** WIRED

```python
# src/training/main.py:1603, 1657
self.config.get("evaluation", {}).get("anti_regression_floor", 0.35)
```

---

### `evaluation.max_consecutive_rejections: 1`
**Status:** WIRED

```python
# src/training/main.py:1392, 1550
max_rejections = self.config["evaluation"].get("max_consecutive_rejections", 5)
```

---

### `evaluation.sprt.enabled`, `.p0`, `.p1`, `.alpha`, `.beta`, `.min_games`, `.max_games`
**Status:** WIRED

```python
# src/training/main.py:1235-1239
sprt_cfg = self.config.get("evaluation", {}).get("sprt", {}) or {}
if bool(sprt_cfg.get("enabled", False)):
    prev_best_results = self.evaluator.evaluate_sprt(..., sprt_cfg)
# src/training/evaluation.py:461-466
p0, p1, alpha, beta, min_games, max_games from sprt_cfg
```

---

### `evaluation.micro_gate.enabled`, `.start_games`, `.step_games`, `.max_games`, `.anti_regression_threshold`
**Status:** WIRED

```python
# src/training/main.py:1236-1243
micro_cfg = self.config.get("evaluation", {}).get("micro_gate", {}) or {}
prev_best_results = self._run_micro_gate(..., micro_cfg)
# src/training/main.py:1293-1297
start_games = int(micro_cfg.get("start_games", 20))
max_games = int(micro_cfg.get("max_games", 80))
step_games = int(micro_cfg.get("step_games", 20))
threshold = float(micro_cfg.get("anti_regression_threshold", 0.45))
```

---

### `evaluation.eval_mcts.simulations`, `.temperature`, `.cpuct`
**Status:** WIRED

```python
# src/training/evaluation.py:354-359, 483-488
eval_config = self.config["evaluation"]["eval_mcts"]
model_mcts.config.simulations = int(eval_config["simulations"])
model_mcts.config.cpuct = float(eval_config["cpuct"])
```

---

### `evaluation.frozen_ladder.enabled`, `.simulations`, `.temperature`, `.root_noise_weight`, `.games_anchor`, `.games_rolling_initial`, `.games_rolling_topup`, `.wr_ambiguous_lo`, `.wr_ambiguous_hi`
**Status:** WIRED — `main` passes through to metadata; ladder logic in `tools/run_frozen_ladder.py`.

```python
# src/training/main.py:507, 540
fladder = (eval_cfg.get("frozen_ladder") or {}) or {}
# ... writes "frozen_ladder": {...} to metadata
# tools/run_frozen_ladder.py:100-107
ladder_cfg = (config.get("evaluation", {}) or {}).get("frozen_ladder", config.get("frozen_ladder", {})) or {}
sims = int(ladder_cfg.get("simulations", 192))
games_anchor = int(ladder_cfg.get("games_anchor", 100))
games_rolling_base = int(ladder_cfg.get("games_rolling_initial", ladder_cfg.get("games_rolling_base", 50)))
games_rolling_topup = int(ladder_cfg.get("games_rolling_topup", 50))
wr_ambiguous_lo = float(ladder_cfg.get("wr_ambiguous_lo", 0.45))
wr_ambiguous_hi = float(ladder_cfg.get("wr_ambiguous_hi", 0.55))
```

---

### `evaluation.elo.enabled`, `.initial_rating`, `.k_factor`
**Status:** WIRED

```python
# src/training/main.py:429-430
initial_rating=self.config["evaluation"]["elo"]["initial_rating"],
k_factor=self.config["evaluation"]["elo"]["k_factor"],
# main.py:1193
if self.config["evaluation"]["elo"]["enabled"] and ...
```

---

## logging

### `logging.log_level: INFO`
**Status:** WIRED

```python
# src/training/main.py:403-405
log_cfg = self.config.get("logging", {}) or {}
log_level = str(log_cfg.get("log_level", "INFO")).upper()
logging.getLogger().setLevel(getattr(logging, log_level, logging.INFO))
```

---

### `logging.tensorboard.enabled: true`
**Status:** WIRED

```python
# src/training/main.py:854-858
tb_enabled = bool((self.config.get("logging", {}) or {}).get("tensorboard", {}).get("enabled", True))
if tb_enabled:
    self.writer = SummaryWriter(...)
else:
    self.writer = _NoOpSummaryWriter()
```

---

### `logging.tensorboard.log_dir: logs/tensorboard`
**Status:** WIRED

```python
# src/training/main.py:466-468
tb_cfg = (self.config.get("logging", {}) or {}).get("tensorboard", {}) or {}
self.tb_log_dir = Path(tb_cfg.get("log_dir") or str(Path(...) / "tensorboard"))
```

---

### `logging.wandb.enabled`, `.project`, `.entity`
**Status:** NOT WIRED — Weights & Biases not integrated.

---

### `logging.log_file: logs/training.log`
**Status:** WIRED

```python
# src/training/main.py:1800-1801
self.config.get("logging", {}).get("log_file")
or str(Path(self.config.get("paths", {}).get("logs_dir", "logs")) / "training.log")
```

---

## paths

### `paths.checkpoints_dir`, `paths.logs_dir`, `paths.run_root`, `paths.run_id`
**Status:** WIRED

```python
# src/training/run_layout.py:74-75
paths = config.get("paths", {})
run_id from paths.get("run_id") or paths.get("run_dir") or paths.get("run_name")
# run_layout.py:106-112
paths.get("run_root", "runs")
run_id = get_run_id(config, cli_run_id)
# src/training/main.py:468, 487-490
Path(config["paths"]["checkpoints_dir"])
Path(self.config["paths"]["logs_dir"])
```

---

## iteration

### `iteration.max_iterations: 400`
**Status:** WIRED

```python
# src/training/main.py:823
total_iterations = self.config["iteration"]["max_iterations"]
```

---

### `iteration.auto_resume: true`
**Status:** WIRED

```python
# src/training/main.py:687
auto_resume = bool(self.config.get("iteration", {}).get("auto_resume", True))
```

---

### `iteration.lr_schedule`
**Status:** WIRED

```python
# src/training/trainer.py:522-523
lr_entries = sorted(
    self.config.get("iteration", {}).get("lr_schedule", []) or [{"iteration": 0, "lr": ...}],
```

---

### `iteration.temperature_schedule`
**Status:** WIRED

```python
# src/training/selfplay_optimized_integration.py:672-675
temp_schedule = self.config.get("iteration", {}).get("temperature_schedule", [])
config["selfplay"]["mcts"]["temperature"] = entry["temperature"]
```

---

### `iteration.mcts_schedule`
**Status:** WIRED

```python
# src/training/selfplay_optimized_integration.py:679-682
mcts_schedule = self.config.get("iteration", {}).get("mcts_schedule", [])
config["selfplay"]["mcts"]["simulations"] = entry["simulations"]
```

---

### `iteration.dirichlet_alpha_schedule`
**Status:** WIRED

```python
# src/training/selfplay_optimized_integration.py:686-689
alpha_schedule = self.config.get("iteration", {}).get("dirichlet_alpha_schedule", [])
config["selfplay"]["mcts"]["root_dirichlet_alpha"] = entry["alpha"]
```

---

### `iteration.q_value_weight_schedule`
**Status:** WIRED

```python
# src/training/main.py:345-348
qvw_sched = iter_cfg.get("q_value_weight_schedule", [])
qvw = float(_step_schedule_lookup(qvw_sched, iteration, "q_value_weight", qvw_base))
```

---

### `iteration.cpuct_schedule`
**Status:** WIRED

```python
# src/training/main.py:351-354
cpuct_sched = iter_cfg.get("cpuct_schedule", [])
cpuct = float(_step_schedule_lookup(cpuct_sched, iteration, "cpuct", cpuct_base))
```

---

### `iteration.noise_weight_schedule`
**Status:** WIRED

```python
# src/training/selfplay_optimized_integration.py:693-696
noise_schedule = self.config.get("iteration", {}).get("noise_weight_schedule", [])
config["selfplay"]["mcts"]["root_noise_weight"] = entry["weight"]
```

---

### `iteration.games_schedule`
**Status:** WIRED

```python
# src/training/main.py:331-333, selfplay_optimized_integration.py:651-655
sched = config.get("iteration", {}).get("games_schedule", [])
# _step_schedule_lookup for games per iteration
```

---

### `iteration.window_iterations_schedule`
**Status:** WIRED

```python
# src/training/main.py:324-325, 819
sched = config.get("iteration", {}).get("window_iterations_schedule", [])
return int(_step_schedule_lookup(sched, iteration, "window_iterations", base))
self.replay_buffer.window_size = _get_window_iterations_for_iteration(self.config, iteration)
```

---

## inference

### `inference.batch_size`, `max_batch_wait_ms`, `use_amp`, `amp_dtype`, `torch_compile`, `allow_tf32`
**Status:** WIRED

```python
# src/network/gpu_inference_server.py:57-64
inf = config.get("inference", {}) or {}
self.settings = InferenceSettings(
    batch_size=int(inf.get("batch_size", 256)),
    max_batch_wait_ms=int(inf.get("max_batch_wait_ms", 2)),
    use_amp=bool(inf.get("use_amp", True)),
    amp_dtype=str(inf.get("amp_dtype", "float16")),
    torch_compile=bool(inf.get("torch_compile", False)),
    allow_tf32=bool(inf.get("allow_tf32", True)),
)
```

---

## league

### All `league.*` keys
**Status:** WIRED via `LeagueConfig.from_config()`

```python
# src/training/league.py:86-93
@classmethod
def from_config(cls, config: dict) -> "LeagueConfig":
    lc = config.get("league", {}) or {}
    kwargs = {}
    for f in cls.__dataclass_fields__:
        if f in lc:
            kwargs[f] = type(...)(lc[f])
    return cls(**kwargs)
```

League fields: `pfsp_alpha`, `pfsp_uniform_prob`, `sp_frac_vs_best`, `sp_frac_vs_pfsp`, `sp_frac_vs_uniform`, `gate_threshold`, `suite_threshold`, `worst_threshold`, `gate_eval_games`, `suite_eval_games`, `max_pool_size`, `anchor_size`, `anchor_refresh_interval`, `payoff_max_models`, `exploiter_enabled`, `exploiter_train_games`, `exploiter_temp_boost`, `recency_newest_frac`, `recency_newest_window`.

---

## Summary

- **WIRED:** Most parameters, including all iteration schedules and core training/selfplay/eval logic.
- **NOT WIRED:** `api_url`, `stream_to_disk`, `write_buffer_positions`, `mcts.num_threads`, `best_metric`, `early_stopping.*`, `activation`, `prioritized`, `storage_format`, `wandb.*`.
