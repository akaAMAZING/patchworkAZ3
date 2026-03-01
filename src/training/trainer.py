"""
Training Pipeline for Patchwork AlphaZero

BATCHED HDF5 INPUT PIPELINE (high-throughput)
- Lazy-open HDF5 per worker (no open-per-sample)
- Batched HDF5 reads (one HDF5 read per batch, not per item)
- Within-batch index sorting to reduce random I/O on shuffled batches
- AMP is CUDA-only (won't crash on CPU/MPS)
- Optional replay buffer integration: train on sliding window of recent iterations

"""

import logging
import math
import os
import queue
import random
import threading
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, IterableDataset, Sampler
from torch.utils.tensorboard import SummaryWriter

from src.network.model import PatchworkNetwork, ValueHead, create_network, load_model_checkpoint
from src.training.replay_buffer import _validate_score_margins, SCORE_MARGIN_MAX_ABS
from src.network.d4_augmentation import (
    apply_d4_augment_batch,
    apply_ownership_transform_batch,
)
from src.network.d4_augmentation_gpu import apply_d4_augment_batch_gpu

logger = logging.getLogger(__name__)


def make_gaussian_score_targets(
    score_margins_tanh: torch.Tensor,
    score_utility_scale: float,
    score_min: int,
    score_max: int,
    sigma: float,
    bin_vals: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build (B, 201) soft label distributions from tanh-normalised score_margins (replay schema).

    Replay stores score_margins in [-1, 1] (tanh). Derive point margin: margin_points = scale * atanh(m).
    Then Gaussian over bins centred at margin_points, normalised to sum to 1.
    """
    m = score_margins_tanh.float().clamp(-0.999999, 0.999999)
    margin_points = (score_utility_scale * torch.atanh(m)).round().clamp(float(score_min), float(score_max))
    if bin_vals is None:
        bin_vals = torch.arange(
            score_min, score_max + 1,
            device=margin_points.device, dtype=torch.float32,
        )
    # (B,) and (201,) -> (B, 201)
    diff = bin_vals.unsqueeze(0) - margin_points.unsqueeze(1)
    weights = torch.exp(-0.5 * (diff / sigma) ** 2)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    return weights


IndexLike = Union[int, List[int], Tuple[int, ...], np.ndarray, torch.Tensor]

# Named batch keys for dict-based API (avoids brittle 11-element tuple unpacking)
BATCH_KEYS = (
    "states",
    "action_masks",
    "policies",
    "values",
    "score_margins",
    "ownerships",
    "x_global",
    "x_track",
    "shop_ids",
    "shop_feats",
    "slot_piece_ids",
)


_TUPLE_BATCH_WARNED = False


def batch_to_dict(batch: Union[Tuple, Dict]) -> Dict[str, Any]:
    """
    Normalize batch from tuple (11 elements) or dict to a key-based dict.
    Enables maintainable, position-independent batch handling.
    """
    global _TUPLE_BATCH_WARNED
    if isinstance(batch, dict):
        return batch
    if isinstance(batch, (tuple, list)) and len(batch) >= 11:
        if not _TUPLE_BATCH_WARNED:
            _TUPLE_BATCH_WARNED = True
            warnings.warn(
                "Legacy tuple batch format is deprecated; dataset should return dict.",
                DeprecationWarning,
                stacklevel=2,
            )
        return dict(zip(BATCH_KEYS, batch[:11]))
    if isinstance(batch, (tuple, list)) and len(batch) >= 6:
        # Legacy 6-element (no multimodal): pad with None
        legacy = list(batch[:6]) + [None] * 5
        return dict(zip(BATCH_KEYS, legacy[:11]))
    raise ValueError(
        f"Batch must be dict or tuple with >=6 elements, got {type(batch).__name__} len={len(batch) if hasattr(batch, '__len__') else 'N/A'}"
    )


def _prefetch_generator(loader: DataLoader, prefetch_batches: int = 2) -> Iterable:
    """
    Wrap a DataLoader to prefetch batches in a background thread.
    Overlaps CPU augmentation (next(loader)) with GPU compute on the previous batch.
    With prefetch_batches=2: worker prepares batch N+1 while main thread runs batch N on GPU.
    """
    q: queue.Queue = queue.Queue(maxsize=max(1, prefetch_batches))

    def worker():
        try:
            for batch in loader:
                q.put(batch)
        except Exception as e:
            q.put(e)
        finally:
            q.put(None)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    while True:
        item = q.get()
        if item is None:
            break
        if isinstance(item, Exception):
            raise item
        yield item
    t.join()


def _atomic_torch_save(obj: dict, path: Path) -> None:
    """
    Atomically save a checkpoint by writing to a temp file and replacing.
    On failure: remove partial .tmp, re-raise. Prevents corrupted checkpoint on disk full.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


def _capture_rng_state() -> Dict[str, object]:
    state: Dict[str, object] = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Dict[str, object]) -> None:
    if not state:
        return
    if "python_random_state" in state:
        random.setstate(state["python_random_state"])
    if "numpy_random_state" in state:
        np.random.set_state(state["numpy_random_state"])
    if "torch_rng_state" in state:
        torch.set_rng_state(state["torch_rng_state"])
    if "torch_cuda_rng_state_all" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda_rng_state_all"])


def _ensure_float32_contig(x: np.ndarray) -> np.ndarray:
    """Avoid unnecessary copies: cast/contiguate only when needed."""
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    if not x.flags['C_CONTIGUOUS']:
        x = np.ascontiguousarray(x)
    return x


class PatchworkDataset(Dataset):
    """
    In-memory dataset loaded from HDF5.

    [PERF FIX] The previous implementation used lazy HDF5 random-access reads
    during training. With LZF-compressed chunks, random access to 500K rows
    required decompressing ~18MB chunks for each batch, causing 5+ seconds per
    training step (100x slower than GPU compute). Now we load everything into
    RAM once at init time. With max_size=300K, memory usage is ~6GB which fits
    comfortably in 16GB RAM.

    Supports batched indexing: dataset[[i1, i2, ...]] returns a full batch.
    Supports optional ownership targets (backward-compatible with old HDF5 files).
    """

    def __init__(self, h5_path: str, config: Optional[dict] = None):
        self.h5_path = str(h5_path)
        train_cfg = (config or {}).get("training", {}) or {}
        self.d4_dynamic = str(train_cfg.get("d4_augmentation", "store")).lower() == "dynamic"
        self.d4_on_gpu = bool(train_cfg.get("d4_on_gpu", False))

        # Load entire dataset into RAM (one sequential read, fast even with LZF)
        logger.debug("Loading training data into RAM from %s ...", h5_path)
        load_start = time.time()
        with h5py.File(self.h5_path, "r") as f:
            states_key = "spatial_states" if "spatial_states" in f else "states"
            self.num_samples = int(f[states_key].shape[0])
            self._multimodal = states_key == "spatial_states"
            for key in ("action_masks", "policies", "values"):
                if key not in f:
                    raise ValueError(f"HDF5 file {h5_path} missing dataset '{key}'")
                if int(f[key].shape[0]) != self.num_samples:
                    raise ValueError(
                        f"HDF5 shape mismatch: states has {self.num_samples} rows "
                        f"but {key} has {f[key].shape[0]} rows in {h5_path}"
                    )
            self.has_ownership = "ownerships" in f
            if self.has_ownership and int(f["ownerships"].shape[0]) != self.num_samples:
                raise ValueError(
                    f"HDF5 shape mismatch: states has {self.num_samples} rows "
                    f"but ownerships has {f['ownerships'].shape[0]} rows in {h5_path}"
                )
            self.has_score_margins = "score_margins" in f
            if self.has_score_margins and int(f["score_margins"].shape[0]) != self.num_samples:
                raise ValueError(
                    f"HDF5 shape mismatch: states has {self.num_samples} rows "
                    f"but score_margins has {f['score_margins'].shape[0]} rows in {h5_path}"
                )

            # Sequential bulk read — fast even with compression
            self._states = np.array(f[states_key], dtype=np.float32)
            if self._multimodal:
                self._global_states = np.array(f["global_states"], dtype=np.float32)
                self._track_states = np.array(f["track_states"], dtype=np.float32)
                self._shop_ids = np.array(f["shop_ids"], dtype=np.int16)
                self._shop_feats = np.array(f["shop_feats"], dtype=np.float32)
            else:
                self._global_states = None
                self._track_states = None
                self._shop_ids = None
                self._shop_feats = None
            self._action_masks = np.array(f["action_masks"], dtype=np.uint8)
            self._policies = np.array(f["policies"], dtype=np.float32)
            self._values = np.array(f["values"], dtype=np.float32)
            if self.has_ownership:
                self._ownerships = np.array(f["ownerships"], dtype=np.float32)
            else:
                self._ownerships = np.full(
                    (self.num_samples, 2, 9, 9), -1.0, dtype=np.float32
                )
            if self.has_score_margins:
                self._score_margins = np.array(f["score_margins"], dtype=np.float32)
                _validate_score_margins(
                    self._score_margins,
                    max_abs=SCORE_MARGIN_MAX_ABS,
                    source=str(h5_path),
                )
            else:
                self._score_margins = np.zeros(self.num_samples, dtype=np.float32)
            if "slot_piece_ids" in f:
                self._slot_piece_ids = np.array(f["slot_piece_ids"], dtype=np.int16)
                if self._slot_piece_ids.shape != (self.num_samples, 3):
                    raise ValueError(
                        f"HDF5 slot_piece_ids shape {self._slot_piece_ids.shape} "
                        f"!= (num_samples={self.num_samples}, 3)"
                    )
            else:
                self._slot_piece_ids = None

        # Ensure values are (N, 1)
        if self._values.ndim == 1:
            self._values = self._values.reshape(-1, 1)

        # Validate: every position must have >= 1 legal action
        # Cast to int32 for sum to avoid uint8 overflow (max legal actions > 255)
        mask_sums = self._action_masks.reshape(self.num_samples, -1).astype(np.int32).sum(axis=1)
        if (mask_sums == 0).any():
            n_bad = int((mask_sums == 0).sum())
            raise ValueError(f"Found {n_bad} sample(s) with zero legal actions in {h5_path}")

        mem_mb = (
            self._states.nbytes + self._action_masks.nbytes +
            self._policies.nbytes + self._values.nbytes +
            self._ownerships.nbytes + self._score_margins.nbytes
        ) / (1024 * 1024)
        if self._global_states is not None:
            mem_mb += (self._global_states.nbytes + self._track_states.nbytes +
                      self._shop_ids.nbytes + self._shop_feats.nbytes) / (1024 * 1024)
        logger.info(
            "Loaded %d positions into RAM (%.0f MB) in %.1fs",
            self.num_samples, mem_mb, time.time() - load_start,
        )

    def close(self) -> None:
        pass  # No file handles to close (all in RAM)

    def __del__(self):
        pass

    def __len__(self) -> int:
        return self.num_samples

    @staticmethod
    def _to_index_list(idx: IndexLike) -> List[int]:
        if isinstance(idx, int):
            return [idx]
        if isinstance(idx, torch.Tensor):
            return idx.flatten().tolist()
        if isinstance(idx, np.ndarray):
            return idx.flatten().tolist()
        return list(idx)

    def _apply_dynamic_d4(
        self,
        states: np.ndarray,
        masks: np.ndarray,
        policies: np.ndarray,
        ownerships: np.ndarray,
        slot_ids: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply random D4 transform per sample. Vectorized batch for speed."""
        n = states.shape[0]
        transform_indices = np.random.randint(0, 8, size=n, dtype=np.int32)
        states, policies, masks = apply_d4_augment_batch(
            states, policies, masks, slot_ids, transform_indices
        )
        ownerships = apply_ownership_transform_batch(ownerships, transform_indices)
        return states, masks, policies, ownerships

    def __getitem__(self, idx: IndexLike) -> Dict[str, Any]:
        is_scalar = isinstance(idx, int)
        if is_scalar:
            idxs = [idx]
        else:
            idxs = self._to_index_list(idx)

        states = self._states[idxs]
        masks = self._action_masks[idxs].astype(np.float32)
        if self._multimodal:
            global_s = self._global_states[idxs]
            track_s = self._track_states[idxs]
            shop_i = self._shop_ids[idxs]
            shop_f = self._shop_feats[idxs]
        else:
            global_s = track_s = shop_i = shop_f = None
        policies = self._policies[idxs]
        values = self._values[idxs]
        ownerships = self._ownerships[idxs]
        score_margins = self._score_margins[idxs]

        # [PERF] When d4_on_gpu: return canonical data + slot_piece_ids; D4 applied on GPU after H2D.
        # Otherwise: apply D4 in __getitem__ (CPU path).
        slot_piece_ids_out = None
        if self.d4_dynamic and self._slot_piece_ids is not None:
            slot_ids = self._slot_piece_ids[idxs]
            if self.d4_on_gpu:
                slot_piece_ids_out = slot_ids
            else:
                states, masks, policies, ownerships = self._apply_dynamic_d4(
                    states, masks, policies, ownerships, slot_ids
                )

        states = torch.from_numpy(states)
        action_masks = torch.from_numpy(masks)
        policies = torch.from_numpy(policies)
        values = torch.from_numpy(values)
        ownerships = torch.from_numpy(ownerships)
        score_margins = torch.from_numpy(score_margins)
        if self._multimodal and global_s is not None:
            global_states = torch.from_numpy(global_s)
            track_states = torch.from_numpy(track_s)
            shop_ids_t = torch.from_numpy(shop_i)
            shop_feats_t = torch.from_numpy(shop_f)
        else:
            global_states = track_states = shop_ids_t = shop_feats_t = None

        if is_scalar:
            states = states.squeeze(0)
            action_masks = action_masks.squeeze(0)
            policies = policies.squeeze(0)
            values = values.squeeze(0)
            ownerships = ownerships.squeeze(0)
            score_margins = score_margins.squeeze(0)
            if self._multimodal:
                global_states = global_states.squeeze(0)
                track_states = track_states.squeeze(0)
                shop_ids_t = shop_ids_t.squeeze(0)
                shop_feats_t = shop_feats_t.squeeze(0)

        return {
            "states": states,
            "action_masks": action_masks,
            "policies": policies,
            "values": values,
            "score_margins": score_margins,
            "ownerships": ownerships,
            "x_global": global_states,
            "x_track": track_states,
            "shop_ids": shop_ids_t,
            "shop_feats": shop_feats_t,
            "slot_piece_ids": slot_piece_ids_out,
        }


class BatchIndexSampler(Sampler[List[int]]):
    """
    Sampler that yields lists of indices (one list per batch).
    """

    def __init__(self, indices: Sequence[int], batch_size: int, shuffle: bool, seed: int):
        self.indices = list(indices)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return math.ceil(len(self.indices) / self.batch_size)

    def __iter__(self) -> Iterable[List[int]]:
        idxs = self.indices.copy()
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            perm = torch.randperm(len(idxs), generator=g).tolist()
            idxs = [idxs[i] for i in perm]

        for i in range(0, len(idxs), self.batch_size):
            yield idxs[i : i + self.batch_size]


class _BatchIterableDataset(IterableDataset):
    """
    Yields one dataset[indices] call per batch instead of 1024 per-sample calls.
    DataLoader's default path does dataset[i] for each i, causing ~10s/batch on in-memory data.
    This wrapper yields pre-batched dicts so we get O(1) dataset calls per batch.
    """

    def __init__(self, dataset: Dataset, batch_sampler: BatchIndexSampler):
        self.dataset = dataset
        self.batch_sampler = batch_sampler

    def set_epoch(self, epoch: int) -> None:
        self.batch_sampler.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.batch_sampler)

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        for indices in self.batch_sampler:
            yield self.dataset[indices]


class Trainer:
    """Training manager for Patchwork AlphaZero."""

    def __init__(
        self,
        network: PatchworkNetwork,
        config: dict,
        device: torch.device,
        log_dir: Path,
        total_train_steps: int = 100_000,
        writer: Optional[SummaryWriter] = None,
        global_step_offset: int = 0,
        optimizer_state_checkpoint: Optional[str] = None,
        model_source_checkpoint: Optional[str] = None,
        current_iteration: int = 0,
        *,
        force_resume_optimizer_state: bool = False,
        force_resume_scheduler_state: bool = False,
        force_resume_scaler_state: bool = False,
        force_resume_ema: bool = False,
    ):
        self.network = network
        self.config = config
        self.device = device

        self.network.to(device)
        train_config = config["training"]

        # Apply training CUDA settings from config
        if device.type == "cuda":
            allow_tf32 = bool(train_config.get("allow_tf32", True))
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            torch.backends.cudnn.allow_tf32 = allow_tf32
            torch.backends.cudnn.benchmark = bool(train_config.get("cudnn_benchmark", True))
            matmul_prec = str(train_config.get("matmul_precision", "high")).lower()
            if matmul_prec in ("high", "highest", "medium"):
                torch.set_float32_matmul_precision(matmul_prec)

        self.optimizer = self._create_optimizer(train_config)
        self.total_train_steps = total_train_steps
        self.current_iteration = current_iteration
        self.scheduler = self._create_scheduler(train_config)

        requested_amp = bool(train_config.get("use_amp", False))
        self.use_amp = requested_amp and (self.device.type == "cuda")
        if requested_amp and not self.use_amp:
            logger.info("AMP disabled (config has use_amp=True but device is %s).", self.device.type)
        amp_dtype_str = str(train_config.get("amp_dtype", "bfloat16")).lower()
        self._autocast_dtype = torch.bfloat16 if amp_dtype_str == "bfloat16" else torch.float16
        # GradScaler only for fp16 (prevents overflow); bf16 has sufficient dynamic range
        use_scaler = self.use_amp and (self._autocast_dtype != torch.bfloat16)
        self.scaler = GradScaler("cuda", enabled=True, init_scale=2**12) if use_scaler else None

        self.policy_weight = train_config["policy_loss_weight"]
        self.value_weight = train_config["value_loss_weight"]
        self.score_loss_weight = float(train_config.get("score_loss_weight", 0.02))
        self.score_utility_scale = float(train_config.get("score_utility_scale", 30.0))
        self.score_bins_min = int(train_config.get("score_bins_min", getattr(ValueHead, "SCORE_MIN", -100)))
        self.score_bins_max = int(train_config.get("score_bins_max", getattr(ValueHead, "SCORE_MAX", 100)))
        self.score_target_sigma = float(train_config.get("score_target_sigma", 1.5))
        self._score_bin_vals: Optional[torch.Tensor] = None
        self.ownership_weight = float(train_config.get("ownership_loss_weight", 0.0))
        self.max_grad_norm = train_config["max_grad_norm"]

        # Use shared SummaryWriter if provided, otherwise create one (for standalone use)
        if writer is not None:
            self.writer = writer
            self._owns_writer = False
        else:
            self.writer = SummaryWriter(log_dir=log_dir / "tensorboard")
            self._owns_writer = True
        self.global_step = global_step_offset
        self._lr_log_step_ceiling = global_step_offset + 5  # Log LR/scheduler for first 5 steps of this iteration
        self.current_epoch = 0
        self.best_val_loss = float("inf")

        self.checkpoint_dir = Path(config["paths"]["checkpoints_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # EMA (Exponential Moving Average) for selfplay/eval strength
        ema_cfg = train_config.get("ema", {}) or {}
        self.ema_enabled = bool(ema_cfg.get("enabled", False))
        self.ema_decay = float(ema_cfg.get("decay", 0.999))
        self.ema_use_for_selfplay = bool(ema_cfg.get("use_for_selfplay", True))
        self.ema_use_for_eval = bool(ema_cfg.get("use_for_eval", True))
        self.ema_state_dict: Optional[Dict[str, torch.Tensor]] = None
        self._ema_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        if self.ema_enabled:
            self.ema_state_dict = {
                k: v.clone().detach()
                for k, v in self.network.state_dict().items()
            }
            # Cache (ema_tensor, param_tensor) for fast per-step update; avoid state_dict() + dict lookup
            sd = self.network.state_dict()
            for k in self.ema_state_dict:
                if k in sd and sd[k].is_floating_point():
                    self._ema_pairs.append((self.ema_state_dict[k], sd[k]))
            # Pre-split into flat lists for _foreach_lerp_ (avoids per-step tuple unpacking)
            self._ema_tensors: List[torch.Tensor] = [p[0] for p in self._ema_pairs]
            self._param_tensors: List[torch.Tensor] = [p[1] for p in self._ema_pairs]
            logger.debug(
                "[EMA] enabled decay=%.4f use_for_selfplay=%s use_for_eval=%s (%d params)",
                self.ema_decay, self.ema_use_for_selfplay, self.ema_use_for_eval, len(self._ema_pairs),
            )
        else:
            logger.debug("[EMA] disabled")
            self._ema_tensors: List[torch.Tensor] = []
            self._param_tensors: List[torch.Tensor] = []

        # Resume optimizer/scheduler/scaler/EMA from checkpoint.
        # Main sets force_resume_* (e.g. False for seed warm-start); that is the definitive decision.
        resume_opt = force_resume_optimizer_state
        resume_sched = force_resume_scheduler_state
        resume_scaler = force_resume_scaler_state
        if resume_opt or resume_sched or resume_scaler or force_resume_ema:
            if optimizer_state_checkpoint and Path(optimizer_state_checkpoint).exists():
                self._try_load_optimizer_state(
                    optimizer_state_checkpoint,
                    resume_opt=resume_opt,
                    resume_sched=resume_sched,
                    resume_scaler=resume_scaler,
                    resume_ema=force_resume_ema,
                    model_source=model_source_checkpoint or optimizer_state_checkpoint,
                )

        # Log phase LR and scheduler state at iteration start (debug only, not terminal)
        iter_lr = float(train_config.get("learning_rate", 0.0))
        opt_lr = self.optimizer.param_groups[0]["lr"] if self.optimizer.param_groups else 0.0
        sched_step = getattr(self.scheduler, "last_epoch", None)
        logger.debug(
            "[LR] iteration_start current_iteration=%d iter_lr=%.2e optimizer_lr=%.2e scheduler_last_epoch=%s",
            self.current_iteration, iter_lr, opt_lr, sched_step,
        )

    def close(self) -> None:
        """Release resources (TensorBoard writer if owned by this Trainer)."""
        if self._owns_writer:
            try:
                self.writer.close()
            except Exception:
                pass
        else:
            # Flush shared writer to ensure all pending events are written
            try:
                self.writer.flush()
            except Exception:
                pass

    def _create_optimizer(self, config: dict) -> optim.Optimizer:
        opt_name = config["optimizer"].lower()
        lr = config["learning_rate"]
        weight_decay = self.config["network"]["weight_decay"]

        if opt_name == "adamw":
            return optim.AdamW(
                self.network.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8,
            )
        if opt_name == "adam":
            return optim.Adam(self.network.parameters(), lr=lr, weight_decay=weight_decay)
        if opt_name == "sgd":
            return optim.SGD(
                self.network.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay,
            )
        raise ValueError(f"Unknown optimizer: {opt_name}")

    def _is_phase_boundary(self) -> bool:
        """True if current_iteration is the start of a new LR phase (iteration in lr_schedule).
        Used to skip loading scheduler state so the new phase gets full warmup."""
        entries = sorted(
            self.config.get("iteration", {}).get("lr_schedule", []) or [{"iteration": 0}],
            key=lambda x: x["iteration"],
        )
        phase_starts = {ent["iteration"] for ent in entries}
        return self.current_iteration in phase_starts

    def _try_load_optimizer_state(
        self,
        checkpoint_path: str,
        resume_opt: bool,
        resume_sched: bool,
        resume_scaler: bool,
        resume_ema: bool = False,
        model_source: Optional[str] = None,
    ) -> None:
        """Load optimizer/scheduler/scaler/EMA from checkpoint with compatibility checks.
        Requires source=train_base (optimizer and model weights from same checkpoint).
        """
        try:
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        except Exception as e:
            logger.warning("[OPT_RESUME] Failed to load checkpoint %s: %s", checkpoint_path, e)
            return
        opt_loaded = sched_loaded = scaler_loaded = ema_loaded = False
        if resume_opt and "optimizer_state_dict" in ckpt:
            # Defensive: skip load if checkpoint optimizer state has different param shapes (e.g. seed from another arch).
            try:
                od = ckpt["optimizer_state_dict"]
                ckpt_states = od.get("state", [])
                if isinstance(ckpt_states, dict):
                    ckpt_states = list(ckpt_states.values())
                current_params = [p for pg in self.optimizer.param_groups for p in pg["params"]]
                if len(ckpt_states) != len(current_params):
                    raise ValueError(
                        f"optimizer state param count {len(ckpt_states)} != current {len(current_params)}"
                    )
                for i, p in enumerate(current_params):
                    if i >= len(ckpt_states):
                        break
                    st = ckpt_states[i]
                    exp_avg = st.get("exp_avg")
                    if exp_avg is not None and hasattr(exp_avg, "shape") and exp_avg.shape != p.shape:
                        raise ValueError(
                            f"param {i} shape mismatch: ckpt {exp_avg.shape} vs current {p.shape}"
                        )
                self.optimizer.load_state_dict(od)
                opt_loaded = True
                # Peak LR must follow iteration.lr_schedule; load_state_dict restores checkpoint LR
                # and overwrites the phase LR we set in config. Force phase LR after load.
                iter_lr = float(self.config.get("training", {}).get("learning_rate", 0.0))
                for pg in self.optimizer.param_groups:
                    pg["lr"] = iter_lr
                opt_lr_after = self.optimizer.param_groups[0]["lr"] if self.optimizer.param_groups else 0.0
                logger.debug(
                    "[OPT_RESUME] LR after load: iter_lr=%.2e optimizer_lr_after_load=%.2e checkpoint=%s",
                    iter_lr, opt_lr_after, checkpoint_path,
                )
            except Exception as e:
                logger.warning("[OPT_RESUME] Optimizer state mismatch, using fresh: %s", e)
        # At phase boundaries we want warmup for the new phase; loading scheduler state would
        # continue last_epoch and skip warmup. Skip loading scheduler when starting a new phase.
        at_phase_boundary = self._is_phase_boundary()
        if at_phase_boundary and resume_sched:
            logger.debug(
                "[OPT_RESUME] phase boundary iter=%d: not loading scheduler state (fresh warmup for this phase)",
                self.current_iteration,
            )
        if resume_sched and "scheduler_state_dict" in ckpt and not at_phase_boundary:
            try:
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                sched_loaded = True
            except Exception as e:
                logger.warning("[OPT_RESUME] Scheduler state mismatch, using fresh: %s", e)
        if resume_scaler and self.scaler is not None and "scaler_state_dict" in ckpt:
            try:
                self.scaler.load_state_dict(ckpt["scaler_state_dict"])
                scaler_loaded = True
            except Exception as e:
                logger.warning("[OPT_RESUME] Scaler state mismatch, using fresh: %s", e)
        if resume_ema and self.ema_enabled and self.ema_state_dict is not None and "ema_state_dict" in ckpt:
            try:
                for k, v in ckpt["ema_state_dict"].items():
                    if k in self.ema_state_dict:
                        self.ema_state_dict[k].copy_(v.to(self.device))
                ema_loaded = True
            except Exception as e:
                logger.warning("[OPT_RESUME] EMA state mismatch, using fresh: %s", e)
        # Only overwrite global_step when we actually resumed (keeps TensorBoard continuity)
        if opt_loaded or sched_loaded:
            self.global_step = int(ckpt.get("global_step", self.global_step))
        train_base = model_source if model_source is not None else checkpoint_path
        logger.info(
            "[OPT_RESUME] source=%s train_base=%s (must match)  enabled=opt:%s sched:%s scaler:%s ema:%s  loaded=opt:%s sched:%s scaler:%s ema:%s",
            checkpoint_path,
            train_base,
            resume_opt,
            resume_sched,
            resume_scaler,
            resume_ema,
            opt_loaded,
            sched_loaded,
            scaler_loaded,
            ema_loaded,
        )

    def _create_scheduler(self, config: dict):
        schedule_type = config["lr_schedule"]

        if schedule_type == "cosine_warmup":
            warmup_steps_cfg = config["warmup_steps"]
            min_lr = config["min_lr"]
            base_lr = config["learning_rate"]
            total = max(1, self.total_train_steps)

            # [C2 FIX] LambdaLR multiplies base_lr * lambda(step).
            # We need the lambda to return a *ratio* so that:
            #   actual_lr = base_lr * ratio
            # At minimum: actual_lr = min_lr  =>  ratio = min_lr / base_lr
            min_lr_ratio = min_lr / base_lr if base_lr > 0 else 0.0

            # [M1 FIX] Clamp warmup to at most 10% of total steps so warmup
            # always completes even on small per-iteration datasets.
            warmup_steps = min(warmup_steps_cfg, max(1, total // 10))
            if warmup_steps < warmup_steps_cfg:
                logger.debug(
                    f"Clamped warmup_steps from {warmup_steps_cfg} to {warmup_steps} "
                    f"(10% of {total} total steps)"
                )

            def lr_lambda(step):
                if step < warmup_steps:
                    # Linear warmup from 0 to 1.0 (i.e., base_lr)
                    return step / max(1, warmup_steps)
                # Cosine decay from 1.0 down to min_lr_ratio
                progress = (step - warmup_steps) / max(1, total - warmup_steps)
                progress = min(progress, 1.0)
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

            return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        if schedule_type == "cosine_warmup_per_phase":
            # Phases from iteration.lr_schedule; warmup once per phase, no restart each iteration
            lr_entries = sorted(
                self.config.get("iteration", {}).get("lr_schedule", [])
                or [{"iteration": 0, "lr": config["learning_rate"]}],
                key=lambda x: x["iteration"],
            )
            iter_lr = config["learning_rate"]
            phase_start_iter = 0
            phase_end_iter = 999999
            for i, ent in enumerate(lr_entries):
                if self.current_iteration >= ent["iteration"]:
                    iter_lr = ent["lr"]
                    phase_start_iter = ent["iteration"]
                    phase_end_iter = lr_entries[i + 1]["iteration"] if i + 1 < len(lr_entries) else 999999
            iters_in_phase = max(1, phase_end_iter - phase_start_iter)
            phase_total_steps = iters_in_phase * self.total_train_steps
            warmup_steps_cfg = config.get("warmup_steps", 200)
            warmup_steps = min(warmup_steps_cfg, max(1, phase_total_steps // 10))
            min_lr = config.get("min_lr", iter_lr / 50)
            min_lr_ratio = min_lr / iter_lr if iter_lr > 0 else 0.0

            def lr_lambda(step):
                if step < warmup_steps:
                    return step / max(1, warmup_steps)
                progress = (step - warmup_steps) / max(1, phase_total_steps - warmup_steps)
                progress = min(progress, 1.0)
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

            return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        if schedule_type == "step":
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=10000, gamma=0.1)

        if schedule_type == "none":
            return optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: 1.0)

        raise ValueError(f"Unknown schedule: {schedule_type}")

    def train_epoch(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> Dict[str, float]:
        self.network.train()

        epoch_metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "score_loss": 0.0,
            "ownership_loss": 0.0,
            "total_loss": 0.0,
            "policy_accuracy": 0.0,
            "policy_top5_accuracy": 0.0,
            "value_mse": 0.0,
            "grad_norm": 0.0,
            "grad_norm_count": 0,
            "steps_skipped": 0,
        }
        num_batches = 0

        # [PERF] Background prefetch to overlap next(data_iter) with GPU compute
        hw = self.config.get("hardware", {}) or {}
        prefetch_batches = int(hw.get("prefetch_batches", 2))
        use_prefetch = prefetch_batches > 0 and self.device.type == "cuda"
        batch_iter = _prefetch_generator(train_loader, prefetch_batches) if use_prefetch else train_loader

        train_cfg = self.config.get("training", {}) or {}
        d4_on_gpu = bool(train_cfg.get("d4_on_gpu", False))
        d4_deterministic = bool(train_cfg.get("d4_gpu_deterministic_test_mode", False))
        profile_steps = int(train_cfg.get("profile_training_steps", 0))

        for batch_idx, batch_data in enumerate(batch_iter):
            _t0 = time.perf_counter() if profile_steps > 0 and batch_idx < profile_steps else None
            batch = batch_to_dict(batch_data)
            states = batch["states"].to(self.device, non_blocking=True)
            action_masks = batch["action_masks"].to(self.device, non_blocking=True)
            policies = batch["policies"].to(self.device, non_blocking=True)
            values = batch["values"].to(self.device, non_blocking=True)
            score_margins = batch["score_margins"].to(self.device, non_blocking=True)
            ownerships = batch["ownerships"].to(self.device, non_blocking=True)
            _t1 = time.perf_counter() if _t0 is not None else None

            slot_piece_ids = batch["slot_piece_ids"]
            # [PERF] Apply D4 after H2D: GPU path when CUDA, else CPU fallback (same semantics)
            if slot_piece_ids is not None and train_cfg.get("d4_augmentation") == "dynamic":
                n = states.shape[0]
                if d4_on_gpu and self.device.type == "cuda":
                    if d4_deterministic:
                        g = torch.Generator(device=self.device).manual_seed(self.global_step + batch_idx)
                        transform_indices = torch.randint(0, 8, (n,), device=self.device, dtype=torch.long, generator=g)
                    else:
                        transform_indices = torch.randint(0, 8, (n,), device=self.device, dtype=torch.long)
                    slot_ids_t = slot_piece_ids.to(self.device, non_blocking=True)
                    states, policies, action_masks, ownerships = apply_d4_augment_batch_gpu(
                        states, policies, action_masks, ownerships,
                        slot_ids_t, transform_indices, self.device,
                    )
                else:
                    # CPU fallback when d4_on_gpu=False or CUDA unavailable
                    if d4_deterministic:
                        rng = np.random.default_rng(self.global_step + batch_idx)
                        transform_indices = rng.integers(0, 8, size=n, dtype=np.int32)
                    else:
                        transform_indices = np.random.randint(0, 8, size=n, dtype=np.int32)
                    slot_ids_np = slot_piece_ids.cpu().numpy() if slot_piece_ids.is_cuda else slot_piece_ids.numpy()
                    if slot_ids_np.ndim == 1:
                        slot_ids_np = np.broadcast_to(slot_ids_np[:, None], (n, 3)).copy()
                    st, pol, msk = apply_d4_augment_batch(
                        states.cpu().numpy(), policies.cpu().numpy(), action_masks.cpu().numpy(),
                        slot_ids_np, transform_indices,
                    )
                    own_np = ownerships.cpu().numpy()
                    own_np = apply_ownership_transform_batch(own_np, transform_indices)
                    states = torch.from_numpy(st).to(self.device, non_blocking=True)
                    policies = torch.from_numpy(pol).to(self.device, non_blocking=True)
                    action_masks = torch.from_numpy(msk).to(self.device, non_blocking=True)
                    ownerships = torch.from_numpy(own_np).to(self.device, non_blocking=True)
            _t2 = time.perf_counter() if _t0 is not None else None

            # Ownership: use per-sample masking so mixed old/new data still trains
            own_target = None
            own_weight = 0.0
            own_valid_mask = None
            if self.ownership_weight > 0 and self.network.ownership_head is not None:
                # Per-sample valid mask: check each sample for -1 sentinel (CPU, no sync)
                # Shape: (B, 2, 9, 9) -> check min per sample across all ownership dims
                valid_mask = ownerships.view(ownerships.shape[0], -1).min(dim=1).values >= 0
                if valid_mask.any():
                    ownerships = ownerships.to(self.device, non_blocking=True)
                    own_target = ownerships
                    own_weight = self.ownership_weight
                    # Pass validity mask to get_loss for per-sample masking
                    own_valid_mask = valid_mask.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            x_global = batch["x_global"]
            x_track = batch["x_track"]
            shop_ids = batch["shop_ids"]
            shop_feats = batch["shop_feats"]
            xg = x_global.to(self.device, non_blocking=True) if x_global is not None else None
            xt = x_track.to(self.device, non_blocking=True) if x_track is not None else None
            si = shop_ids.to(self.device, non_blocking=True) if shop_ids is not None else None
            sf = shop_feats.to(self.device, non_blocking=True) if shop_feats is not None else None
            # 201-bin: build soft target from tanh score_margins (replay schema unchanged).
            target_score_dist = None
            if self.score_loss_weight > 0:
                if self._score_bin_vals is None or self._score_bin_vals.device != score_margins.device:
                    self._score_bin_vals = torch.arange(
                        self.score_bins_min, self.score_bins_max + 1,
                        device=score_margins.device, dtype=torch.float32,
                    )
                target_score_dist = make_gaussian_score_targets(
                    score_margins, self.score_utility_scale,
                    self.score_bins_min, self.score_bins_max,
                    self.score_target_sigma, bin_vals=self._score_bin_vals,
                )
            if self.use_amp:
                with autocast(device_type="cuda", dtype=self._autocast_dtype):
                    loss, metrics = self.network.get_loss(
                        states, action_masks, policies, values,
                        self.policy_weight, self.value_weight,
                        target_score_dist=target_score_dist,
                        score_loss_weight=self.score_loss_weight,
                        target_ownership=own_target,
                        ownership_weight=own_weight,
                        ownership_valid_mask=own_valid_mask,
                        x_global=xg, x_track=xt, shop_ids=si, shop_feats=sf,
                    )
            else:
                loss, metrics = self.network.get_loss(
                        states, action_masks, policies, values,
                        self.policy_weight, self.value_weight,
                        target_score_dist=target_score_dist,
                        score_loss_weight=self.score_loss_weight,
                        target_ownership=own_target,
                        ownership_weight=own_weight,
                        ownership_valid_mask=own_valid_mask,
                        x_global=xg, x_track=xt, shop_ids=si, shop_feats=sf,
                    )
            _t3 = time.perf_counter() if _t0 is not None else None

            step_skipped = False
            if self.use_amp and self.scaler is not None:
                scale_before = self.scaler.get_scale()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                step_skipped = self.scaler.get_scale() < scale_before
            elif self.use_amp and self.scaler is None:
                # bf16 AMP: no scaler, direct backward
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
            _t4 = time.perf_counter() if _t0 is not None else None

            # EMA update: single fused _foreach_lerp_ call instead of 290 individual kernel launches.
            # 290 separate lerp_() calls saturate the CUDA kernel queue (backpressure from optimizer
            # step already queuing ~870 kernels), causing the CPU to block for 130-180ms waiting for
            # queue capacity. _foreach fuses all 290 updates into 1-2 kernels → ~5ms.
            if self.ema_enabled and self._ema_tensors and not step_skipped:
                alpha = 1.0 - self.ema_decay
                with torch.no_grad():
                    torch._foreach_lerp_(self._ema_tensors, self._param_tensors, alpha)
            _t5 = time.perf_counter() if _t0 is not None else None

            if not step_skipped:
                self.scheduler.step()

            for k, v in metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0.0) + float(v)
            gn = float(grad_norm.item())
            if math.isfinite(gn):
                epoch_metrics["grad_norm"] = epoch_metrics["grad_norm"] + gn
                epoch_metrics["grad_norm_count"] = epoch_metrics["grad_norm_count"] + 1
            epoch_metrics["steps_skipped"] = epoch_metrics["steps_skipped"] + int(step_skipped)

            num_batches += 1
            self.global_step += 1

            # Log LR/scheduler for first few steps (debug only, not terminal). warmup is heuristic:
            # lr < 0.999*peak can mislabel post-warmup as True when cosine has already decayed slightly.
            if self.global_step < self._lr_log_step_ceiling:
                step_in_iter = self.global_step - (self._lr_log_step_ceiling - 5)
                current_lr = self.scheduler.get_last_lr()[0]
                sched_step = getattr(self.scheduler, "last_epoch", None)
                iter_lr = float(self.config.get("training", {}).get("learning_rate", 0.0))
                in_warmup = current_lr < iter_lr * 0.999 if iter_lr > 0 else False
                logger.debug(
                    "[LR] step %d (step_in_iter=%d) lr=%.2e scheduler_last_epoch=%s warmup=%s",
                    self.global_step, step_in_iter, current_lr, sched_step, in_warmup,
                )

            if _t0 is not None and _t1 is not None and _t2 is not None and _t3 is not None and _t4 is not None and _t5 is not None:
                total_ms = (time.perf_counter() - _t0) * 1000
                logger.info(
                    "[PERF] step %d: h2d=%.0fms d4=%.0fms fwd=%.0fms bwd=%.0fms ema=%.0fms other=%.0fms total=%.0fms",
                    self.global_step,
                    (_t1 - _t0) * 1000,
                    (_t2 - _t1) * 1000,
                    (_t3 - _t2) * 1000,
                    (_t4 - _t3) * 1000,
                    (_t5 - _t4) * 1000,
                    (time.perf_counter() - _t5) * 1000,
                    total_ms,
                )

            if self.global_step % 10 == 0:
                # Core training metrics
                for k in ("total_loss", "policy_loss", "value_loss", "policy_accuracy", "policy_entropy",
                          "policy_top5_accuracy"):
                    if k in metrics:
                        self.writer.add_scalar(f"train/{k}", metrics[k], self.global_step)
                # Auxiliary head losses (only log when non-zero)
                for k in ("ownership_loss", "score_loss"):
                    if metrics.get(k, 0.0) > 0:
                        self.writer.add_scalar(f"train/{k}", metrics[k], self.global_step)
                self.writer.add_scalar("train/grad_norm", grad_norm.item(), self.global_step)
                self.writer.add_scalar("train/learning_rate", self.scheduler.get_last_lr()[0], self.global_step)

            if val_loader is not None and self.global_step % self.config["training"]["val_frequency"] == 0:
                val_metrics = self.validate(val_loader)

                for k in ("total_loss", "policy_loss", "value_loss", "score_loss",
                          "policy_accuracy", "policy_entropy", "policy_top5_accuracy",
                          "ownership_loss"):
                    if k in val_metrics:
                        self.writer.add_scalar(f"val/{k}", val_metrics[k], self.global_step)

                logger.debug(f"Step {self.global_step}: Val loss = {val_metrics['total_loss']:.4f}")

                if val_metrics["total_loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["total_loss"]
                    self.save_checkpoint("best.pt", is_best=True)

                self.network.train()

            if self.global_step % self.config["training"]["checkpoint_frequency"] == 0:
                self.save_checkpoint(f"step_{self.global_step}.pt")

        cnt = epoch_metrics.pop("grad_norm_count", 0)
        epoch_metrics["grad_norm"] = epoch_metrics["grad_norm"] / max(1, cnt)
        steps_skipped_sum = epoch_metrics.pop("steps_skipped", 0)
        epoch_metrics["step_skip_rate"] = steps_skipped_sum / max(1, num_batches)

        for k in epoch_metrics:
            if k in ("grad_norm", "step_skip_rate"):
                continue
            epoch_metrics[k] /= max(1, num_batches)

        self.current_epoch += 1
        return epoch_metrics

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.network.eval()

        val_metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "score_loss": 0.0,
            "ownership_loss": 0.0,
            "total_loss": 0.0,
            "policy_accuracy": 0.0,
            "policy_top5_accuracy": 0.0,
            "value_mse": 0.0,
        }
        num_batches = 0

        with torch.no_grad():
            for batch_data in val_loader:
                batch = batch_to_dict(batch_data)
                states = batch["states"].to(self.device, non_blocking=True)
                action_masks = batch["action_masks"].to(self.device, non_blocking=True)
                policies = batch["policies"].to(self.device, non_blocking=True)
                values = batch["values"].to(self.device, non_blocking=True)
                score_margins = batch["score_margins"].to(self.device, non_blocking=True)
                ownerships = batch["ownerships"]

                own_target = None
                own_weight = 0.0
                own_valid_mask = None
                if self.ownership_weight > 0 and self.network.ownership_head is not None:
                    # [PERF FIX] Check sentinel on CPU before GPU transfer
                    valid_mask = ownerships.view(ownerships.shape[0], -1).min(dim=1).values >= 0
                    if valid_mask.any():
                        ownerships = ownerships.to(self.device, non_blocking=True)
                        own_target = ownerships
                        own_weight = self.ownership_weight
                        own_valid_mask = valid_mask.to(self.device)

                x_global = batch["x_global"]
                x_track = batch["x_track"]
                shop_ids = batch["shop_ids"]
                shop_feats = batch["shop_feats"]
                xg = x_global.to(self.device, non_blocking=True) if x_global is not None else None
                xt = x_track.to(self.device, non_blocking=True) if x_track is not None else None
                si = shop_ids.to(self.device, non_blocking=True) if shop_ids is not None else None
                sf = shop_feats.to(self.device, non_blocking=True) if shop_feats is not None else None
                target_score_dist = None
                if self.score_loss_weight > 0:
                    if self._score_bin_vals is None or self._score_bin_vals.device != score_margins.device:
                        self._score_bin_vals = torch.arange(
                            self.score_bins_min, self.score_bins_max + 1,
                            device=score_margins.device, dtype=torch.float32,
                        )
                    target_score_dist = make_gaussian_score_targets(
                        score_margins, self.score_utility_scale,
                        self.score_bins_min, self.score_bins_max,
                        self.score_target_sigma, bin_vals=self._score_bin_vals,
                    )
                if self.use_amp:
                    with autocast(device_type="cuda", dtype=self._autocast_dtype):
                        _, metrics = self.network.get_loss(
                            states, action_masks, policies, values,
                            self.policy_weight, self.value_weight,
                            target_score_dist=target_score_dist,
                            score_loss_weight=self.score_loss_weight,
                            target_ownership=own_target,
                            ownership_weight=own_weight,
                            ownership_valid_mask=own_valid_mask,
                            x_global=xg, x_track=xt, shop_ids=si, shop_feats=sf,
                        )
                else:
                    _, metrics = self.network.get_loss(
                        states, action_masks, policies, values,
                        self.policy_weight, self.value_weight,
                        target_score_dist=target_score_dist,
                        score_loss_weight=self.score_loss_weight,
                        target_ownership=own_target,
                        ownership_weight=own_weight,
                        ownership_valid_mask=own_valid_mask,
                        x_global=xg, x_track=xt, shop_ids=si, shop_feats=sf,
                    )

                for k, v in metrics.items():
                    if k in val_metrics:
                        val_metrics[k] += float(v)
                num_batches += 1

        for k in val_metrics:
            val_metrics[k] /= max(1, num_batches)

        return val_metrics

    def save_checkpoint(
        self, filename: str, is_best: bool = False, target_dir: Optional[Path] = None
    ) -> None:
        # CRITICAL: model_state_dict is ALWAYS the active training weights (raw/noisy).
        # EMA is saved separately under ema_state_dict for inference (selfplay/eval).
        # This prevents checkpoint corruption: train_iteration resumes from raw weights.
        checkpoint = {
            "model_state_dict": {k: v.cpu().clone() for k, v in self.network.state_dict().items()},
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
            "rng_state": _capture_rng_state(),
            "steps_per_iteration": self.total_train_steps,
        }
        if self.ema_enabled and self.ema_state_dict is not None:
            checkpoint["ema_state_dict"] = {k: v.cpu().clone() for k, v in self.ema_state_dict.items()}
            checkpoint["ema_decay"] = self.ema_decay
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        out_dir = target_dir if target_dir is not None else self.checkpoint_dir
        save_path = Path(out_dir) / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_torch_save(checkpoint, save_path)
        logger.debug("Saved checkpoint: %s", save_path)

        if target_dir is None:
            self._cleanup_checkpoints()

    def load_checkpoint(self, checkpoint_path: str) -> None:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        except Exception as e:
            logger.error("Failed to load checkpoint %s: %s", checkpoint_path, e)
            raise RuntimeError(
                f"Cannot load checkpoint {checkpoint_path}. File may be corrupted or missing."
            ) from e

        for key in ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict"):
            if key not in checkpoint:
                raise RuntimeError(
                    f"Checkpoint {checkpoint_path} missing required key '{key}' — possibly corrupted."
                )

        load_model_checkpoint(self.network, checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.current_epoch = checkpoint["current_epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]

        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if "rng_state" in checkpoint:
            _restore_rng_state(checkpoint["rng_state"])
        # Restore EMA from checkpoint when resuming (model_state_dict is raw; ema stays paired)
        if self.ema_enabled and self.ema_state_dict is not None and "ema_state_dict" in checkpoint:
            for k, v in checkpoint["ema_state_dict"].items():
                if k in self.ema_state_dict:
                    self.ema_state_dict[k].copy_(v.to(self.device))

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(f"Resuming from step {self.global_step}, epoch {self.current_epoch}")

    def _cleanup_checkpoints(self) -> None:
        keep_n = self.config["training"]["keep_last_n_checkpoints"]

        # PROTECTED files — never delete these
        protected = {"best_model.pt", "latest_model.pt", "best.pt"}

        # Clean step checkpoints
        checkpoints = sorted(
            self.checkpoint_dir.glob("step_*.pt"),
            key=lambda p: int(p.stem.split("_")[1]),
        )

        if len(checkpoints) > keep_n:
            for checkpoint in checkpoints[:-keep_n]:
                if checkpoint.name not in protected:
                    checkpoint.unlink()
                    logger.debug(f"Removed old checkpoint: {checkpoint}")

        # Clean old iteration checkpoints
        iter_checkpoints = sorted(
            self.checkpoint_dir.glob("iteration_*.pt"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        if len(iter_checkpoints) > keep_n:
            for checkpoint in iter_checkpoints[:-keep_n]:
                if checkpoint.name not in protected:
                    checkpoint.unlink()
                    logger.debug(f"Removed old iteration checkpoint: {checkpoint}")


def _split_indices(n: int, val_split: float, seed: int) -> Tuple[List[int], List[int]]:
    val_size = int(n * val_split)
    train_size = n - val_size

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()

    train_indices = perm[:train_size]
    val_indices = perm[train_size:]
    return train_indices, val_indices


def _estimate_total_train_steps(
    n_samples: int,
    val_split: float,
    batch_size: int,
    epochs: int,
) -> int:
    """Estimate total training steps for the LR scheduler."""
    n_train = int(n_samples * (1.0 - val_split))
    batches_per_epoch = math.ceil(n_train / batch_size)
    return batches_per_epoch * epochs


def train_iteration(
    iteration: int,
    data_path: str,
    config: dict,
    device: torch.device,
    previous_checkpoint: Optional[str] = None,
    replay_buffer=None,
    writer: Optional[SummaryWriter] = None,
    global_step_offset: int = 0,
    iteration_output_dir: Optional[Path] = None,
    merged_output_path: Optional[str] = None,
    *,
    force_resume_optimizer_state: bool = False,
    force_resume_scheduler_state: bool = False,
    force_resume_scaler_state: bool = False,
    force_resume_ema: bool = False,
) -> Tuple[str, Dict, int]:
    """
    Train the network for one iteration.

    CRITICAL FIX (v2): Each iteration gets a FRESH optimizer and scheduler
    (KataGo-style warm restart). Only model weights carry over from the
    previous checkpoint. This prevents the LR schedule collapse bug where
    accumulated global_step >> per-iteration total_train_steps caused LR
    to permanently collapse to min_lr after a few iterations.

    Args:
        iteration: Current iteration number.
        data_path: Path to the current iteration's self-play HDF5 data.
        config: Full configuration dict.
        device: torch device.
        previous_checkpoint: Path to previous model checkpoint (warm start).
        replay_buffer: Optional ReplayBuffer instance. If provided, training
                       uses merged data from the replay buffer instead of
                       just the current iteration's data.
        writer: Optional shared SummaryWriter for TensorBoard continuity.
                If provided, this single writer is reused across iterations
                so all metrics appear on one continuous line.
        global_step_offset: Starting global_step for this iteration. Passed
                            from the caller to ensure monotonically increasing
                            steps across iterations (even if models are rejected).
        iteration_output_dir: If provided, save iteration checkpoint here
                              (for transactional staging); else use checkpoint_dir.
        merged_output_path: If provided (with replay_buffer), write merged HDF5 here
                            for atomic staging; else use buffer_dir default.

    Returns:
        (checkpoint_path, avg_metrics, final_global_step)
    """
    logger.debug("Training iteration %d", iteration)

    network = create_network(config)

    if previous_checkpoint:
        logger.debug(f"Initializing model weights from {previous_checkpoint}")
        checkpoint = torch.load(previous_checkpoint, map_location=device, weights_only=False)
        load_model_checkpoint(network, checkpoint["model_state_dict"])

    # --- Replay buffer integration ---
    # If a replay buffer is provided, register the new data and use the
    # merged dataset for training. Otherwise fall back to single-iteration.
    effective_data_path = data_path
    if replay_buffer is not None:
        ok = replay_buffer.add_iteration(iteration, data_path)
        if not ok:
            raise RuntimeError(f"ReplayBuffer failed to register data_path={data_path}")
        if replay_buffer.has_enough_data():
            effective_data_path = replay_buffer.get_training_data(
                seed=config.get("seed", 42) + iteration,
                output_path=merged_output_path,
            )
            logger.debug(
                f"Replay buffer: {replay_buffer.total_positions} raw positions "
                f"from {replay_buffer.num_iterations} iterations (will cap/subsample to max_size={replay_buffer.max_size})"
            )
        else:
            logger.debug(
                f"Replay buffer has {replay_buffer.total_positions} positions "
                f"(min {replay_buffer.min_size}), using current iteration only"
            )

    dataset = PatchworkDataset(effective_data_path, config=config)
    if len(dataset) == 0:
        raise RuntimeError(
            f"Training dataset is empty (path={effective_data_path}). "
            "Cannot train. Check self-play output or replay buffer merge."
        )
    if replay_buffer is not None and replay_buffer.has_enough_data():
        logger.debug(f"Effective training dataset size (post-merge/subsample): {len(dataset)} positions")

    train_indices, val_indices = _split_indices(
        n=len(dataset),
        val_split=config["training"]["val_split"],
        seed=config["seed"] + iteration,  # FIX: vary val split seed per iteration
    )

    logger.debug(f"Train size: {len(train_indices)}, Val size: {len(val_indices)}")

    batch_size = int(config["training"]["batch_size"])
    num_epochs = config["training"]["epochs_per_iteration"]

    total_train_steps = _estimate_total_train_steps(
        n_samples=len(dataset),
        val_split=config["training"]["val_split"],
        batch_size=batch_size,
        epochs=num_epochs,
    )
    logger.debug(f"Estimated total training steps: {total_train_steps}")

    train_sampler = BatchIndexSampler(
        indices=train_indices,
        batch_size=batch_size,
        shuffle=True,
        seed=int(config["seed"]) + iteration,  # FIX: vary shuffle seed per iteration
    )
    val_sampler = BatchIndexSampler(
        indices=val_indices,
        batch_size=batch_size,
        shuffle=False,
        seed=int(config["seed"]),
    )

    # [PERF FIX] Use _BatchIterableDataset to call dataset[indices] once per batch
    # instead of 1024 per-sample calls. Default DataLoader does dataset[i] for each i,
    # which caused ~10s/batch (60x slower than expected) on in-memory data.
    hw = config.get("hardware", {}) or {}
    num_workers = 0  # In-memory: 0 is fastest (no pickle overhead)
    train_batch_ds = _BatchIterableDataset(dataset, train_sampler)
    val_batch_ds = _BatchIterableDataset(dataset, val_sampler)
    train_loader = DataLoader(
        train_batch_ds,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=hw.get("pin_memory", config["hardware"]["pin_memory"]),
        persistent_workers=bool(hw.get("persistent_workers", False)) if num_workers > 0 else False,
    )
    val_loader = DataLoader(
        val_batch_ds,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=hw.get("pin_memory", config["hardware"]["pin_memory"]),
        persistent_workers=bool(hw.get("persistent_workers", False)) if num_workers > 0 else False,
    )

    log_dir = Path(config["paths"]["logs_dir"])

    # CRITICAL: Optimizer state must come from the SAME checkpoint as model weights.
    # Adam moments for weights A applied to weights B = silent degradation/destabilization.
    optimizer_state_ckpt = previous_checkpoint if previous_checkpoint else None
    if optimizer_state_ckpt and previous_checkpoint:
        assert str(optimizer_state_ckpt) == str(previous_checkpoint), (
            "optimizer_state_checkpoint must equal model source (train_base); "
            "got optimizer=%s model=%s"
        ) % (optimizer_state_ckpt, previous_checkpoint)
    trainer = Trainer(
        network,
        config,
        device,
        log_dir,
        total_train_steps=total_train_steps,
        writer=writer,
        global_step_offset=global_step_offset,
        optimizer_state_checkpoint=optimizer_state_ckpt,
        model_source_checkpoint=previous_checkpoint,
        current_iteration=iteration,
        force_resume_optimizer_state=force_resume_optimizer_state,
        force_resume_scheduler_state=force_resume_scheduler_state,
        force_resume_scaler_state=force_resume_scaler_state,
        force_resume_ema=force_resume_ema,
    )
    logger.debug(f"TensorBoard logging from global_step={trainer.global_step}")

    all_metrics = []
    loss_exploded = False

    try:
        for epoch in range(num_epochs):
            if hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)
            elif hasattr(train_loader.dataset, "set_epoch"):
                train_loader.dataset.set_epoch(epoch)

            start_time = time.time()
            metrics = trainer.train_epoch(train_loader, val_loader)
            epoch_time = time.time() - start_time

            current_lr = trainer.scheduler.get_last_lr()[0]
            own_str = ""
            if metrics.get("ownership_loss", 0.0) > 0:
                own_str = f"  own_loss={metrics['ownership_loss']:.4f}  own_acc={metrics.get('ownership_accuracy', 0.0):.1%}"
            skip_str = f"  skip={metrics.get('step_skip_rate', 0):.1%}" if metrics.get("step_skip_rate", 0) > 0 else ""
            logger.info(
                f"  Epoch {epoch+1} done in {epoch_time:.1f}s | "
                f"loss={metrics['total_loss']:.4f}  pol_loss={metrics['policy_loss']:.4f}  val_loss={metrics['value_loss']:.4f}{own_str} | "
                f"pol_acc={metrics['policy_accuracy']:.1%}  top5={metrics['policy_top5_accuracy']:.1%} | "
                f"val_mse={metrics['value_mse']:.4f}  grad={metrics['grad_norm']:.3f}{skip_str} | "
                f"LR={current_lr:.2e}"
            )
            all_metrics.append(metrics)

            # Policy entropy monitoring
            if "policy_entropy" in metrics:
                if metrics["policy_entropy"] < 0.5:
                    logger.warning(f"  Low policy entropy ({metrics['policy_entropy']:.3f}) — potential policy collapse!")

            # Loss explosion detection: abort iteration if loss is NaN/Inf
            if math.isnan(metrics['total_loss']) or math.isinf(metrics['total_loss']):
                logger.error(f"  Loss explosion at epoch {epoch+1}! Aborting iteration.")
                loss_exploded = True
                break
    finally:
        # [m7 FIX] Always close the SummaryWriter to avoid file handle leaks
        trainer.close()

    # CRITICAL: Never save checkpoint if loss exploded — model may be corrupted
    if loss_exploded:
        raise RuntimeError(
            "Training aborted due to NaN/Inf loss. No checkpoint saved. "
            "Check learning rate, gradient scaling, or data quality."
        )

    if iteration_output_dir is not None:
        checkpoint_path = Path(iteration_output_dir) / f"iteration_{iteration:03d}.pt"
        trainer.save_checkpoint(checkpoint_path.name, target_dir=Path(iteration_output_dir))
    else:
        checkpoint_path = trainer.checkpoint_dir / f"iteration_{iteration:03d}.pt"
        trainer.save_checkpoint(checkpoint_path.name)

    avg_metrics: Dict[str, float] = {}
    if all_metrics:
        for key in all_metrics[0].keys():
            avg_metrics[key] = float(np.mean([m[key] for m in all_metrics]))
    else:
        avg_metrics = {"total_loss": float("inf"), "policy_loss": 0.0, "value_loss": 0.0,
                       "policy_accuracy": 0.0, "policy_top5_accuracy": 0.0, "value_mse": 0.0,
                       "grad_norm": 0.0}

    return str(checkpoint_path), avg_metrics, trainer.global_step


if __name__ == "__main__":
    print("Training pipeline implementation complete.")