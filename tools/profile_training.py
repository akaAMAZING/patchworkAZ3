"""
Performance profiling script for Patchwork AlphaZero training pipeline.

Usage:
  python tools/profile_training.py --config configs/config_best.yaml [--data path/to/merged_training.h5] [--steps 250] [--profiler]
  
  --steps: number of training steps to profile (default 250, ~200 for timing stats)
  --profiler: run torch.profiler and export trace
  --data: override data path (default: auto-detect from runs or replay buffer)
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.network.model import create_network, load_model_checkpoint
from src.network.d4_augmentation_gpu import apply_d4_augment_batch_gpu
from src.training.trainer import (
    PatchworkDataset,
    BatchIndexSampler,
    _BatchIterableDataset,
    Trainer,
    _split_indices,
    _estimate_total_train_steps,
    batch_to_dict,
)
from torch.utils.data import DataLoader


def find_training_data(config: dict) -> str | None:
    """Find merged_training.h5 or selfplay.h5 for profiling."""
    run_root = Path(config.get("paths", {}).get("run_root", "runs")) / config.get("paths", {}).get("run_id", "patchwork_production")
    # Staging
    for staging in run_root.glob("staging/iter_*/merged_training.h5"):
        if staging.exists():
            return str(staging)
    for sp in run_root.glob("staging/iter_*/selfplay.h5"):
        if sp.exists():
            return str(sp)
    # Committed
    for comm in run_root.glob("committed/iter_*/selfplay.h5"):
        if comm.exists():
            return str(comm)
    # E2E artifacts
    artifact = Path("artifacts/e2e_check/merged_training.h5")
    if artifact.exists():
        return str(artifact)
    return None


def create_minimal_h5(path: Path, n_positions: int = 2048, config: dict = None) -> str:
    """Create minimal synthetic HDF5 for profiling when no real data exists.
    gold_v2_32ch: shop_ids (N, 33) -1/0..21, shop_feats (N, 33, 10)."""
    import h5py
    from src.network.gold_v2_constants import C_SPATIAL_ENC, F_GLOBAL, C_TRACK, TRACK_LEN, NMAX, F_SHOP
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    enc = (config or {}).get("data", {}) or {}
    multimodal = str(enc.get("encoding_version", "")).lower() in ("gold_v2_32ch", "gold_v2_multimodal")
    c_spatial = C_SPATIAL_ENC
    c_global = F_GLOBAL
    track_shape = (C_TRACK, TRACK_LEN)
    with h5py.File(path, "w") as f:
        f.create_dataset("spatial_states" if multimodal else "states",
                         data=np.random.randn(n_positions, c_spatial, 9, 9).astype(np.float32),
                         compression="lzf")
        if multimodal:
            f.create_dataset("global_states", data=np.random.randn(n_positions, c_global).astype(np.float32), compression="lzf")
            f.create_dataset("track_states", data=np.random.randn(n_positions, *track_shape).astype(np.float32), compression="lzf")
            shop_ids = np.full((n_positions, NMAX), -1, dtype=np.int16)
            shop_ids[:, :min(3, NMAX)] = np.random.randint(1, 22, (n_positions, min(3, NMAX)), dtype=np.int16)
            f.create_dataset("shop_ids", data=shop_ids, compression="lzf")
            f.create_dataset("shop_feats", data=np.random.randn(n_positions, NMAX, F_SHOP).astype(np.float32), compression="lzf")
        mask = np.zeros((n_positions, 2026), dtype=np.uint8)
        mask[:, 0] = 1  # pass always legal
        mask[:, 1:82] = 1  # some patch positions
        f.create_dataset("action_masks", data=mask, compression="lzf")
        policy = np.zeros((n_positions, 2026), dtype=np.float32)
        policy[:, 0] = 0.8
        policy[:, 1:10] = 0.02
        f.create_dataset("policies", data=policy, compression="lzf")
        f.create_dataset("values", data=np.random.randn(n_positions, 1).astype(np.float32) * 0.3, compression="lzf")
        f.create_dataset("ownerships", data=np.random.rand(n_positions, 2, 9, 9).astype(np.float32), compression="lzf")
        f.create_dataset("score_margins", data=np.random.randn(n_positions).astype(np.float32) * 5, compression="lzf")
        f.create_dataset("slot_piece_ids", data=np.random.randint(1, 22, (n_positions, 3), dtype=np.int16), compression="lzf")
    return str(path)


def _prefetch_iter(loader, prefetch_batches: int = 2):
    """Prefetch batches in background thread."""
    import queue
    import threading
    q = queue.Queue(maxsize=max(1, prefetch_batches))

    def worker():
        for batch in loader:
            q.put(batch)
        q.put(None)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    while True:
        batch = q.get()
        if batch is None:
            break
        yield batch
    t.join()


def run_instrumented(config: dict, data_path: str, device: torch.device, n_steps: int = 250, run_profiler: bool = False, use_prefetch: bool = False):
    """Run training with micro-timing and optional torch.profiler."""
    dataset = PatchworkDataset(data_path, config=config)
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty")
    
    batch_size = int(config["training"]["batch_size"])
    val_split = config["training"].get("val_split", 0.08)
    train_indices, val_indices = _split_indices(len(dataset), val_split, config.get("seed", 42))
    
    total_train_steps = _estimate_total_train_steps(len(dataset), val_split, batch_size, 1)
    train_sampler = BatchIndexSampler(train_indices, batch_size, shuffle=True, seed=config.get("seed", 42))
    train_batch_ds = _BatchIterableDataset(dataset, train_sampler)

    hw = config.get("hardware", {}) or {}
    loader = DataLoader(
        train_batch_ds,
        batch_size=None,
        num_workers=0,
        pin_memory=hw.get("pin_memory", config.get("hardware", {}).get("pin_memory", True)),
        persistent_workers=False,
    )
    
    network = create_network(config)
    log_dir = Path(config.get("paths", {}).get("logs_dir", "logs"))
    trainer = Trainer(network, config, device, log_dir, total_train_steps=total_train_steps)
    
    # Micro-timing accumulators
    t_next = []
    t_h2d = []
    t_forward_backward = []
    t_logging = []
    t_total = []
    
    if use_prefetch and device.type == "cuda":
        data_iter = _prefetch_iter(loader, prefetch_batches=2)
    else:
        data_iter = iter(loader)
    trainer.network.train()
    
    if run_profiler:
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=2, warmup=5, active=15, repeat=1),
            on_trace_ready=lambda p: p.export_chrome_trace("tools/profile_trace.json"),
            record_shapes=False,
            profile_memory=False,
        )
        prof.start()
    
    for step in range(min(n_steps, len(loader))):
        step_start = time.perf_counter()
        
        t0 = time.perf_counter()
        try:
            batch_data = next(data_iter)
        except StopIteration:
            if use_prefetch and device.type == "cuda":
                data_iter = _prefetch_iter(loader, prefetch_batches=2)
            else:
                data_iter = iter(loader)
            batch_data = next(data_iter)
        t1 = time.perf_counter()
        t_next.append(t1 - t0)

        batch = batch_to_dict(batch_data)
        states = batch["states"]
        action_masks = batch["action_masks"]
        policies = batch["policies"]
        values = batch["values"]
        score_margins = batch["score_margins"]
        ownerships = batch["ownerships"]
        x_global = batch["x_global"]
        x_track = batch["x_track"]
        shop_ids = batch["shop_ids"]
        shop_feats = batch["shop_feats"]
        slot_piece_ids = batch.get("slot_piece_ids")

        t2 = time.perf_counter()
        states = states.to(device, non_blocking=True)
        action_masks = action_masks.to(device, non_blocking=True)
        policies = policies.to(device, non_blocking=True)
        values = values.to(device, non_blocking=True)
        score_margins = score_margins.to(device, non_blocking=True)
        ownerships = ownerships.to(device, non_blocking=True) if trainer.ownership_weight > 0 else ownerships
        xg = x_global.to(device, non_blocking=True) if x_global is not None else None
        xt = x_track.to(device, non_blocking=True) if x_track is not None else None
        si = shop_ids.to(device, non_blocking=True) if shop_ids is not None else None
        sf = shop_feats.to(device, non_blocking=True) if shop_feats is not None else None
        d4_on_gpu = bool(config.get("training", {}).get("d4_on_gpu", False))
        if d4_on_gpu and slot_piece_ids is not None and device.type == "cuda":
            transform_indices = torch.randint(0, 8, (states.shape[0],), device=device, dtype=torch.long)
            slot_ids_t = slot_piece_ids.to(device, non_blocking=True)
            states, policies, action_masks, ownerships = apply_d4_augment_batch_gpu(
                states, policies, action_masks, ownerships,
                slot_ids_t, transform_indices, device,
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        t3 = time.perf_counter()
        t_h2d.append(t3 - t2)

        t4 = time.perf_counter()
        trainer.optimizer.zero_grad(set_to_none=True)
        own_target = ownerships if (trainer.ownership_weight > 0 and trainer.network.ownership_head) else None
        own_weight = trainer.ownership_weight if own_target is not None else 0.0
        own_valid_mask = None
        if own_target is not None:
            valid_mask = ownerships.view(ownerships.shape[0], -1).min(dim=1).values >= 0
            if valid_mask.any():
                own_valid_mask = valid_mask.to(device)
        
        from torch.amp import autocast
        if trainer.use_amp:
            with autocast(device_type="cuda", dtype=trainer._autocast_dtype):
                loss, metrics = trainer.network.get_loss(
                    states, action_masks, policies, values,
                    trainer.policy_weight, trainer.value_weight,
                    target_score=score_margins, score_loss_weight=trainer.score_loss_weight,
                    target_ownership=own_target, ownership_weight=own_weight,
                    ownership_valid_mask=own_valid_mask,
                    x_global=xg, x_track=xt, shop_ids=si, shop_feats=sf,
                )
        else:
            loss, metrics = trainer.network.get_loss(
                states, action_masks, policies, values,
                trainer.policy_weight, trainer.value_weight,
                target_score=score_margins, score_loss_weight=trainer.score_loss_weight,
                target_ownership=own_target, ownership_weight=own_weight,
                ownership_valid_mask=own_valid_mask,
                x_global=xg, x_track=xt, shop_ids=si, shop_feats=sf,
            )
        
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainer.network.parameters(), trainer.max_grad_norm)
        trainer.optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t5 = time.perf_counter()
        t_forward_backward.append(t5 - t4)
        
        t6 = time.perf_counter()
        _ = grad_norm.item()
        for k, v in metrics.items():
            _ = float(v)
        t7 = time.perf_counter()
        t_logging.append(t7 - t6)
        
        t_total.append(time.perf_counter() - step_start)
        trainer.global_step += 1
        trainer.scheduler.step()
        
        if run_profiler:
            prof.step()
    
    if run_profiler:
        prof.stop()
        print("\n[PROFILER] Trace exported to tools/profile_trace.json")
    
    t_next = np.array(t_next) * 1000
    t_h2d = np.array(t_h2d) * 1000
    t_fb = np.array(t_forward_backward) * 1000
    t_log = np.array(t_logging) * 1000
    t_total = np.array(t_total) * 1000
    
    return {
        "t_next_mean_ms": float(np.mean(t_next)),
        "t_next_std_ms": float(np.std(t_next)),
        "t_h2d_mean_ms": float(np.mean(t_h2d)),
        "t_h2d_std_ms": float(np.std(t_h2d)),
        "t_forward_backward_mean_ms": float(np.mean(t_fb)),
        "t_forward_backward_std_ms": float(np.std(t_fb)),
        "t_logging_mean_ms": float(np.mean(t_log)),
        "t_logging_std_ms": float(np.std(t_log)),
        "t_total_mean_ms": float(np.mean(t_total)),
        "t_total_std_ms": float(np.std(t_total)),
        "steps_per_sec": 1000.0 / float(np.mean(t_total)),
        "positions_per_sec": batch_size * (1000.0 / float(np.mean(t_total))),
        "n_steps": len(t_total),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config_best.yaml")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--profiler", action="store_true")
    parser.add_argument("--synthetic", action="store_true", help="Create synthetic data if no real data")
    parser.add_argument("--prefetch", action="store_true", help="Use background thread prefetch (for validation)")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    data_path = args.data
    if not data_path:
        data_path = find_training_data(config)
    if not data_path or not Path(data_path).exists():
        if args.synthetic:
            data_path = create_minimal_h5(
                Path("artifacts/profile_tmp/merged_training.h5"),
                n_positions=8192,
                config=config,
            )
            print(f"Created synthetic data: {data_path}")
        else:
            print("No training data found. Use --data <path> or --synthetic")
            sys.exit(1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prefetch_str = " (with prefetch)" if args.prefetch else ""
    print(f"Profiling on {device} for {args.steps} steps{prefetch_str}...")

    results = run_instrumented(
        config, data_path, device,
        n_steps=args.steps,
        run_profiler=args.profiler,
        use_prefetch=args.prefetch,
    )
    
    print("\n" + "=" * 70)
    print("MICRO-TIMING RESULTS (per-step averages over ~{} steps)".format(results["n_steps"]))
    print("=" * 70)
    print(f"  (i) next(data_iter) [CPU aug/collation]: {results['t_next_mean_ms']:.2f} ± {results['t_next_std_ms']:.2f} ms")
    print(f"  (ii) H2D transfer:                       {results['t_h2d_mean_ms']:.2f} ± {results['t_h2d_std_ms']:.2f} ms")
    print(f"  (iii) forward+backward+step:              {results['t_forward_backward_mean_ms']:.2f} ± {results['t_forward_backward_std_ms']:.2f} ms")
    print(f"  (iv) logging/sync (.item etc):           {results['t_logging_mean_ms']:.2f} ± {results['t_logging_std_ms']:.2f} ms")
    print(f"  TOTAL per step:                          {results['t_total_mean_ms']:.2f} ± {results['t_total_std_ms']:.2f} ms")
    print()
    print(f"  Throughput: {results['steps_per_sec']:.2f} steps/sec | {results['positions_per_sec']:.0f} positions/sec")
    
    fb = results["t_forward_backward_mean_ms"]
    cpu_total = results["t_next_mean_ms"] + results["t_logging_mean_ms"]
    print(f"\n  GPU time / CPU overhead ratio: {fb:.1f} / {cpu_total:.1f} = {fb/max(cpu_total,0.01):.2f}")
    if fb < cpu_total:
        print("  >>> GPU compute is SMALLER than CPU overhead — data/sync bound!")
    
    return results


if __name__ == "__main__":
    main()
