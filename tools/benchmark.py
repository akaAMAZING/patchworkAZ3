#!/usr/bin/env python
"""
Performance benchmark for Patchwork AlphaZero training hotspots.

Measures the throughput of critical path components:
  1. State encoding (game state → 56x9x9 spatial tensor)
  2. Legal action encoding (action list → indices + mask)
  3. MCTS simulation rate (pure MCTS, no neural network)
  4. Network inference throughput (forward pass on GPU/CPU)
  5. Shard merge (NPZ files → HDF5)

USAGE:
    python tools/benchmark.py --config configs/overnight_strong.yaml
    python tools/benchmark.py --config configs/overnight_strong.yaml --quick
    python tools/benchmark.py --config configs/overnight_strong.yaml --output bench_results.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.game.patchwork_engine import (
    apply_action_unchecked,
    current_player_fast,
    legal_actions_fast,
    new_game,
    terminal_fast,
)
from src.mcts.alphazero_mcts_optimized import encode_legal_actions_fast
from src.network.encoder import GoldV2StateEncoder
from src.network.model import create_network


# ---------------------------------------------------------------------------
# 1. State Encoder
# ---------------------------------------------------------------------------
def bench_state_encoder(num_states: int, seed: int) -> dict:
    """Measure state encoding throughput (states/sec). 56ch gold_v2 spatial."""
    enc = GoldV2StateEncoder()
    rng = random.Random(seed)
    states, players = [], []
    s = new_game(seed=seed)
    while len(states) < num_states:
        states.append(s.copy())
        players.append(current_player_fast(s))
        legal = legal_actions_fast(s)
        if not legal:
            s = new_game(seed=rng.randint(1, 10_000_000))
            continue
        s = apply_action_unchecked(s, rng.choice(legal))

    # Warmup
    for st, p in zip(states[:4], players[:4]):
        enc.encode_state_multimodal(st, p)

    t0 = time.perf_counter()
    for st, p in zip(states, players):
        enc.encode_state_multimodal(st, p)
    dt = time.perf_counter() - t0

    return {
        "state_encoder_s": round(dt, 4),
        "state_encoder_per_s": round(num_states / dt, 1),
    }


# ---------------------------------------------------------------------------
# 2. Legal Action Encoding
# ---------------------------------------------------------------------------
def bench_legal_encoding(num_states: int, seed: int) -> dict:
    """Measure legal action encoding throughput (actions/sec)."""
    rng = random.Random(seed + 11)
    s = new_game(seed=seed + 11)
    action_lists = []
    for _ in range(num_states):
        legal = legal_actions_fast(s)
        if not legal:
            s = new_game(seed=rng.randint(1, 10_000_000))
            legal = legal_actions_fast(s)
        action_lists.append(legal)
        s = apply_action_unchecked(s, rng.choice(legal))

    t0 = time.perf_counter()
    total_actions = 0
    for legal in action_lists:
        _, _ = encode_legal_actions_fast(legal)
        total_actions += len(legal)
    dt = time.perf_counter() - t0

    return {
        "legal_encode_s": round(dt, 4),
        "legal_encode_actions_per_s": round(total_actions / dt, 1) if dt > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# 3. Pure MCTS (no network)
# ---------------------------------------------------------------------------
def bench_pure_mcts(num_games: int, sims: int, seed: int) -> dict:
    """Measure pure MCTS throughput (simulations/sec)."""
    from src.training.evaluation import PureMCTSEvaluator

    rng = random.Random(seed + 33)
    total_sims = 0
    total_moves = 0
    t0 = time.perf_counter()

    for g in range(num_games):
        game_seed = seed + g * 7
        evaluator = PureMCTSEvaluator(simulations=sims, seed=game_seed)
        s = new_game(seed=game_seed)
        for move in range(100):
            if terminal_fast(s):
                break
            action = evaluator.get_move(s, seed_offset=move)
            s = apply_action_unchecked(s, action)
            total_moves += 1
            total_sims += sims

    dt = time.perf_counter() - t0
    return {
        "pure_mcts_s": round(dt, 4),
        "pure_mcts_sims_per_s": round(total_sims / dt, 1) if dt > 0 else 0.0,
        "pure_mcts_moves_per_s": round(total_moves / dt, 1) if dt > 0 else 0.0,
        "pure_mcts_total_games": num_games,
        "pure_mcts_total_sims": total_sims,
    }


# ---------------------------------------------------------------------------
# 4. Network Inference
# ---------------------------------------------------------------------------
def bench_network_inference(config: dict, batch_sizes: list[int], num_batches: int) -> dict:
    """Measure neural network inference throughput (samples/sec). 56ch gold_v2."""
    from tools.deep_preflight import _dummy_gold_v2_batch

    device_name = config.get("hardware", {}).get("device", "cpu")
    device = torch.device(device_name if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    inf = config.get("inference", {}) or {}
    use_amp = inf.get("use_amp", True) and device.type == "cuda"
    amp_dtype_str = str(inf.get("amp_dtype", "float16")).lower()
    amp_dtype = torch.bfloat16 if amp_dtype_str == "bfloat16" else torch.float16

    network = create_network(config)
    network = network.to(device)
    network.eval()

    def _forward(bs):
        st, xg, xt, si, sf = _dummy_gold_v2_batch(bs, device)
        mask = torch.ones(bs, 2026, device=device)
        return network(st, mask, x_global=xg, x_track=xt, shop_ids=si, shop_feats=sf)

    results = {}
    for bs in batch_sizes:
        # Warmup (3 forward passes)
        with torch.inference_mode():
            for _ in range(3):
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        _ = _forward(bs)
                else:
                    _ = _forward(bs)

        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        with torch.inference_mode():
            for _ in range(num_batches):
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        _ = _forward(bs)
                else:
                    _ = _forward(bs)

        if device.type == "cuda":
            torch.cuda.synchronize()

        dt = time.perf_counter() - t0
        samples = bs * num_batches
        results[f"inference_bs{bs}_s"] = round(dt, 4)
        results[f"inference_bs{bs}_per_s"] = round(samples / dt, 1) if dt > 0 else 0.0

    results["inference_device"] = str(device)
    results["inference_amp"] = use_amp
    results["inference_amp_dtype"] = amp_dtype_str if use_amp else "none"

    del network
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# 5. Training Steps/sec (training loop throughput)
# ---------------------------------------------------------------------------
def bench_training_steps(
    config: dict, num_steps: int, batch_size: int, seed: int
) -> dict:
    """Measure training step throughput (steps/sec, wall-clock per epoch estimate)."""
    from src.training.trainer import PatchworkDataset, BatchIndexSampler, Trainer
    from torch.utils.data import DataLoader

    device_name = config.get("hardware", {}).get("device", "cuda")
    device = torch.device(
        device_name if device_name == "cuda" and torch.cuda.is_available() else "cpu"
    )

    # Create minimal in-memory HDF5 for benchmark (56ch gold_v2)
    from src.network.gold_v2_constants import C_SPATIAL, F_GLOBAL, C_TRACK, TRACK_LEN, NMAX, F_SHOP
    with tempfile.TemporaryDirectory(prefix="pw_train_bench_") as td:
        root = Path(td)
        h5_path = root / "bench_train.h5"
        rng = np.random.default_rng(seed)
        n_samples = max(batch_size * num_steps + 1024, 10000)
        with __import__("h5py").File(h5_path, "w") as f:
            f.create_dataset(
                "spatial_states", data=rng.standard_normal((n_samples, C_SPATIAL, 9, 9)).astype(np.float32)
            )
            f.create_dataset("global_states", data=rng.standard_normal((n_samples, F_GLOBAL)).astype(np.float32))
            f.create_dataset("track_states", data=rng.standard_normal((n_samples, C_TRACK, TRACK_LEN)).astype(np.float32))
            f.create_dataset("shop_ids", data=np.full((n_samples, NMAX), -1, dtype=np.int16))
            f.create_dataset("shop_feats", data=rng.standard_normal((n_samples, NMAX, F_SHOP)).astype(np.float32))
            f.create_dataset(
                "action_masks",
                data=(rng.random((n_samples, 2026)) > 0.1).astype(np.float32),
            )
            f.create_dataset(
                "policies",
                data=rng.dirichlet(np.ones(2026) * 0.01, size=n_samples).astype(np.float32),
            )
            f.create_dataset(
                "values",
                data=np.clip(rng.standard_normal((n_samples,)), -0.99, 0.99).astype(np.float32),
            )
            f.create_dataset("score_margins", data=np.zeros(n_samples, dtype=np.float32))
            f.create_dataset(
                "ownerships",
                data=rng.random((n_samples, 2, 9, 9)).astype(np.float32),
            )

        dataset = PatchworkDataset(str(h5_path), config)
        train_cfg = config.get("training", {})
        batch_size_actual = int(train_cfg.get("batch_size", batch_size))
        total_samples = len(dataset)
        n_train = int(total_samples * (1 - train_cfg.get("val_split", 0.08)))
        train_indices = list(range(n_train))
        steps_to_run = min(num_steps, (n_train + batch_size_actual - 1) // batch_size_actual)
        total_train_steps = steps_to_run * 2  # epochs estimate for scheduler

        sampler = BatchIndexSampler(
            indices=train_indices,
            batch_size=batch_size_actual,
            shuffle=True,
            seed=seed,
        )
        loader = DataLoader(
            dataset,
            batch_size=None,
            sampler=sampler,
            num_workers=0,
            pin_memory=config.get("hardware", {}).get("pin_memory", True),
        )

        network = create_network(config)
        log_dir = root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        # Disable EMA for benchmark (pure throughput)
        cfg_no_ema = dict(config)
        if "training" not in cfg_no_ema:
            cfg_no_ema["training"] = {}
        cfg_no_ema["training"]["ema"] = {"enabled": False}

        trainer = Trainer(
            network, cfg_no_ema, device, log_dir, total_train_steps=total_train_steps
        )

        # Warmup
        for i, batch in enumerate(loader):
            if i >= 3:
                break
            _ = trainer.train_epoch(loader, val_loader=None)
            break  # train_epoch consumes full loader

        # Reset loader for actual benchmark
        sampler.set_epoch(0)
        loader = DataLoader(
            dataset,
            batch_size=None,
            sampler=sampler,
            num_workers=0,
            pin_memory=config.get("hardware", {}).get("pin_memory", True),
        )

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        steps_done = 0
        for epoch in range(2):
            sampler.set_epoch(epoch)
            for _, batch in enumerate(loader):
                if steps_done >= steps_to_run:
                    break
                from src.training.trainer import batch_to_dict
                b = batch_to_dict(batch)
                states = b["states"].to(device, non_blocking=True)
                action_masks = b["action_masks"].to(device, non_blocking=True)
                policies = b["policies"].to(device, non_blocking=True)
                values = b["values"].to(device, non_blocking=True)
                action_masks = action_masks.to(device, non_blocking=True)
                policies = policies.to(device, non_blocking=True)
                values = values.to(device, non_blocking=True)
                trainer.optimizer.zero_grad(set_to_none=True)
                if trainer.use_amp:
                    with torch.amp.autocast(device_type="cuda", dtype=trainer._autocast_dtype):
                        loss, _ = trainer.network.get_loss(
                            states, action_masks, policies, values,
                            trainer.policy_weight, trainer.value_weight,
                        )
                    if trainer.scaler is not None:
                        trainer.scaler.scale(loss).backward()
                        trainer.scaler.step(trainer.optimizer)
                        trainer.scaler.update()
                    else:
                        loss.backward()
                        trainer.optimizer.step()
                else:
                    loss, _ = trainer.network.get_loss(
                        states, action_masks, policies, values,
                        trainer.policy_weight, trainer.value_weight,
                    )
                    loss.backward()
                    trainer.optimizer.step()
                steps_done += 1
                if steps_done >= steps_to_run:
                    break
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

    steps_per_sec = steps_done / dt if dt > 0 else 0.0
    return {
        "train_steps": steps_done,
        "train_wall_s": round(dt, 4),
        "train_steps_per_s": round(steps_per_sec, 2),
        "train_batch_size": batch_size_actual,
    }


# ---------------------------------------------------------------------------
# 6. Shard Merge (formerly 5)
# ---------------------------------------------------------------------------
def _write_fake_shards(shard_dir: Path, num_shards: int, pos_per_shard: int, seed: int) -> None:
    from src.network.gold_v2_constants import C_SPATIAL, F_GLOBAL, C_TRACK, TRACK_LEN, NMAX, F_SHOP
    rng = np.random.default_rng(seed)
    shard_dir.mkdir(parents=True, exist_ok=True)
    for i in range(num_shards):
        n = pos_per_shard
        np.savez(
            shard_dir / f"bench_{i:03d}.npz",
            spatial_states=rng.standard_normal((n, C_SPATIAL, 9, 9)).astype(np.float32),
            global_states=rng.standard_normal((n, F_GLOBAL)).astype(np.float32),
            track_states=rng.standard_normal((n, C_TRACK, TRACK_LEN)).astype(np.float32),
            shop_ids=np.full((n, NMAX), -1, dtype=np.int16),
            shop_feats=rng.standard_normal((n, NMAX, F_SHOP)).astype(np.float32),
            action_masks=(rng.random((n, 2026)) > 0.95).astype(np.float32),
            policies=rng.random((n, 2026)).astype(np.float32),
            values=rng.standard_normal((n,)).astype(np.float32),
            ownerships=rng.random((n, 2, 9, 9)).astype(np.float32),
        )


def bench_shard_merge(config: dict, quick: bool, seed: int) -> dict:
    """Measure shard → HDF5 merge throughput (positions/sec)."""
    from src.training.selfplay_optimized_integration import SelfPlayGenerator

    num_shards = 8 if quick else 24
    pos_per_shard = 24 if quick else 64

    with tempfile.TemporaryDirectory(prefix="pw_bench_") as td:
        root = Path(td)
        cfg = dict(config)
        cfg["paths"] = dict(config["paths"])
        cfg["paths"]["selfplay_dir"] = str(root / "selfplay")

        gen = SelfPlayGenerator(cfg)
        shard_dir = Path(cfg["paths"]["selfplay_dir"]) / "bench_shards"
        _write_fake_shards(shard_dir, num_shards=num_shards,
                           pos_per_shard=pos_per_shard, seed=seed)
        summaries = [{
            "num_positions": pos_per_shard,
            "game_length": 40,
            "winner": 0,
            "final_score_diff": 1.0,
        }] * num_shards

        out_h5 = Path(cfg["paths"]["selfplay_dir"]) / "bench_out.h5"
        t0 = time.perf_counter()
        gen._merge_shards(shard_dir, out_h5, summaries)
        dt = time.perf_counter() - t0

        total_positions = num_shards * pos_per_shard
        return {
            "shard_merge_s": round(dt, 4),
            "shard_merge_positions_per_s": round(total_positions / dt, 1) if dt > 0 else 0.0,
            "shard_merge_total_positions": int(total_positions),
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _print_section(title: str) -> None:
    print(f"\n{'-'*50}")
    print(f"  {title}")
    print(f"{'-'*50}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Patchwork training hotspots")
    parser.add_argument("--config", default="configs/config_best.yaml", help="Config path")
    parser.add_argument("--quick", action="store_true", help="Run fast benchmark subset")
    parser.add_argument("--output", default="", help="Optional JSON output file")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = int(config.get("seed", 42))

    num_states = 64 if args.quick else 256
    mcts_games = 2 if args.quick else 6
    mcts_sims = 16 if args.quick else 32
    batch_sizes = [1, 64, 256] if args.quick else [1, 32, 128, 256, 512]
    num_batches = 10 if args.quick else 40

    metrics: dict = {}

    # 1. State encoder
    _print_section("State Encoder")
    result = bench_state_encoder(num_states=num_states, seed=seed)
    metrics.update(result)
    print(f"  {result['state_encoder_per_s']:.0f} states/sec ({num_states} states)")

    # 2. Legal encoding
    _print_section("Legal Action Encoding")
    result = bench_legal_encoding(num_states=num_states, seed=seed)
    metrics.update(result)
    print(f"  {result['legal_encode_actions_per_s']:.0f} actions/sec")

    # 3. Pure MCTS
    _print_section("Pure MCTS (no network)")
    result = bench_pure_mcts(num_games=mcts_games, sims=mcts_sims, seed=seed)
    metrics.update(result)
    print(f"  {result['pure_mcts_sims_per_s']:.0f} sims/sec "
          f"({result['pure_mcts_total_sims']} total over {mcts_games} games)")

    # 4. Network inference
    _print_section("Network Inference")
    result = bench_network_inference(config, batch_sizes=batch_sizes, num_batches=num_batches)
    metrics.update(result)
    amp_str = f", dtype={result.get('inference_amp_dtype', 'float16')}" if result.get("inference_amp") else ""
    print(f"  Device: {result['inference_device']}, AMP: {result['inference_amp']}{amp_str}")
    for bs in batch_sizes:
        key = f"inference_bs{bs}_per_s"
        if key in result:
            print(f"  batch_size={bs:>4d}: {result[key]:>8.0f} samples/sec")

    # 5. Training steps/sec
    _print_section("Training Throughput")
    try:
        batch_size = int(config.get("training", {}).get("batch_size", 1024))
        result = bench_training_steps(
            config, num_steps=50 if args.quick else 100, batch_size=batch_size, seed=seed
        )
        metrics.update(result)
        print(f"  {result['train_steps_per_s']:.1f} steps/sec ({result['train_steps']} steps, "
              f"batch={result['train_batch_size']})")
    except Exception as e:
        print(f"  SKIPPED: {e}")

    # 6. Shard merge
    _print_section("Shard Merge (HDF5)")
    result = bench_shard_merge(config=config, quick=args.quick, seed=seed)
    metrics.update(result)
    print(f"  {result['shard_merge_positions_per_s']:.0f} positions/sec "
          f"({result['shard_merge_total_positions']} total)")

    # Summary
    _print_section("Summary")
    print(json.dumps(metrics, indent=2))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
