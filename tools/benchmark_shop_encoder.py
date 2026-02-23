#!/usr/bin/env python
"""Profile ShopEncoder attention on RTX 3080 (or any CUDA device).

Runs ShopEncoder forward for batch sizes [256, 512, 1024], Nmax=33, F=10,
100 iterations each. Reports avg ms/iter and % of total model forward time.

Also tests sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=True).

Usage:
    python tools/benchmark_shop_encoder.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.network.model import ShopEncoder, PatchworkNetwork, create_network


def _benchmark_shop_encoder(batch_sizes, n_iters=100, device=None):
    """Benchmark ShopEncoder alone."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = ShopEncoder(num_piece_ids=34, feat_dim=10, embed_dim=32, out_dim=128).to(device).eval()
    NMAX, F = 33, 10

    results = []
    with torch.inference_mode():
        for B in batch_sizes:
            shop_ids = torch.randint(-1, 34, (B, NMAX), device=device, dtype=torch.int64)
            shop_ids[:, 10:] = -1  # pad
            shop_feats = torch.randn(B, NMAX, F, device=device, dtype=torch.float32)

            torch.cuda.synchronize() if device.type == "cuda" else None
            t0 = time.perf_counter()
            for _ in range(n_iters):
                _ = encoder(shop_ids, shop_feats)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000 / n_iters
            results.append((B, elapsed_ms))
    return results


def _benchmark_full_model(batch_sizes, n_iters=100, device=None):
    """Benchmark full PatchworkNetwork forward (gold_v2 with FiLM)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = {
        "network": {
            "input_channels": 56,
            "num_res_blocks": 18,
            "channels": 128,
            "policy_channels": 48,
            "policy_hidden": 512,
            "use_factorized_policy_head": True,
            "value_channels": 48,
            "value_hidden": 512,
            "max_actions": 2026,
            "use_batch_norm": True,
            "se_ratio": 0.0625,
            "dropout": 0.0,
            "use_film": True,
            "film_hidden": 256,
            "film_global_dim": 61,
            "film_track_dim": 432,
            "film_shop_dim": 128,
            "film_per_block": True,
        }
    }
    net = create_network(cfg).to(device).eval()
    C, NMAX, F = 56, 33, 10
    F_GLOBAL, C_TRACK, TRACK_LEN = 61, 8, 54

    results = []
    with torch.inference_mode():
        for B in batch_sizes:
            state = torch.randn(B, C, 9, 9, device=device, dtype=torch.float32)
            action_mask = torch.ones(B, 2026, device=device, dtype=torch.float32)
            x_global = torch.randn(B, F_GLOBAL, device=device, dtype=torch.float32)
            x_track = torch.randn(B, C_TRACK, TRACK_LEN, device=device, dtype=torch.float32)
            shop_ids = torch.randint(-1, 34, (B, NMAX), device=device, dtype=torch.int64)
            shop_ids[:, 10:] = -1
            shop_feats = torch.randn(B, NMAX, F, device=device, dtype=torch.float32)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_iters):
                _ = net(state, action_mask, x_global=x_global, x_track=x_track, shop_ids=shop_ids, shop_feats=shop_feats)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000 / n_iters
            results.append((B, elapsed_ms))
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_sizes = [256, 512, 1024]
    n_iters = 100

    print("=" * 60)
    print("ShopEncoder + Full Model Benchmark")
    print(f"Device: {device}")
    print(f"Batch sizes: {batch_sizes}, {n_iters} iters each")
    print("=" * 60)

    # Default SDP
    print("\n[Default SDP kernel]")
    shop_results = _benchmark_shop_encoder(batch_sizes, n_iters, device)
    full_results = _benchmark_full_model(batch_sizes, n_iters, device)

    print("\nShopEncoder (standalone):")
    for (B, ms) in shop_results:
        print(f"  B={B}: {ms:.2f} ms/iter")

    print("\nFull model (gold_v2 FiLM):")
    for (B, ms) in full_results:
        print(f"  B={B}: {ms:.2f} ms/iter")

    print("\nShopEncoder % of full model:")
    for i, B in enumerate(batch_sizes):
        shop_ms = shop_results[i][1]
        full_ms = full_results[i][1]
        pct = 100.0 * shop_ms / full_ms if full_ms > 0 else 0
        print(f"  B={B}: {pct:.1f}%")

    # Try sdp_kernel override (context manager)
    if device.type == "cuda" and hasattr(torch.backends.cuda, "sdp_kernel"):
        print("\n" + "-" * 60)
        print("[SDP kernel: flash=False, math=True, mem_efficient=True]")
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=True):
            shop_results2 = _benchmark_shop_encoder(batch_sizes, n_iters, device)
            full_results2 = _benchmark_full_model(batch_sizes, n_iters, device)

        print("\nShopEncoder:")
        for (B, ms) in shop_results2:
            print(f"  B={B}: {ms:.2f} ms/iter")

        print("\nFull model:")
        for (B, ms) in full_results2:
            print(f"  B={B}: {ms:.2f} ms/iter")

        print("\nShopEncoder % of full model:")
        for i, B in enumerate(batch_sizes):
            shop_ms = shop_results2[i][1]
            full_ms = full_results2[i][1]
            pct = 100.0 * shop_ms / full_ms if full_ms > 0 else 0
            print(f"  B={B}: {pct:.1f}%")

    print("\n" + "=" * 60)
    if shop_results:
        max_pct = max(100.0 * shop_results[i][1] / full_results[i][1] for i in range(len(batch_sizes)))
        if max_pct > 25:
            print("ShopEncoder >25% of forward time. Suggestion: reduce n_layers (2->1) or dim_feedforward (64->32).")
        else:
            print("ShopEncoder <=25% of forward time. No architecture change suggested.")
    print("=" * 60)


if __name__ == "__main__":
    main()
