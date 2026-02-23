"""Test GPU inference server with gold_v2_multimodal payloads.

Spins up server in subprocess, sends 5 gold_v2 requests, verifies output shapes.
Runs on CPU when CUDA unavailable (CI).
"""

from __future__ import annotations

import multiprocessing as mp
import tempfile
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import torch

from src.network.model import create_network
from src.network.gold_v2_constants import C_SPATIAL, F_GLOBAL, C_TRACK, TRACK_LEN, NMAX, F_SHOP
from src.mcts.gpu_eval_client import GPUEvalClient


def _make_gold_v2_config():
    return {
        "data": {"encoding_version": "gold_v2_multimodal"},
        "network": {
            "input_channels": 56,
            "num_res_blocks": 2,
            "channels": 32,
            "policy_channels": 24,
            "policy_hidden": 128,
            "use_factorized_policy_head": True,
            "value_channels": 24,
            "value_hidden": 128,
            "max_actions": 2026,
            "use_batch_norm": True,
            "se_ratio": 0.0625,
            "dropout": 0.0,
            "use_film": True,
            "film_hidden": 64,
            "film_global_dim": 61,
            "film_track_dim": 432,
            "film_shop_dim": 128,
            "film_per_block": True,
        },
    }


def _run_server_process(config, ckpt_path, req_q, resp_qs, stop_evt, ready_q, device):
    from src.network.gpu_inference_server import run_gpu_inference_server
    run_gpu_inference_server(config, ckpt_path, req_q, resp_qs, stop_evt, ready_q, device)


@pytest.mark.slow
def test_gpu_server_gold_v2_payloads():
    """Send 5 gold_v2 payloads to server, verify output shapes."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = _make_gold_v2_config()

    # Create minimal checkpoint
    net = create_network(config)
    ckpt = {"model_state_dict": net.state_dict(), "config_hash": "test"}
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = f.name
    try:
        torch.save(ckpt, ckpt_path)

        ctx = mp.get_context("spawn")
        req_q = ctx.Queue(maxsize=64)
        resp_qs = [ctx.Queue(maxsize=64)]
        stop_evt = ctx.Event()
        ready_q = ctx.Queue(maxsize=1)

        proc = ctx.Process(
            target=_run_server_process,
            args=(config, ckpt_path, req_q, resp_qs, stop_evt, ready_q, device),
            daemon=True,
        )
        proc.start()

        status = ready_q.get(timeout=15.0)
        assert status == "ready", f"Server failed: {status}"

        client = GPUEvalClient(req_q, resp_qs[0], worker_id=0, timeout_s=10.0)

        for _ in range(5):
            x_spatial = np.random.randn(C_SPATIAL, 9, 9).astype(np.float32)
            x_global = np.random.randn(F_GLOBAL).astype(np.float32)
            x_track = np.random.randn(C_TRACK, TRACK_LEN).astype(np.float32)
            shop_ids = np.full(NMAX, -1, dtype=np.int64)
            shop_ids[:5] = np.random.randint(0, 34, size=5)
            shop_feats = np.random.randn(NMAX, F_SHOP).astype(np.float32)
            mask = np.zeros(2026, dtype=np.float32)
            mask[:20] = 1.0
            legal_idxs = np.arange(20, dtype=np.int32)

            rid = client.submit_multimodal(
                x_spatial, x_global, x_track, shop_ids, shop_feats, mask, legal_idxs
            )
            priors, value, score = client.receive(rid)

            assert priors.shape == (20,)
            assert np.isfinite(priors).all()
            # float32 softmax sum can be ~0.9997 due to precision; use 1e-3 tolerance
            assert abs(float(priors.sum()) - 1.0) < 1e-3
            assert isinstance(value, (float, np.floating))
            assert isinstance(score, (float, np.floating))

        stop_evt.set()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate()
    finally:
        Path(ckpt_path).unlink(missing_ok=True)
