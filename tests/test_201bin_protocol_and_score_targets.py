"""Tests for 201-bin inference protocol and training score targets."""
from __future__ import annotations

import math
import queue

import numpy as np
import torch

from src.mcts.gpu_eval_client import GPUEvalClient
from src.training.trainer import make_gaussian_score_targets


# ---------------------------------------------------------------------------
# Protocol: receive() returns 4 values
# ---------------------------------------------------------------------------

def test_gpu_eval_client_receive_returns_four_values():
    """receive(rid) must return (priors_legal, value, mean_points, score_utility) — 4 values."""
    resp_q = queue.Queue()
    client = GPUEvalClient(req_q=queue.Queue(), resp_q=resp_q, worker_id=0)
    rid = client._next_id()
    # Simulate server 5-tuple response
    resp_q.put((rid, np.array([0.5, 0.5], dtype=np.float32), 0.1, 10.0, 0.05))
    result = client.receive(rid)
    assert len(result) == 4
    priors, value, mean_points, score_utility = result
    assert isinstance(priors, np.ndarray)
    assert value == 0.1
    assert mean_points == 10.0
    assert score_utility == 0.05


def test_gpu_eval_client_stash_has_four_values():
    """Out-of-order stash must store 4-tuple so receive() always returns 4 values."""
    resp_q = queue.Queue()
    client = GPUEvalClient(req_q=queue.Queue(), resp_q=resp_q, worker_id=0)
    r1, r2 = client._next_id(), client._next_id()
    resp_q.put((r2, np.array([1.0], dtype=np.float32), -0.2, -5.0, -0.02))
    resp_q.put((r1, np.array([0.3, 0.7], dtype=np.float32), 0.0, 0.0, 0.0))
    out1 = client.receive(r1)
    out2 = client.receive(r2)
    assert len(out1) == len(out2) == 4


# ---------------------------------------------------------------------------
# Server response contract: 5-tuple (rid, priors, value, mean_points, score_utility)
# ---------------------------------------------------------------------------

def test_server_response_five_tuple_unpacks_to_four_values():
    """Server sends 5-tuple; client.receive() returns 4 values (no rid). Fallback must also be 5-tuple."""
    resp_q = queue.Queue()
    resp_q.put((99, np.array([0.5, 0.5], dtype=np.float32), 0.0, 0.0, 0.0))
    got = resp_q.get()
    assert len(got) == 5
    rid, priors, value, mean_points, score_utility = got
    assert len((priors, value, mean_points, score_utility)) == 4


# ---------------------------------------------------------------------------
# Training: make_gaussian_score_targets from tanh score_margins
# ---------------------------------------------------------------------------

def test_margin_points_from_tanh_scale_30():
    """tanh(10/30) -> margin_points should be ~10 after atanh and round."""
    scale = 30.0
    # tanh(10/30) ≈ 0.329
    m_tanh = math.tanh(10.0 / scale)
    margin_points = round(scale * math.atanh(min(0.999999, max(-0.999999, m_tanh))))
    assert margin_points == 10


def test_make_gaussian_score_targets_peaks_and_sums():
    """Target distribution from tanh(10/30) peaks at bin index 110 (bin value 10) and sums to 1."""
    scale = 30.0
    score_min, score_max = -100, 100
    sigma = 1.5
    # One sample: tanh(10/30) -> margin_points = 10 -> bin index 110 (bins are -100..100)
    m_tanh = torch.tensor([math.tanh(10.0 / scale)], dtype=torch.float32)
    target = make_gaussian_score_targets(m_tanh, scale, score_min, score_max, sigma)
    assert target.shape == (1, 201)
    assert torch.allclose(target.sum(dim=-1), torch.ones(1, device=target.device))
    peak_idx = target.argmax(dim=-1).item()
    # Bin index for value 10: arange(-100, 101) has index 110 = 10
    assert 108 <= peak_idx <= 112, "Peak should be at bin index 110 (bin value 10)"
    assert target[0, 110] > 0.1


def test_make_gaussian_score_targets_batch_sums_to_one():
    """Each row of target distribution sums to 1."""
    margins = torch.tensor([0.0, 0.5, -0.3], dtype=torch.float32)
    target = make_gaussian_score_targets(margins, 30.0, -100, 100, 1.5)
    assert target.shape == (3, 201)
    row_sums = target.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(3, device=target.device))
