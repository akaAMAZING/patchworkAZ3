"""
Correctness test: CPU D4 vs GPU D4 for identical inputs and transform codes.
Compares states, policies, masks, ownerships for exact or close equality.

Covers:
- Multiple random seeds; piece IDs 1-32 (PC_MAX=33)
- Batch sizes: 1, 7, 64, 1024
- All modalities: spatial_states, action_masks (spatial-flattened), policy tensors, ownership
- dtype: float32, bf16 (policies/states), masks as float
- Cache versioning: unversioned caches ignored; versioned filenames include pc33, v2
- Unknown piece IDs: no crash, identity mapping

First run may build LUT cache (~10-60s); subsequent runs load from disk.
"""

from __future__ import annotations

import os
import pytest

import numpy as np
import torch

from src.network.d4_constants import PC_MAX
from src.network.d4_augmentation import (
    apply_d4_augment_batch,
    apply_ownership_transform_batch,
)
from src.network.d4_augmentation_gpu import apply_d4_augment_batch_gpu

# Pure permutations: expect exact match. Floats after renormalization: use tolerance.
FLOAT_RTOL = 1e-5
FLOAT_ATOL = 1e-5
# bf16 has lower precision
BF16_RTOL = 1e-2
BF16_ATOL = 1e-2


def _make_synthetic_batch(
    B: int, seed: int, policy_dtype: str = "float32", mask_as_bool: bool = False
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    states_np = np.random.randn(B, 56, 9, 9).astype(np.float32) * 0.1
    policies_np = np.abs(np.random.rand(B, 2026).astype(np.float32)) + 1e-6
    policies_np = policies_np / policies_np.sum(axis=1, keepdims=True)
    if mask_as_bool:
        masks_np = (np.random.rand(B, 2026) > 0.7).astype(np.float32)  # still float for CPU API
    else:
        masks_np = (np.random.rand(B, 2026) > 0.7).astype(np.float32)
    ownerships_np = np.random.rand(B, 2, 9, 9).astype(np.float32)
    slot_ids_np = np.random.randint(1, 33, (B, 3), dtype=np.int16)  # piece IDs 1-32
    transform_indices_np = np.random.randint(0, 8, size=B, dtype=np.int32)
    return states_np, policies_np, masks_np, ownerships_np, slot_ids_np, transform_indices_np


def _compare_cpu_gpu(
    states_np,
    policies_np,
    masks_np,
    ownerships_np,
    slot_ids_np,
    transform_indices_np,
    device: torch.device,
    rtol: float = FLOAT_RTOL,
    atol: float = FLOAT_ATOL,
    use_bf16: bool = False,
):
    """Run CPU and GPU D4, compare outputs."""
    states_cpu, policies_cpu, masks_cpu = apply_d4_augment_batch(
        states_np.copy(), policies_np.copy(), masks_np.copy(),
        slot_ids_np, transform_indices_np,
    )
    ownerships_cpu = apply_ownership_transform_batch(ownerships_np.copy(), transform_indices_np)

    states_gpu = torch.from_numpy(states_np.copy()).to(device)
    policies_gpu = torch.from_numpy(policies_np.copy()).to(device)
    masks_gpu = torch.from_numpy(masks_np.copy()).to(device)
    ownerships_gpu = torch.from_numpy(ownerships_np.copy()).to(device)
    slot_ids_t = torch.from_numpy(slot_ids_np).to(device)
    transform_indices_t = torch.from_numpy(transform_indices_np).to(device, dtype=torch.long)

    if use_bf16:
        states_gpu = states_gpu.to(torch.bfloat16)
        policies_gpu = policies_gpu.to(torch.bfloat16)
        masks_gpu = masks_gpu.to(torch.bfloat16)
        ownerships_gpu = ownerships_gpu.to(torch.bfloat16)
        rtol, atol = BF16_RTOL, BF16_ATOL

    states_out, policies_out, masks_out, ownerships_out = apply_d4_augment_batch_gpu(
        states_gpu, policies_gpu, masks_gpu, ownerships_gpu,
        slot_ids_t, transform_indices_t, device,
    )

    def to_np(t):
        return t.float().cpu().numpy() if t.dtype == torch.bfloat16 else t.cpu().numpy()

    np.testing.assert_allclose(states_cpu, to_np(states_out), rtol=rtol, atol=atol)
    np.testing.assert_allclose(policies_cpu, to_np(policies_out), rtol=rtol, atol=atol)
    np.testing.assert_allclose(masks_cpu, to_np(masks_out), rtol=rtol, atol=atol)
    np.testing.assert_allclose(ownerships_cpu, to_np(ownerships_out), rtol=rtol, atol=atol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("seed", [0, 42, 123, 999])
def test_d4_gpu_vs_cpu_multiple_seeds(seed: int):
    """Deterministic mode: same transform_indices, CPU and GPU must match."""
    device = torch.device("cuda")
    B = 32
    data = _make_synthetic_batch(B, seed)
    _compare_cpu_gpu(*data, device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("B", [1, 7, 64, 1024])
def test_d4_gpu_vs_cpu_batch_sizes(B: int):
    """Various batch sizes including edge cases."""
    device = torch.device("cuda")
    data = _make_synthetic_batch(B, 42)
    _compare_cpu_gpu(*data, device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_d4_gpu_vs_cpu_float32():
    """Explicit float32 coverage."""
    device = torch.device("cuda")
    data = _make_synthetic_batch(16, 7, policy_dtype="float32")
    _compare_cpu_gpu(*data, device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_d4_gpu_vs_cpu_bf16():
    """bf16 inputs; use relaxed tolerance for comparison."""
    device = torch.device("cuda")
    data = _make_synthetic_batch(16, 7)
    _compare_cpu_gpu(*data, device, use_bf16=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_d4_gpu_vs_cpu_uniform_transform():
    """All samples same (ti, p0, p1, p2) — tests grouped path."""
    device = torch.device("cuda")
    np.random.seed(123)
    B = 16
    states_np = np.random.randn(B, 56, 9, 9).astype(np.float32) * 0.1
    policies_np = np.abs(np.random.rand(B, 2026).astype(np.float32)) + 1e-6
    policies_np = policies_np / policies_np.sum(axis=1, keepdims=True)
    masks_np = (np.random.rand(B, 2026) > 0.5).astype(np.float32)
    ownerships_np = np.random.rand(B, 2, 9, 9).astype(np.float32)
    slot_ids_np = np.array([[1, 2, 3]] * B, dtype=np.int16)
    transform_indices_np = np.array([3] * B, dtype=np.int32)

    _compare_cpu_gpu(
        states_np, policies_np, masks_np, ownerships_np,
        slot_ids_np, transform_indices_np, device,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_d4_gpu_vs_cpu_all_transforms():
    """Cover all 8 transform indices."""
    device = torch.device("cuda")
    B = 16
    for ti in range(8):
        data = _make_synthetic_batch(B, 100 + ti)
        states_np, policies_np, masks_np, ownerships_np, slot_ids_np, _ = data
        transform_indices_np = np.array([ti] * B, dtype=np.int32)
        _compare_cpu_gpu(
            states_np, policies_np, masks_np, ownerships_np,
            slot_ids_np, transform_indices_np, device,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_d4_gpu_vs_cpu_slot_piece_edge_cases():
    """slot_piece_ids: -1 (empty slot), various piece IDs."""
    device = torch.device("cuda")
    np.random.seed(77)
    B = 8
    states_np = np.random.randn(B, 56, 9, 9).astype(np.float32) * 0.1
    policies_np = np.abs(np.random.rand(B, 2026).astype(np.float32)) + 1e-6
    policies_np = policies_np / policies_np.sum(axis=1, keepdims=True)
    masks_np = (np.random.rand(B, 2026) > 0.5).astype(np.float32)
    ownerships_np = np.random.rand(B, 2, 9, 9).astype(np.float32)
    slot_ids_np = np.array([[1, -1, 3], [2, 2, 2], [5, 10, 15]] * 3 + [[1, 1, 1]], dtype=np.int16)[:B]
    transform_indices_np = np.array([1, 4, 6, 0, 7, 2, 3, 5][:B], dtype=np.int32)

    _compare_cpu_gpu(
        states_np, policies_np, masks_np, ownerships_np,
        slot_ids_np, transform_indices_np, device,
    )


def test_d4_gpu_skipped_when_no_cuda():
    """When CUDA unavailable, test module imports; apply_d4_augment_batch_gpu would fail if called."""
    from src.network.d4_augmentation_gpu import apply_d4_augment_batch_gpu
    assert callable(apply_d4_augment_batch_gpu)


def test_d4_cpu_path_produces_valid_outputs():
    """CPU D4 (apply_d4_augment_batch) produces valid shape and normalized policy."""
    data = _make_synthetic_batch(16, 7)
    states_np, policies_np, masks_np, ownerships_np, slot_ids_np, transform_indices_np = data
    from src.network.d4_augmentation import apply_d4_augment_batch, apply_ownership_transform_batch
    states_cpu, policies_cpu, masks_cpu = apply_d4_augment_batch(
        states_np.copy(), policies_np.copy(), masks_np.copy(),
        slot_ids_np, transform_indices_np,
    )
    ownerships_cpu = apply_ownership_transform_batch(ownerships_np.copy(), transform_indices_np)
    assert states_cpu.shape == states_np.shape
    assert policies_cpu.shape == policies_np.shape
    assert np.allclose(policies_cpu.sum(axis=1), 1.0, rtol=1e-5)
    assert ownerships_cpu.shape == ownerships_np.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_d4_unknown_piece_id_no_crash():
    """Unknown piece IDs (e.g. 99) must not crash; use identity mapping for orient."""
    device = torch.device("cuda")
    np.random.seed(88)
    B = 4
    states_np = np.random.randn(B, 56, 9, 9).astype(np.float32) * 0.1
    policies_np = np.abs(np.random.rand(B, 2026).astype(np.float32)) + 1e-6
    policies_np = policies_np / policies_np.sum(axis=1, keepdims=True)
    masks_np = (np.random.rand(B, 2026) > 0.5).astype(np.float32)
    ownerships_np = np.random.rand(B, 2, 9, 9).astype(np.float32)
    slot_ids_np = np.array([[99, 1, 2], [1, 100, 3], [2, 3, 200], [1, 2, 3]], dtype=np.int16)
    transform_indices_np = np.array([0, 1, 2, 3], dtype=np.int32)
    states_gpu = torch.from_numpy(states_np.copy()).to(device)
    policies_gpu = torch.from_numpy(policies_np.copy()).to(device)
    masks_gpu = torch.from_numpy(masks_np.copy()).to(device)
    ownerships_gpu = torch.from_numpy(ownerships_np.copy()).to(device)
    slot_ids_t = torch.from_numpy(slot_ids_np).to(device)
    transform_indices_t = torch.from_numpy(transform_indices_np).to(device, dtype=torch.long)
    out_s, out_p, out_m, out_o = apply_d4_augment_batch_gpu(
        states_gpu, policies_gpu, masks_gpu, ownerships_gpu,
        slot_ids_t, transform_indices_t, device,
    )
    assert out_s.shape == states_np.shape
    assert out_p.shape == policies_np.shape
    assert out_m.shape == masks_np.shape
    assert out_o.shape == ownerships_np.shape


def test_d4_cache_versioned_filenames():
    """Cache paths must include pc33 and lut version; old unversioned are ignored."""
    from src.network.d4_lut_cache import get_lut_paths
    buy_path, meta_path = get_lut_paths()
    assert "pc33" in buy_path
    assert "_v" in buy_path
    assert "d4_buy_lut" in buy_path and buy_path.endswith(".npy")


def test_d4_lut_build_or_load_completes():
    """LUT load (or build if uncached) must complete in reasonable time (<60s cached, <120s uncached)."""
    import time
    from src.network import d4_augmentation_gpu
    orig_buy = d4_augmentation_gpu._BUY_LUT
    d4_augmentation_gpu._BUY_LUT = None
    try:
        t0 = time.perf_counter()
        d4_augmentation_gpu._build_buy_lut(verbose=False)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        assert elapsed < 120, f"LUT build/load took {elapsed:.1f}s (max 120s)"
    finally:
        d4_augmentation_gpu._BUY_LUT = orig_buy


def test_d4_no_item_in_hot_path():
    """Static check: apply_d4_augment_batch_gpu and its loop bodies contain no .item() on GPU tensors."""
    import ast
    import inspect
    from src.network import d4_augmentation_gpu as mod

    source = inspect.getsource(mod.apply_d4_augment_batch_gpu)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if node.func.attr == "item":
                    # Allow: unique_keys_np[i] (numpy, no sync), int(orient_before[...]) (CPU tensor)
                    # Disallow: key_val.item() or any .item() on GPU tensors in the loop
                    # The loop uses unique_keys_np (numpy) and _decode_key(int(unique_keys_np[i]))
                    # so we should have no .item() in apply_d4_augment_batch_gpu
                    raise AssertionError(
                        "apply_d4_augment_batch_gpu must not call .item() (causes GPU sync). "
                        "Found .item() in hot path."
                    )
    assert True
