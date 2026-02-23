"""
GPU-side D4 augmentation for gold_v2 spatial encoding.

Applies the same semantics as d4_augmentation.apply_d4_augment_batch but on
CUDA tensors after H2D transfer. Preserves piece-aware remapping via slot_piece_ids.

Transform indices: 0=id, 1=r90, 2=r180, 3=r270, 4=m, 5=m_r90, 6=m_r180, 7=m_r270.

ZERO HOST SYNC: No .cpu(), .numpy(), .item(), or .tolist() in the augmentation hot path.
Uses prebuilt GPU LUTs indexed by packed (ti,p0,p1,p2) for buy/legal perms.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch

from src.network import d4_augmentation as _d4_cpu
from src.network.d4_augmentation import (
    BUY_START,
    D4_COUNT,
    NUM_ORIENTS,
    NUM_SLOTS,
    PATCH_START,
    SLOT_ORIENT_BASE,
)
from src.network.d4_constants import COMPACT_SIZE, MULT, PC_MAX
from src.network.d4_lut_cache import build_and_save_luts, load_luts_if_valid
from src.network.gold_v2_constants import C_SPATIAL_ENC

BS = 9

_BUY_LUT: Optional[torch.Tensor] = None
_GPU_TABLES_CACHE: Dict[torch.device, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _build_buy_lut(verbose: bool = True) -> torch.Tensor:
    """Load or build (COMPACT_SIZE, 1944) int32 buy LUT. Versioned disk cache."""
    global _BUY_LUT
    if _BUY_LUT is not None:
        return _BUY_LUT
    loaded = load_luts_if_valid()
    if loaded is not None:
        _BUY_LUT = torch.from_numpy(loaded)
        return _BUY_LUT
    buy_arr = build_and_save_luts(verbose=verbose)
    _BUY_LUT = torch.from_numpy(buy_arr)
    return _BUY_LUT


def _get_gpu_tables(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """spatial_perm, patch_perm, orient_before (GPU), buy_lut. Cached per device."""
    if device in _GPU_TABLES_CACHE:
        return _GPU_TABLES_CACHE[device]
    _d4_cpu._init_inv_index_2d()
    patch_src = _d4_cpu._get_patch_src_index()
    inv_2d = _d4_cpu._INV_INDEX_2D
    spatial_perms = []
    for ti in range(D4_COUNT):
        row_idx, col_idx = inv_2d[ti]
        perm = (row_idx.astype(np.int64).flatten() * BS + col_idx.astype(np.int64).flatten())
        spatial_perms.append(perm)
    spatial_perm = torch.from_numpy(np.stack(spatial_perms)).to(device, dtype=torch.long)
    patch_perm = torch.from_numpy(patch_src.astype(np.int64)).to(device)

    from src.game.patchwork_engine import PIECE_BY_ID
    _d4_cpu._get_orient_maps()
    max_pid = max(PIECE_BY_ID.keys()) if PIECE_BY_ID else 0
    # orient_tbl: (PC_MAX+2, 8, 8). Row 0..PC_MAX for pids 0..PC_MAX; row PC_MAX+1 for unknown/empty identity
    orient_tbl = torch.zeros(PC_MAX + 2, D4_COUNT, 8, dtype=torch.long, device=device)
    for pid in PIECE_BY_ID:
        pid = int(pid)
        if pid < orient_tbl.shape[0]:
            for ti in range(D4_COUNT):
                for o in range(8):
                    o_old = _d4_cpu.get_orient_before_transform(pid, ti, o)
                    orient_tbl[pid, ti, o] = o_old
    id_row = torch.arange(8, device=device, dtype=torch.long).view(1, 8).expand(D4_COUNT, 8)
    for r in range(max_pid + 1, orient_tbl.shape[0]):
        orient_tbl[r, :, :] = id_row

    buy_lut = _build_buy_lut().to(device)
    _GPU_TABLES_CACHE[device] = (spatial_perm, patch_perm, orient_tbl, buy_lut)
    return spatial_perm, patch_perm, orient_tbl, buy_lut


def _key_to_compact_idx(key_val: torch.Tensor) -> torch.Tensor:
    """
    Decode packed key to compact LUT index. Pure tensor ops.
    Encoding: p_enc 0=empty, 2..33=pieces 1..32. p0c = (p0_enc-1).clamp(0, PC_MAX-1).
    """
    ti = key_val // (MULT**3)
    rem = key_val % (MULT**3)
    p0_enc = rem // (MULT**2)
    rem = rem % (MULT**2)
    p1_enc = rem // MULT
    p2_enc = rem % MULT
    p0c = (p0_enc - 1).clamp(0, PC_MAX - 1)
    p1c = (p1_enc - 1).clamp(0, PC_MAX - 1)
    p2c = (p2_enc - 1).clamp(0, PC_MAX - 1)
    return (ti * (PC_MAX**3) + p0c * (PC_MAX**2) + p1c * PC_MAX + p2c).clamp(0, COMPACT_SIZE - 1)


def _transform_spatial_plane_gpu(
    plane: torch.Tensor,
    ti: torch.Tensor,
    spatial_perm: torch.Tensor,
) -> torch.Tensor:
    """plane: (B, 9, 9). ti: 0-dim long tensor."""
    B = plane.shape[0]
    flat = plane.reshape(B, -1)
    perm = spatial_perm[ti]
    out_flat = flat[:, perm]
    return out_flat.reshape(B, BS, BS)


def _transform_state_group_gpu(
    states: torch.Tensor,
    ti: torch.Tensor,
    p0: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    spatial_perm: torch.Tensor,
    orient_before: torch.Tensor,
    buy_src: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Transform states (B, 32, 9, 9). All args on GPU; no host sync."""
    out = states.clone()
    pids = torch.stack([p0, p1, p2], dim=0)
    max_pid = orient_before.shape[0] - 2
    # Clamp pids to valid range; unknown pids use identity row (PC_MAX+1)
    pid_safe = torch.where((pids >= 0) & (pids <= max_pid), pids, max_pid + 1)

    for ch in range(8):
        out[:, ch] = _transform_spatial_plane_gpu(states[:, ch], ti, spatial_perm)

    for slot in range(NUM_SLOTS):
        pid = pid_safe[slot]
        for o in range(NUM_ORIENTS):
            o_old = orient_before[pid, ti, o]
            ch_src = (SLOT_ORIENT_BASE + slot * NUM_ORIENTS + o_old).long().clamp(0, C_SPATIAL_ENC - 1)
            ch_dst = SLOT_ORIENT_BASE + slot * NUM_ORIENTS + o
            src_plane = states.index_select(1, ch_src.unsqueeze(0)).squeeze(1)
            out[:, ch_dst] = _transform_spatial_plane_gpu(src_plane, ti, spatial_perm)

    return out


def _transform_action_group_gpu(
    arr: torch.Tensor,
    ti: torch.Tensor,
    buy_src: torch.Tensor,
    patch_perm: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Transform policy or mask."""
    out = torch.zeros_like(arr, device=arr.device)
    out[:, 0] = arr[:, 0]
    patch_src = patch_perm[ti]
    out[:, PATCH_START : PATCH_START + 81] = arr[:, PATCH_START + patch_src.long()]
    out[:, BUY_START:] = arr[:, BUY_START + buy_src.long()]
    return out


def _transform_ownership_group_gpu(
    ownerships: torch.Tensor,
    ti: torch.Tensor,
    spatial_perm: torch.Tensor,
) -> torch.Tensor:
    """Transform ownership."""
    out = ownerships.clone()
    for ch in range(2):
        out[:, ch] = _transform_spatial_plane_gpu(ownerships[:, ch], ti, spatial_perm)
    return out


def apply_d4_augment_batch_gpu(
    states: torch.Tensor,
    policies: torch.Tensor,
    masks: torch.Tensor,
    ownerships: torch.Tensor,
    slot_ids: torch.Tensor,
    transform_indices: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply D4 augmentation on GPU. ZERO host sync: no .cpu(), .numpy(), .item(), .tolist().
    """
    B = states.shape[0]
    spatial_perm, patch_perm, orient_before, buy_lut = _get_gpu_tables(device)

    slot_ids_int = slot_ids.to(device, dtype=torch.long)
    slot_ids_int = torch.where(slot_ids_int >= 0, slot_ids_int, torch.tensor(-1, device=device, dtype=torch.long))
    if slot_ids_int.dim() == 1:
        slot_ids_int = slot_ids_int.unsqueeze(-1).expand(-1, 3)
    ti_t = transform_indices.to(device, dtype=torch.long)

    out_states = states.clone()
    out_policies = policies.clone()
    out_masks = masks.clone()
    out_ownerships = ownerships.clone()

    # Encoding: slot_id -1..32 -> p_enc 0..33
    p0_enc = (slot_ids_int[:, 0] + 1).clamp(0, 33)
    p1_enc = (slot_ids_int[:, 1] + 1).clamp(0, 33)
    p2_enc = (slot_ids_int[:, 2] + 1).clamp(0, 33)
    keys_flat = ti_t * (MULT**3) + p0_enc * (MULT**2) + p1_enc * MULT + p2_enc
    unique_keys = torch.unique(keys_flat)

    for i in range(unique_keys.shape[0]):
        key_val = unique_keys[i]
        mask = keys_flat == key_val
        idx = mask.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue

        compact_idx = _key_to_compact_idx(key_val)
        buy_src = buy_lut[compact_idx]

        ti = key_val // (MULT**3)
        rem = key_val % (MULT**3)
        p0_enc_v = rem // (MULT**2)
        rem = rem % (MULT**2)
        p1_enc_v = rem // MULT
        p2_enc_v = rem % MULT
        # p0/p1/p2: clamp(-1, PC_MAX-1) for piece ids 1..32 and empty (-1)
        p0 = (p0_enc_v - 1).clamp(-1, PC_MAX - 1)
        p1 = (p1_enc_v - 1).clamp(-1, PC_MAX - 1)
        p2 = (p2_enc_v - 1).clamp(-1, PC_MAX - 1)

        sub_states = states[idx]
        sub_pol = policies[idx]
        sub_masks = masks[idx]
        sub_own = ownerships[idx]

        out_states[idx] = _transform_state_group_gpu(
            sub_states, ti, p0, p1, p2, spatial_perm, orient_before, buy_src, device
        )
        out_policies[idx] = _transform_action_group_gpu(sub_pol, ti, buy_src, patch_perm, device)
        out_masks[idx] = _transform_action_group_gpu(sub_masks, ti, buy_src, patch_perm, device)
        out_ownerships[idx] = _transform_ownership_group_gpu(sub_own, ti, spatial_perm)

    totals = out_policies.sum(dim=1, keepdim=True).clamp(min=1e-8)
    out_policies = out_policies / totals
    out_masks = (out_masks > 0).to(out_masks.dtype)

    return out_states, out_policies, out_masks, out_ownerships
