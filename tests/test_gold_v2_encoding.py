"""Gold v2 Multimodal encoding tests (G0.2, G0.3, H2, H3, H4)."""
from __future__ import annotations

import pytest
import numpy as np
import torch

from src.network.encoder import encode_state_multimodal, GoldV2StateEncoder
from src.network.gold_v2_constants import (
    BUY_START,
    MAX_ACTIONS,
    C_SPATIAL_ENC,
    F_GLOBAL,
    C_TRACK,
    NMAX,
    F_SHOP,
    TRACK_LEN,
)
from src.game.patchwork_engine import new_game, legal_actions_fast, apply_action
from src.mcts.alphazero_mcts_optimized import encode_legal_actions_fast
from src.network.d4_augmentation import (
    transform_action_vector,
    transform_state,
)
from src.network.encoder import get_slot_piece_id
from src.network.model import DeterministicLegalityModule, _build_piece_masks_5x5


def _make_det_legality() -> DeterministicLegalityModule:
    masks_np, areas_np = _build_piece_masks_5x5()
    return DeterministicLegalityModule(masks_np, areas_np)


def test_encode_state_multimodal_shapes():
    """G0.1: Assert output shapes."""
    st = new_game(seed=42)
    xs, xg, xt, sid, sfeat = encode_state_multimodal(st, 0)
    assert xs.shape == (C_SPATIAL_ENC, 9, 9)
    assert xg.shape == (F_GLOBAL,)
    assert xt.shape == (C_TRACK, 54)
    assert sid.shape == (NMAX,)
    assert sfeat.shape == (NMAX, F_SHOP)


def test_encode_into_matches_encode_state_multimodal():
    """encode_into writes identical output to encode_state_multimodal (zero-copy SHM path)."""
    enc = GoldV2StateEncoder()
    st = new_game(seed=123)
    to_move = 0

    # Reference output from encode_state_multimodal
    xs_ref, xg_ref, xt_ref, sid_ref, sfeat_ref = enc.encode_state_multimodal(st, to_move)

    # Pre-allocated buffers (simulating WorkerSharedBuffer views)
    xs_buf = np.zeros((C_SPATIAL_ENC, 9, 9), dtype=np.float32)
    xg_buf = np.zeros(F_GLOBAL, dtype=np.float32)
    xt_buf = np.zeros((C_TRACK, TRACK_LEN), dtype=np.float32)
    sid_buf = np.full(NMAX, -1, dtype=np.int16)
    sfeat_buf = np.zeros((NMAX, F_SHOP), dtype=np.float32)

    enc.encode_into(st, to_move, xs_buf, xg_buf, xt_buf, sid_buf, sfeat_buf)

    np.testing.assert_array_equal(xs_buf, xs_ref, err_msg="encode_into x_spatial mismatch")
    np.testing.assert_array_equal(xg_buf, xg_ref, err_msg="encode_into x_global mismatch")
    np.testing.assert_array_equal(xt_buf, xt_ref, err_msg="encode_into x_track mismatch")
    np.testing.assert_array_equal(sid_buf, sid_ref, err_msg="encode_into shop_ids mismatch")
    np.testing.assert_array_equal(sfeat_buf, sfeat_ref, err_msg="encode_into shop_feats mismatch")


def test_encode_into_worker_shared_buffer():
    """encode_into works with WorkerSharedBuffer views (MCTS SHM path at line 652)."""
    from src.mcts.shared_state_buffer import WorkerSharedBuffer

    enc = GoldV2StateEncoder()
    st = new_game(seed=456)
    to_move = 1

    xs_ref, xg_ref, xt_ref, sid_ref, sfeat_ref = enc.encode_state_multimodal(st, to_move)

    buf = WorkerSharedBuffer(n_slots=2, worker_id=0, create=True)
    try:
        slot = 0
        enc.encode_into(
            st, to_move,
            buf.spatial_view(slot),
            buf.global_view(slot),
            buf.track_view(slot),
            buf.shopids_view(slot),
            buf.shopfeats_view(slot),
        )
        xs_buf = np.copy(buf.spatial_view(slot))
        xg_buf = np.copy(buf.global_view(slot))
        xt_buf = np.copy(buf.track_view(slot))
        sid_buf = np.copy(buf.shopids_view(slot))
        sfeat_buf = np.copy(buf.shopfeats_view(slot))

        np.testing.assert_array_equal(xs_buf, xs_ref, err_msg="SHM encode_into x_spatial mismatch")
        np.testing.assert_array_equal(xg_buf, xg_ref, err_msg="SHM encode_into x_global mismatch")
        np.testing.assert_array_equal(xt_buf, xt_ref, err_msg="SHM encode_into x_track mismatch")
        np.testing.assert_array_equal(sid_buf, sid_ref, err_msg="SHM encode_into shop_ids mismatch")
        np.testing.assert_array_equal(sfeat_buf, sfeat_ref, err_msg="SHM encode_into shop_feats mismatch")
    finally:
        buf.destroy()


def test_legalTL_matches_action_mask_buy():
    """G0.2: GPU DeterministicLegalityModule legalTL matches action_mask BUY portion.

    The encoder no longer fills channels 32-55 (legalTL); those are computed
    on-the-fly by DeterministicLegalityModule in the network forward pass.
    This test verifies the GPU module produces correct legalTL from the board.
    """
    det_legality = _make_det_legality()

    np.random.seed(123)
    for _ in range(5):
        st = new_game(seed=np.random.randint(0, 10000))
        legal = legal_actions_fast(st)
        if not legal:
            continue
        to_move = 0 if st[6] <= st[0] else 1
        xs, _, _, shop_ids, shop_feats = encode_state_multimodal(st, to_move)
        _, mask = encode_legal_actions_fast(legal)

        # Encoder produces 32ch; legalTL (was channels 32-55) is now computed exclusively on GPU.
        assert xs.shape == (C_SPATIAL_ENC, 9, 9), "encoder must output 32ch (gold_v2_32ch)"

        # GPU module: current player's free cells -> legalTL for all 264 (pid, orient) pairs.
        board_free = torch.from_numpy((1.0 - xs[0:1]).astype(np.float32)).unsqueeze(0)  # (1,1,9,9)
        legal_all = det_legality(board_free)  # (1, 264, 9, 9)

        vis_ids = torch.from_numpy(shop_ids[:3].astype(np.int64)).unsqueeze(0)    # (1, 3)
        afford = torch.from_numpy(shop_feats[:3, 6].astype(np.float32)).unsqueeze(0)  # (1, 3)
        legalTL_24 = det_legality.extract_vis_legalTL(legal_all, vis_ids, afford)  # (1, 24, 9, 9)
        legalTL = legalTL_24[0].numpy().reshape(3, 8, 9, 9)

        mask_buy = mask[BUY_START:MAX_ACTIONS].reshape(3, 8, 9, 9)
        np.testing.assert_array_equal(
            mask_buy, legalTL,
            err_msg="GPU DeterministicLegalityModule legalTL must match action_mask BUY portion",
        )


def test_d4_gpu_legalTL_equivalence_after_transform():
    """H2: GPU DeterministicLegalityModule is D4-equivariant.

    After applying D4 transform to the board spatial state, the GPU module
    computes legalTL that matches the D4-transformed action mask.  This verifies
    that computing legalTL from the already-transformed board (as happens during
    training data augmentation) is correct.
    """
    det_legality = _make_det_legality()

    st = new_game(seed=99)
    legal = legal_actions_fast(st)
    if not legal:
        pytest.skip("No legal actions")
    to_move = 0
    xs, _, _, shop_ids, shop_feats = encode_state_multimodal(st, to_move)
    _, mask = encode_legal_actions_fast(legal)
    slot_ids = [get_slot_piece_id(st, i) for i in range(3)]
    slot_ids_int = [p if p is not None else -1 for p in slot_ids]
    ti = 3  # r270

    # D4-transform the spatial state (channels 0-31 rotate; 32-55 stay zero).
    xs_t = transform_state(xs, ti, slot_ids_int)

    # GPU module on D4-transformed board.
    board_free_t = torch.from_numpy((1.0 - xs_t[0:1]).astype(np.float32)).unsqueeze(0)
    legal_all_t = det_legality(board_free_t)  # (1, 264, 9, 9)

    vis_ids = torch.from_numpy(shop_ids[:3].astype(np.int64)).unsqueeze(0)
    afford = torch.from_numpy(shop_feats[:3, 6].astype(np.float32)).unsqueeze(0)
    legalTL_t = det_legality.extract_vis_legalTL(legal_all_t, vis_ids, afford)
    legalTL_t_3d = legalTL_t[0].numpy().reshape(3, 8, 9, 9)

    # D4-transform the action mask (slot order unchanged; orient + spatial remapped).
    mask_t = transform_action_vector(mask, ti, slot_ids_int)
    mask_t_buy = mask_t[BUY_START:MAX_ACTIONS].reshape(3, 8, 9, 9)

    np.testing.assert_array_equal(
        mask_t_buy, legalTL_t_3d,
        err_msg="After D4 transform, GPU legalTL must match D4-transformed action mask",
    )


def test_tie_player_and_bonus_7x7_perspective_swap():
    """H3: tie_player_is_current and bonus_7x7_status correct under to-move swap."""
    from src.game.patchwork_engine import TIE_PLAYER, BONUS_OWNER  # noqa: F401
    st = new_game(seed=77)
    for to_move in (0, 1):
        _, xg, _, _, _ = encode_state_multimodal(st, to_move)
        tie_player = int(st[TIE_PLAYER])
        assert xg[21] == (1.0 if tie_player == to_move else 0.0)
        bonus_owner = int(st[BONUS_OWNER])
        if bonus_owner == -1:
            assert xg[11] == 0.0
        elif bonus_owner == to_move:
            assert xg[11] == 1.0
        else:
            assert xg[11] == 0.5


def test_shop_list_reconstruction_stable():
    """Shop list from engine state is exact and stable (clockwise from neutral+1)."""
    st = new_game(seed=11)
    _, _, _, shop_ids, _ = encode_state_multimodal(st, 0)
    n = int(st[12])
    neutral = int(st[13])
    for i in range(min(n, NMAX)):
        piece_idx = (neutral + 1 + i) % n
        expected_pid = int(st[18 + piece_idx])
        assert shop_ids[i] == expected_pid
    for i in range(n, NMAX):
        assert shop_ids[i] == -1


def test_replay_buffer_ram_estimate():
    """H4: Replay buffer RAM estimate for 300k samples."""
    from src.network.gold_v2_constants import C_SPATIAL_ENC, F_GLOBAL, C_TRACK, TRACK_LEN, NMAX, F_SHOP
    N = 300_000
    spatial = N * C_SPATIAL_ENC * 9 * 9 * 4
    global_ = N * F_GLOBAL * 2
    track = N * C_TRACK * TRACK_LEN * 2
    shop_ids = N * NMAX * 2
    shop_feats = N * NMAX * F_SHOP * 2
    masks = N * 2026 * 4
    policies = N * 2026 * 4
    values = N * 4
    ownerships = N * 2 * 9 * 9 * 4
    total_mb = (spatial + global_ + track + shop_ids + shop_feats + masks + policies + values + ownerships) / (1024 * 1024)
    assert total_mb < 16 * 1024, f"300k samples should use < 16 GB, estimated {total_mb/1024:.1f} GB"
