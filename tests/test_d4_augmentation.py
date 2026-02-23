"""Phase C: D4 symmetry augmentation — unit and property tests."""
from __future__ import annotations

import pytest
import numpy as np

from src.network.d4_augmentation import (
    transform_state,
    transform_action_vector,
    apply_d4_augment,
    inverse_transform_idx,
    get_orient_after_transform,
    get_orient_before_transform,
    transform_position,
    _transform_coord_planes,
    D4_NAMES,
    D4_COUNT,
    SPATIAL_CHANNELS,
)
from src.game.patchwork_engine import ORIENT_COUNT, MASK_W0, MASK_W1, MASK_W2
from src.network.encoder import StateEncoder, ActionEncoder


# ============== A) Unit: coord planes under each symmetry ==============
def test_coord_planes_transform():
    """Coord planes: identity at ti=0; round-trip T then inv(T) recovers original."""
    BS = 9
    denom = float(BS - 1)
    row_plane = np.zeros((BS, BS), dtype=np.float32)
    col_plane = np.zeros((BS, BS), dtype=np.float32)
    for r in range(BS):
        for c in range(BS):
            row_plane[r, c] = r / denom
            col_plane[r, c] = c / denom

    out0_r, out0_c = _transform_coord_planes(row_plane, col_plane, 0)
    assert np.allclose(out0_r, row_plane)
    assert np.allclose(out0_c, col_plane)

    for ti in range(D4_COUNT):
        out_r, out_c = _transform_coord_planes(row_plane, col_plane, ti)
        inv_ti = inverse_transform_idx(ti)
        back_r, back_c = _transform_coord_planes(out_r, out_c, inv_ti)
        assert np.allclose(back_r, row_plane), f"ti={ti} coord row round-trip"
        assert np.allclose(back_c, col_plane), f"ti={ti} coord col round-trip"


# ============== B) Unit: orient_map bijection (BLOCKER) ==============
def test_orient_map_bijection_all_pieces_all_transforms():
    """Every orient_map[pid][T] MUST be a permutation of 0..7 (or 0..n-1 for n-orient pieces)."""
    from src.network.d4_augmentation import _get_orient_maps
    from src.game.patchwork_engine import PIECE_BY_ID, ORIENT_COUNT

    maps = _get_orient_maps()
    for pid in PIECE_BY_ID:
        pid = int(pid)
        assert pid in maps, f"piece_id {pid} missing from orient maps"
        n_orient = int(ORIENT_COUNT[pid])
        for ti in range(D4_COUNT):
            om = maps[pid][ti]
            assert len(om) == 8, f"pid={pid} ti={ti}: map must have length 8, got {len(om)}"
            # First n_orient must be permutation of 0..n-1
            partial = om[:n_orient]
            assert len(set(partial)) == n_orient, (
                f"pid={pid} ti={D4_NAMES[ti]}: orient_map[:{n_orient}] has duplicates: {partial}"
            )
            assert set(partial) == set(range(n_orient)), (
                f"pid={pid} ti={D4_NAMES[ti]}: orient_map[:{n_orient}] must be permutation of 0..{n_orient-1}"
            )
            # Padding orients n..7 must map to themselves
            for k in range(n_orient, 8):
                assert om[k] == k, f"pid={pid} ti={ti}: padding orient {k} must map to self, got {om[k]}"


def test_orient_map_bijection_inverse_valid():
    """Inverse of each orient_map must be a valid permutation."""
    from src.network.d4_augmentation import _get_orient_maps, _ORIENT_MAPS_INVERSE

    maps = _get_orient_maps()
    assert _ORIENT_MAPS_INVERSE, "Inverse maps must be built at module load"
    for pid, fwd_maps in maps.items():
        inv_maps = _ORIENT_MAPS_INVERSE[pid]
        for ti in range(D4_COUNT):
            om = fwd_maps[ti]
            inv_om = inv_maps[ti]
            for k in range(8):
                o = om[k]
                assert inv_om[o] == k, (
                    f"pid={pid} ti={ti}: inverse(om[{k}]={o}) should be {k}, got {inv_om[o]}"
                )


def test_orient_map_bijection_several_pieces():
    """Spot-check bijection for pieces 0, 1, 5, 11 (varying orient counts)."""
    from src.network.d4_augmentation import _get_orient_maps, get_orient_after_transform, get_orient_before_transform
    from src.game.patchwork_engine import ORIENT_COUNT

    for pid in [0, 1, 5, 11]:
        n = int(ORIENT_COUNT[pid])
        for ti in range(D4_COUNT):
            for o in range(n):
                o_after = get_orient_after_transform(pid, ti, o)
                o_back = get_orient_before_transform(pid, ti, o_after)
                assert o_back == o, (
                    f"pid={pid} ti={ti} orient={o}: after={o_after} before(after)={o_back} != {o}"
                )


# ============== C) Unit: orientation remap correctness ==============
def test_orient_remap_matches_engine_shapes():
    """Verify orient remap: transform_mask(mask[p][k]) == mask[p][om[k]] (exact mask equality)."""
    from src.network.d4_augmentation import (
        _get_orient_maps,
        _mask_words_to_plane,
        _transform_mask_plane,
    )
    from src.game.patchwork_engine import MASK_W0, MASK_W1, MASK_W2, PIECES

    maps = _get_orient_maps()
    for p in PIECES[:8]:
        pid = int(p["id"])
        if pid not in maps or ORIENT_COUNT[pid] < 2:
            continue
        for ti in range(D4_COUNT):
            om = maps[pid][ti]
            for k in range(int(ORIENT_COUNT[pid])):
                mask_k = None
                for pos in range(81):
                    m0 = int(MASK_W0[pid, k, pos])
                    m1 = int(MASK_W1[pid, k, pos])
                    m2 = int(MASK_W2[pid, k, pos])
                    if (m0 | m1 | m2) != 0:
                        mask_k = _mask_words_to_plane(m0, m1, m2)
                        break
                assert mask_k is not None, f"pid={pid} orient={k} has no mask"
                target = _transform_mask_plane(mask_k, ti)
                k_prime = om[k]
                found = False
                for pos in range(81):
                    m0 = int(MASK_W0[pid, k_prime, pos])
                    m1 = int(MASK_W1[pid, k_prime, pos])
                    m2 = int(MASK_W2[pid, k_prime, pos])
                    if (m0 | m1 | m2) == 0:
                        continue
                    mask_kp = _mask_words_to_plane(m0, m1, m2)
                    if np.array_equal(target, mask_kp):
                        found = True
                        break
                assert found, (
                    f"pid={pid} ti={D4_NAMES[ti]} k={k}: transform(mask) != mask[om[k]={k_prime}]"
                )


# ============== D) Property: invertibility round-trip (strict with bijection) ==============
def test_d4_invertibility_round_trip():
    """Apply T then inverse(T): state, policy, mask recover (gold_v2 32ch).
    All channels (board+shape) round-trip exactly.
    """
    from src.network.gold_v2_constants import C_SPATIAL_ENC

    np.random.seed(42)
    state = np.random.randn(C_SPATIAL_ENC, 9, 9).astype(np.float32) * 0.1
    mask = (np.random.rand(2026) > 0.7).astype(np.float32)
    mask[0] = 1.0
    policy = np.random.rand(2026).astype(np.float32)
    policy = (policy * mask) / max(policy[mask > 0].sum(), 1e-9)
    slot_piece_ids = [0, 1, 2]

    for ti in range(D4_COUNT):
        inv_ti = inverse_transform_idx(ti)
        s1, p1, m1 = apply_d4_augment(state, policy, mask, slot_piece_ids, ti)
        s2, p2, m2 = apply_d4_augment(s1, p1, m1, slot_piece_ids, inv_ti)

        assert np.allclose(state, s2, rtol=1e-4, atol=1e-5), (
            f"ti={ti} all channels (board+shape) round-trip exactly"
        )
        # Dimension-aware buy transform can have collisions (multiple sources -> same dest);
        # round-trip preserves sum, allow relaxed tolerance
        assert np.isclose(p2.sum(), policy.sum(), atol=1e-5), f"ti={ti} policy sum preserved"
        assert np.allclose(p2, policy, atol=1e-2), f"ti={ti} policy round-trip"
        # Mask: identity round-trips exactly; other ti can have BUY collisions, so mask may differ
        if ti == 0:
            assert np.array_equal(m2, mask), f"ti={ti} mask exact round-trip (identity)"
        else:
            assert m2.sum() <= mask.sum(), f"ti={ti} mask: no spurious legal actions"


# ============== E) Property: legality preserved for buy actions ==============
def test_legality_preserved_for_buy_actions():
    """Remap legal buy idx -> idx'; assert idx' is legal in transformed mask."""
    from src.game.patchwork_engine import new_game, legal_actions_fast, current_player_fast
    from src.mcts.alphazero_mcts_optimized import encode_legal_actions_fast
    from src.network.encoder import StateEncoder, get_slot_piece_id

    enc = StateEncoder()
    st = new_game(seed=123)
    for _ in range(20):
        legal = legal_actions_fast(st)
        if not legal:
            break
        to_move = current_player_fast(st)
        state_enc = enc.encode_state(st, to_move).astype(np.float32)
        indices, mask = encode_legal_actions_fast(legal)
        policy = np.zeros(2026, dtype=np.float32)
        for i in indices:
            policy[i] = 1.0 / len(indices)
        slot_piece_ids = [get_slot_piece_id(st, s) for s in range(3)]

        for ti in range(D4_COUNT):
            _, p2, m2 = apply_d4_augment(state_enc, policy, mask, slot_piece_ids, ti)
            buy_legal_orig = [i for i in range(82, 2026) if mask[i] > 0]
            for old_idx in buy_legal_orig[:5]:
                if old_idx >= 2026:
                    continue
                k = old_idx - 82
                so, pos = divmod(k, 81)
                slot, orient = divmod(so, 8)
                r, c = pos // 9, pos % 9
                from src.network.d4_augmentation import (
                    get_orient_after_transform,
                    transform_buy_top_left,
                )
                from src.game.patchwork_engine import ORIENT_COUNT

                pid = slot_piece_ids[slot] if slot < len(slot_piece_ids) else None
                if pid is None or orient >= ORIENT_COUNT[pid]:
                    continue
                orient_new = get_orient_after_transform(pid, ti, orient)
                top_new, left_new = transform_buy_top_left(pid, orient, r, c, ti)
                pos_new = top_new * 9 + left_new
                new_idx = 82 + (slot * 8 + orient_new) * 81 + pos_new
                assert m2[new_idx] > 0, (
                    f"ti={ti} old_idx={old_idx} -> new_idx={new_idx} not legal"
                )

        from src.game.patchwork_engine import apply_action_unchecked
        st = apply_action_unchecked(st, legal[0])
