"""
Tests for legacy StateEncoder (deprecated; production uses 56ch gold_v2).
"""
import numpy as np
import pytest

from src.game.patchwork_engine import (
    new_game,
    apply_action_unchecked,
    legal_actions_fast,
    current_player_fast,
    P0_POS,
    P1_POS,
    BUTTONS_AFTER,
    PATCHES_AFTER,
)
from src.network.encoder import StateEncoder, ActionEncoder, get_slot_piece_id

_LEGACY_SPATIAL = StateEncoder.NUM_CHANNELS  # legacy encoder output size


def test_encoder_output_shape():
    """Legacy encoder output shape."""
    encoder = StateEncoder()
    state = new_game(seed=42)
    enc = encoder.encode_state(state, 0)
    assert enc.shape == (_LEGACY_SPATIAL, 9, 9), f"Expected ({_LEGACY_SPATIAL},9,9), got {enc.shape}"
    assert enc.dtype == np.float32


def test_coord_planes_correctness():
    """Channels 2 and 3: coord_row_norm and coord_col_norm."""
    encoder = StateEncoder()
    state = new_game(seed=42)
    enc = encoder.encode_state(state, 0)
    denom = 8.0  # BOARD_SIZE-1
    for r in range(9):
        for c in range(9):
            assert enc[2, r, c] == r / denom, f"coord_row at ({r},{c})"
            assert enc[3, r, c] == c / denom, f"coord_col at ({r},{c})"


def test_slot_orient_shapes_non_empty_for_buyable():
    """Slot×orient shape planes non-empty when slot has buyable piece."""
    encoder = StateEncoder()
    state = new_game(seed=42)
    enc = encoder.encode_state(state, 0)
    for slot_idx in range(3):
        pid = get_slot_piece_id(state, slot_idx)
        if pid is not None:
            base = 4 + slot_idx * 8
            total = 0
            for o in range(8):
                total += enc[base + o].sum()
            assert total > 0, f"Slot {slot_idx} has piece {pid} but no non-zero shape plane"


def test_dist_to_next_income_correctness():
    """dist_to_next_income: steps to next trigger > pos; 0 if at/after last."""
    # Hand-constructed: pos=0 -> next income at 5, dist=5
    state = new_game(seed=42)
    enc = __encode_positions(state, 0, 0)  # c_pos=0, o_pos=0
    # ch36 = dist_to_next_income_current_norm
    dist_norm = enc[36, 0, 0]
    assert 0 < dist_norm <= 1.0  # 5/53
    # pos=53 -> at end, dist=0
    enc_end = __encode_positions(state, 53, 53)
    assert enc_end[36, 0, 0] == 0.0
    assert enc_end[37, 0, 0] == 0.0


def __encode_positions(state, c_pos, o_pos):
    """Helper: encode state with c_pos and o_pos (to_move=0)."""
    s = state.copy()
    s[P0_POS] = c_pos
    s[P1_POS] = o_pos
    encoder = StateEncoder()
    return encoder.encode_state(s, 0)


def test_dist_to_next_patch_correctness():
    """dist_to_next_patch: same logic with PATCHES_AFTER."""
    state = new_game(seed=42)
    enc = __encode_positions(state, 20, 20)  # before first patch at 26
    dist_norm = enc[38, 0, 0]
    assert dist_norm > 0  # 6/53 to 26
    enc_end = __encode_positions(state, 53, 53)
    assert enc_end[38, 0, 0] == 0.0
    assert enc_end[39, 0, 0] == 0.0


def test_pass_effects_controlled_state():
    """pass_* correctness: c_pos=0, o_pos=0 -> pass moves to 1, 0 income marks."""
    state = new_game(seed=42)
    enc = __encode_positions(state, 0, 0)
    pass_steps = enc[40, 0, 0]  # pass_steps_norm
    pass_income = enc[41, 0, 0]
    pass_patches = enc[42, 0, 0]
    # Pass from (0,0): new_pos=min(0+1,53)=1, steps=1
    assert pass_steps > 0
    assert np.isclose(pass_steps, 1.0 / 53.0), f"pass_steps_norm should be 1/53, got {pass_steps}"
    # No income marks crossed (prev=0, new=1, triggers at 5,11,...; 0<5<=1 false)
    assert pass_income == 0.0
    # No patches (prev=0, new=1, first patch at 26)
    assert pass_patches == 0.0


def test_hdf5_shape_and_attrs():
    """HDF5 writing uses legacy spatial shape and attrs."""
    import tempfile
    import h5py
    from src.training.run_layout import (
        SELFPLAY_SCHEMA_VERSION_ATTR,
        SELFPLAY_EXPECTED_CHANNELS_ATTR,
        ENCODING_VERSION_ATTR,
    )

    encoder = StateEncoder()
    states = []
    for _ in range(5):
        s = new_game(seed=42 + _)
        pl = current_player_fast(s)
        enc = encoder.encode_state(s, int(pl))
        states.append(enc)
    arr = np.stack(states, axis=0)
    assert arr.shape == (5, _LEGACY_SPATIAL, 9, 9)

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        path = f.name
    try:
        with h5py.File(path, "w") as f:
            f.create_dataset("states", data=arr.astype(np.float32))
            f.attrs[SELFPLAY_SCHEMA_VERSION_ATTR] = 2
            f.attrs[SELFPLAY_EXPECTED_CHANNELS_ATTR] = _LEGACY_SPATIAL
            f.attrs[ENCODING_VERSION_ATTR] = "full_clarity_v1"
        with h5py.File(path, "r") as f:
            assert f["states"].shape == (5, _LEGACY_SPATIAL, 9, 9)
            assert f.attrs[SELFPLAY_EXPECTED_CHANNELS_ATTR] == _LEGACY_SPATIAL
            assert f.attrs[ENCODING_VERSION_ATTR] == "full_clarity_v1"
    finally:
        import os
        os.unlink(path)


def test_scalar_channels_full_plane():
    """Channels 28 to end are full-plane constants."""
    encoder = StateEncoder()
    state = new_game(seed=99)
    enc = encoder.encode_state(state, 0)
    for ch in range(28, _LEGACY_SPATIAL):
        plane = enc[ch]
        assert np.all(plane == plane.flat[0]), f"Channel {ch} must be full-plane constant"
