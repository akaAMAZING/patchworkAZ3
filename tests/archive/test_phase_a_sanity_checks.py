"""
Phase A sanity checks before Phase B.

1) Slot×orient planes match engine orientation indexing used by buy actions.
2) Buy "crossed marks" uses correct move semantics (prev, new_pos, o_pos).
"""
import numpy as np
import pytest

from src.game.patchwork_engine import (
    new_game,
    legal_actions_fast,
    apply_action_unchecked,
    MASK_W0,
    MASK_W1,
    MASK_W2,
    ORIENT_COUNT,
    PIECE_COST_TIME,
    BUTTONS_AFTER,
    PATCHES_AFTER,
    SCOREBOARD_LENGTH,
    EDITION_CODE,
)
from src.network.encoder import (
    StateEncoder,
    ActionEncoder,
    get_slot_piece_id,
    _button_marks_crossed,
    _patches_crossed,
    _clamp_pos,
    TRACK_LENGTH,
)


def _decode_mask(m0: int, m1: int, m2: int) -> np.ndarray:
    """Decode mask words to 9x9 float32 (reuse encoder logic)."""
    from src.network.encoder import StateEncoder
    return StateEncoder._decode_occ_words(m0, m1, m2)


def _engine_mask_at_placement(pid: int, orient: int, top: int, left: int) -> np.ndarray:
    """Get engine's placement mask at (top, left)."""
    pos = top * 9 + left
    return _decode_mask(
        int(MASK_W0[pid, orient, pos]),
        int(MASK_W1[pid, orient, pos]),
        int(MASK_W2[pid, orient, pos]),
    )


def _encoder_shape_at_origin(pid: int, orient: int) -> np.ndarray:
    """Get encoder's shape for (pid, orient) - top-left anchored (pos=0 when valid)."""
    for pos in range(81):
        m0 = int(MASK_W0[pid, orient, pos])
        m1 = int(MASK_W1[pid, orient, pos])
        m2 = int(MASK_W2[pid, orient, pos])
        if (m0 | m1 | m2) != 0:
            return _decode_mask(m0, m1, m2)
    return np.zeros((9, 9), dtype=np.float32)


def test_slot_orient_matches_engine_buy_actions():
    """
    Check 1: For 2-3 asymmetric pieces, decode(index) for buy with orient=k uses
    the exact same mask plane slot*_orientk_shape the encoder fills.
    """
    encoder = StateEncoder()
    action_encoder = ActionEncoder()
    state = new_game(seed=42)

    checked = 0
    seen_pieces = set()

    for _ in range(50):  # Play a few moves to get varied positions
        if checked >= 6:
            break
        legal = legal_actions_fast(state)
        if not legal:
            break
        # Pick a buy action if any
        buy_actions = [a for a in legal if (isinstance(a[0], int) and int(a[0]) == 2) or (isinstance(a[0], str) and a[0] in ("buy", "buy_slot"))]
        if not buy_actions:
            state = apply_action_unchecked(state, legal[0])
            continue

        for action in buy_actions[:3]:
            if checked >= 6:
                break
            if isinstance(action[0], int) and int(action[0]) == 2:
                offset, piece_id, orient, top, left = int(action[1]), int(action[2]), int(action[3]), int(action[4]), int(action[5])
            elif isinstance(action[0], str) and action[0] == "buy_slot":
                slot_index = int(action[1])
                orient = int(action[2])
                top, left = int(action[3]), int(action[4])
                pid_from_slot = get_slot_piece_id(state, slot_index)
                if pid_from_slot is None:
                    continue
                piece_id = pid_from_slot
                offset = slot_index + 1
            else:
                continue

            if piece_id in seen_pieces and orient in (0, 1, 2):
                continue
            if ORIENT_COUNT[piece_id] < 2:
                continue

            seen_pieces.add(piece_id)
            from src.game.patchwork_engine import current_player_fast
            to_move = int(current_player_fast(state))
            enc = encoder.encode_state(state, to_move)
            slot_index = offset - 1
            ch = 4 + slot_index * 8 + orient
            encoder_plane = enc[ch]

            engine_mask = _engine_mask_at_placement(piece_id, orient, top, left)
            encoder_shape = _encoder_shape_at_origin(piece_id, orient)

            assert encoder_shape.sum() > 0, f"Piece {piece_id} orient {orient} should have non-empty shape"
            assert engine_mask.sum() > 0, f"Piece {piece_id} orient {orient} at ({top},{left}) should have placement"

            for r in range(9):
                for c in range(9):
                    if encoder_shape[r, c] > 0.5:
                        tr, tc = top + r, left + c
                        assert 0 <= tr < 9 and 0 <= tc < 9, f"Shape at ({r},{c}) out of bounds when placed at ({top},{left})"
                        assert engine_mask[tr, tc] > 0.5, (
                            f"Piece {piece_id} orient {orient}: encoder shape has ({r},{c}) but engine mask at "
                            f"({top},{left}) has 0 at ({tr},{tc})"
                        )
                for tr in range(9):
                    for tc in range(9):
                        if engine_mask[tr, tc] > 0.5:
                            r, c = tr - top, tc - left
                            assert 0 <= r < 9 and 0 <= c < 9
                            assert encoder_shape[r, c] > 0.5, (
                                f"Piece {piece_id} orient {orient}: engine has ({tr},{tc}) but encoder shape "
                                f"at origin has 0 at ({r},{c})"
                            )

            assert np.allclose(encoder_plane, encoder_shape), (
                f"Encoder channel {ch} (slot{slot_index}_orient{orient}) should match shape at origin for piece {piece_id}"
            )
            checked += 1

        state = apply_action_unchecked(state, buy_actions[0])

    assert checked >= 2, "Should have verified at least 2 buy actions across asymmetric pieces"


def test_buy_crossed_marks_same_as_engine():
    """
    Check 2: buy_*crossed uses prev=c_pos, new_pos=min(c_pos+piece_time, TRACK_END),
    opponent_pos for patches, and same trigger inclusion (prev < m <= new).
    """
    assert TRACK_LENGTH == SCOREBOARD_LENGTH == 53

    for c_pos in [0, 4, 5, 6, 10, 26, 27, 52, 53]:
        for piece_time in [1, 2, 6]:
            buy_new = _clamp_pos(c_pos + piece_time)
            enc_income = _button_marks_crossed(c_pos, buy_new)
            enc_patches_5 = _patches_crossed(c_pos, buy_new, 5, 0)
            enc_patches_ahead = _patches_crossed(c_pos, buy_new, 60, 0)

            for m in BUTTONS_AFTER.tolist():
                if c_pos < m <= buy_new:
                    assert enc_income >= 1, f"c_pos={c_pos} buy_new={buy_new}: trigger {m} should be crossed"
            manual_income = sum(1 for m in BUTTONS_AFTER.tolist() if c_pos < m <= buy_new)
            assert enc_income == manual_income, f"Button crossing: c_pos={c_pos} buy_new={buy_new} expected {manual_income} got {enc_income}"

            for m in PATCHES_AFTER.tolist():
                if c_pos < m <= buy_new and 5 < m:
                    assert enc_patches_5 >= 1
                if c_pos < m <= buy_new and 60 < m:
                    assert enc_patches_ahead >= 1
            manual_patches_5 = sum(1 for m in PATCHES_AFTER.tolist() if c_pos < m <= buy_new and 5 < m)
            manual_patches_ahead = sum(1 for m in PATCHES_AFTER.tolist() if c_pos < m <= buy_new and 60 < m)
            assert enc_patches_5 == manual_patches_5
            assert enc_patches_ahead == manual_patches_ahead

    # Boundary: prev=5, new=5 crosses nothing; prev=4, new=5 crosses m=5
    assert _button_marks_crossed(5, 5) == 0
    assert _button_marks_crossed(4, 5) == 1
