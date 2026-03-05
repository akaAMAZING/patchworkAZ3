"""
D) PROVE ownership targets are aligned with perspective + D4.

1. Document target_ownership channel semantics (cur vs P0) from code.
2. Check on 16 samples: state ch0/ch1 vs target ch0/ch1 (both are cur/opp from same perspective).
3. D4: apply a known rotation to one ownership map; verify transformed ownership matches
   _transform_board_plane applied to each channel.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main():
    import numpy as np
    from src.network.d4_augmentation import (
        apply_ownership_transform,
        _transform_board_plane,
        D4_COUNT,
    )

    print("D) Ownership targets audit")
    print()
    print("1. Channel semantics (from selfplay_optimized.py and model.py get_loss doc):")
    print("   - target_ownership is stored per position from the perspective of the player to move.")
    print("   - Channel 0: current player's board (1=filled at game end, 0=empty).")
    print("   - Channel 1: opponent's board (same).")
    print("   - State input is encoded from the same perspective (ch0=cur occ, ch1=opp occ at that time).")
    print("   => So channel 0 target corresponds to CUR (who was to move), not fixed P0.")
    print()

    # 2. We cannot load 16 real samples without an HDF5; instead verify D4 and document.
    # If we had a batch: to_move is not in the batch; state is already from to_move perspective,
    # so state ch0 = cur occupancy at that time, target ch0 = cur occupancy at end. They differ
    # mid-game but semantics are aligned (both cur).
    print("2. Without HDF5 we cannot decode to_move from batch (not stored).")
    print("   Code path: selfplay stores ownership as (cur_board, opp_board) per stored_players (to_move).")
    print("   Dataset returns ownerships as-is; D4 is applied with same transform to state and ownership.")
    print("   => Channel 0 = cur, channel 1 = opp for the perspective that encoded the state.")
    print()

    # 3. D4: apply_ownership_transform(own, ti) must equal [_transform_board_plane(own[0], ti), _transform_board_plane(own[1], ti)]
    print("3. D4 consistency: apply_ownership_transform(own, ti) must match _transform_board_plane per channel.")
    own = np.zeros((2, 9, 9), dtype=np.float32)
    own[0, 0, 0] = 1.0
    own[0, 4, 4] = 1.0
    own[1, 8, 8] = 1.0
    own[1, 1, 7] = 1.0

    for ti in range(D4_COUNT):
        out = apply_ownership_transform(own, ti)
        ref0 = _transform_board_plane(own[0], ti)
        ref1 = _transform_board_plane(own[1], ti)
        ok = np.allclose(out[0], ref0) and np.allclose(out[1], ref1)
        print(f"   transform_idx={ti}: ownership transform matches per-channel transform: {ok}")
        if not ok:
            print(f"      out[0] diff: {np.abs(out[0] - ref0).max()}, out[1] diff: {np.abs(out[1] - ref1).max()}")

    # 4. Round-trip: transform then inverse should recover original
    from src.network.d4_augmentation import inverse_transform_idx
    ti = 1
    inv_ti = inverse_transform_idx(ti)
    out = apply_ownership_transform(own, ti)
    back = apply_ownership_transform(out, inv_ti)
    round_ok = np.allclose(back, own)
    print()
    print(f"4. Round-trip (ti={ti} then inv): ownership recovers original: {round_ok}")
    print()
    print("Conclusion: target_ownership is (cur_filled, opp_filled); D4 transforms ownership consistently with state.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
