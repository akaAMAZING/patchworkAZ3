"""
End-to-end invariance audit for D4 augmentation: POLICY + MASK + ACTION encoding.

1) Samples N random mid-game states (by playing games and saving states).
2) For each state s:
   a) L(s) = legal_actions_fast(s) [same legality source as MCTS]
   b) Pick up to 10 random actions from L(s)
   c) For each D4 transform t in 0..7:
      - s_t = transform_engine_state_boards(s, t)
      - a_t = transform_engine_action(a, s, t)
      - s_next = apply_action_unchecked(s, a)
      - s_t_next = apply_action_unchecked(s_t, a_t)
      - s_next_transformed = transform_engine_state_boards(s_next, t)
      - Assert s_t_next == s_next_transformed (field-by-field; report first diff)
3) Mask validation: mask(s_t) should equal transform_action_vector(mask(s), t) for all t.
4) Fail fast and print first counterexample with full decoded action components.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# State index names for error reporting
STATE_INDEX_NAMES = [
    "P0_POS", "P0_BUTTONS", "P0_INCOME", "P0_OCC0", "P0_OCC1", "P0_OCC2",
    "P1_POS", "P1_BUTTONS", "P1_INCOME", "P1_OCC0", "P1_OCC1", "P1_OCC2",
    "CIRCLE_LEN", "NEUTRAL", "BONUS_OWNER", "PENDING_PATCHES", "PENDING_OWNER", "TIE_PLAYER",
] + [f"CIRCLE_{i}" for i in range(33)] + ["EDITION_CODE"]


def _state_diff(s1: "np.ndarray", s2: "np.ndarray") -> Optional[Tuple[int, str, int, int]]:
    """Return first (index, name, val1, val2) where s1[i] != s2[i], else None."""
    n = min(len(s1), len(s2), len(STATE_INDEX_NAMES))
    for i in range(n):
        if int(s1[i]) != int(s2[i]):
            name = STATE_INDEX_NAMES[i] if i < len(STATE_INDEX_NAMES) else f"state[{i}]"
            return (i, name, int(s1[i]), int(s2[i]))
    if len(s1) != len(s2):
        return (n, f"len", len(s1), len(s2))
    return None


def _decode_action_tuple(action: Tuple) -> str:
    """Human-readable decoded action for error output."""
    atype = int(action[0])
    if atype == 0:  # AT_PASS
        return "PASS"
    if atype == 1:  # AT_PATCH
        return f"PATCH(pos={action[1]})"
    if atype == 2:  # AT_BUY
        return f"BUY(offset={action[1]}, piece_id={action[2]}, orient={action[3]}, top={action[4]}, left={action[5]})"
    return str(action)


def main() -> int:
    import numpy as np
    from src.game.patchwork_engine import (
        new_game,
        apply_action_unchecked,
        terminal_fast,
        current_player_fast,
        legal_actions_fast,
        AT_PASS,
        AT_PATCH,
        AT_BUY,
        STATE_SIZE,
    )
    from src.mcts.alphazero_mcts_optimized import encode_legal_actions_fast, engine_action_to_flat_index
    from src.network.encoder import get_slot_piece_id
    from src.network.d4_augmentation import (
        transform_action_vector,
        D4_COUNT,
        D4_NAMES,
    )
    # Reuse engine-state and engine-action transform from existing equivariance test
    from tests.test_d4_action_equivariance import (
        transform_engine_state_boards,
        transform_engine_action,
    )

    ap = argparse.ArgumentParser(description="D4 policy+mask+action equivariance audit")
    ap.add_argument("--n-states", type=int, default=20, help="Number of mid-game states to sample")
    ap.add_argument("--actions-per-state", type=int, default=10, help="Random actions to test per state")
    ap.add_argument("--seed", type=int, default=12345, help="RNG seed for sampling")
    ap.add_argument("--move-min", type=int, default=8, help="Min moves before saving state")
    ap.add_argument("--move-max", type=int, default=45, help="Max moves (stop game)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    # ----- 1) Sample N mid-game states -----
    states: List[Tuple[np.ndarray, int, Optional[int]]] = []  # (state, seed_used, move_count)
    games_played = 0
    max_games = 500
    while len(states) < args.n_states and games_played < max_games:
        seed = rng.randint(0, 999999)
        st = new_game(seed=seed)
        move_count = 0
        while not terminal_fast(st) and move_count < args.move_max:
            legal = legal_actions_fast(st)
            if not legal:
                break
            if args.move_min <= move_count <= args.move_max - 2:
                states.append((st.copy(), seed, move_count))
                if len(states) >= args.n_states:
                    break
            a = rng.choice(legal)
            st = apply_action_unchecked(st, a)
            move_count += 1
        games_played += 1

    if len(states) < args.n_states:
        print(f"Warning: collected only {len(states)} states after {games_played} games")
    states = states[: args.n_states]
    print(f"D4 policy+mask+action audit: {len(states)} states, up to {args.actions_per_state} actions each, 8 transforms")
    print()

    # ----- 2) For each state, test action equivariance -----
    for state_idx, (s, game_seed, move_count) in enumerate(states):
        legal = legal_actions_fast(s)
        if not legal:
            continue
        to_move = current_player_fast(s)
        slot_piece_ids: List[Optional[int]] = [
            get_slot_piece_id(s, slot) for slot in range(3)
        ]
        n_actions = min(args.actions_per_state, len(legal))
        chosen = rng.sample(legal, n_actions)

        for action in chosen:
            for ti in range(D4_COUNT):
                s_t = transform_engine_state_boards(s, ti)
                a_t = transform_engine_action(action, s, ti)
                s_next = apply_action_unchecked(s.copy(), action)
                s_t_next = apply_action_unchecked(s_t.copy(), a_t)
                s_next_transformed = transform_engine_state_boards(s_next, ti)

                diff = _state_diff(s_t_next, s_next_transformed)
                if diff is not None:
                    idx, name, v1, v2 = diff
                    print("FAIL: state/action/transform equivariance")
                    print(f"  state_seed={game_seed}  move_count={move_count}  state_index_in_batch={state_idx}")
                    print(f"  action={_decode_action_tuple(action)}")
                    print(f"  flat_index={engine_action_to_flat_index(action)}")
                    print(f"  transform_index={ti} ({D4_NAMES[ti]})")
                    print(f"  transformed_action={_decode_action_tuple(a_t)}")
                    print(f"  field_diff: {name}  s_t_next={v1}  transform(s_next)={v2}")
                    return 1

    print("PASS: state/action equivariance (apply(a,s)->s_next; apply(a_t,s_t)->s_t_next; s_t_next == transform(s_next,t))")
    print()

    # ----- 3) Mask validation: mask(s_t) == transform_action_vector(mask(s), t) -----
    for state_idx, (s, game_seed, move_count) in enumerate(states):
        legal = legal_actions_fast(s)
        if not legal:
            continue
        _, mask_s = encode_legal_actions_fast(legal)
        slot_piece_ids = [get_slot_piece_id(s, slot) for slot in range(3)]

        for ti in range(D4_COUNT):
            s_t = transform_engine_state_boards(s, ti)
            legal_t = legal_actions_fast(s_t)
            _, mask_s_t_actual = encode_legal_actions_fast(legal_t)
            mask_s_t_expected = transform_action_vector(mask_s.copy(), ti, slot_piece_ids)
            mask_s_t_expected_bin = (mask_s_t_expected > 0).astype(np.float32)
            if mask_s_t_actual.shape != mask_s_t_expected_bin.shape:
                print("FAIL: mask shape mismatch")
                print(f"  state_seed={game_seed}  transform_index={ti} ({D4_NAMES[ti]})")
                print(f"  mask(s_t).shape={mask_s_t_actual.shape}  transform_mask(s).shape={mask_s_t_expected_bin.shape}")
                return 1
            diff = np.abs(mask_s_t_actual - mask_s_t_expected_bin)
            if diff.max() > 0.5:
                where = np.where(diff > 0.5)
                first_flat = int(np.flatnonzero(diff > 0.5)[0])
                print("FAIL: mask(s_t) != transform_mask(mask(s), t)")
                print(f"  state_seed={game_seed}  move_count={move_count}  state_index={state_idx}")
                print(f"  transform_index={ti} ({D4_NAMES[ti]})")
                print(f"  first_differing_flat_index={first_flat}  mask(s_t)[i]={mask_s_t_actual.flat[first_flat]}  transformed[i]={mask_s_t_expected_bin.flat[first_flat]}")
                return 1

    print("PASS: mask(s_t) == transform_action_vector(mask(s), t) for all sampled states and t in 0..7")
    print()
    print("All D4 policy+mask+action invariance checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
