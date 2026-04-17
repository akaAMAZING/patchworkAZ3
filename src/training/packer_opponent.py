"""
Greedy "packer" opponent: human proxy that chooses moves by board-packing quality.

No search. For each legal move we evaluate by:
  (a) minimize number of empty connected components
  (b) minimize empty squares
  (c) maximize buttons (tie-break)
PASS/PATCH handled normally (PATCH placements evaluated by (a)(b); PASS when only option or no better BUY).
"""

from typing import Tuple

import numpy as np

from src.game.patchwork_engine import (
    AT_PASS,
    AT_PATCH,
    AT_BUY,
    BOARD_SIZE,
    BOARD_CELLS,
    P0_OCC0,
    P0_OCC1,
    P0_OCC2,
    P0_BUTTONS,
    P1_OCC0,
    P1_OCC1,
    P1_OCC2,
    P1_BUTTONS,
    apply_action_unchecked,
    current_player_fast,
    empty_count_from_occ,
    legal_actions_fast,
    new_game,
)


def _is_occupied(occ0: int, occ1: int, occ2: int, idx: int) -> bool:
    """True if cell idx is occupied (has a piece)."""
    word = idx >> 5
    bit = idx & 31
    mask = 1 << bit
    if word == 0:
        return (occ0 & mask) != 0
    if word == 1:
        return (occ1 & mask) != 0
    return (occ2 & mask) != 0


def count_empty_connected_components(occ0: int, occ1: int, occ2: int) -> int:
    """Count connected components of empty cells on the 9x9 board. Uses BFS."""
    occ0, occ1, occ2 = int(occ0), int(occ1), int(occ2)
    empty = np.array(
        [not _is_occupied(occ0, occ1, occ2, i) for i in range(BOARD_CELLS)],
        dtype=np.bool_,
    )
    visited = np.zeros(BOARD_CELLS, dtype=np.bool_)
    components = 0
    for start in range(BOARD_CELLS):
        if not empty[start] or visited[start]:
            continue
        components += 1
        stack = [start]
        visited[start] = True
        while stack:
            i = stack.pop()
            r, c = i // BOARD_SIZE, i % BOARD_SIZE
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    ni = nr * BOARD_SIZE + nc
                    if empty[ni] and not visited[ni]:
                        visited[ni] = True
                        stack.append(ni)
    return components


def _get_occ_after_move(state: np.ndarray, action: tuple, pl: int) -> Tuple[int, int, int, int]:
    """Apply action and return (occ0, occ1, occ2, buttons) for player pl. Uses engine layout constants only."""
    # Call on a copy so we never mutate the caller's state if the engine ever mutates in place.
    new_state = apply_action_unchecked(np.asarray(state, dtype=state.dtype).copy(), action)
    if pl == 0:
        occ0 = int(new_state[P0_OCC0])
        occ1 = int(new_state[P0_OCC1])
        occ2 = int(new_state[P0_OCC2])
        buttons = int(new_state[P0_BUTTONS])
    else:
        occ0 = int(new_state[P1_OCC0])
        occ1 = int(new_state[P1_OCC1])
        occ2 = int(new_state[P1_OCC2])
        buttons = int(new_state[P1_BUTTONS])
    return occ0, occ1, occ2, buttons


class PackerOpponent:
    """Greedy opponent: choose BUY/PATCH by (a) min empty components (b) min empty count (c) max buttons."""

    def get_move(self, state: np.ndarray, seed_offset: int = 0) -> tuple:
        """Pick a legal move that best improves packing (min components, min empty, max buttons)."""
        legal = legal_actions_fast(state)
        if not legal:
            return (AT_PASS, 0, 0, 0, 0, 0)
        if len(legal) == 1:
            return legal[0]

        pl = int(current_player_fast(state))
        pass_actions = [a for a in legal if int(a[0]) == AT_PASS]
        patch_actions = [a for a in legal if int(a[0]) == AT_PATCH]
        buy_actions = [a for a in legal if int(a[0]) == AT_BUY]

        if patch_actions:
            best = patch_actions[0]
            best_comp, best_empty = float("inf"), float("inf")
            for a in patch_actions:
                occ0, occ1, occ2, _ = _get_occ_after_move(state, a, pl)
                comp = count_empty_connected_components(occ0, occ1, occ2)
                empty = empty_count_from_occ(occ0, occ1, occ2)
                if (comp, empty) < (best_comp, best_empty):
                    best_comp, best_empty = comp, empty
                    best = a
            return best

        if buy_actions:
            best = buy_actions[0]
            best_key = (float("inf"), float("inf"), -float("inf"))
            for a in buy_actions:
                occ0, occ1, occ2, buttons = _get_occ_after_move(state, a, pl)
                comp = count_empty_connected_components(occ0, occ1, occ2)
                empty = empty_count_from_occ(occ0, occ1, occ2)
                key = (comp, empty, -buttons)
                if key < best_key:
                    best_key = key
                    best = a
            return best

        return pass_actions[0] if pass_actions else legal[0]


def _sanity_check(num_trials: int = 100) -> None:
    """Run packer on random states; verify return is legal and state unchanged. Print PASS if ok."""
    import random
    from src.game.patchwork_engine import apply_action_unchecked, terminal_fast

    packer = PackerOpponent()
    rng = random.Random(42)
    for trial in range(num_trials):
        seed = rng.randint(0, 2**31 - 1)
        state = new_game(seed=seed)
        # Advance a few random moves to get non-trivial states
        for _ in range(rng.randint(0, 20)):
            if terminal_fast(state):
                break
            legal = legal_actions_fast(state)
            if not legal:
                break
            action = rng.choice(legal)
            state = apply_action_unchecked(state, action)
        if terminal_fast(state):
            continue
        legal = legal_actions_fast(state)
        if not legal:
            continue
        state_before = np.array(state, copy=True)
        move = packer.get_move(state, seed_offset=trial)
        if not np.array_equal(state, state_before):
            raise AssertionError(f"Packer mutated state (trial {trial})")
        if move not in legal:
            raise AssertionError(f"Packer returned illegal move (trial {trial}): {move} not in legal")
    print("PASS")


if __name__ == "__main__":
    _sanity_check(100)
