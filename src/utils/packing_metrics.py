"""
Packing / quilt quality metrics for "beat humans" observability.

Shared by selfplay (terminal states), A/B tests, and TensorBoard.
Uses 4-neighbor BFS for empty components; isolated 1x1 = empty cell with zero empty neighbors.
"""

from typing import List, Tuple

import numpy as np

# Reuse engine constants so definitions stay in sync
from src.game.patchwork_engine import (
    BOARD_CELLS,
    BOARD_SIZE,
    empty_count_from_occ,
)


def empties_from_occ_words(occ0: int, occ1: int, occ2: int) -> int:
    """Empty square count from occupancy words (9x9 board)."""
    return empty_count_from_occ(int(occ0), int(occ1), int(occ2))


def fragmentation_from_occ_words(occ0: int, occ1: int, occ2: int) -> Tuple[int, int]:
    """
    Fragmentation for one board from occupancy words.
    Returns (empty_components, isolated_1x1_holes).
    - empty_components: connected components of empty cells (4-neighbor BFS).
    - isolated_1x1_holes: empty cells with zero empty neighbors (4-neighbor).
    """
    occ0_i, occ1_i, occ2_i = int(occ0), int(occ1), int(occ2)
    empty = np.zeros(BOARD_CELLS, dtype=np.bool_)
    for idx in range(BOARD_CELLS):
        word = idx >> 5
        bit = idx & 31
        mask = 1 << bit
        if word == 0:
            occupied = (occ0_i & mask) != 0
        elif word == 1:
            occupied = (occ1_i & mask) != 0
        else:
            occupied = (occ2_i & mask) != 0
        empty[idx] = not occupied

    visited = np.zeros(BOARD_CELLS, dtype=np.bool_)
    components = 0
    isolated = 0

    for start in range(BOARD_CELLS):
        if not empty[start] or visited[start]:
            continue
        components += 1
        stack = [start]
        visited[start] = True
        size = 0
        has_neighbour = False
        while stack:
            i = stack.pop()
            size += 1
            r, c = divmod(i, BOARD_SIZE)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    ni = nr * BOARD_SIZE + nc
                    if empty[ni]:
                        has_neighbour = True
                        if not visited[ni]:
                            visited[ni] = True
                            stack.append(ni)
        if size == 1 and not has_neighbour:
            isolated += 1

    return components, isolated


def aggregate_packing_over_games(
    per_game: List[Tuple[int, int, int, int, int, int]],
) -> dict:
    """
    per_game: list of (empty_p0, empty_p1, comp_p0, comp_p1, iso_p0, iso_p1) per game.
    Returns dict with:
      - *_mean: mean over all 2*G values (both players, all games)
      - *_(p50|p90)_final_empty_squares_mean: percentile of per-game mean (e.g. (e0+e1)/2) across games
      - *_abs_diff: mean over games of |p0 - p1|
    """
    if not per_game:
        return {
            "selfplay_avg_final_empty_squares_mean": 0.0,
            "selfplay_avg_final_empty_components_mean": 0.0,
            "selfplay_avg_final_isolated_1x1_holes_mean": 0.0,
            "selfplay_p50_final_empty_squares_mean": 0.0,
            "selfplay_p90_final_empty_squares_mean": 0.0,
            "selfplay_p50_final_empty_components_mean": 0.0,
            "selfplay_p90_final_empty_components_mean": 0.0,
            "selfplay_p50_final_isolated_1x1_holes_mean": 0.0,
            "selfplay_p90_final_isolated_1x1_holes_mean": 0.0,
            "selfplay_avg_final_empty_squares_abs_diff": 0.0,
            "selfplay_avg_final_empty_components_abs_diff": 0.0,
            "selfplay_avg_final_isolated_1x1_holes_abs_diff": 0.0,
        }

    arr = np.array(per_game, dtype=np.float64)  # (G, 6)
    e0, e1 = arr[:, 0], arr[:, 1]
    c0, c1 = arr[:, 2], arr[:, 3]
    i0, i1 = arr[:, 4], arr[:, 5]

    # Aggregate means (all 2G values)
    all_empty = np.concatenate([e0, e1])
    all_comp = np.concatenate([c0, c1])
    all_iso = np.concatenate([i0, i1])

    # Per-game mean across players (G values)
    game_mean_empty = (e0 + e1) / 2.0
    game_mean_comp = (c0 + c1) / 2.0
    game_mean_iso = (i0 + i1) / 2.0

    # Abs diff per game
    abs_diff_empty = np.abs(e0 - e1)
    abs_diff_comp = np.abs(c0 - c1)
    abs_diff_iso = np.abs(i0 - i1)

    return {
        "selfplay_avg_final_empty_squares_mean": float(np.mean(all_empty)),
        "selfplay_avg_final_empty_components_mean": float(np.mean(all_comp)),
        "selfplay_avg_final_isolated_1x1_holes_mean": float(np.mean(all_iso)),
        "selfplay_p50_final_empty_squares_mean": float(np.percentile(game_mean_empty, 50)),
        "selfplay_p90_final_empty_squares_mean": float(np.percentile(game_mean_empty, 90)),
        "selfplay_p50_final_empty_components_mean": float(np.percentile(game_mean_comp, 50)),
        "selfplay_p90_final_empty_components_mean": float(np.percentile(game_mean_comp, 90)),
        "selfplay_p50_final_isolated_1x1_holes_mean": float(np.percentile(game_mean_iso, 50)),
        "selfplay_p90_final_isolated_1x1_holes_mean": float(np.percentile(game_mean_iso, 90)),
        "selfplay_avg_final_empty_squares_abs_diff": float(np.mean(abs_diff_empty)),
        "selfplay_avg_final_empty_components_abs_diff": float(np.mean(abs_diff_comp)),
        "selfplay_avg_final_isolated_1x1_holes_abs_diff": float(np.mean(abs_diff_iso)),
    }


def aggregate_root_over_moves(
    root_legal_counts: List[int],
    root_expanded_counts: List[int],
) -> dict:
    """
    Flattened lists: one (legal, expanded) per model move across all games.
    Returns dict with avg_root_legal_count, avg_root_expanded_count, avg_ratio, p90_legal, p90_ratio.
    """
    if not root_legal_counts or not root_expanded_counts:
        return {
            "selfplay_avg_root_legal_count": 0.0,
            "selfplay_avg_root_expanded_count": 0.0,
            "selfplay_avg_root_expanded_ratio": 0.0,
            "selfplay_p90_root_legal_count": 0.0,
            "selfplay_p90_root_expanded_ratio": 0.0,
        }

    legal = np.array(root_legal_counts, dtype=np.float64)
    expanded = np.array(root_expanded_counts, dtype=np.float64)
    ratio = expanded / np.maximum(legal, 1.0)

    return {
        "selfplay_avg_root_legal_count": float(np.mean(legal)),
        "selfplay_avg_root_expanded_count": float(np.mean(expanded)),
        "selfplay_avg_root_expanded_ratio": float(np.mean(ratio)),
        "selfplay_p90_root_legal_count": float(np.percentile(legal, 90)),
        "selfplay_p90_root_expanded_ratio": float(np.percentile(ratio, 90)),
    }
