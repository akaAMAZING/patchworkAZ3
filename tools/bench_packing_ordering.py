#!/usr/bin/env python3
"""
Benchmark packing ordering: baseline (prior only) vs optimized (bitboard + top-M).

Samples random midgame states, generates legal BUY actions, times:
  - Baseline: sort BUY by prior only.
  - Packing: decode occupancy once, compute heuristic for top M only, sort by rank_key.

Reports: avg ms per root ordering, BUY count, effective M.
"""

import random
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.game.patchwork_engine import (
    new_game,
    legal_actions_list,
    apply_action_unchecked,
    terminal_fast,
    current_player_fast,
    AT_BUY,
    P0_OCC0,
    P0_OCC1,
    P0_OCC2,
    P1_OCC0,
    P1_OCC1,
    P1_OCC2,
)
from src.mcts.packing_heuristic import (
    occ_words_to_bitboard_for_node,
    packing_heuristic_score_fast,
    packing_heuristic_scores_batch,
    cache_index,
    packing_heuristic_score,
    _CYTHON_AVAILABLE,
    PLACEMENT_LO,
    PLACEMENT_HI,
    WINDOW_LO,
    WINDOW_HI,
)
# For Cython-only timing (allocation vs call)
try:
    from src.mcts.packing_heuristic import packing_scores_batch_cy
    _CYTHON_DIRECT_AVAILABLE = _CYTHON_AVAILABLE and packing_scores_batch_cy is not None
except ImportError:
    packing_scores_batch_cy = None
    _CYTHON_DIRECT_AVAILABLE = False
_SCALE = 3.0

# Config-like (match production)
WEIGHTS = {"adj_edges": 1.0, "corner_bonus": 0.5, "iso_hole_penalty": 2.0, "frontier_penalty": 0.25}
LOCAL_RADIUS = 2
ALPHA = 0.15
USE_LOG_PRIOR = True
K_ROOT = 64
K0 = 32


def sample_midgame_states(n_states: int, seed: int = 42, min_moves: int = 15, max_moves: int = 80) -> list:
    """Play random legal moves to get midgame states; collect states that have BUY actions."""
    rng = random.Random(seed)
    states = []
    for _ in range(n_states * 6):  # over-sample then take n_states with BUY
        state = new_game(seed=rng.randint(0, 2**31 - 1))
        moves = rng.randint(min_moves, max_moves)
        for _ in range(moves):
            if terminal_fast(state):
                break
            legal = legal_actions_list(state)
            if not legal:
                break
            a = rng.choice(legal)
            state = apply_action_unchecked(state, a)
        buy_actions = [a for a in legal_actions_list(state) if int(a[0]) == AT_BUY]
        if buy_actions and not terminal_fast(state):
            states.append((state, buy_actions))
            if len(states) >= n_states:
                break
    return states


def baseline_ordering(buy_actions: list, priors: np.ndarray, action_to_idx: dict) -> list:
    """Sort BUY by prior descending. Returns ordered list of actions."""
    with_prior = [(a, float(priors[action_to_idx[a]])) for a in buy_actions]
    with_prior.sort(key=lambda x: -x[1])
    return [a for a, _ in with_prior]


def packing_ordering_legacy(
    state,
    buy_actions: list,
    priors: np.ndarray,
    action_to_idx: dict,
) -> list:
    """Legacy: decode state + full numpy heuristic for every BUY (pre-optimization behavior)."""
    with_rank = []
    for a in buy_actions:
        prior = max(float(priors[action_to_idx[a]]), 1e-10)
        base = np.log(prior) if USE_LOG_PRIOR else prior
        pack_score = packing_heuristic_score(state, a, WEIGHTS, LOCAL_RADIUS)
        rank_key = base + ALPHA * pack_score
        with_rank.append((a, rank_key))
    with_rank.sort(key=lambda x: -x[1])
    return [a for a, _ in with_rank]


def packing_ordering(
    state,
    buy_actions: list,
    priors: np.ndarray,
    action_to_idx: dict,
    M: int,
    return_timings: bool = False,
    use_cython: bool = True,
    use_indices_buffer: bool = True,
):
    """Optimized: decode once, heuristic for top M only. use_cython=False forces Python batch.
    use_indices_buffer=True passes a prealloc int32 array (no per-call list→array). Returns (ordered actions, M) or (ordered actions, M, timings_dict)."""
    t0 = time.perf_counter()
    pl = current_player_fast(state)
    if pl == 0:
        o0, o1, o2 = int(state[P0_OCC0]), int(state[P0_OCC1]), int(state[P0_OCC2])
    else:
        o0, o1, o2 = int(state[P1_OCC0]), int(state[P1_OCC1]), int(state[P1_OCC2])
    filled_bb = occ_words_to_bitboard_for_node(o0, o1, o2)
    t_decode = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    with_prior = [(a, float(priors[action_to_idx[a]])) for a in buy_actions]
    with_prior.sort(key=lambda x: -x[1])
    M = min(len(buy_actions), M)
    t_sort_prior = (time.perf_counter() - t0) * 1000

    radius_index = min(max(0, LOCAL_RADIUS), 3)
    w_adj = float(WEIGHTS.get("adj_edges", 1.0))
    w_corner = float(WEIGHTS.get("corner_bonus", 0.5))
    w_iso = float(WEIGHTS.get("iso_hole_penalty", 2.0))
    w_front = float(WEIGHTS.get("frontier_penalty", 0.25))
    if use_indices_buffer:
        indices_buf = np.empty(M, dtype=np.int32)
        for i, (a, _) in enumerate(with_prior[:M]):
            indices_buf[i] = cache_index(int(a[2]), int(a[3]), int(a[4]) * 9 + int(a[5]))
        indices_arg = indices_buf
    else:
        indices_arg = [cache_index(int(a[2]), int(a[3]), int(a[4]) * 9 + int(a[5])) for a, _ in with_prior[:M]]

    t0 = time.perf_counter()
    pack_scores = packing_heuristic_scores_batch(
        filled_bb, indices_arg, radius_index, w_adj, w_corner, w_iso, w_front, 0.0, _use_cython=use_cython
    )
    buy_with_rank = []
    for i, (a, prior) in enumerate(with_prior):
        base = np.log(max(prior, 1e-10)) if USE_LOG_PRIOR else max(prior, 1e-10)
        pack_score = pack_scores[i] if i < M else 0.0
        rank_key = base + ALPHA * pack_score
        buy_with_rank.append((a, rank_key))
    t_heuristic = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    buy_with_rank.sort(key=lambda x: -x[1])
    ordered = [a for a, _ in buy_with_rank]
    t_sort_rank = (time.perf_counter() - t0) * 1000

    if return_timings:
        return ordered, M, {
            "decode_ms": t_decode,
            "sort_prior_ms": t_sort_prior,
            "heuristic_ms": t_heuristic,
            "sort_rank_ms": t_sort_rank,
            "total_ms": t_decode + t_sort_prior + t_heuristic + t_sort_rank,
        }
    return ordered, M


def main():
    n_states = 200
    print("Sampling", n_states, "midgame states with BUY actions...")
    state_action_list = sample_midgame_states(n_states)
    if len(state_action_list) < n_states:
        print(f"  Got {len(state_action_list)} states (increase over-sample if needed)")
    n_states = len(state_action_list)

    # M formula: same as MCTS (root K_cap+16, here we use root)
    M_formula = lambda L: min(L, K_ROOT + 16)

    rng = np.random.default_rng(123)
    timings_baseline = []
    timings_legacy = []
    timings_packing = []
    timings_cython = [] if _CYTHON_AVAILABLE else None
    timings_heuristic_cython = [] if _CYTHON_AVAILABLE else None
    timings_heuristic_list_path = [] if _CYTHON_AVAILABLE else None  # heuristic ms when passing list (alloc inside)
    timings_alloc_list = [] if _CYTHON_AVAILABLE else None  # list build + np.array
    timings_cython_call_only = [] if _CYTHON_DIRECT_AVAILABLE else None  # raw Cython call
    buy_counts = []
    M_used = []
    timings_decode = []
    timings_sort_prior = []
    timings_heuristic = []
    timings_sort_rank = []

    for state, buy_actions in state_action_list:
        L = len(buy_actions)
        buy_counts.append(L)
        action_to_idx = {a: i for i, a in enumerate(buy_actions)}
        priors = rng.exponential(0.1, size=L).astype(np.float64)
        priors /= priors.sum()

        # Baseline (prior only)
        t0 = time.perf_counter()
        for _ in range(10):
            baseline_ordering(buy_actions, priors, action_to_idx)
        t1 = time.perf_counter()
        timings_baseline.append((t1 - t0) / 10 * 1000)

        # Legacy (decode + numpy heuristic for every BUY) - skip if L very large
        if L <= 200:
            t0 = time.perf_counter()
            for _ in range(2):
                packing_ordering_legacy(state, buy_actions, priors, action_to_idx)
            t1 = time.perf_counter()
            timings_legacy.append((t1 - t0) / 2 * 1000)
        else:
            timings_legacy.append(np.nan)

        # Packing optimized (Python batch when measuring; Cython when available for main timing)
        M = M_formula(L)
        # Time Python batch path (for comparison when Cython available)
        t0 = time.perf_counter()
        for _ in range(10):
            packing_ordering(state, buy_actions, priors, action_to_idx, M, use_cython=False)
        t1 = time.perf_counter()
        timings_packing.append((t1 - t0) / 10 * 1000)
        M_used.append(min(L, M))

        # Time Cython path when available
        if _CYTHON_AVAILABLE:
            t0 = time.perf_counter()
            for _ in range(10):
                packing_ordering(state, buy_actions, priors, action_to_idx, M, use_cython=True)
            t1 = time.perf_counter()
            timings_cython.append((t1 - t0) / 10 * 1000)

        # One run with timings breakdown (uses Cython if available, buffer path by default)
        _, _, tt = packing_ordering(state, buy_actions, priors, action_to_idx, M, return_timings=True)
        timings_decode.append(tt["decode_ms"])
        timings_sort_prior.append(tt["sort_prior_ms"])
        timings_heuristic.append(tt["heuristic_ms"])
        timings_sort_rank.append(tt["sort_rank_ms"])
        if _CYTHON_AVAILABLE:
            _, _, tt_cy = packing_ordering(state, buy_actions, priors, action_to_idx, M, return_timings=True, use_cython=True)
            timings_heuristic_cython.append(tt_cy["heuristic_ms"])
            # Heuristic time when passing list (includes list→array allocation inside)
            _, _, tt_list = packing_ordering(state, buy_actions, priors, action_to_idx, M, return_timings=True, use_cython=True, use_indices_buffer=False)
            timings_heuristic_list_path.append(tt_list["heuristic_ms"])
            # Allocation only: list build + np.array(indices)
            pl = current_player_fast(state)
            if pl == 0:
                o0, o1, o2 = int(state[P0_OCC0]), int(state[P0_OCC1]), int(state[P0_OCC2])
            else:
                o0, o1, o2 = int(state[P1_OCC0]), int(state[P1_OCC1]), int(state[P1_OCC2])
            with_prior = [(a, float(priors[action_to_idx[a]])) for a in buy_actions]
            with_prior.sort(key=lambda x: -x[1])
            t0 = time.perf_counter()
            list_indices = [cache_index(int(a[2]), int(a[3]), int(a[4]) * 9 + int(a[5])) for a, _ in with_prior[:M]]
            _ = np.array(list_indices, dtype=np.int32)
            t1 = time.perf_counter()
            timings_alloc_list.append((t1 - t0) * 1000)
        if _CYTHON_DIRECT_AVAILABLE:
            pl = current_player_fast(state)
            if pl == 0:
                o0, o1, o2 = int(state[P0_OCC0]), int(state[P0_OCC1]), int(state[P0_OCC2])
            else:
                o0, o1, o2 = int(state[P1_OCC0]), int(state[P1_OCC1]), int(state[P1_OCC2])
            filled_bb = occ_words_to_bitboard_for_node(o0, o1, o2)
            filled_lo = int(filled_bb) & 0xFFFFFFFFFFFFFFFF
            filled_hi = (int(filled_bb) >> 64) & 0x1FFFF
            radius_index = min(max(0, LOCAL_RADIUS), 3)
            w_adj = float(WEIGHTS.get("adj_edges", 1.0))
            w_corner = float(WEIGHTS.get("corner_bonus", 0.5))
            w_iso = float(WEIGHTS.get("iso_hole_penalty", 2.0))
            w_front = float(WEIGHTS.get("frontier_penalty", 0.25))
            buf = np.empty(M, dtype=np.int32)
            with_prior = [(a, float(priors[action_to_idx[a]])) for a in buy_actions]
            with_prior.sort(key=lambda x: -x[1])
            for i, (a, _) in enumerate(with_prior[:M]):
                buf[i] = cache_index(int(a[2]), int(a[3]), int(a[4]) * 9 + int(a[5]))
            n_rep = 20
            t0 = time.perf_counter()
            for _ in range(n_rep):
                packing_scores_batch_cy(
                    filled_lo, filled_hi, buf, radius_index,
                    w_adj, w_corner, w_iso, w_front, 0.0,
                    PLACEMENT_LO, PLACEMENT_HI, WINDOW_LO, WINDOW_HI, _SCALE,
                )
            t1 = time.perf_counter()
            timings_cython_call_only.append((t1 - t0) / n_rep * 1000)

    timings_baseline = np.array(timings_baseline)
    timings_legacy = np.array(timings_legacy)
    timings_packing = np.array(timings_packing)
    if _CYTHON_AVAILABLE:
        timings_cython = np.array(timings_cython)
        timings_heuristic_cython = np.array(timings_heuristic_cython)
        timings_heuristic_list_path = np.array(timings_heuristic_list_path)
        timings_alloc_list = np.array(timings_alloc_list)
        heuristic_cython_avg = timings_heuristic_cython.mean()
    if _CYTHON_DIRECT_AVAILABLE:
        timings_cython_call_only = np.array(timings_cython_call_only)
    buy_counts = np.array(buy_counts)
    M_used = np.array(M_used)
    timings_decode = np.array(timings_decode)
    timings_sort_prior = np.array(timings_sort_prior)
    timings_heuristic = np.array(timings_heuristic)
    timings_sort_rank = np.array(timings_sort_rank)

    # When Cython available: timings_packing = Python batch; timings_cython = Cython batch
    total_avg = timings_packing.mean()
    decode_avg = timings_decode.mean()
    sort_prior_avg = timings_sort_prior.mean()
    heuristic_avg = timings_heuristic.mean()
    sort_rank_avg = timings_sort_rank.mean()

    print()
    print("=== Packing ordering benchmark ===")
    print(f"  Cython extension:             {'available' if _CYTHON_AVAILABLE else 'not built (using Python fallback)'}")
    print(f"  States: {n_states}  |  Avg L_buy: {buy_counts.mean():.1f}  (min={buy_counts.min()}, max={buy_counts.max()})")
    print(f"  Chosen M (K_root+16): avg {M_used.mean():.1f}  (max={M_used.max()})")
    print()
    print(f"  Baseline (prior only)         avg ms/order: {timings_baseline.mean():.3f}")
    legacy_mean = np.nanmean(timings_legacy)
    if not np.isnan(legacy_mean):
        print(f"  Legacy (decode+numpy per BUY) avg ms/order: {legacy_mean:.3f}")
    print(f"  Optimized (Python batch)      avg ms/order: {total_avg:.3f}")
    if _CYTHON_AVAILABLE:
        print(f"  Optimized (Cython batch)      avg ms/order: {timings_cython.mean():.3f}")
        print(f"  Speedup Cython vs Python:     {total_avg / max(timings_cython.mean(), 1e-6):.1f}x")
    if not np.isnan(legacy_mean):
        print(f"  Speedup vs legacy:             {legacy_mean / max(total_avg, 1e-6):.1f}x")
    print()
    print("  Time breakdown (optimized):")
    print(f"    decode (occ->bitboard)     {decode_avg:.3f} ms  ({100*decode_avg/max(total_avg,1e-6):.0f}%)")
    print(f"    sort by prior              {sort_prior_avg:.3f} ms  ({100*sort_prior_avg/max(total_avg,1e-6):.0f}%)")
    print(f"    heuristic (batch)          {heuristic_avg:.3f} ms  ({100*heuristic_avg/max(total_avg,1e-6):.0f}%)")
    print(f"    sort by rank_key           {sort_rank_avg:.3f} ms  ({100*sort_rank_avg/max(total_avg,1e-6):.0f}%)")
    if _CYTHON_AVAILABLE:
        print()
        print("  Time breakdown (Cython):")
        print(f"    heuristic (Cython batch)    {heuristic_cython_avg:.3f} ms  ({100*heuristic_cython_avg/max(timings_cython.mean(),1e-6):.0f}%)")
    # Allocation vs Cython call (optimized path)
    CYTHON_BEFORE_MS = 0.088  # reference: previous Cython ~0.088 ms/order heuristic
    if _CYTHON_AVAILABLE:
        print()
        print("  Cython optimizations (fast popcount, no per-call indices alloc, skip zero weights):")
        print(f"    Cython before (reference)   ~{CYTHON_BEFORE_MS:.3f} ms/order (heuristic)")
        print(f"    Cython after (optimized)    {heuristic_cython_avg:.3f} ms/order (heuristic)")
        if heuristic_cython_avg > 1e-6:
            print(f"    Speedup vs previous Cython  {CYTHON_BEFORE_MS / heuristic_cython_avg:.2f}x")
        print("  Allocation vs Cython call (coarse timing):")
        print(f"    Allocation (list + np.array)  avg {timings_alloc_list.mean():.3f} ms")
        print(f"    Heuristic total (list path)   avg {timings_heuristic_list_path.mean():.3f} ms")
        print(f"    Heuristic total (buffer path) avg {heuristic_cython_avg:.3f} ms")
    if _CYTHON_DIRECT_AVAILABLE:
        cython_only_avg = timings_cython_call_only.mean()
        print(f"    Cython call only              avg {cython_only_avg:.3f} ms")
        if _CYTHON_AVAILABLE and timings_alloc_list is not None:
            print(f"    -> allocation share (list path) ~{100 * timings_alloc_list.mean() / max(timings_heuristic_list_path.mean(), 1e-6):.0f}%")
    print()
    print("  Target: ~0.05 ms/order or better. Build Cython: python setup.py build_ext --inplace")


if __name__ == "__main__":
    main()
