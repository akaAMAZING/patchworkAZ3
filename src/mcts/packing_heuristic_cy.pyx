# cython: language_level=3
# Cython extension for packing heuristic batch scoring (81-bit bitboard as two uint64).

import numpy as np
cimport numpy as cnp
cimport cython

cnp.import_array()

# Fast popcount: CPU intrinsics when available, fallback to bit-clearing loop.
cdef extern from *:
    """
    #if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_AMD64) || defined(_M_ARM64))
    #include <intrin.h>
    static inline int popcount64_fast(unsigned long long x) {
        return (int)__popcnt64(x);
    }
    #elif defined(__GNUC__) || defined(__clang__)
    static inline int popcount64_fast(unsigned long long x) {
        return (int)__builtin_popcountll(x);
    }
    #else
    static inline int popcount64_fast(unsigned long long x) {
        int n = 0;
        while (x) { n++; x &= x - 1; }
        return n;
    }
    #endif
    """
    int popcount64_fast(unsigned long long x) nogil

cdef inline int popcount64(unsigned long long x) nogil:
    return popcount64_fast(x)

# 81-bit board: lo = bits 0-63, hi = bits 64-80 (only 17 bits used)
cdef unsigned long long MASK64 = 0xFFFFFFFFFFFFFFFF
cdef unsigned long long MASK_HI = 0x1FFFF  # 17 bits

# Column masks (9x9: cell (r,c) = bit r*9+c)
cdef unsigned long long COL0_LO = 0x8040201008040201  # bits 0,9,18,27,36,45,54,63
cdef unsigned long long COL0_HI = 0x100               # bit 72 (index 8 of hi)
cdef unsigned long long COL8_LO = 0x0102040810204080  # bits 8,17,26,35,44,53,62
cdef unsigned long long COL8_HI = 0x10080            # bits 71,80 -> hi bits 7,16

cdef unsigned long long FULL_LO = 0xFFFFFFFFFFFFFFFF
cdef unsigned long long FULL_HI = 0x1FFFF

# Corner mask only: (0,0)=bit0, (0,8)=bit8, (8,0)=bit72, (8,8)=bit80. Bonus only for touching a corner cell.
cdef unsigned long long CORNER_LO = 0x101   # bits 0 and 8
cdef unsigned long long CORNER_HI = 0x10100  # bits 72 and 80 in hi (indices 8 and 16)


cdef inline int popcount81(unsigned long long lo, unsigned long long hi) nogil:
    return popcount64(lo) + popcount64(hi & MASK_HI)


cdef inline void shift_up(unsigned long long lo, unsigned long long hi,
                         unsigned long long *out_lo, unsigned long long *out_hi) nogil:
    """Neighbors above: bb >> 9."""
    out_hi[0] = hi >> 9
    out_lo[0] = (lo >> 9) | ((hi & 0x1FF) << 55)


cdef inline void shift_down(unsigned long long lo, unsigned long long hi,
                           unsigned long long *out_lo, unsigned long long *out_hi) nogil:
    """Neighbors below: (bb << 9) & FULL."""
    out_lo[0] = (lo << 9) & MASK64
    out_hi[0] = ((hi << 9) | (lo >> 55)) & MASK_HI


cdef inline void shift_left(unsigned long long lo, unsigned long long hi,
                           unsigned long long *out_lo, unsigned long long *out_hi) nogil:
    """Neighbors left: (bb & ~COL0) >> 1."""
    out_lo[0] = ((lo & ~COL0_LO) >> 1) | ((hi & 1) << 63)
    out_hi[0] = (hi & ~COL0_HI) >> 1


cdef inline void shift_right(unsigned long long lo, unsigned long long hi,
                            unsigned long long *out_lo, unsigned long long *out_hi) nogil:
    """Neighbors right: (bb & ~COL8) << 1."""
    out_lo[0] = ((lo & ~COL8_LO) << 1) & MASK64
    out_hi[0] = ((hi & ~COL8_HI) << 1 | (lo >> 63)) & MASK_HI


cdef inline void neigh(unsigned long long lo, unsigned long long hi,
                       unsigned long long *n_lo, unsigned long long *n_hi) nogil:
    cdef unsigned long long u_lo, u_hi, d_lo, d_hi, l_lo, l_hi, r_lo, r_hi
    shift_up(lo, hi, &u_lo, &u_hi)
    shift_down(lo, hi, &d_lo, &d_hi)
    shift_left(lo, hi, &l_lo, &l_hi)
    shift_right(lo, hi, &r_lo, &r_hi)
    n_lo[0] = u_lo | d_lo | l_lo | r_lo
    n_hi[0] = u_hi | d_hi | l_hi | r_hi


cdef inline double score_one(
    unsigned long long filled_lo, unsigned long long filled_hi,
    unsigned long long placed_lo, unsigned long long placed_hi,
    unsigned long long window_lo, unsigned long long window_hi,
    double w_adj, double w_corner, double w_iso, double w_front, double w_area,
    double scale,
) nogil:
    cdef unsigned long long filled_after_lo, filled_after_hi
    cdef unsigned long long empty_lo, empty_hi
    cdef unsigned long long placed_up_lo, placed_up_hi, placed_down_lo, placed_down_hi
    cdef unsigned long long placed_left_lo, placed_left_hi, placed_right_lo, placed_right_hi
    cdef unsigned long long empty_neigh_lo, empty_neigh_hi
    cdef unsigned long long filled_neigh_lo, filled_neigh_hi
    cdef unsigned long long iso_lo, iso_hi, frontier_lo, frontier_hi
    cdef int adj_edges, iso_count, frontier_count, area
    cdef double s

    if placed_lo == 0 and placed_hi == 0:
        return 0.0

    filled_after_lo = filled_lo | placed_lo
    filled_after_hi = (filled_hi | placed_hi) & MASK_HI
    empty_lo = (FULL_LO ^ filled_after_lo) & MASK64
    empty_hi = (FULL_HI ^ filled_after_hi) & MASK_HI

    # Adjacency: placed touches filled_before
    shift_up(placed_lo, placed_hi, &placed_up_lo, &placed_up_hi)
    shift_down(placed_lo, placed_hi, &placed_down_lo, &placed_down_hi)
    shift_left(placed_lo, placed_hi, &placed_left_lo, &placed_left_hi)
    shift_right(placed_lo, placed_hi, &placed_right_lo, &placed_right_hi)
    adj_edges = (
        popcount81(filled_lo & placed_up_lo, filled_hi & placed_up_hi)
        + popcount81(filled_lo & placed_down_lo, filled_hi & placed_down_hi)
        + popcount81(filled_lo & placed_left_lo, filled_hi & placed_left_hi)
        + popcount81(filled_lo & placed_right_lo, filled_hi & placed_right_hi)
    )
    s = w_adj * <double>adj_edges

    # Corner bonus only (skip if weight is zero): placement touches one of the four corner cells
    if w_corner != 0.0 and ((placed_lo & CORNER_LO) != 0 or (placed_hi & CORNER_HI) != 0):
        s += w_corner

    # Iso and frontier (skip entirely when both weights are zero)
    if w_iso != 0.0 or w_front != 0.0:
        neigh(empty_lo, empty_hi, &empty_neigh_lo, &empty_neigh_hi)
        iso_lo = empty_lo & ~empty_neigh_lo
        iso_hi = empty_hi & ~empty_neigh_hi
        neigh(filled_after_lo, filled_after_hi, &filled_neigh_lo, &filled_neigh_hi)
        frontier_lo = empty_lo & filled_neigh_lo
        frontier_hi = empty_hi & filled_neigh_hi
        iso_count = popcount81(iso_lo & window_lo, iso_hi & window_hi)
        frontier_count = popcount81(frontier_lo & window_lo, frontier_hi & window_hi)
        s -= w_iso * <double>iso_count
        s -= w_front * <double>frontier_count

    # Area bonus: +w_area * (number of cells in placed piece); prefer larger pieces when w_area > 0
    if w_area != 0.0:
        area = popcount81(placed_lo, placed_hi)
        s += w_area * <double>area

    return s / scale


@cython.boundscheck(False)
@cython.wraparound(False)
def packing_scores_batch_cy(
    unsigned long long filled_lo,
    unsigned long long filled_hi,
    cnp.ndarray[cnp.int32_t, ndim=1] indices,
    int radius_index,
    double w_adj,
    double w_corner,
    double w_iso,
    double w_front,
    double w_area,
    cnp.ndarray[cnp.uint64_t, ndim=1] placement_lo,
    cnp.ndarray[cnp.uint64_t, ndim=1] placement_hi,
    cnp.ndarray[cnp.uint64_t, ndim=2] window_lo,
    cnp.ndarray[cnp.uint64_t, ndim=2] window_hi,
    double scale,
):
    """
    Compute packing scores for a batch of placement indices. Returns float64 array.
    """
    cdef Py_ssize_t n = indices.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] out = np.empty(n, dtype=np.float64)
    cdef Py_ssize_t i
    cdef int idx
    cdef unsigned long long plo, phi, wlo, whi

    if radius_index < 0:
        radius_index = 0
    elif radius_index > 3:
        radius_index = 3

    for i in range(n):
        idx = indices[i]
        plo = placement_lo[idx]
        phi = placement_hi[idx]
        wlo = window_lo[radius_index, idx]
        whi = window_hi[radius_index, idx]
        out[i] = score_one(
            filled_lo, filled_hi, plo, phi, wlo, whi,
            w_adj, w_corner, w_iso, w_front, w_area, scale,
        )
    return out
