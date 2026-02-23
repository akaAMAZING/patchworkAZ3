# Invariant Verification Report

This document records the verification of two critical invariants in the Patchwork AlphaZero codebase: (1) shop order encoding alignment with engine semantics, and (2) D4 augmentation consistency for spatial planes, orientation channels, and policy/action indices.

---

## Part A: Shop Order Encoding

### Pawn / Shop Indexing Convention

**Authoritative engine logic:** [src/game/patchwork_engine.py](src/game/patchwork_engine.py)

- **Neutral pawn:** `state[NEUTRAL]` = index of the 1×1 neutral pawn (piece_id=32) in the circle array.
- **Visible slots:** For circle length `n`, the three buyable slots are:
  - `slot0` = `circle[(neutral + 1) % n]`
  - `slot1` = `circle[(neutral + 2) % n]`
  - `slot2` = `circle[(neutral + 3) % n]`
- **Pawn semantics:** The pawn index points to **"piece before slot0"** — i.e., the neutral piece itself. The three visible slots are the next three pieces clockwise after the pawn.

**After buy at offset `k` (1, 2, or 3):**

- `idx = (neutral + offset) % n` — index of the bought piece
- Piece is removed from the circle
- `new_neutral = (idx - 1) % (n - 1)` — pawn moves to the position before the removed piece (or wraps)

**Example:** Circle = [A, B, C, D, E], neutral = 0 (pawn at A). Slots: slot0=B, slot1=C, slot2=D. Buy slot1 (C): idx=2, remove C, new circle = [A, B, D, E], new_neutral = (2-1) % 4 = 1 (pawn at B).

### Alignment Property

`remaining_after_pawn` = piece IDs in clockwise order starting from the cell after the pawn (`(neutral+1)%n` onward).

**Invariant:** `remaining_after_pawn[0:3] == [slot0_id, slot1_id, slot2_id]`

### Verification

- **Helper:** [src/network/shop_debug.py](src/network/shop_debug.py) — `debug_dump_shop_state()`, `get_remaining_after_pawn()`, `assert_shop_order_alignment()`
- **Tests:** [tests/test_shop_order_markov_alignment.py](tests/test_shop_order_markov_alignment.py) — all 7 tests pass.

---

## Part B: D4 Augmentation

### Transform Order

**Transform indices:** 0=id, 1=r90, 2=r180, 3=r270, 4=m, 5=m_r90, 6=m_r180, 7=m_r270
**Source:** [src/network/d4_augmentation.py](src/network/d4_augmentation.py)

- **Mirror:** horizontal flip `(r, c) -> (r, 8-c)`
- **Composition:** m_r90 = mirror then r90; m_r180 = mirror then r180; m_r270 = mirror then r270

### Position Transforms

| ti | Name | Formula |
|----|------|---------|
| 0 | id | (r, c) |
| 1 | r90 | (c, 8-r) |
| 2 | r180 | (8-r, 8-c) |
| 3 | r270 | (8-c, r) |
| 4 | m | (r, 8-c) |
| 5 | m_r90 | (8-c, 8-r) |
| 6 | m_r180 | (8-r, c) |
| 7 | m_r270 | (c, r) |

### Orient Remap

- **Function:** `get_orient_after_transform(piece_id, transform_idx, orient)` in [src/network/d4_augmentation.py](src/network/d4_augmentation.py)
- **Logic:** Uses `_ORIENT_MAPS[piece_id][transform_idx]` — built by **exact mask matching**: for each orient o, `transform_mask(mask[p][o])` is compared to `mask[p][o2]` for all o2; o2 is chosen deterministically (min of candidates) with a bijection constraint so every output has a preimage.
- **Status:** Buy equivariance is now verified. The previous orient remap bug (canonical-cell greedy assignment) has been resolved by switching to mask-based matching.

### Buy Position Transform (Dimension-Aware)

For non-1×1 pieces, the simple `(r,c) -> T(r,c)` is incorrect. The correct new top-left is obtained by either (see Implementation below):

- **Bulletproof** (training): From mask at `(top, left)`, transform occupied cells; min row/col = new top-left.
- **Corners** (engine tests): Min of four transformed corners of bounding box.

**Implementation:**
- **Training:** `transform_buy_top_left()` in [src/network/d4_augmentation.py](src/network/d4_augmentation.py)
- **Engine equivariance tests:** `_transform_buy_position()` in [tests/test_d4_action_equivariance.py](tests/test_d4_action_equivariance.py)

### Action Transform

- **Patch:** `(r, c) -> transform_position(r, c, ti)` (single cell)
- **Buy:** `orient_new = get_orient_after_transform(pid, ti, orient)`; `(top_new, left_new) = transform_buy_top_left(pid, orient, top, left, ti)` (dimension-aware, mask-based)

### Test Status

| Test | Status |
|------|--------|
| `test_shop_order_*` | All pass |
| `test_d4_action_equivariance_patch` | Pass |
| `test_d4_action_equivariance_buy_legal_in_transformed` | **Pass** — orient remap fixed (mask-based) |
| `test_d4_action_equivariance_many_states` | **Pass** |
| `test_legalTL_transform_consistency` | **Pass** — scatter-map legalTL transform (dimension-aware) |
| `test_transform_action_vector_buy_matches_engine_dimension_aware` | **Pass** — policy remap matches engine equivariance |

---

## Part C: Where to Plug in gold_v2_multimodal

1. **Shop encoding:** Use `get_remaining_after_pawn()` / `remaining_after_pawn[0:3]` for slot piece IDs. The encoder’s `get_slot_piece_id()` is aligned with engine semantics (verified by shop tests).

2. **D4 augmentation:** Apply only to spatial planes (0–27 in full_clarity; 0–55 in gold_v2 spatial). Use `transform_state()` and `transform_action_vector()` from [src/network/d4_augmentation.py](src/network/d4_augmentation.py). Buy equivariance uses mask-based orient remap and dimension-aware `transform_buy_top_left()` for policy/action remapping.

3. **Action encoding:** Use the same policy layout (PASS, PATCH, BUY slot×orient×81). The D4 module’s `transform_action_vector()` uses the dimension-aware `transform_buy_top_left()` for BUY actions, matching engine equivariance.

---

## Recommendations

1. **D4 orient remap:** Resolved. The orient map now uses exact mask matching with bijection constraint. Buy equivariance tests pass.
2. **Training pipeline:** D4 augmentation uses dimension-aware `transform_buy_top_left()` for buy policy targets, matching engine-level equivariance.
3. **gold_v2 integration:** Proceed with gold_v2_multimodal using the verified shop convention. For D4, rely on the existing `transform_state` / `transform_action_vector`; buy equivariance is now verified.
