# Fully Markov Input Channels (Gold v2 Multimodal)

The game state is **fully observable** (Markov): the network sees everything needed to reconstruct the board, track, and shop from the **current player’s perspective**. All inputs are from the encoder and the GPU legality module; no hidden state.

Encoding version: **`gold_v2_32ch`**. Config: `expected_spatial_channels: 32`, `expected_global_dim: 61`, `expected_track_shape: [8, 54]`, `expected_shop_shapes: [33, 10]`. Trunk input: **56 channels** (32 encoder + 24 legalTL computed on GPU).

---

## 1. Spatial input to the trunk: **(56, 9, 9)**

The trunk receives **56 channels** of shape 9×9 (one 9×9 board per player in Patchwork; spatial features are board-sized).

### 1.1 Channels 0–31 (encoder output, `C_SPATIAL_ENC = 32`)

Produced by **GoldV2StateEncoder** (`src/network/encoder.py`). Stored in replay and sent over IPC as **(32, 9, 9)**; at inference/training the model concatenates 24 legalTL channels to get **(56, 9, 9)**.

| Channel(s) | Name | Description |
|------------|------|-------------|
| **0** | cur_occ | Current player’s board occupancy (0/1 per cell). Decoded from 3×32-bit words (P0_OCC0/1/2 or P1_OCC0/1/2). |
| **1** | opp_occ | Opponent’s board occupancy (0/1 per cell). |
| **2** | coord_row | Row index / 8 (constant plane, 0..1). |
| **3** | coord_col | Column index / 8 (constant plane). |
| **4** | frontier_cur | Frontier cells of current player’s board (empty cells adjacent to filled). |
| **5** | frontier_opp | Frontier cells of opponent’s board. |
| **6** | valid_7x7_cur | 1 at (r,c) if the 7×7 with top-left at (r,c) is fully empty (only (0,0)..(2,2) can be 1). |
| **7** | valid_7x7_opp | Same for opponent. |
| **8–31** | slot×orient shape | 3 slots × 8 orientations = 24 planes. Channel `8 + slot*8 + orient` = shape mask (9×9, top-left anchored) for the piece in that slot and orientation; 0 if slot empty or orient invalid. |

All of 0–31 are **spatial** (9×9) and are **D4-augmented** in training (rotations/flips); the encoder always outputs from the **current player’s view** (cur = to_move, opp = opponent).

### 1.2 Channels 32–55 (legalTL, computed on GPU)

Computed by **DeterministicLegalityModule** in `src/network/model.py`. Not stored in replay; derived from **board_free** (1 − (cur_occ | opp_occ)) and the 264 piece×orientation filters.

- **264 internal map:** For each of 33 pieces × 8 orientations, a 9×9 plane = 1 at (r,c) iff placing that piece at top-left (r,c) is legal (all cells under the piece are empty and within board).
- **24 channels fed to trunk:** **3 visible slots × 8 orientations** = 24. For each visible slot, the 8 orientation planes are taken from the 264 map (by slot piece ID); then **masked by affordability** (current player’s buttons ≥ piece cost). So channels 32–55 = “legal placement for visible slot i, orientation j, and affordable”.

So:

- **32–39:** Slot 0, orientations 0–7 (legal TL placement, affordability-masked).
- **40–47:** Slot 1, orientations 0–7.
- **48–55:** Slot 2, orientations 0–7.

**Summary:** 56 = 32 (encoder spatial) + 24 (legalTL for 3 slots × 8 orients). Fully Markov: no hidden state, only current state and current player’s view.

---

## 2. Global vector: **(61,)**

**`x_global`** in `src/network/encoder.py` (`_encode_scalars_shop_jit`). All from current state; normalized scalars.

| Index | Description |
|-------|-------------|
| 0 | c_pos / 54 (current player track position) |
| 1 | o_pos / 54 (opponent) |
| 2 | (c_pos - o_pos) / 54 |
| 3–4 | log1p(c_buttons), log1p(o_buttons) normalized by log1p(200) |
| 5 | tanh((c_buttons - o_buttons) / 40) |
| 6–7 | tanh(c_income/15), tanh(o_income/15) |
| 8 | tanh((c_income - o_income) / 10) |
| 9–10 | cur_filled/81, opp_filled/81 (board fill fraction) |
| 11 | Bonus tile owner: 0 = none, 1 = to_move, 0.5 = opponent |
| 12–13 | 7×7 completion distance (cur, opp), 0–1 (0 = completed) |
| 14 | pending patches (capped 5) / 5 |
| 15 | remaining patches ahead of opponent (capped 5) / 5 |
| 16–20 | One-hot “opponent ahead of patch k” (5 patch markers) |
| 21 | 1 if tie_player == to_move else 0 |
| 22–27 | Pass action: pass_steps/54, pass_income_marks/9, pass_patches/2, tanh(pass_btn_delta/30), pass_new/54, 1 if pass_new==o_pos |
| 28–60 | Per-slot (3×11) buy features: can_afford, log1p(cost), time/6, income/3, area/9, buy_inc/9, tanh(ibg/80), buy_pat/2, tanh((-cost+ibg)/40), buy_new/54, 1 if buy_new==o_pos |

All from observable game state; no history.

---

## 3. Track tensor: **(8, 54)**

**`x_track`**: 8 channels × 54 positions (track length 0..53).

| Channel | Description |
|---------|-------------|
| 0 | Current player position one-hot (1 at c_pos) |
| 1 | Opponent position one-hot (1 at o_pos) |
| 2 | Income button markers (1 at each button position) |
| 3 | Patch markers where opponent is ahead (1 at patch position if o_pos < marker) |
| 4 | Pass landing: 1 at clamp(o_pos+1) |
| 5–7 | Buy landing for slot 0, 1, 2: 1 at c_pos + piece_time for that slot’s piece |

Used by FiLM as `film_track` (e.g. Conv1d over 8×54). Fully Markov: only current positions and markers.

---

## 4. Shop: **shop_ids (33,)**, **shop_feats (33, 10)**

- **shop_ids:** int16, piece_id for each of up to 33 circle positions; -1 if unused.
- **shop_feats (33, 10):** Per-item features: position in circle (0..1), log1p(cost), time/6, income/3, area/9, n_orient/8, can_afford_cur, can_afford_opp. Indices 8–9 (e.g. num_legal placements) can be filled on GPU.

Shop order is from neutral+1 clockwise. All from current state.

---

## 5. How it’s used in the model

- **Trunk:** `state` = **(B, 56, 9, 9)** (32 encoder + 24 legalTL). Optional **DeterministicLegalityModule** builds 24 from 32-channel encoder output (board_free) and overwrites/writes channels 32–55.
- **FiLM:** Conditions trunk with **x_global (61)**, **x_track (8×54)** (e.g. flattened or Conv1d), **shop_feats (33×10)** (or pooled). Config: `film_global_dim: 61`, `film_track_dim: 432` (= 8×54), `film_shop_dim: 128`.
- **Policy/value/ownership:** Consume trunk features (and optionally global inject). No extra hidden state.

So the entire input is **Markov**: only the current state and current player’s perspective; no previous moves or private information.

---

## 6. Source-of-truth constants (`src/network/gold_v2_constants.py`)

```text
C_SPATIAL     = 56   # Trunk input: 32 + 24 legalTL
C_SPATIAL_ENC = 32   # Encoder/replay/IPC spatial channels
C_TRACK       = 8
F_GLOBAL      = 61
F_SHOP        = 10
NMAX          = 33
TRACK_LEN     = 54
BOARD_CHANNELS    = [0, 1]
COORD_CHANNELS   = [2, 3]
FRONTIER_CHANNELS = [4, 5]
VALID_7X7_CHANNELS = [6, 7]
SLOT_ORIENT_SHAPE_BASE = 8   # 8..31
SLOT_ORIENT_LEGAL_BASE = 32  # 32..55 (24 ch)
```

Policy indexing (PASS_INDEX, PATCH_START, BUY_START, MAX_ACTIONS=2026) is also in that file and must match the encoder and model.

---

## 7. Summary table

| Input | Shape | Markov? | Description |
|-------|--------|--------|-------------|
| x_spatial (encoder) | (32, 9, 9) | Yes | Occupancy, coords, frontier, valid_7x7, slot×orient shapes |
| legalTL (GPU) | (24, 9, 9) | Yes | Legal TL placement for 3 slots × 8 orients, affordability-masked |
| **Trunk input** | **(56, 9, 9)** | **Yes** | 32 + 24 concatenated |
| x_global | (61,) | Yes | Position, buttons, income, fill, bonus, 7×7, pass/buy features |
| x_track | (8, 54) | Yes | Position and event markers on time track |
| shop_ids | (33,) | Yes | Piece IDs in circle order |
| shop_feats | (33, 10) | Yes | Per-item cost, time, income, area, affordability |

Everything is derived from the **current state** and **current player’s perspective**; there is no history or hidden state, so the input is fully Markov.
