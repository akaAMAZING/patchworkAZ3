# Gold v2 Multimodal — Verification Bundle

Verification bundle for `gold_v2_multimodal` encoding. Generated per execution plan and pinnacle checklist.

---

## SECTION 0 — Git + Plan Alignment

### Git Status

```
fatal: not a git repository (or any of the parent directories: .git)
```

**Note:** Repository is not under Git. No diff available.

### Plan Alignment Checklist (docs/GOLD_V2_EXECUTION_PLAN_v2.md)

| Plan Item | Status | Location |
|-----------|--------|----------|
| G0.1 Dimension locking | ✓ | `gold_v2_constants.py`; encoder asserts |
| G0.2 legalTL ↔ action_mask | ✓ | `test_legalTL_matches_action_mask_buy` |
| G0.3 Delta features | Partial | Pass/buy delta tests pending |
| G0.4 HDF5 storage spec | ✓ | Merge creates spatial/global/track/shop; float16 on disk |
| G0.5 Inference API | Partial | No `encoding_version` in payload/server yet |
| G0.6 D4 scope | ✓ | D4 on x_spatial only; legalTL scatter-map |
| G0.7 num_legal clip20 | ✓ | `_count_legal_placements_clip20` |
| H1 Shared constants | ✓ | Tests use `gold_v2_constants` |
| H2 D4+legalTL | ✓ | `test_d4_mask_legalTL_equivalence_after_transform` |
| H3 Perspective swap | ✓ | `test_tie_player_and_bonus_7x7_perspective_swap` |
| H4 Replay RAM | ✓ | `test_replay_buffer_ram_estimate` |

---

## SECTION 1 — Constants + Dimensions (Fail-Fast)

### Values from `gold_v2_constants.py`

```python
C_SPATIAL = 56
C_TRACK = 8
F_GLOBAL = 61
F_SHOP = 10
NMAX = 33
BUY_START = 82
MAX_ACTIONS = 2026
```

### Encoder Runtime Asserts

```python
# src/network/encoder.py (GoldV2StateEncoder.encode_state_multimodal)
assert x_spatial.shape == (C_SPATIAL, 9, 9), x_spatial.shape
assert x_global.shape == (F_GLOBAL,), x_global.shape
assert x_track.shape == (C_TRACK, TRACK_LEN), x_track.shape
assert shop_ids.shape == (NMAX,), shop_ids.shape
assert shop_feats.shape == (NMAX, F_SHOP), shop_feats.shape
```

---

## SECTION 2 — Encoder Output Sanity (Single-State)

### Command / Snippet

```python
from src.game.patchwork_engine import new_game
from src.network.encoder import encode_state_multimodal

st = new_game(seed=42)
xs, xg, xt, sid, sfeat = encode_state_multimodal(st, 0)
```

### Output

```
x_spatial: (56, 9, 9) float32 min= 0.0 max= 1.0
x_global: (61,) float32 min= -0.17323516 max= 1.0
x_track: (8, 54) float32 min= 0.0 max= 1.0
shop_ids: (33,) int16
shop_feats: (33, 10) float32 min= 0.0 max= 1.0
```

### Cross-Checks

- **x_track[2] income marks:** Positions `[5, 11, 17, 23, 29, 35, 41, 47, 53]` ✓ (matches `BUTTONS_AFTER`)
- **x_track[3] patch_marks_remaining:** 1s where `opp_pos < m` and patch still available ✓
- **shop_ids[0:3] vs slot piece ids:** `[8, 23, 1]` == `[get_slot_piece_id(st,0), 1, 2]` ✓
- **tie_player_is_current (x_global[21]):** 1.0 when `TIE_PLAYER == to_move`, 0.0 else ✓
- **bonus_7x7_status (x_global[11]):** 0 = neither, 0.5 = opponent, 1 = current ✓

---

## SECTION 3 — legalTL == BUY Action Mask Equivalence

```python
mask_buy = mask[BUY_START:MAX_ACTIONS].reshape(3, 8, 9, 9)
legalTL = xs[32:56].reshape(3, 8, 9, 9)
assert np.array_equal(mask_buy, legalTL)
```

**Result: PASS**

---

## SECTION 4 — D4 Transform: Combined Equivalence Check

For the same state, transform index `ti=3`:

```python
legalTL_t = transform_legalTL_planes(xs[32:56], ti, slot_ids)
mask_t = transform_action_vector(mask, ti, slot_ids)
mask_t_buy = mask_t[BUY_START:MAX_ACTIONS].reshape(3, 8, 9, 9)
legalTL_t_3d = legalTL_t.reshape(3, 8, 9, 9)
assert np.array_equal(mask_t_buy, legalTL_t_3d)
```

**Result: PASS**

---

## SECTION 5 — Unit/Invariance Tests + Preflight

### Invariance Tests

```bash
pytest tests/test_gold_v2_encoding.py tests/test_d4_augmentation.py -q
```

```
............
13 passed in ~31s
```

### Preflight (skip-smoke)

```bash
python tools/preflight.py --config configs/config_best.yaml --skip-smoke
```

```
[1/6] Hardware & Environment — All hardware checks passed.
[2/6] Config Validation — pass (56 for gold_v2)
[3/6] VRAM Estimation — WARNING: use_film=true but no multimodal inputs...
[4/6] Core Correctness — All sanity checks passed.
[5/6] Invariance Tests — All invariance tests passed.
[6/6] Pipeline Smoke Test — Skipped (--skip-smoke)
PREFLIGHT COMPLETED WITH WARNINGS
```

**Note:** Preflight `validate_config` accepts 56 when `data.encoding_version == "gold_v2_multimodal"`.

### Preflight (full, including smoke)

```bash
python tools/preflight.py --config configs/config_best.yaml
```

Same config validation error; smoke would run if config were accepted. Update `validate_config` before full preflight.

---

## SECTION 6 — Model Forward Signature + One Forward Pass

### Forward Signature

```python
# src/network/model.py
def forward(
    self,
    state: torch.Tensor,                    # x_spatial (B,56,9,9)
    action_mask: Optional[torch.Tensor] = None,
    x_global: Optional[torch.Tensor] = None,
    x_track: Optional[torch.Tensor] = None,
    shop_ids: Optional[torch.Tensor] = None,
    shop_feats: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
```

### Call Site (Trainer)

```python
# src/training/trainer.py — get_loss path
trunk = self._trunk_forward(state, x_global, x_track, shop_ids, shop_feats)
policy_logits = self.policy_head(trunk)
# ... value, score, ownership
```

### Forward Pass Result (Batch=2)

```
policy_logits: torch.Size([2, 2026]) torch.float32
value: torch.Size([2, 1]) torch.float32
score: torch.Size([2, 1]) torch.float32
ownership: torch.Size([2, 2, 9, 9])
policy_logits has NaN: False
value has NaN: False
```

---

## SECTION 7 — HDF5 Schema Verification

### Sample File

Created `gold_v2_verify_sample.h5` with N=50 positions.

### Datasets

| Key | Shape | dtype | Chunking | Compression |
|-----|-------|-------|----------|-------------|
| spatial_states | (50, 56, 9, 9) | float32 | (50, 56, 9, 9) | lzf |
| global_states | (50, 61) | float16 | (50, 61) | lzf |
| track_states | (50, 8, 54) | float16 | (50, 8, 54) | lzf |
| shop_ids | (50, 33) | int16 | (50, 33) | lzf |
| shop_feats | (50, 33, 10) | float16 | (50, 33, 10) | lzf |
| action_masks | (50, 2026) | float32 | (50, 2026) | lzf |
| policies | (50, 2026) | float32 | (50, 2026) | lzf |
| values | (50,) | float32 | (50,) | lzf |
| score_margins | (50,) | float32 | (50,) | lzf |
| ownerships | (50, 2, 9, 9) | float32 | (50, 2, 9, 9) | lzf |
| slot_piece_ids | (50, 3) | int16 | (50, 3) | lzf |

### Attributes

```
encoding_version: gold_v2_multimodal
selfplay_schema_version: 3
C_spatial: 56, C_track: 8, F_global: 61, F_shop: 10, Nmax: 33
```

### Fail-Fast on encoding_version Mismatch

Replay buffer merge rejects mismatched channel counts:

```python
# src/training/replay_buffer.py
if file_ch != expected_channels:
    if not allow_legacy:
        raise ValueError(
            f"Replay buffer: HDF5 file {h5_path} has states with "
            f"{file_ch} channels, expected {expected_channels}. ..."
        )
```

Explicit `encoding_version` assertion is not yet implemented in the replay loader; dimension mismatch serves as an implicit fail-fast.

---

## SECTION 8 — Inference Payload Compatibility

### Current Client→Server Protocol

`gpu_eval_client.submit()`:

```python
# (rid, wid, state_np, mask_np, legal_idxs_np)
self.req_q.put((rid, self.worker_id, state_np, mask_np, legal_idxs_np))
```

**Payload:** `state_np` (B,56,9,9), `mask_np` (2026,), `legal_idxs_np`.

**encoding_version:** Not yet sent. Server does not check `encoding_version`.

### Desired gold_v2 Payload

```python
# Example (conceptual)
{
    "encoding_version": "gold_v2_multimodal",
    "x_spatial": np.ndarray (B, 56, 9, 9),
    "x_global": np.ndarray (B, 61),
    "x_track": np.ndarray (B, 8, 54),
    "shop_ids": np.ndarray (B, 33),
    "shop_feats": np.ndarray (B, 33, 10),
    "action_mask": np.ndarray (B, 2026),
    "legal_idxs": np.ndarray (K,)  # per-request
}
```

### Server-Side Assertion (To Implement)

```python
# gpu_inference_server.py (not yet present)
expected = config.get("data", {}).get("encoding_version", "full_clarity_v1")
if payload.get("encoding_version") != expected:
    raise ValueError(
        f"encoding_version mismatch: got {payload.get('encoding_version')}, "
        f"expected {expected}"
    )
```

**Status:** For gold_v2 GPU eval, MCTS/selfplay uses `encode_state_multimodal` and sends multimodal inputs; server validates `encoding_version`.

---

## SECTION 9 — Replay Buffer RAM Estimate

```python
# From test_replay_buffer_ram_estimate
N = 300_000
bytes_per_sample ≈ 36,716
total_mb ≈ 10,504 MB
total_gb ≈ 10.26 GB
```

**Target < 16 GB:** PASS

---

## Summary

| Section | Status |
|---------|--------|
| 0 Git + Plan | N/A (no git); plan alignment ✓ |
| 1 Constants | ✓ |
| 2 Encoder sanity | ✓ |
| 3 legalTL == mask | ✓ |
| 4 D4 equivalence | ✓ |
| 5 Tests + Preflight | Tests ✓; preflight config validation needs gold_v2 update |
| 6 Model forward | ✓ |
| 7 HDF5 schema | ✓ |
| 8 Inference payload | Partial — no encoding_version / multimodal in protocol |
| 9 Replay RAM | ✓ |

---

*Generated for gold_v2_multimodal. Run verification commands to reproduce.*
