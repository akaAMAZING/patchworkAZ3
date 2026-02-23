# Gold v2 Multimodal — Execution Plan

Execution plan for implementing gold_v2_multimodal. **Gotcha checks** are first-class tasks, not afterthoughts. Reference: [gold_v2_multimodal_encoding_6d99e840.plan.md](.cursor/plans/gold_v2_multimodal_encoding_6d99e840.plan.md)

---

## Phase 0: Gotcha Checks (First-Class Tasks)

### G0.1 Dimension Locking

**Task:** Hard-assert and persist dimension constants; validate on load.

- Define in a single module (e.g. `src/network/gold_v2_constants.py`):
  - `C_SPATIAL = 56`
  - `C_TRACK = 8`
  - `F_GLOBAL = 61`
  - `F_SHOP = 10`
  - `NMAX = 33` (from `patchwork_engine.MAX_CIRCLE`)
  - `BUY_START = 82`, `MAX_ACTIONS = 2026` (policy indexing; shared with tests)
- **Encoder:** Assert output shapes against these before returning.
- **HDF5:** Store as attributes: `C_spatial`, `C_track`, `F_global`, `F_shop`, `Nmax`.
- **On load:** Validate `f.attrs["C_spatial"] == 56` etc.; raise on mismatch.
- **Config:** `data.expected_*` must match; preflight validates.

**Files:** `src/network/encoder.py`, `src/network/gold_v2_constants.py` (new), `src/training/selfplay_optimized_integration.py`, `src/training/replay_buffer.py`, `src/training/trainer.py`, `configs/config_best.yaml`, `tools/preflight.py`, `tools/full_integrity_check.py`

---

### G0.2 legalTL ↔ action_mask Equivalence

**Task:** Unit test that BUY portion of action_mask reshaped to (3,8,9,9) equals spatial legalTL (32–55).

- **Test:** `test_legalTL_matches_action_mask_buy()` in `tests/test_gold_v2_encoding.py`
- For random reachable states:
  - Encode state with `encode_state_multimodal()`
  - Obtain legal actions; build action_mask (2026) from legal indices
  - Reshape BUY portion: `mask[BUY_START:MAX_ACTIONS].reshape(3, 8, 9, 9)` (use shared constants)
  - Assert `np.array_equal(mask_buy, x_spatial[32:56])`
- Run across several states and D4-augmented views (legalTL is not augmented at encode time; mask is; compare canonical only).

**File:** `tests/test_gold_v2_encoding.py`

---

### G0.3 Delta Feature Correctness

**Task:** Tests verifying pass and buy deltas match engine step results.

**Pass deltas:**
- `test_pass_deltas_match_engine()`: For a state where pass is legal, compute `apply_action(state, PASS)`. Compare engine `pos`, `buttons`, `patches_crossed` etc. with encoder pass deltas in x_global (indices 22–27).

**Buy slot deltas (placement-independent):**
- `test_buy_deltas_match_engine()`: For each slot 0..2, compute engine values for `cost`, `time`, `income_gain`, `area`, `income_crossings`, `patch_tokens_gained`, etc. Compare with encoder buy deltas (indices 28–60). Use states where buy is legal; iterate placements and verify placement-independent quantities are identical across legal placements.

**Files:** `tests/test_gold_v2_encoding.py`, `src/network/encoder.py`

---

### G0.4 HDF5 Storage Spec

**Task:** Explicit dtype/chunking/compression for each new dataset; consider float16-on-disk.

| Dataset         | Shape        | dtype (disk) | Chunking         | Compression |
|-----------------|-------------|--------------|------------------|-------------|
| spatial_states  | (N, 56, 9, 9) | float32     | (256, 56, 9, 9)  | lzf         |
| global_states   | (N, 61)       | float16*    | (512, 61)        | lzf         |
| track_states    | (N, 8, 54)    | float16*    | (512, 8, 54)     | lzf         |
| shop_ids        | (N, 33)       | int16       | (512, 33)        | lzf         |
| shop_feats      | (N, 33, 10)   | float16*    | (256, 33, 10)    | lzf         |
| action_masks    | (N, 2026)     | float32     | (256, 2026)      | lzf         |
| policies        | (N, 2026)     | float32     | (256, 2026)      | lzf         |
| values          | (N,)          | float32     | (1024,)          | lzf         |
| score_margins   | (N,)          | float32     | (1024,)          | lzf         |
| ownerships      | (N, 2, 9, 9)  | float32     | (256, 2, 9, 9)   | lzf         |
| slot_piece_ids  | (N, 3)        | int16       | (512, 3)         | lzf         |

\* float16-on-disk: Dataset stores float16; Dataset `__getitem__` casts to float32 before D4/training. Reduces I/O.

**Files:** `src/training/selfplay_optimized_integration.py`, `src/training/replay_buffer.py`, `src/training/trainer.py`

---

### G0.5 Inference API Compatibility

**Task:** encoding_version in payload; server-side fail-fast; policy indexing unchanged.

- **Payload:** Add `encoding_version: "gold_v2_multimodal"` to every submit/request.
- **Server:** Assert `payload.get("encoding_version") == expected` (from config); raise clear error on mismatch.
- **Policy indexing:** max_actions=2026; PASS=0, PATCH=1..81, BUY=82..2025. Unchanged.
- **Backward compat:** If `encoding_version` missing or `"full_clarity_v1"`, accept legacy spatial-only and route to legacy model path.

**Files:** `src/network/gpu_inference_server.py`, `src/mcts/gpu_eval_client.py`, `src/mcts/alphazero_mcts_optimized.py`

---

### G0.6 D4 Augmentation Scope

**Task:** Confirm D4 is applied ONLY to x_spatial and policy/action_mask; legalTL via scatter-map.

- **Applied to:** `x_spatial` (all 56 channels), `policies`, `action_masks`, `ownerships` (spatial).
- **NOT applied to:** `x_global`, `x_track`, `shop_ids`, `shop_feats`.
- **legalTL (32–55):** Transform via `transform_legalTL_planes()` — scatter-map, dimension-aware. Do NOT rotate plane images. Same mapping as buy action space.

**Verification:** Add `test_d4_scope_gold_v2()`: Apply D4; assert x_global/track/shop_* unchanged; assert x_spatial and mask transformed.

**Files:** `src/network/d4_augmentation.py`, `src/training/trainer.py`, `tests/test_d4_augmentation.py`

---

### G0.7 num_legal_* Performance Plan

**Task:** Early-exit at 20, reuse precomputed masks, optional caching.

- **Early-exit:** Stop counting at 20; clip `num_legal_clip20_norm = min(count, 20) / 20`.
- **Precomputed masks:** Reuse `MASK_W0/W1/W2` for overlap checks; no per-call shape generation.
- **Optional cache:** Per (piece_id, orient, board_hash) cache legal positions — only if profiling shows bottleneck.
- **Test:** `test_num_legal_accurate_up_to_20()` — verify clip20 matches min(actual, 20) for sample states.

**File:** `src/network/encoder.py`

---

## Phase 0.5: Hardening Additions

### H1. Shared Policy Constants in Tests

**Task:** Replace hardcoded `BUY_START=82` and `2026` in tests with shared constants.

- Import `BUY_START`, `MAX_ACTIONS` from `src/network/d4_augmentation` or `gold_v2_constants`.
- Update: `test_d4_augmentation.py`, `test_d4_action_equivariance.py`, `test_gold_v2_encoding.py`, `test_structured_policy_head.py`, and any other tests using 82/2026 literals for policy indexing.

**Files:** `tests/test_d4_augmentation.py`, `tests/test_d4_action_equivariance.py`, `tests/test_gold_v2_encoding.py`, `tests/test_structured_policy_head.py`, etc.

---

### H2. Combined D4 + legalTL Test

**Task:** After encoding canonical state, apply one random D4 transform to both mask and legalTL; assert equality holds.

- **Test:** `test_d4_mask_legalTL_equivalence_after_transform()`
- Encode canonical state → x_spatial, action_mask
- Pick random `ti` in 1..7
- Transform: `x_spatial_t = transform_state(...)`, `legalTL_t = transform_legalTL_planes(x_spatial[32:56], ti, slot_piece_ids)`, `mask_t = transform_action_vector(mask, ti, slot_piece_ids)`
- Reshape BUY portion of `mask_t` to (3,8,9,9)
- Assert `np.array_equal(mask_t_buy, legalTL_t)` — transformed mask BUY == transformed legalTL

**File:** `tests/test_d4_action_equivariance.py` or `tests/test_gold_v2_encoding.py`

---

### H3. to_move Perspective Swap Test

**Task:** Unit test ensuring `tie_player_is_current` and `bonus_7x7_status` are correct under to-move perspective swapping.

- **Test:** `test_tie_player_and_bonus_7x7_perspective_swap()`
- For states with known `TIE_PLAYER` and bonus owner: encode with `to_move=0` and `to_move=1`.
- Assert `tie_player_is_current` = 1 iff (TIE_PLAYER == to_move)
- Assert `bonus_7x7_status` reflects correct owner from current player's perspective (0/0.5/1)
- Cover: P0 has bonus, P1 has bonus, neither has bonus; tie_player 0 vs 1.

**File:** `tests/test_gold_v2_encoding.py`

---

### H4. Replay Buffer RAM Estimate

**Task:** One-time debug log or test that estimates replay buffer RAM after loading ~N samples; confirm max_size=300k is safe.

- **Option A:** Add `estimate_replay_buffer_ram(N, config)` in `replay_buffer.py` or a test: compute bytes for spatial_states + global_states + track_states + shop_ids + shop_feats + masks + policies + values + ownerships at N samples.
- **Option B:** Test `test_replay_buffer_ram_estimate()`: Load N=1000 (or 10k) samples, measure `sys.getsizeof` or actual memory delta; extrapolate to 300k; assert < threshold (e.g. 16 GB).
- Log: `"Replay buffer at 300k samples: ~X GB (spatial: Y, global: Z, ...)"`

**Files:** `src/training/replay_buffer.py`, `tests/test_gold_v2_encoding.py` or `tests/test_replay_buffer.py`

---

## Phase 1: Implementation Order

1. **Constants + Encoder** — `gold_v2_constants.py` (incl. BUY_START, MAX_ACTIONS), `encode_state_multimodal()` in encoder.py
2. **D4 updates** — SPATIAL_CHANNELS=56, transform_state for 56ch, legalTL scatter-map
3. **HDF5 schema** — New datasets, attrs, dtype/chunk/compression
4. **Selfplay** — Use multimodal encode; write new schema
5. **Replay buffer** — Merge/validate new schema; RAM estimate (H4)
6. **Dataset** — PatchworkDataset multimodal; D4 on spatial only
7. **Model** — Forward signature; FiLM refactor; ShopEncoder
8. **Inference** — Server/client payload; encoding_version
9. **Config** — config_best.yaml updates
10. **Tests + Tools** — All gotcha tests; hardening (H1–H4); preflight; CI

---

## Exact Files / Modules to Change

### Encoder
| File | Changes |
|------|---------|
| `src/network/encoder.py` | GoldV2StateEncoder; `encode_state_multimodal()`; frontier, valid_7x7, legalTL; x_global, x_track, shop; dimension asserts |
| `src/network/gold_v2_constants.py` | **New.** C_SPATIAL, C_TRACK, F_GLOBAL, F_SHOP, NMAX |

### Dataset / HDF5 Schema
| File | Changes |
|------|---------|
| `src/training/selfplay_optimized_integration.py` | Create spatial_states, global_states, track_states, shop_ids, shop_feats; HDF5 attrs; dtype/chunk/compression per G0.4 |
| `src/training/run_layout.py` | SELFPLAY_SCHEMA_VERSION=3; gold_v2 attr names |

### Replay Buffer Loader
| File | Changes |
|------|---------|
| `src/training/replay_buffer.py` | get_training_data for multimodal datasets; validate attrs on load; float16→float32 cast if used |

### Model
| File | Changes |
|------|---------|
| `src/network/model.py` | `forward(x_spatial, x_global, x_track, shop_ids, shop_feats, action_mask)`; remove film_input_plane_indices; ShopEncoder; new FiLM conditioning |

### Inference
| File | Changes |
|------|---------|
| `src/network/gpu_inference_server.py` | Multimodal batch; encoding_version assertion; legacy fallback |
| `src/mcts/gpu_eval_client.py` | submit() with multimodal payload + encoding_version |
| `src/mcts/alphazero_mcts_optimized.py` | encode_state_multimodal(); pass 5 inputs + version |

### D4
| File | Changes |
|------|---------|
| `src/network/d4_augmentation.py` | SPATIAL_CHANNELS=range(56); SLOT_ORIENT_BASE=8; transform_state 0–55; apply_d4_augment_batch spatial-only |

### Config
| File | Changes |
|------|---------|
| `configs/config_best.yaml` | input_channels=56; conditioning; remove film_input_plane_indices; data.expected_* |

### Integrity / Preflight / CI
| File | Changes |
|------|---------|
| `tools/pipeline_audit.py` | encoding_version, expected_* |
| `tools/preflight.py` | gold_v2 config validation; input_channels=56 |
| `tools/deep_preflight.py` | Step A encoder for gold_v2 |
| `tools/full_integrity_check.py` | encoding_version check |
| `.github/workflows/invariance-tests.yml` | Include test_gold_v2_encoding |

### Hardening (Tests)
| File | Changes |
|------|---------|
| `tests/test_d4_augmentation.py` | Use BUY_START, MAX_ACTIONS from shared constants |
| `tests/test_d4_action_equivariance.py` | Use BUY_START, MAX_ACTIONS; add `test_d4_mask_legalTL_equivalence_after_transform()` |
| `tests/test_gold_v2_encoding.py` | Use shared constants; add `test_tie_player_and_bonus_7x7_perspective_swap()` |
| `tests/test_structured_policy_head.py` | Use BUY_START, MAX_ACTIONS |
| `src/training/replay_buffer.py` | Add RAM estimate logging (or test) for N samples |

---

## Verification Runbook

Run these commands in order to prove correctness end-to-end:

```bash
# 1. Unit tests (encoder, legalTL↔mask, deltas, D4 scope, num_legal, hardening H1–H4)
pytest tests/test_gold_v2_encoding.py -v
pytest tests/test_d4_augmentation.py tests/test_d4_action_equivariance.py -v
# H2: combined D4+legalTL; H3: perspective swap; H4: RAM estimate (if test)

# 2. All invariance tests
pytest tests/test_d4_augmentation.py tests/test_d4_action_equivariance.py tests/test_shop_order_markov_alignment.py -v

# 3. Preflight (hardware, config, invariance, smoke)
python tools/preflight.py --config configs/config_best.yaml

# 4. Deep preflight (encoder, net, train, ckpt)
python tools/deep_preflight.py --config configs/config_best.yaml --device auto --tmp_dir /tmp/pw_gold_v2_smoke

# 5. Short selfplay rollout (1 game)
python -c "
from src.training.main import AlphaZeroTrainer
import tempfile
from pathlib import Path
cfg = Path('configs/config_best.yaml').read_text()
# Use minimal games=1, sims=1 for smoke
# Run one iteration selfplay only
"
# Or: run main with max_iterations=1, games_per_iteration=1, then inspect selfplay.h5

# 6. One training batch forward
python -c "
from src.training.trainer import PatchworkDataset
from src.network.model import create_network
import yaml
cfg = yaml.safe_load(open('configs/config_best.yaml'))
# Load dataset from committed/iter_000/selfplay.h5 (or create minimal HDF5)
# Get one batch; model.forward(...); assert no NaN
"
```

**Minimal smoke script (optional):**

```bash
# One-liner: preflight + 1 iter selfplay + 1 train step
python tools/preflight.py --config configs/config_best.yaml
python -m src.training.main --config configs/config_best.yaml  # interrupt after iter 0 train step
```

---

## Checklist Before Merge

- [ ] G0.1 Dimension constants defined, asserted, stored in HDF5, validated on load
- [ ] G0.2 legalTL ↔ action_mask test passes
- [ ] G0.3 Pass and buy delta tests pass
- [ ] G0.4 HDF5 dtype/chunk/compression implemented
- [ ] G0.5 encoding_version in payload; server fail-fast
- [ ] G0.6 D4 scope verified (spatial only; legalTL scatter-map)
- [ ] G0.7 num_legal early-exit and clip
- [ ] H1 Tests use shared BUY_START, MAX_ACTIONS constants
- [ ] H2 Combined D4+legalTL equivalence test passes
- [ ] H3 tie_player_is_current and bonus_7x7_status perspective-swap test passes
- [ ] H4 Replay buffer RAM estimate logged or tested; 300k safe
- [ ] All invariance tests pass
- [ ] Preflight passes
- [ ] One selfplay rollout produces valid HDF5
- [ ] One training batch forward completes without NaN
