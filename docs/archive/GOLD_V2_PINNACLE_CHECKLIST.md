# Gold v2 Multimodal — Pinnacle Checklist

Before claiming "pinnacle" quality when moving to gold_v2_multimodal, ensure:

---

## 1. D4 Machinery on New Spatial Stack (0–55) and legalTL

- [ ] **SPATIAL_CHANNELS** in `d4_augmentation.py`: `range(28)` → `range(56)`
- [ ] **SLOT_ORIENT_BASE**: 4 → 8 (shape planes 8–31; legalTL 32–55)
- [ ] **transform_state**: Apply to all 56 channels; slot×orient permute for 8–55 (shape + legalTL same orient remap)
- [ ] **legalTL transform**: Use `transform_legalTL_planes()` (scatter-map, dimension-aware) — already implemented for full_clarity; extend for gold_v2’s 24 legalTL planes (32–55)
- [ ] **Dataset __getitem__**: If legalTL stored separately, transform via `transform_legalTL_planes()` — do NOT rotate plane images; scatter-map with buy-action mapping

**Reference:** Plan Part E; `docs/INVARIANT_VERIFICATION_REPORT.md`

---

## 2. Invariance Tests in CI and Before Long Training

- [ ] **CI**: `.github/workflows/invariance-tests.yml` runs these on push/PR
- [ ] **Pre-training**: Preflight runs invariance tests (step 5/6); `run_overnight.ps1` invokes preflight by default
- [ ] **Full integrity check**: Use `--run_slow true` before major runs so `test_legalTL_transform_consistency` runs

**Tests:** Shop order (7), D4 augmentation (7), D4 equivariance (5 incl. 1 slow).

---

## 3. Markov-Completeness Bits in x_global

Include these truly-Markov features (required for correctness, not “extra features”):

| Feature | x_global slot | Engine source |
|---------|---------------|---------------|
| **track_patch_available_0..4** | Leather+tie (5 bits) | Patch token still available at each of 5 track positions |
| **tie_player_is_current** | Leather+tie (1 bit) | `TIE_PLAYER` — who wins on draw (first to pass) |

Also in x_track:

| Channel | Name | Semantics |
|---------|------|-----------|
| 3 | patch_marks_remaining | 1s only where patch token still available |

**Reference:** Plan Part B (Leather+tie), Part C (x_track ch.3); `TIE_PLAYER` in `patchwork_engine.py`

---

## Quick Verify

```bash
# Invariance tests (~70s with slow)
pytest tests/test_d4_augmentation.py tests/test_d4_action_equivariance.py tests/test_shop_order_markov_alignment.py -v

# Full preflight (includes invariance as of this checklist)
python tools/preflight.py --config configs/config_best.yaml
```
