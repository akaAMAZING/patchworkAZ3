# Phase A: Full Clarity Encoder Overhaul — Migration

**This is a BREAKING CHANGE.** Delete old replay data, selfplay HDF5, and checkpoints.

## Summary

- **Encoder**: 16 channels → 56 channels (gold_v2)
- **HDF5 schema**: `(N, 16, 9, 9)` → `(N, 56, 9, 9)`
- **Schema version**: 2
- **Encoding version**: `full_clarity_v1`
- **Redundant channels removed**: affordable_fraction, avg_norm_cost, neutral_position

## Config Edits Required

```yaml
network:
  input_channels: 61
  film_input_plane_indices: [28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]

data:
  allow_legacy_state_channels: false  # fail-fast; no padding
```

## Delete Before Restart

1. **Replay buffer**: `data/replay_buffer/`, `replay_state.json`
2. **Selfplay HDF5**: All `*selfplay*.h5`, `staging/iter_*/selfplay.h5`, `committed/iter_*/selfplay.h5`
3. **Checkpoints**: Old `.pt` checkpoints are incompatible (input_channels mismatch)
4. **NPZ shards**: Any `*_shards/*.npz` from previous selfplay runs

## Post-Run Sanity Check

```python
import h5py

path = "runs/<run_id>/committed/iter_000/selfplay.h5"  # or your merged HDF5
with h5py.File(path, "r") as f:
    states = f["states"]
    print("Shape:", states.shape)  # (N, 56, 9, 9)
    print("expected_channels:", f.attrs.get("expected_channels"))
    print("encoding_version:", f.attrs.get("encoding_version"))
    print("selfplay_schema_version:", f.attrs.get("selfplay_schema_version"))
    assert states.shape[1:] == (56, 9, 9), "states must be (N, 56, 9, 9)"
```

## Source-of-Truth Constants

- `TRACK_LENGTH=53` (positions 0..53)
- `BUTTONS_AFTER = [5, 11, 17, 23, 29, 35, 41, 47, 53]`
- `PATCHES_AFTER = [26, 32, 38, 44, 50]`
- `MAX_PIECE_COST=10`, `MAX_PIECE_TIME=6`, `MAX_PIECE_INCOME=2`
- Action space unchanged (2026)
