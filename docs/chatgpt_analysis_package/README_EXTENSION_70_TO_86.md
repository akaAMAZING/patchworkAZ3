# Extension: iterations 70–86

This folder contains an **extension CSV** for the main training metrics, so you can keep the original 0–70 snapshot and add newer iters separately.

## Files

| File | Description |
|------|-------------|
| **training_metrics_per_iteration.csv** | Original: iterations 0–70 (71 rows). Unchanged. |
| **training_metrics_per_iteration_70_to_86.csv** | Extension: iterations **70–86** (17 rows). Same columns as the main CSV where present; includes **beat-humans** columns (e.g. `selfplay_selfplay_avg_final_empty_components_mean`, `selfplay_selfplay_avg_root_expanded_ratio`) for iters 71–86 (iter 70 may have blanks if committed before beat-humans logging). |
| **README_EXTENSION_70_TO_86.md** | This file. |

## How to use

- **Merge in analysis:** Concatenate the extension rows (e.g. drop the header of the extension file and append rows 71–86 to the main CSV, or load both and filter by `iteration`).
- **Regenerate extension:** From repo root:
  ```bash
  python docs/chatgpt_analysis_package/build_training_csv.py --min-iter 70 --max-iter 86 --output training_metrics_per_iteration_70_to_86.csv
  ```
  This reads from `runs/patchwork_production/committed/iter_*/iteration_*.json`. To extend beyond 86, increase `--max-iter` (e.g. `--max-iter 100`) and/or change the output filename.

## Column note

Beat-humans keys in `selfplay_stats` are prefixed in the CSV as `selfplay_selfplay_*` (e.g. `selfplay_selfplay_avg_final_empty_squares_mean`). See **FULL_METRICS.md** for definitions.
