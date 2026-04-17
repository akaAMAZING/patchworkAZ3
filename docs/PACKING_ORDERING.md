# Packing placement-ordering for BUY actions (Progressive Widening)

A **tiny** heuristic used only to **rank** BUY actions when ordering moves for Progressive Widening (PW). It does not change scoring, targets, the network, or legality. It only changes the **order** in which BUY actions are expanded (so PW explores “tighter packing” moves first).

## What it does

- **Scope:** Only the ordering key for BUY actions at PW:  
  `rank_key = (log(prior) or prior) + alpha * packing_heuristic_score(state, action)`  
  BUY actions are sorted by `rank_key` descending instead of by prior alone. PASS and PATCH are unchanged (still at the front).
- **Goal:** Reduce fragmentation (components + isolated 1×1 holes) in practice by favouring placements that look more “compact” in a small local check, matching human “tight packing” shortlisting.
- **Safety:** No pruning, no change to legal moves, value, or policy targets. Only the order of BUY moves for expansion is affected.

## Presets in use

- **Training (self-play + training):** Preset **C** — stable improvement without increasing empties. Used in `configs/config_continue_from_iter70.yaml` (main training config) and `configs/config_best.yaml`: `enabled: true`, alpha 0.25, adj_edges 0.25, corner_bonus 0, iso_hole_penalty 6, frontier_penalty 0, area_bonus 0. No phase gating; packing is on from move 1. `opponent_mix.enabled` is kept **false** for training.
- **GUI / max-strength play:** Preset **E_max_pack** — strongest in practice (beats A2 head-to-head, reduces fragmentation/isolated holes). Used in `configs/config_gui_max_strength.yaml`: `enabled: true`, alpha 0.90, adj_edges 0.05, corner_bonus 0, iso_hole_penalty 10, frontier_penalty 0, area_bonus 0.40. Always on from move 1 (no phase gating). GUI play uses `add_noise=false`, `temperature=0`, deterministic move selection.

## How to enable

In your MCTS config (e.g. `configs/config_continue_from_iter70.yaml` or `configs/config_best.yaml`):

```yaml
selfplay:
  mcts:
    packing_ordering:
      enabled: true
      alpha: 0.25
      use_log_prior: true
      local_check_radius: 2
      weights:
        adj_edges: 0.25
        corner_bonus: 0.0
        iso_hole_penalty: 6.0
        frontier_penalty: 0.0
        area_bonus: 0.0
```

Default is **off** (`enabled: false`).

## Heuristic components

- **adj_edges:** Rewards placements where new cells are 4-adjacent to already-filled cells (more adjacency → more compact).
- **corner_bonus:** Small bonus if the placement touches a board corner (true corner-only in code). 0 = off.
- **area_bonus:** Bonus proportional to the number of cells in the placed piece (prefer larger pieces when > 0). Use 0 for training; E_max_pack uses 0.40 for GUI.
- **iso_hole_penalty:** Penalty for empty cells in a small neighbourhood that end up with zero empty 4-neighbors (isolated 1×1 pockets) after the placement.
- **frontier_penalty:** Mild penalty for increasing the “frontier” (empty cells adjacent to filled) in the local window.

Scores are scaled so typical values sit roughly in the [-5, +5] range; `alpha` then blends this with the policy prior (e.g. 0.15 keeps the heuristic as a gentle nudge).

## Validation

Run head-to-head with:

- **A2:** PW on, packing_ordering off.
- **E:** PW on + packing_ordering E_max_pack.

Compare:

- E vs A2 win rate
- Avg empty components / isolated holes (and p90 isolated holes if logged)

Example: `tools/ab_test_search_variants_iter069.py` with `--packing --only-variant E`.

## Quick validation commands

- **Run GUI with max strength (E_max_pack from move 1):**
  ```bash
  python GUI/patchwork_api.py --model runs/patchwork_production/committed/iter_090/iteration_090.pt --config configs/config_gui_max_strength.yaml --simulations 2048 --host 127.0.0.1 --port 8000
  ```
  Or use the one-click launcher (edit `ITER` in the batch file if needed):
  ```bash
  launch_gui.bat
  ```
  In the GUI, set Config path to `configs/config_gui_max_strength.yaml` so the API loads E_max_pack.

- **Optional head-to-head: A2 (PW, packing off) vs E_max_pack:**
  ```bash
  python tools/ab_test_search_variants_iter069.py --iter 90 --packing --only-variant E --games 200 --progress-interval 10
  ```

## Metrics

- **selfplay_avg_packing_ordering_enabled:** 0 or 1 (from config).
- **selfplay_avg_packing_score_top1:** Optional; mean packing score of the top-ranked BUY at root (if reported by workers).

## Debug

At root, when packing ordering is enabled and log level is DEBUG, the MCTS logs the top 5 BUY actions by prior and by `rank_key` (with prior and packing score) so you can see how the order changes.

## Performance (optimized)

The heuristic is optimized for MCTS throughput:

- **Node-level cache:** Current-player occupancy is decoded once per node to an 81-bit bitboard and reused.
- **Pure bit-ops (no Python loops):** Adjacency and edge bonus use shifts + popcount. Iso-hole and frontier counts use neighborhood expansion `neigh(bb)` and precomputed window masks; no per-cell iteration.
- **Precomputed window masks:** `WINDOW_MASK[radius_index][cache_index]` for radius 0..3 (rectangle around placement + radius, clamped to board). Lookup is O(1).
- **Flat array caches:** `PLACEMENT_BB[cache_index]` and `WINDOW_MASK[r][cache_index]` with `cache_index = piece_id*648 + orient*81 + pos_idx`. No dict/tuple overhead.
- **Top-M only:** M = min(L_buy, K_cap+16) at root, K_cap+8 internally. Heuristic runs only for top M by prior; rest use rank_key = log(prior).
- **Batch API:** MCTS calls `packing_heuristic_scores_batch` once per node with a list of cache indices, so one inlined loop instead of M function calls.

**Cython extension:** An optional compiled module `src/mcts/packing_heuristic_cy` accelerates the batch scorer (81-bit bitboard as two uint64s, neigh/popcount in C). Build with:

```bash
python setup.py build_ext --inplace
```

If the extension is not built, the pure Python batch path is used automatically. The benchmark reports both when Cython is available (target: heuristic portion 2–5× faster).

Benchmark: run `python tools/bench_packing_ordering.py` for avg ms/order and a breakdown (decode, sort prior, heuristic, sort rank). Reports Python batch vs Cython batch when the extension is built.
