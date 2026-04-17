# Beat-Humans Metrics (TensorBoard)

Low-noise observability that correlates with **human strength**: packing quality and search health. Logged each iteration under `selfplay/*` and included in committed iteration JSON / `metadata.jsonl` for downstream use (e.g. `build_training_csv`).

---

## A) Packing / Quilt Quality

Computed at **terminal state** for both players; definitions match `tools/ab_test_*` scripts (shared logic in `src/utils/packing_metrics.py`).

### Why empty_components and isolated_1x1_holes matter

- **Empty squares** — Raw unfilled cells; lower is better. Correlates with score but not with *shape* of remaining space.
- **Empty components** — Connected components of empty cells (4-neighbor BFS). Fewer components usually means fewer “scattered” holes and better packing; humans punish fragmented boards.
- **Isolated 1×1 holes** — Empty cells with **zero** empty neighbors. These are dead squares that can never be filled; they directly hurt score and are a strong signal of packing mistakes. Humans notice and avoid them.

So: **fewer empty squares, fewer components, fewer isolated holes** → better packing and stronger play.

### TensorBoard series

| Series | Meaning |
|--------|--------|
| `selfplay_avg_final_empty_squares_mean` | Mean empty squares over all players/games |
| `selfplay_avg_final_empty_components_mean` | Mean empty components over all players/games |
| `selfplay_avg_final_isolated_1x1_holes_mean` | Mean isolated 1×1 holes over all players/games |
| `selfplay_p50_final_*_mean` | Median (50th percentile) of *per-game* mean (across the two players) |
| `selfplay_p90_final_*_mean` | 90th percentile of per-game mean — **tail risk** |
| `selfplay_avg_final_*_abs_diff` | Mean over games of \|P0 − P1\| — **asymmetry** |

### How to interpret p90 tail metrics

- **p90** answers: “In the worst 10% of games (by this metric), how bad did it get?”
- If **p90** is high while **mean** looks fine, a minority of games have very fragmented boards or many isolated holes — the model still has bad tail cases that humans would punish.
- Improving p90 (e.g. p90 empty components going down) indicates the model is getting more consistent packing, not just better on average.

---

## B) Search Health / Tractability

Collected **per model move** when MCTS runs: root legal count and PW-limited expanded child count.

### TensorBoard series

| Series | Meaning |
|--------|--------|
| `selfplay_avg_root_legal_count` | Mean number of legal actions at root |
| `selfplay_avg_root_expanded_count` | Mean number of children expanded at root (PW cap) |
| `selfplay_avg_root_expanded_ratio` | Mean of `expanded / max(legal, 1)` per move |
| `selfplay_p90_root_legal_count` | 90th percentile of root legal count |
| `selfplay_p90_root_expanded_ratio` | 90th percentile of expanded ratio |

### What expanded_ratio means

- **expanded_ratio** = expanded_children / legal_actions (capped at 1 when PW limits expansion).
- **Too high** (near 1.0 everywhere) — Search is expanding almost every legal move; tree is very wide, possibly slow or unfocused. May need stronger pruning or a lower PW cap.
- **Too low** — Over-pruning; many legal moves never get expanded. Risk of missing good lines; PW or K_buy may be too aggressive.
- **Reasonable range** — Depends on position type; typically you want a middle ground so search is tractable but not blind to important moves. Use p90 to watch for positions where ratio spikes or collapses.

---

## C) Where they’re written

- **TensorBoard:** All series above are logged as `selfplay/<key>` each iteration.
- **Committed iteration:** The iteration summary JSON (e.g. `committed/iter_###/iteration_###.json`) includes `selfplay_stats` with these keys.
- **metadata.jsonl:** Each line’s `selfplay` object includes the same fields for use by `build_training_csv` or other analysis.

No per-player P0/P1 series are logged; only aggregates and asymmetry (`*_abs_diff`) to keep TensorBoard low-noise.
