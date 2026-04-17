# Code Context for Patchwork AlphaZero

This document gives ChatGPT (or any analyst) the key source snippets and optimizations so the codebase is fully understandable. Paths are relative to the repo root.

---

## 1. Score and ownership loss (possible stall points)

We suspect **score loss** and/or **ownership loss / ownership accuracy** may be stalling. These are the exact code paths.

### 1.1 Score targets (replay → training)

Replay stores **tanh-normalised** score margin: `tanh((to_move_score - opp_score) / 30.0)`. Training builds a **201-bin soft label** from that.

**`src/training/value_targets.py`** — how labels are produced in self-play:

```python
_SCORE_NORMALISE_DIVISOR = 30.0

def value_and_score_from_scores(score0, score1, winner, to_move) -> tuple[float, float]:
    value = terminal_value_from_scores(score0, score1, winner, to_move)  # +1/-1/0
    raw_margin = float(score0 - score1) if to_move == 0 else float(score1 - score0)
    score_margin = math.tanh(raw_margin / _SCORE_NORMALISE_DIVISOR)  # in (-1, 1)
    return value, score_margin
```

**`src/training/trainer.py`** — Gaussian soft target for 201-bin CE (used in training step):

```python
def make_gaussian_score_targets(score_margins_tanh, score_utility_scale, score_min, score_max, sigma, bin_vals=None):
    """Build (B, 201) soft label from tanh-normalised score_margins. margin_points = scale * atanh(m)."""
    m = score_margins_tanh.float().clamp(-0.999999, 0.999999)
    margin_points = (score_utility_scale * torch.atanh(m)).round().clamp(float(score_min), float(score_max))
    diff = bin_vals.unsqueeze(0) - margin_points.unsqueeze(1)  # (B, 201)
    weights = torch.exp(-0.5 * (diff / sigma) ** 2)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    return weights
```

Config: `score_loss_weight: 0.1`, `score_utility_scale: 30.0` (must match `_SCORE_NORMALISE_DIVISOR`), `score_target_sigma` from trainer (default 1.5). If **train_score_loss** stops decreasing, consider: sigma too sharp/flat, scale mismatch, or 201-bin head capacity.

### 1.2 Score and ownership loss inside the network

**`src/network/model.py` — `get_loss` (score and ownership):**

```python
# Score loss: distributional CE over 201 bins (soft target from replay score_margins).
score_loss = torch.tensor(0.0, device=state.device)
if score_loss_weight > 0 and self.value_head.score_head is not None:
    if target_score_dist is not None:
        tgt = target_score_dist.float() / target_score_dist.float().sum(dim=-1, keepdim=True).clamp(min=1e-12)
        logp = F.log_softmax(score_logits.float(), dim=-1).clamp(min=-100.0)
        score_loss = -(tgt * logp).sum(dim=-1).mean()

# Ownership loss (auxiliary, KataGo-style dual-board: 2 channels = P0 filled, P1 filled)
ownership_loss = torch.tensor(0.0, device=state.device)
if self.ownership_head is not None and target_ownership is not None and ownership_weight > 0:
    ownership_logits = self.ownership_head(trunk)  # (B, 2, 9, 9)
    if ownership_valid_mask is not None and not ownership_valid_mask.all():
        valid_logits = ownership_logits[ownership_valid_mask]
        valid_targets = target_ownership[ownership_valid_mask]
        if valid_logits.shape[0] > 0:
            ownership_loss = F.binary_cross_entropy_with_logits(valid_logits, valid_targets)
    else:
        ownership_loss = F.binary_cross_entropy_with_logits(ownership_logits, target_ownership)

# Ownership accuracy (binary threshold 0.5), only over valid samples
ownership_accuracy = (ownership_pred == target_ownership).float().mean().item()  # (or masked)
```

Config: `ownership_loss_weight: 0.15`. If **train_ownership_accuracy** plateaus (e.g. ~85–86%), the auxiliary head may be underfitting or targets may be noisy; old replay data can have `-1` sentinel (excluded via `ownership_valid_mask`).

### 1.3 Ownership head architecture

**`src/network/model.py` — OwnershipHead:**

```python
class OwnershipHead(nn.Module):
    """Predicts per-cell board occupancy at game end. Channel 0: P(current player's cell filled), Channel 1: P(opponent's)."""
    def __init__(self, input_channels, ownership_channels, use_batch_norm=True):
        self.conv = nn.Conv2d(input_channels, ownership_channels, 1, ...)
        self.bn = nn.BatchNorm2d(ownership_channels) if use_batch_norm else nn.Identity()
        self.conv_out = nn.Conv2d(ownership_channels, 2, 1)  # 2 binary channels
    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)), inplace=True)
        return self.conv_out(x)  # (B, 2, 9, 9) logits
```

---

## 2. Training pipeline optimizations

### 2.1 Batched HDF5 and prefetch

**`src/training/trainer.py`** — DataLoader uses lazy HDF5 per worker, batched reads, and optional prefetch to overlap next batch with GPU:

```python
# [PERF] Background prefetch to overlap next(data_iter) with GPU compute
prefetch_batches = int(hw.get("prefetch_batches", 2))
use_prefetch = prefetch_batches > 0 and self.device.type == "cuda"
batch_iter = _prefetch_generator(train_loader, prefetch_batches) if use_prefetch else train_loader
```

Config: `hardware.prefetch_batches: 2`.

### 2.2 D4 augmentation (dynamic, CPU by default)

States/policies/masks/ownerships are augmented with the dihedral group (8 transforms). Applied **per batch** at training time; stored state is canonical (one state per move). Config: `d4_augmentation: dynamic`, `d4_on_gpu: false` (CPU vectorized numpy is faster; GPU path has high per-batch overhead).

**`src/network/d4_augmentation.py`** — `apply_d4_augment_batch`, `apply_ownership_transform_batch`; **`src/network/d4_augmentation_gpu.py`** — GPU variant when `d4_on_gpu: true`.

### 2.3 EMA (Polyak)

EMA is applied every optimizer step; checkpoint saves EMA weights for self-play and eval. Config: `ema.decay: 0.999`, `ema.use_for_selfplay: true`, `ema.use_for_eval: true`.

### 2.4 Win-first MCTS (Stockfish-like)

Value and score utility are blended so that we prioritise win/loss over margin when value is decisive. Config: `selfplay.mcts.win_first.enabled: true`, with `value_delta_*`, `gate_dsu_*`, `tiebreak: score_then_visits`, `filter_before_sampling: true`. Implemented in `src/mcts/alphazero_mcts_optimized.py`.

### 2.5 FiLM conditioning

Trunk is conditioned on global (61-dim), track (8×54 → 432), and shop (128-dim) via FiLM (scale/shift per block). Config: `use_film: true`, `film_per_block: true`, `film_track_use_conv: true`, `film_global_inject_dim: 64` (also injected into value and pass-logit heads). Policy head is factorized (pass + patch 81 + buy 24×81 = 2026 actions).

### 2.6 AMP and gradient clipping

Training uses `use_amp: true`, `amp_dtype: bfloat16`. Gradients are clipped with `max_grad_norm: 1.0` before the optimizer step.

---

## 3. Progressive widening (PW) — MCTS

**Progressive widening** limits how many root children are expanded so MCTS can deepen search with limited sims. Config: `selfplay.mcts.progressive_widening.enabled: true`, `k_root: 64`, `k0: 32`, `k_sqrt_coef: 8`, `always_include_pass: true`, `always_include_patch: true`.

**`src/mcts/alphazero_mcts_optimized.py`** — Root expansion is capped; `get_root_legal_count()` and `get_root_expanded_count()` return legal and expanded counts for the last search. Selfplay records these per move and aggregates into `selfplay_avg_root_legal_count`, `selfplay_avg_root_expanded_count`, `selfplay_avg_root_expanded_ratio`, and p90 variants (see **FULL_METRICS.md**). Too high ratio ⇒ too wide; too low ⇒ over-pruning.

---

## 4. Beat-humans metrics (packing + search health)

**Packing / quilt quality** (terminal state, both players):

- **`src/utils/packing_metrics.py`:** `empties_from_occ_words(occ0,occ1,occ2)`, `fragmentation_from_occ_words(occ0,occ1,occ2)` → (empty_components, isolated_1x1_holes). Same 4-neighbor BFS and isolated-hole definition as `tools/ab_test_*`.
- **`src/training/selfplay_optimized.py`:** At game end, for each player we call `fragmentation_from_occ_words` on the board occupancy words; we add to the game output: `empty_squares_p0/p1`, `empty_components_p0/p1`, `isolated_1x1_holes_p0/p1`, and (per MCTS move) `root_legal_counts`, `root_expanded_counts`.
- **`src/training/selfplay_optimized_integration.py`:** `_compute_stats()` builds per-game lists, then calls `aggregate_packing_over_games()` and `aggregate_root_over_moves()` from `src/utils/packing_metrics.py` to produce all `selfplay_avg_final_*`, `selfplay_p50_*`, `selfplay_p90_*`, and `*_abs_diff` / root stats. These are written into `selfplay_stats` and thus into iteration JSON and metadata.

No per-player P0/P1 series are logged; only aggregates and asymmetry to keep TensorBoard low-noise. Full key list and interpretation: **FULL_METRICS.md**.

---

## 5. No automated evaluation (design choice)

**Evaluation is intentionally off** in production: `evaluation.games_vs_best: 0`, `evaluation.games_vs_pure_mcts: 0`. We **manually evaluate** (head-to-head, A/B tests, GUI). So no eval_* win-rate or Elo series are produced; strength is judged via selfplay stats and beat-humans metrics.

---

## 6. Metrics logged per iteration (all in CSV)

From **`src/training/trainer.py`** (epoch aggregation) and **`src/training/main.py`** (iter summary → JSON):

- **train_metrics:** `total_loss`, `policy_loss`, `value_loss`, **`score_loss`**, **`ownership_loss`**, `value_mse`, `policy_accuracy`, `policy_top5_accuracy`, **`ownership_accuracy`**, `grad_norm`, `policy_entropy`, `kl_divergence`, `step_skip_rate`, plus ownership diagnostics and `approx_identity_check`. Full list: **FULL_METRICS.md**.
- **selfplay_stats:** `num_games`, `num_positions`, `avg_game_length`, `p0_wins`, `p1_wins`, `games_per_minute`, `avg_policy_entropy`, `avg_top1_prob`, `avg_num_legal`, `avg_redundancy`, `unique_positions`, `avg_root_q`, plus **beat-humans:** all `selfplay_avg_final_*`, `selfplay_p50_*`, `selfplay_p90_*`, `*_abs_diff`, and root legal/expanded/ratio (see **FULL_METRICS.md**).
- **applied_settings:** Flattened selfplay/training/replay/adaptive_games (games, lr, cpuct, dynamic_score_utility_weight, PW params, etc.).

If **score_loss** or **ownership_loss** / **ownership_accuracy** flatten over iterations, that is the suspected stall; the code paths above are where to reason about fixes (targets, weights, head size, or data validity).

---

## 7. File map (where to look)

| Concern | File |
|--------|------|
| Score/ownership loss, total loss, metrics | `src/network/model.py` (`get_loss`, `OwnershipHead`) |
| Score target building (Gaussian 201-bin) | `src/training/trainer.py` (`make_gaussian_score_targets`, training step) |
| Score margin labelling (tanh) | `src/training/value_targets.py` |
| Ownership masking (valid samples) | `src/training/trainer.py` (ownership_valid_mask), `model.get_loss` |
| D4 augmentation | `src/network/d4_augmentation.py`, `d4_augmentation_gpu.py` |
| MCTS win-first, score utility, PW, root legal/expanded | `src/mcts/alphazero_mcts_optimized.py` |
| Packing metrics (empties, components, isolated), aggregation | `src/utils/packing_metrics.py` |
| Terminal packing + root stats per game | `src/training/selfplay_optimized.py` |
| Selfplay stats aggregation (incl. beat-humans) | `src/training/selfplay_optimized_integration.py` |
| Training loop, prefetch, EMA step | `src/training/trainer.py` |
| Iteration loop, commit, JSON summary | `src/training/main.py` |
