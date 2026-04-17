# In-Depth Change Log: Progressive Widening + Margin Tie-Break (from iter_069)

**Purpose:** One strong, compute-efficient change set to improve Patchwork playing strength vs decent humans, starting from iteration_069. Primary objective: maximize win rate (margin is secondary / tie-break only). Limited compute: Ryzen 5900X + RTX 3080; training ~28 games/min at iter69.

---

## Overview of Changes

1. **Progressive widening (top-K BUY expansion)** in MCTS so search is tractable with ~208 sims when root can have hundreds of legal BUY placements.
2. **Margin as small, always-on tie-break**: win-first kept; DSU gating disabled; dynamic score utility weight reduced (0.30 → 0.12) and schedule capped at 0.12 by iter40.
3. **A/B test script** to compare baseline vs new MCTS with the same checkpoint (no training).
4. **Training continuation** from iter_069 with a new run_id and optional replay/optimizer reset.

---

## 1. File: `src/mcts/alphazero_mcts_optimized.py`

### 1.1 New helper: `_apply_progressive_widening_order` (inserted before `encode_legal_actions_fast`)

**Location:** After `engine_action_to_flat_index` / before `encode_legal_actions_fast`.

**What it does:**
- Takes a node that already has `legal_actions` set and priors filled (from one NN evaluation).
- Partitions legal actions by type: PASS (`action[0] == AT_PASS`), PATCH (`AT_PATCH`), BUY (`AT_BUY`).
- Respects config: `always_include_pass` / `always_include_patch` (if false, those lists are empty).
- Sorts BUY actions by policy prior descending: `buy_with_prior = [(a, prior[a]) for a in buy_actions]` then `sort(key=lambda x: -x[1])`.
- Builds new order: `ordered_legal = pass_actions + patch_actions + [a for a, _ in buy_with_prior]`.
- Reassigns on the node:
  - `node.legal_actions = ordered_legal`
  - `node._prior` = priors in this new order (same length).
  - `node._action_to_idx` = `{a: i for i, a in enumerate(ordered_legal)}`
  - `node._pw_n_always = len(pass_actions) + len(patch_actions)`
  - `node._pw_L_buy = len(buy_actions)`
  - `node._pw_k0_buy = pw_config.k_root if is_root else pw_config.k0`
- Reallocates per-action arrays to length `len(ordered_legal)` (all zeroed): `_visit_count`, `_total_value`, `_virtual_loss`, `_value_sum`, `_score_sum`.

**Code (exact):**

```python
def _apply_progressive_widening_order(
    node: "MCTSNode",
    pw_config: ProgressiveWideningConfig,
    is_root: bool,
) -> None:
    """
    Reorder node.legal_actions to [PASS, PATCH..., top BUY by prior], set _pw_*.
    Call after _init_arrays and after priors are set. Mutates node.legal_actions and node._prior.
    """
    if not pw_config.enabled or not node.legal_actions or node._prior is None:
        return
    legal = node.legal_actions
    action_to_idx = node._action_to_idx
    pass_actions = [a for a in legal if int(a[0]) == AT_PASS] if pw_config.always_include_pass else []
    patch_actions = [a for a in legal if int(a[0]) == AT_PATCH] if pw_config.always_include_patch else []
    buy_actions = [a for a in legal if int(a[0]) == AT_BUY]
    buy_with_prior = [(a, float(node._prior[action_to_idx[a]])) for a in buy_actions]
    buy_with_prior.sort(key=lambda x: -x[1])
    ordered_legal = pass_actions + patch_actions + [a for a, _ in buy_with_prior]
    n_always = len(pass_actions) + len(patch_actions)
    L_buy = len(buy_actions)
    k0_buy = pw_config.k_root if is_root else pw_config.k0
    node.legal_actions = ordered_legal
    new_prior = np.array([node._prior[action_to_idx[a]] for a in ordered_legal], dtype=np.float64)
    node._prior = new_prior
    node._action_to_idx = {a: i for i, a in enumerate(ordered_legal)}
    node._pw_n_always = n_always
    node._pw_L_buy = L_buy
    node._pw_k0_buy = k0_buy
    n_total = len(ordered_legal)
    node._visit_count = np.zeros(n_total, dtype=np.int32)
    node._total_value = np.zeros(n_total, dtype=np.float64)
    node._virtual_loss = np.zeros(n_total, dtype=np.float64)
    node._value_sum = np.zeros(n_total, dtype=np.float64)
    node._score_sum = np.zeros(n_total, dtype=np.float64)
```

**Note:** Engine action types come from `patchwork_engine`: `AT_PASS`, `AT_PATCH`, `AT_BUY` (ints). The 2026 action encoding is unchanged; this only reorders and limits which children are *selectable*.

---

### 1.2 New config: `ProgressiveWideningConfig` and `_parse_progressive_widening`

**Location:** In the "MCTS Configuration" section, before `WinFirstConfig`.

**ProgressiveWideningConfig (dataclass):**
- `enabled: bool = False`
- `k_root: int = 64`   — at root, K for BUY uses this as base
- `k0: int = 32`       — at internal nodes, K for BUY uses this as base
- `k_sqrt_coef: float = 16.0` — growth term: `K_buy = min(L_buy, k0_buy + floor(k_sqrt_coef * sqrt(n_total)))`
- `always_include_pass: bool = True`
- `always_include_patch: bool = True`

**Parser:** `_parse_progressive_widening(cfg: dict)` reads `cfg.get("enabled", False)` etc. and returns a `ProgressiveWideningConfig` instance.

---

### 1.3 MCTSConfig change

**Addition:** One new field on `MCTSConfig`:

```python
progressive_widening: ProgressiveWideningConfig = field(default_factory=ProgressiveWideningConfig)
```

(Existing fields like `win_first`, `dynamic_score_utility_weight`, etc. unchanged in the class.)

---

### 1.4 MCTSNode changes

**New slots (in `__slots__`):**
- `"_pw_n_always"` — number of always-included actions (PASS + PATCH)
- `"_pw_L_buy"`     — number of legal BUY actions at this node
- `"_pw_k0_buy"`    — K0 or K_root used in the widening formula

**New attributes in `__init__`:**
- `self._pw_n_always: Optional[int] = None`
- `self._pw_L_buy: Optional[int] = None`
- `self._pw_k0_buy: Optional[int] = None`

**New method: `get_expanded_count(self, pw_config: Optional[ProgressiveWideningConfig]) -> int`**
- If `legal_actions` or `_visit_count` is None → return 0.
- If `pw_config` is None or not enabled or `_pw_n_always` is None → return `len(self.legal_actions)` (all actions selectable).
- Otherwise:  
  `K_buy = min(L_buy, k0_buy + floor(k_sqrt_coef * sqrt(max(0, n_total))))`  
  Return `min(L, n_always + K_buy)` where `L = len(self.legal_actions)`.

**Change to `select_action`:**
- Signature now includes `pw_config: Optional[ProgressiveWideningConfig] = None`.
- First line: `cap = self.get_expanded_count(pw_config)`.
- All array operations use slices to index only `0..cap-1`:  
  `n = self._visit_count[:cap].astype(np.float64)`, `vl = self._virtual_loss[:cap]`, `prior_slice = self._prior[:cap]`, `total_value_slice = self._total_value[:cap]`.  
  UCB is computed on these slices; `best_idx` is in `0..cap-1`; return `self.legal_actions[best_idx]`.

So when progressive widening is enabled, selection (PUCT) runs only over the first `cap` actions (always-include + top-K BUY, with K growing with visits).

---

### 1.5 Where `_apply_progressive_widening_order` is called

- **Single-node expand (`_expand_and_evaluate`):**  
  After setting `node._prior[:]` from the NN (both eval_client and local-network paths), if `self.config.progressive_widening.enabled`: call  
  `_apply_progressive_widening_order(node, self.config.progressive_widening, is_root=(self._root is not None and node is self._root))`.  
  Then call `node.normalize_priors()`.

- **Batch expand (`_batch_expand_and_evaluate`):**  
  For each non-terminal node, after `node._prior[:] = priors_legal[...]` (or from local network), if `self.config.progressive_widening.enabled`: call  
  `_apply_progressive_widening_order(node, self.config.progressive_widening, is_root=False)`  
  (batch expands leaves, so no node is the search root). Then `node.normalize_priors()`.

---

### 1.6 Traversal: passing `pw_config` into `select_action`

- **`_select_leaf`:**  
  Before the `while node.is_expanded() and not node.terminal:` loop, set  
  `pw = self.config.progressive_widening if self.config.progressive_widening.enabled else None`.  
  In the loop, replace  
  `action = node.select_action(self.config.cpuct, self.config.fpu_reduction)`  
  with  
  `action = node.select_action(self.config.cpuct, self.config.fpu_reduction, pw_config=pw)`.

- **`_simulate`:**  
  Same: compute `pw` once, then call `node.select_action(..., pw_config=pw)` inside the loop.

---

### 1.7 New methods on `OptimizedAlphaZeroMCTS`

- **`get_root_legal_count(self) -> int`**  
  Returns `len(self._root.legal_actions)` if `self._root` and `self._root.legal_actions` exist, else 0.

- **`get_root_expanded_count(self) -> int`**  
  If `self._root` is None, return 0. Else set  
  `pw = self.config.progressive_widening if self.config.progressive_widening.enabled else None`  
  and return `self._root.get_expanded_count(pw)`.

Used by the A/B script to report average root legal count and average root expanded count (to confirm widening is active).

---

### 1.8 Factory: `create_optimized_mcts`

- After parsing `win_first`, add:  
  `progressive_widening = _parse_progressive_widening(mcts_cfg.get("progressive_widening"))`.
- When building `MCTSConfig`, add the keyword argument:  
  `progressive_widening=progressive_widening`.

No other factory logic changed (simulations, cpuct, etc. still come from config as before).

---

## 2. File: `configs/config_best.yaml`

### 2.1 selfplay.mcts

- **`dynamic_score_utility_weight`:** value changed from `0.3` to `0.12` (margin as small tie-break; win remains primary).

- **New block `progressive_widening`** (inserted after `score_utility_scale`, before `cpuct`):
  ```yaml
  progressive_widening:
    enabled: true
    k_root: 64
    k0: 32
    k_sqrt_coef: 16
    always_include_pass: true
    always_include_patch: true
  ```

- **Under `win_first`:**  
  `gate_dsu_enabled` changed from `true` to `false` so DSU is always applied as a tie-break (no gating by root value). Win-first selection logic is unchanged; only the DSU gate is disabled.

### 2.2 iteration.dynamic_score_utility_weight_schedule

**Before (conceptually):** ramp to 0.30 by iter 40 (e.g. 0, 0.10, 0.20, 0.24, 0.27, 0.30 at iters 0, 10, 25, 32, 36, 40).

**After:** cap at 0.12 by iter 40 and hold:
```yaml
dynamic_score_utility_weight_schedule:
- iteration: 0
  dynamic_score_utility_weight: 0.00
- iteration: 10
  dynamic_score_utility_weight: 0.06
- iteration: 25
  dynamic_score_utility_weight: 0.10
- iteration: 40
  dynamic_score_utility_weight: 0.12
- iteration: 500
  dynamic_score_utility_weight: 0.12
```

So by iter 40 the weight is 0.12 and stays there. The schedule is applied in `main.py` via `_apply_q_value_weight_and_cpuct_schedules` (which does `_step_schedule_lookup` on `dynamic_score_utility_weight_schedule` and writes the result into `config["selfplay"]["mcts"]["dynamic_score_utility_weight"]`).

---

## 3. New file: `tools/ab_test_widening_iter069.py`

**Purpose:** Compare baseline MCTS (A) vs new MCTS (B) using the *same* checkpoint, without training.

**CLI arguments:**
- `--config` (default: `configs/config_best.yaml`)
- `--checkpoint` (default: None → resolved to `runs/<run_id>/committed/iter_069/iteration_069.pt` via `get_run_root(config)` and `committed_dir(run_root, 69)`)
- `--games` (default: 200)
- `--sims` (default: 208)
- `--seed` (default: 42)

**Config variants:**
- **A (baseline):** `make_config_baseline(config)` — `dynamic_score_utility_weight: 0.30`, `progressive_widening.enabled: False`, `win_first.gate_dsu_enabled: True`.
- **B (new):** `make_config_new(config)` — `dynamic_score_utility_weight: 0.12`, `progressive_widening.enabled: True` (with k_root=64, k0=32, k_sqrt_coef=16, always_include_* true), `win_first.gate_dsu_enabled: False`.

**Opponent:**  
If `committed_dir(run_root, 68) / "iteration_068.pt"` exists, load it as a second model and use it as MCTS opponent (64 sims). Otherwise use `PureMCTSEvaluator(simulations=64, ...)`.

**Flow:**
- Load main config; resolve checkpoint and opponent.
- Create one network and load the chosen checkpoint into it.
- Build `mcts_a` from `config_a`, `mcts_b` from `config_b` (same network, different MCTS configs).
- For each arm (A, B): run `run_games(model_mcts, opponent_mcts, pure_mcts_opponent, num_games, base_seed, sims, device, config, label)`.

**`run_games` behavior:**
- Build paired eval schedule (seeds and who plays first).
- For each game: `new_game(seed)`, then loop until terminal or max_moves. On model’s turn: `model_mcts.search(...)`, `model_mcts.select_action(visit_counts, temperature=0, deterministic=True)`, then record `model_mcts.get_root_legal_count()` and `model_mcts.get_root_expanded_count()`. On opponent’s turn: same with opponent MCTS or pure MCTS.
- At terminal state: record win/loss, game length, and `_empty_squares_both_players(state)` (uses `empty_count_from_occ` for P0 and P1 from engine state indices `P0_OCC0/1/2`, `P1_OCC0/1/2`).
- Return dict: `win_rate`, `wins`, `total_games`, `avg_game_length`, `avg_empty_p0`, `avg_empty_p1`, `avg_root_legal_count`, `avg_root_expanded_count`.

**Output:** Prints results for A and B: win rate vs opponent, avg game length, avg final empty squares (P0, P1), avg root legal count, avg root expanded count. Comment in script: “Widening active when B’s avg root expanded count is noticeably below avg root legal count.”

**Dependencies:** Same as evaluation (patchwork_engine, mcts, network, evaluation.build_eval_schedule and PureMCTSEvaluator, run_layout.get_run_root and committed_dir). No training or replay code.

---

## 4. New file: `docs/RUN_WIDENING_ITER069.md`

Short “how to run” doc:

- **Run A/B test:**  
  `python tools/ab_test_widening_iter069.py`  
  Optional: `--checkpoint ... --games 200 --sims 208 --seed 42`.

- **Start training from iter_069 with new run_id:**  
  Set `paths.run_id: patchwork_production_widening` (or use `--run-id`).  
  Command:  
  `python -m src.training.main --config configs/config_best.yaml --start-iteration 69 --resume runs/patchwork_production/committed/iter_069/iteration_069.pt --run-id patchwork_production_widening`.  
  Notes on optional replay buffer reset or smaller window for first 2 iters, and optional optimizer reset (e.g. `resume_optimizer_state: false` for first iter of new run).

- **Config diff summary:** Lists the selfplay.mcts and iteration schedule changes (as in section 2 above).

---

## 5. Behaviour summary (for ChatGPT)

- **Progressive widening:**  
  When enabled, at each node we still get *all* legal moves and one NN evaluation (priors for all). We then reorder to [PASS, PATCH…, BUY sorted by prior]. Only the first `n_always + K_buy` actions are selectable, with `K_buy = min(L_buy, k0_buy + floor(k_sqrt_coef * sqrt(n_total)))`. So at root we start with up to 64 BUY actions (plus pass/patch); at internal nodes up to 32 BUY; as visits increase, more BUY actions become selectable. Legality and 2026 encoding are unchanged; only the set of children considered in PUCT is restricted.

- **Margin / DSU:**  
  Win-first remains the primary decision (value-delta and tie-break by score/visits). `gate_dsu_enabled: false` means the DSU gate in the MCTS is always 1.0, so `dynamic_score_utility_weight` (now 0.12, capped by schedule at 0.12 by iter40) is always applied as a tie-break / small margin term. So: win first; when win chances are similar, DSU slightly prefers better packing.

- **Training continuation:**  
  Same codebase; only config and CLI change. New run_id keeps logs/checkpoints separate; resume from iter_069 checkpoint; optional replay/optimizer reset as described in the doc.

---

## 6. Files touched (checklist)

| File | Action |
|------|--------|
| `src/mcts/alphazero_mcts_optimized.py` | Modified: new helper, new config, MCTSNode slots/methods, select_action signature, expand paths, traversal, root stats, factory |
| `configs/config_best.yaml` | Modified: dynamic_score_utility_weight, progressive_widening block, win_first.gate_dsu_enabled, dynamic_score_utility_weight_schedule |
| `tools/ab_test_widening_iter069.py` | Created: A/B script |
| `docs/RUN_WIDENING_ITER069.md` | Created: how to run + config diff summary |
| `docs/CHANGES_WIDENING_ITER069_DETAILED.md` | Created: this in-depth changelog |

No changes to: legal move generation, 2026 action encoding, value targets, training loop (except via config), or GPU inference server (it already uses `dynamic_score_utility_weight` from config).
