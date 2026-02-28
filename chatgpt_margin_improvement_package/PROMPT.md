# Task: Optimize the codebase for margin in all situations

## Goal

We want the agent to **maximize score margin from its own perspective in every situation**:

- **When winning**: win by as much as possible (maximize positive margin).
- **When losing**: lose by as little as possible (minimize the negative margin — i.e. maximize the margin from the current player’s perspective, which is negative but closer to 0).

The hypothesis: by training the agent to always optimize for “largest possible margin from my perspective” (whether that’s +50 or -5 instead of -78), it will both win more games and, when it loses, lose by smaller margins. So the single objective is: **maximize (my_score − opponent_score)** in expectation, which implies winning more and losing by less.

---

## Current behavior (what to change)

The codebase uses a **KataGo-style dual-head** setup:

1. **Value head**
   - **Target**: strictly binary: +1 (win), -1 (loss), 0 (tie).
   - **Training**: MSE between value head output and this binary target.
   - **Effect**: The value head gets no signal about *how much* the agent won or lost. A loss by 1 and a loss by 78 both use target -1. So the agent is not encouraged to “lose by less.”

2. **Score head**
   - **Target**: tanh-normalised margin: `tanh((my_score - opp_score) / 30)` in (-1, 1).
   - **Training**: Huber loss in logit space on this target.
   - **Effect**: The score head *does* learn margin. But the **policy and value targets** are driven by:
     - Value target = binary win/loss (no margin when losing).
     - MCTS utility = value + dynamic_w * (score - root_score), so MCTS prefers “win by more” when ahead, but the **value backup** for terminal nodes still uses binary value + score utility; the value-head *training target* remains binary.

3. **MCTS**
   - Terminal backup: `utility = value + dynamic_w * (score - root_score)` where `value` is ±1/0 and `score` is tanh(margin/30). So when losing, a loss by 5 gives a *better* (less negative) utility than a loss by 78, and MCTS does prefer lines that “lose by less” during search. But the **training labels** for the value head are still binary (all losses → -1), so the network is not trained to predict “I will lose by 5” vs “I will lose by 78.”

4. **Self-play**
   - For each position, `value_and_score_from_scores()` is used to set:
     - `values[i]` = binary (+1 / -1 / 0), optionally mixed with root Q.
     - `score_margins[i]` = tanh(margin/30).
   - So the **value target** fed to the value head is never the margin; it’s always ±1/0.

**Summary of the gap**:  
The value head is trained only on win/loss/tie. So the agent has no gradient to “lose by less.” We want the agent to maximize margin in all cases, which means the **value target** (or the way we use value + score) should reflect “margin from my perspective” so that:
- Winning by more is better than winning by less.
- Losing by less is better than losing by more.

---

## Constraints and compatibility

- **MCTS**: Must remain consistent. Terminal backup and non-terminal evaluation use `utility = value + score_utility`. If we change what “value” means (e.g. to a margin-based target), we must either:
  - Change the value head to predict margin (or tanh margin) and possibly blend with a win-probability notion, or
  - Keep value head for win/loss and add a **single scalar target** that is “margin from my perspective” and train the value head (or a combined target) on that.
- **Replay data**: Existing HDF5 has `values` (float) and `score_margins` (tanh-normalised). Any change to the *semantics* of `values` (e.g. from ±1/0 to margin-based) will make old replays incompatible unless we support a migration or a “legacy” path.
- **Evaluation**: Eval currently uses win rate and sometimes average score margin. We still care about win rate; the hope is that margin-maximizing behavior improves it.
- **Scale**: Value head output is tanh, so in [-1, 1]. Score head already uses tanh(margin/30). Any new value target should stay in a comparable range to avoid gradient saturation.

---

## What we need from you

1. **Concrete design**
   - Propose a clear rule for the **value target** (and if needed the **score target**) so that:
     - The agent is trained to maximize margin from its own perspective in all situations (win by more, lose by less).
     - MCTS and self-play stay consistent (same formula for terminal value and for labeling).
   - Options to consider (you may propose others):
     - **Option A**: Replace value target with tanh(margin/30) for *all* terminal states (so value head learns “margin” directly; win/loss is implied by sign). MCTS would then use this same quantity (or a blend with current value) for backup.
     - **Option B**: Keep value head as win/loss but add a **combined loss** or **target** that combines value and score so that the gradient pushes “lose by less” (e.g. value target = sign(margin) * f(|margin|) so that -1 with small |margin| gets a less negative target).
     - **Option C**: Use a **single head** for “expected margin” (tanh-normalised) and derive win probability from it for logging/eval only.
   - Consider impact on: Q-value mixing (currently blends root Q with binary outcome), FPU, and any place that assumes value in {-1, 0, 1}.

2. **Exact code changes**
   - List every file and function to change, with:
     - Current behavior (one sentence).
     - New behavior (one sentence).
     - Suggested code edits (or clear pseudocode). Prefer minimal, localized changes.
   - Key files (see `context/` in this package):
     - `value_targets.py`: defines `terminal_value_from_scores` and `value_and_score_from_scores` (value and score targets).
     - `selfplay_optimized.py`: assigns `values` and `score_margins` per position using the above; Q-value mixing applies to `values`.
     - `alphazero_mcts_optimized.py`: `_get_terminal_value` and `_compute_score_utility`; backup uses utility = value + score_utility.
     - `model.py`: value loss (MSE on value target), score loss (Huber on score target); value head output is tanh.
     - Config: `value_loss_weight`, `score_loss_weight`, MCTS `static_score_utility_weight`, `dynamic_score_utility_weight`.

3. **Backward compatibility**
   - If new replays will have a different `values` semantics, either:
     - Propose a version flag or dataset attribute so old data is rejected or converted, or
     - Propose a transition that works with existing data (e.g. keep old value target as an optional branch).

4. **Testing / sanity**
   - Suggest one or two sanity checks (e.g. after the change: in self-play, positions that lead to “lose by 5” should have a less negative value target than “lose by 78”; or a simple unit test on `value_and_score_from_scores` or the new target function).

---

## Delivered context

The package includes:

- **PROMPT.md** (this file): task and constraints.
- **context/value_targets.py**: full file — defines binary value and tanh score margin.
- **context/model_value_and_score_loss.py**: value head output and loss (value MSE, score Huber).
- **context/selfplay_value_assignment.py**: how `values` and `score_margins` are filled per position; Q mixing.
- **context/mcts_utility_and_terminal.py**: MCTS config (score weights), `_compute_score_utility`, `_get_terminal_value`, backup convention.
- **context/config_snippet.yaml**: training and MCTS weights (value_weight, score_weight, static/dynamic score utility).

Use these to propose the minimal, consistent set of changes so that the agent maximizes margin in all situations (win by more, lose by less) and we can expect win rate to improve as a result.
