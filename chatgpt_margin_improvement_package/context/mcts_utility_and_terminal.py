# EXCERPT from src/mcts/alphazero_mcts_optimized.py
# MCTS score utility and terminal value; backup uses utility = value + score_utility.
# Full path in repo: src/mcts/alphazero_mcts_optimized.py

# --- MCTSConfig (lines ~172-179) ---
# utility = value + static_w * score + dynamic_w * (score - root_score)
# static_score_utility_weight: float = 0.0   # disabled in self-play
# dynamic_score_utility_weight: float = 0.3   # KataGo default
# score_utility_scale: float = 30.0           # tanh divisor; matches value_targets

# --- _compute_score_utility(score, is_root_player) (lines ~398-427) ---
# static = static_w * score
# root_s = _search_root_score if is_root_player else -_search_root_score
# dynamic = dynamic_w * (score - root_s)
# return static + dynamic

# --- _get_terminal_value(state, to_move) (lines ~981-1007) ---
# Used when backup reaches a terminal node. Returns utility from to_move perspective.
# 1) value = terminal_value_from_scores(score0, score1, winner, to_move)  # ±1 or 0
# 2) raw_margin = (score0 - score1) if to_move==0 else (score1 - score0)
# 3) score = tanh(raw_margin / score_utility_scale)
# 4) return value + _compute_score_utility(score, is_root)
# So terminal utility = binary_value + dynamic_w * (score - root_s). When losing,
# a smaller |margin| gives a less negative score term, so MCTS does prefer "lose by less"
# during search — but the VALUE part is still ±1, and that's what gets mixed into
# root Q and then (if q_value_weight) into value-head targets. The value-head
# training target itself remains binary from self-play.

# --- _backup_path(search_path, leaf_value, leaf_to_move) ---
# leaf_value is the utility (value + score_utility) from leaf's to_move perspective.
# Backed up to parents with sign flip for opponent. parent_node.update(action, value_for_parent).

# --- Root Q (lines ~506-514) ---
# root_q = sum(_total_value) / total_visits. So root Q is the average of (value + score_utility)
# over all simulations. This can be outside [-1,1]. When used in self-play for Q mixing,
# it is clamped to [-1, 1] before blending with z_value.

# Actual _get_terminal_value implementation:
"""
    def _get_terminal_value(self, state, to_move: int) -> float:
        score0 = compute_score_fast(state, 0)
        score1 = compute_score_fast(state, 1)
        winner = int(get_winner_fast(state))
        value = terminal_value_from_scores(
            score0=score0, score1=score1, winner=winner, to_move=int(to_move),
        )
        raw_margin = float(int(score0) - int(score1)) if int(to_move) == 0 else float(int(score1) - int(score0))
        score = math.tanh(raw_margin / self.config.score_utility_scale)
        is_root = (int(to_move) == int(self._root.to_move)) if self._root is not None else True
        return value + self._compute_score_utility(score, is_root)
"""

# terminal_value_from_scores is imported from src.training.value_targets (same as self-play).
