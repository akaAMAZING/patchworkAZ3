# EXCERPT from src/training/selfplay_optimized.py
# How values and score_margins are assigned per stored position after a game.
# Full path in repo: src/training/selfplay_optimized.py

# After game ends: score0, score1, terminal_winner, final_score_diff are computed.
# For each stored position i (with stored_players[i] = to_move at that position):

# 1) value_and_score_from_scores(score0, score1, winner, to_move=p) returns:
#    z_value: +1 (win), -1 (loss), 0 (tie)
#    z_score: tanh((to_move_score - opp_score) / 30.0)

# 2) score_margins[i] = z_score   # always the tanh-normalised margin

# 3) values[i] is set as follows:
#    If q_value_weight > 0 and move_root_qs is available:
#        q_clamped = clamp(root_q from MCTS, -1, 1)
#        values[i] = (1 - q_value_weight) * z_value + q_value_weight * q_clamped
#    Else:
#        values[i] = z_value   # pure binary outcome

# So: the VALUE target fed to the value head is always binary (or mix with Q);
# the SCORE target is always tanh(margin/30). There is no "lose by less" in values.

# Relevant code (lines ~528-556):
"""
        # Fill values and score_margins per stored state's to-move perspective
        # value: strictly 1.0 (Win), -1.0 (Loss), 0.0 (Tie)
        # score_margins: integer margin (Current Player Score - Opponent Score)
        # Q-value mixing (Oracle Part 4): blend MCTS root Q-value with game outcome for value only.
        q_w = self.q_value_weight
        aug_factor = 1 if self.store_canonical_only else (4 if self.augmentation == "flip" else (8 if self.augmentation == "d4" else 1))
        for i, p in enumerate(stored_players):
            z_value, z_score = value_and_score_from_scores(
                score0=int(score0),
                score1=int(score1),
                winner=int(terminal_winner),
                to_move=int(p),
            )
            score_margins[i] = z_score
            if q_w > 0 and move_root_qs:
                move_idx = i // aug_factor
                if move_idx < len(move_root_qs):
                    q = move_root_qs[move_idx]
                    q_clamped = max(-1.0, min(1.0, q))
                    values[i] = (1.0 - q_w) * z_value + q_w * q_clamped
                else:
                    values[i] = z_value
            else:
                values[i] = z_value
"""

# Imports used: value_and_score_from_scores from src.training.value_targets
