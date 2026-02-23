"""
Shared terminal value utilities for self-play and MCTS.

KataGo Dual-Head Architecture:
  Value head:  strictly  1.0 (win), -1.0 (loss), 0.0 (tie)
  Score head:  raw integer margin  (current_player_score - opponent_score)

  The old score_tanh(diff / scale) logic has been removed.  Instead the
  network learns win-probability (value head) and score margin (score head)
  as two independent targets.  During MCTS the two signals are blended
  via ``utility = value + score_utility_weight * score`` so the agent
  maximises margin while maintaining wins.
"""

from __future__ import annotations


def terminal_value_from_scores(
    score0: int,
    score1: int,
    winner: int,
    to_move: int,
) -> float:
    """Binary terminal value from ``to_move``'s perspective.

    Returns:
        +1.0 if to_move won, -1.0 if to_move lost, 0.0 on exact tie.

    CRITICAL: This function must be called identically in:
      - selfplay_optimized.py  (labeling training data via value_and_score_from_scores)
      - alphazero_mcts_optimized.py  (MCTS terminal backup)
    Any divergence will cause the value head to learn inconsistent targets.
    """
    if int(score0) == int(score1):
        return 0.0
    return 1.0 if int(winner) == int(to_move) else -1.0


def value_and_score_from_scores(
    score0: int,
    score1: int,
    winner: int,
    to_move: int,
) -> tuple[float, float]:
    """Return ``(value, score_margin)`` for a terminal state.

    value:        +1 / -1 / 0   (binary win/loss/tie from to_move perspective)
    score_margin: integer margin (to_move's score  minus  opponent's score)

    Called by selfplay_optimized.py to label both heads independently.
    """
    value = terminal_value_from_scores(score0, score1, winner, to_move)
    if int(to_move) == 0:
        score_margin = float(int(score0) - int(score1))
    else:
        score_margin = float(int(score1) - int(score0))
    return value, score_margin
