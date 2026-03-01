"""
Shared terminal value utilities for self-play and MCTS.

KataGo Dual-Head Architecture (Categorical Score Distribution):
  Value head:  strictly  1.0 (win), -1.0 (loss), 0.0 (tie)
  Score head:  categorical distribution over integer margins [-100, +100]
               (201 bins).  Training targets are Gaussian-smoothed soft labels
               centred on the actual integer margin.

  Raw integer margins are stored directly in the replay buffer.
  The training loop converts them to soft label distributions via
  make_gaussian_score_targets() before passing to the model.

  During MCTS the distribution is used to compute an expected saturating
  score utility:  E_s[sat((s - center) / scale)]  where
  sat(x) = (2/pi) * atan(x).

  IMPORTANT: score targets stored in the replay buffer are RAW INTEGER
  MARGINS (points).  Any old replay data with tanh-normalised targets
  is INCOMPATIBLE and must be discarded.
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
    """Return ``(value, score_margin_points)`` for a terminal state.

    value:               +1 / -1 / 0   (binary win/loss/tie from to_move perspective)
    score_margin_points: raw integer margin (to_move_score - opponent_score) in POINTS.
                         NOT normalised.  Stored directly in the replay buffer.

    Called by selfplay_optimized.py to label both heads independently.
    """
    value = terminal_value_from_scores(score0, score1, winner, to_move)
    if int(to_move) == 0:
        raw_margin = int(score0) - int(score1)
    else:
        raw_margin = int(score1) - int(score0)
    return value, float(raw_margin)
