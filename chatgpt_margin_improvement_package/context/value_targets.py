"""
Shared terminal value utilities for self-play and MCTS.

KataGo Dual-Head Architecture:
  Value head:  strictly  1.0 (win), -1.0 (loss), 0.0 (tie)
  Score head:  tanh-normalised margin  tanh(margin / 30.0)  in (-1, 1)

  Raw integer margins are normalised via tanh so the score head and value
  head operate on the same numeric scale.  The divisor 30.0 was chosen so
  that a 30-point margin (a large Patchwork lead) maps to ~0.76, while
  small differences (~5 pts) map to ~0.16 — preserving meaningful gradient
  signal throughout.

  During MCTS the two signals are blended via KataGo's dual formula:
    utility = value + static_w * score + dynamic_w * (score - root_score)
  where root_score is the network's score prediction at the root of each search.
  KataGo self-play defaults: static_w=0.0, dynamic_w=0.3 (pure dynamic).
  Dynamic centering means MCTS always pushes to exceed the root's predicted margin.

  IMPORTANT: score targets stored in the replay buffer are tanh-normalised.
  Any existing replay data with raw integer targets is incompatible and must
  be discarded before training with this version.
"""

from __future__ import annotations
import math


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


_SCORE_NORMALISE_DIVISOR = 30.0


def value_and_score_from_scores(
    score0: int,
    score1: int,
    winner: int,
    to_move: int,
) -> tuple[float, float]:
    """Return ``(value, score_margin)`` for a terminal state.

    value:        +1 / -1 / 0   (binary win/loss/tie from to_move perspective)
    score_margin: tanh-normalised margin, tanh((to_move_score - opp_score) / 30.0)
                  in the range (-1, 1).

    Called by selfplay_optimized.py to label both heads independently.
    """
    value = terminal_value_from_scores(score0, score1, winner, to_move)
    if int(to_move) == 0:
        raw_margin = float(int(score0) - int(score1))
    else:
        raw_margin = float(int(score1) - int(score0))
    score_margin = math.tanh(raw_margin / _SCORE_NORMALISE_DIVISOR)
    return value, score_margin
