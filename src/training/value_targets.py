"""
Shared terminal value utilities for self-play and MCTS.

KataGo Clean Dual-Head Architecture:
  Value head:  pure binary outcome: +1.0 / -1.0
               Predicts P(win) — one job, no objective conflict.
               No draws in Patchwork (tied scores resolved by TIE_PLAYER).
  Score head:  tanh-normalised margin  tanh(margin / 30.0)  in (-1, 1)
               Predicts score margin — independently, via 201-bin distribution.

  Each head has ONE job. Combination happens ONLY in search:
    utility = value + static_w * score + dynamic_w * (score - root_score)
  This eliminates gradient conflict between win-probability and margin
  objectives that occurs with a blended value target.

  Raw integer margins are normalised via tanh so the score head operates
  on a bounded scale.  The divisor 30.0 was chosen so that a 30-point
  margin (a large Patchwork lead) maps to ~0.76, while small differences
  (~5 pts) map to ~0.16 — preserving meaningful gradient signal throughout.

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
        +1.0 if to_move won, -1.0 if to_move lost.

    Patchwork has NO draws: tied scores are resolved by the TIE_PLAYER
    tiebreaker (the player who most recently reached or passed the opponent
    on the time track).  The ``winner`` argument is already resolved by
    the game engine, so we use it directly.

    CRITICAL: This function must be called identically in:
      - selfplay_optimized.py  (labeling training data via value_and_score_from_scores)
      - alphazero_mcts_optimized.py  (MCTS terminal backup)
    Any divergence will cause the value head to learn inconsistent targets.
    """
    return 1.0 if int(winner) == int(to_move) else -1.0


_SCORE_NORMALISE_DIVISOR = 30.0


def value_and_score_from_scores(
    score0: int,
    score1: int,
    winner: int,
    to_move: int,
) -> tuple[float, float]:
    """Return ``(value, score_margin)`` for a terminal state.

    value:         pure binary outcome: +1.0 / -1.0
                   The value head's sole job is predicting P(win).
                   No blending with margin — that's the score head's job.
                   No 0.0 — Patchwork always has a winner (tiebreak on equal scores).
    score_margin:  tanh-normalised margin, tanh((to_move_score - opp_score) / 30.0)
                   in the range (-1, 1).

    Called by selfplay_optimized.py to label both heads independently.
    """
    binary = terminal_value_from_scores(score0, score1, winner, to_move)
    if int(to_move) == 0:
        raw_margin = float(int(score0) - int(score1))
    else:
        raw_margin = float(int(score1) - int(score0))
    score_margin = math.tanh(raw_margin / _SCORE_NORMALISE_DIVISOR)
    return binary, score_margin
