"""
Glicko-2 rating system for model evaluation.
Sophisticated Elo-style system with:
  - Rating deviation (RD): uncertainty in each model's strength
  - Volatility: how much the rating fluctuates over time
  - Per-game updates: each game outcome updates both models

When glicko2 package is installed, uses it. Otherwise uses a built-in
Glicko-2 implementation.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Built-in Glicko-2 (no external deps)
# ---------------------------------------------------------------------------
# Based on http://www.glicko.net/glicko/glicko2.pdf
# Scale: display rating = 173.7178 * mu + 1500, display RD = 173.7178 * phi

_GLICKO_SCALE = 173.7178
_BASE_RATING = 1500
_TAU = 0.5  # volatility constraint


def _mu_from_rating(rating: float) -> float:
    return (rating - _BASE_RATING) / _GLICKO_SCALE


def _phi_from_rd(rd: float) -> float:
    return rd / _GLICKO_SCALE


def _rating_from_mu(mu: float) -> float:
    return mu * _GLICKO_SCALE + _BASE_RATING


def _rd_from_phi(phi: float) -> float:
    return phi * _GLICKO_SCALE


def _g(phi: float) -> float:
    """Glicko-2 g(φ) function."""
    return 1 / math.sqrt(1 + 3 * phi**2 / math.pi**2)


def _E(mu: float, mu_j: float, phi_j: float) -> float:
    """Expected outcome against opponent j."""
    return 1 / (1 + math.exp(-_g(phi_j) * (mu - mu_j)))


def _v(mu: float, opponent_mus: list[float], opponent_phis: list[float]) -> float:
    """Estimated variance of performance."""
    total = 0.0
    for mu_j, phi_j in zip(opponent_mus, opponent_phis):
        E = _E(mu, mu_j, phi_j)
        total += _g(phi_j) ** 2 * E * (1 - E)
    return 1 / total if total > 0 else 1e6


def _delta(
    mu: float,
    phi: float,
    opponent_mus: list[float],
    opponent_phis: list[float],
    outcomes: list[float],
) -> float:
    """Improvement term."""
    v = _v(mu, opponent_mus, opponent_phis)
    total = 0.0
    for mu_j, phi_j, s in zip(opponent_mus, opponent_phis, outcomes):
        total += _g(phi_j) * (s - _E(mu, mu_j, phi_j))
    return v * total


def _new_volatility(
    mu: float,
    phi: float,
    vol: float,
    opponent_mus: list[float],
    opponent_phis: list[float],
    outcomes: list[float],
    v: float,
) -> float:
    """Iterative volatility update (simplified one-step)."""
    delta = _delta(mu, phi, opponent_mus, opponent_phis, outcomes)
    a = math.log(vol**2)
    eps = 1e-6

    def f(x: float) -> float:
        ex = math.exp(x)
        numer = ex * (delta**2 - phi**2 - v - ex)
        denom = 2 * (phi**2 + v + ex) ** 2
        return numer / denom - (x - a) / (_TAU**2)

    B = a
    if delta**2 > phi**2 + v:
        B = math.log(delta**2 - phi**2 - v)
    else:
        k = 1
        while f(a - k * math.sqrt(_TAU**2)) < 0:
            k += 1
        B = a - k * math.sqrt(_TAU**2)

    A = a
    fA, fB = f(A), f(B)
    for _ in range(20):  # max iterations
        if abs(B - A) < eps:
            break
        C = A + (A - B) * fA / (fB - fA)
        fC = f(C)
        if fC * fB <= 0:
            A, fA = B, fB
        else:
            fA /= 2
        B, fB = C, fC
    return math.exp(A / 2)


def glicko2_update_builtin(
    rating: float,
    rd: float,
    vol: float,
    opponent_ratings: list[float],
    opponent_rds: list[float],
    outcomes: list[float],
) -> tuple[float, float, float]:
    """
    Update one player's Glicko-2 rating after matches.
    outcomes: 1.0 = win, 0.0 = loss (for each game).
    Returns (new_rating, new_rd, new_vol).
    """
    mu = _mu_from_rating(rating)
    phi = _phi_from_rd(rd)
    opp_mus = [_mu_from_rating(r) for r in opponent_ratings]
    opp_phis = [_phi_from_rd(r) for r in opponent_rds]

    phi_sq = math.sqrt(phi**2 + vol**2)
    v = _v(mu, opp_mus, opp_phis)
    new_vol = _new_volatility(mu, phi_sq, vol, opp_mus, opp_phis, outcomes, v)
    phi_sq = math.sqrt(phi**2 + new_vol**2)
    v = _v(mu, opp_mus, opp_phis)
    delta_val = _delta(mu, phi_sq, opp_mus, opp_phis, outcomes)
    new_phi = 1 / math.sqrt(1 / phi_sq**2 + 1 / v)
    total = sum(_g(ph) * (s - _E(mu, mj, ph)) for mj, ph, s in zip(opp_mus, opp_phis, outcomes))
    new_mu = mu + new_phi**2 * total

    return (
        _rating_from_mu(new_mu),
        _rd_from_phi(new_phi),
        new_vol,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def model_elo_id(path: Path) -> str:
    """Canonical ELO ID for a checkpoint path. Same logical model = same ID."""
    name = path.name
    # iteration_009.pt, best_model_iter009.pt -> iter_9
    m = re.search(r"iteration_(\d+)", name)
    if m:
        return f"iter_{int(m.group(1))}"
    m = re.search(r"best_model_iter(\d+)", name)
    if m:
        return f"iter_{int(m.group(1))}"
    if name == "best_model.pt":
        return "best"
    if name == "latest_model.pt":
        return "latest"
    return path.stem


def load_ratings(elo_file: Path) -> dict[str, dict[str, Any]]:
    """Load ELO ratings from JSON. Returns {model_id: {rating, rd, vol, games, ...}}."""
    if not elo_file.exists():
        return {}
    with open(elo_file, "r") as f:
        return json.load(f)


def save_ratings(elo_file: Path, ratings: dict[str, dict[str, Any]]) -> None:
    """Save ELO ratings to JSON."""
    elo_file.parent.mkdir(parents=True, exist_ok=True)
    with open(elo_file, "w") as f:
        json.dump(ratings, f, indent=2)


def get_or_init_player(
    ratings: dict[str, dict[str, Any]],
    model_id: str,
    default_rating: float = 1500.0,
    default_rd: float = 350.0,
) -> tuple[float, float, float]:
    """Get (rating, rd, vol) for a model. Initialize if new."""
    if model_id in ratings:
        r = ratings[model_id]
        return (r["rating"], r["rd"], r.get("vol", 0.06))
    return (default_rating, default_rd, 0.06)


def expected_score(rating_a: float, rating_b: float, rd_a: float = 350, rd_b: float = 350) -> float:
    """Expected win probability for A vs B (0 to 1). Elo-style formula."""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def wr_to_elo_gap(win_rate: float) -> float:
    """
    Standard Elo: implied rating gap from observed win rate (A - B).
    64.5% WR for A -> ~104 Elo. Uses P = 1/(1+10^(-d/400)) => d = 400*log10(p/(1-p)).
    """
    p = max(0.01, min(0.99, win_rate))
    return 400.0 * math.log10(p / (1.0 - p))


def update_ratings_standard_elo_batch(
    ratings: dict[str, dict[str, Any]],
    id_a: str,
    id_b: str,
    outcomes: list[float],
    k_factor: float = 32.0,
) -> tuple[dict[str, dict[str, Any]], tuple[float, float], tuple[float, float]]:
    """
    Standard Elo batch update. Produces gaps consistent with wr_to_elo_gap.
    One update per match with K chosen so gap ≈ 400*log10(wr/(1-wr)).
    """
    n = len(outcomes)
    if n == 0:
        ra = ratings.get(id_a, {}).get("rating", 1500.0)
        rb = ratings.get(id_b, {}).get("rating", 1500.0)
        rda = ratings.get(id_a, {}).get("rd", 350.0)
        rdb = ratings.get(id_b, {}).get("rd", 350.0)
        return ratings, (ra, rda), (rb, rdb)

    ra, rda, _ = get_or_init_player(ratings, id_a)
    rb, rdb, _ = get_or_init_player(ratings, id_b)

    wr = sum(outcomes) / n
    ea = expected_score(ra, rb)
    implied_gap = wr_to_elo_gap(wr)
    # K so that new_gap = implied_gap: (ra-rb) + 2*K*(wr-E) = implied_gap => K = (implied_gap - (ra-rb)) / (2*(wr-E))
    diff = wr - ea
    current_gap = ra - rb
    if abs(diff) < 1e-6:
        effective_k = 0.0
    else:
        effective_k = (implied_gap - current_gap) / (2.0 * diff)
        effective_k = min(max(effective_k, 0.0), 400.0)  # non-negative, cap for stability

    new_ra = ra + effective_k * (wr - ea)
    new_rb = rb + effective_k * ((1.0 - wr) - (1.0 - ea))

    games_a = ratings.get(id_a, {}).get("games", 0) + n
    games_b = ratings.get(id_b, {}).get("games", 0) + n

    ratings[id_a] = {
        "rating": round(new_ra, 1),
        "rd": round(rda, 1),
        "games": games_a,
    }
    ratings[id_b] = {
        "rating": round(new_rb, 1),
        "rd": round(rdb, 1),
        "games": games_b,
    }

    return ratings, (new_ra, rda), (new_rb, rdb)


def update_ratings_from_match(
    ratings: dict[str, dict[str, Any]],
    id_a: str,
    id_b: str,
    outcomes: list[float],
    margins: list[float],
) -> tuple[dict[str, dict[str, Any]], tuple[float, float], tuple[float, float]]:
    """
    Update both models' ratings after a match.
    outcomes: list of 1.0 (A wins) or 0.0 (B wins)
    margins: list of (A_score - B_score) for each game

    Returns (updated_ratings, (new_rating_a, new_rd_a), (new_rating_b, new_rd_b)).
    """
    ra, rda, vol_a = get_or_init_player(ratings, id_a)
    rb, rdb, vol_b = get_or_init_player(ratings, id_b)

    # A's outcomes: 1.0 = A won, 0.0 = A lost
    # B's outcomes: opposite
    outcomes_b = [1.0 - x for x in outcomes]

    # Opponent ratings/RDs for each game
    opp_ratings_a = [rb] * len(outcomes)
    opp_rds_a = [rdb] * len(outcomes)
    opp_ratings_b = [ra] * len(outcomes)
    opp_rds_b = [rda] * len(outcomes)

    new_ra, new_rda, new_vol_a = glicko2_update_builtin(
        ra, rda, vol_a, opp_ratings_a, opp_rds_a, outcomes
    )
    new_rb, new_rdb, new_vol_b = glicko2_update_builtin(
        rb, rdb, vol_b, opp_ratings_b, opp_rds_b, outcomes_b
    )

    games_a = ratings.get(id_a, {}).get("games", 0) + len(outcomes)
    games_b = ratings.get(id_b, {}).get("games", 0) + len(outcomes)

    ratings[id_a] = {
        "rating": round(new_ra, 1),
        "rd": round(new_rda, 1),
        "vol": round(new_vol_a, 4),
        "games": games_a,
    }
    ratings[id_b] = {
        "rating": round(new_rb, 1),
        "rd": round(new_rdb, 1),
        "vol": round(new_vol_b, 4),
        "games": games_b,
    }

    return (
        ratings,
        (new_ra, new_rda),
        (new_rb, new_rdb),
    )
