"""
League-Based Training System for Patchwork AlphaZero

Addresses non-transitive cycling (rock-paper-scissors dynamics) by maintaining
a pool of historical checkpoints and using Prioritized Fictitious Self-Play
(PFSP) opponent sampling, robustness-based promotion gating, and explicit
non-transitivity tracking via a payoff matrix.

Components:
  - PFSPSampler: selects opponents biased toward matchups where the candidate
    is weakest (near-50% or below), preventing blind spots.
  - PayoffMatrix: rolling winrate matrix over a pool of checkpoints, with
    non-transitivity diagnostics (cycle counting, exploitability proxy).
  - AnchorSet: curated subset of pool maximizing diversity (covers old strategies).
  - PromotionGate: candidate must beat previous best AND meet robustness
    thresholds vs an anchor suite (mean + worst-case winrate).
  - LeagueManager: orchestrates pool, anchors, PFSP, gating, and exploiters.

Design:
  - Single-GPU friendly: no extra processes for exploiter training.
  - Config-driven: all thresholds and percentages are tunable.
  - State is persisted to league_state.json for crash recovery.
  - Integrates with existing Evaluator for match play.
"""

import json
import logging
import math
import os
import random
import re
import shutil
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.training.run_layout import committed_dir

logger = logging.getLogger(__name__)


# =========================================================================
# Configuration
# =========================================================================

@dataclass
class LeagueConfig:
    """All league hyperparameters with sensible defaults for single-GPU."""

    # --- PFSP opponent sampling ---
    pfsp_alpha: float = 2.0           # Exponent for PFSP weighting: (1 - |0.5 - w|)^alpha
    pfsp_uniform_prob: float = 0.10   # Probability of uniform random pool sample (exploration)

    # --- Self-play matchup composition ---
    sp_frac_vs_best: float = 0.40     # Fraction of games vs best_model
    sp_frac_vs_pfsp: float = 0.50     # Fraction of games vs PFSP-selected pool opponents
    sp_frac_vs_uniform: float = 0.10  # Fraction of games vs uniform random pool

    # --- Promotion gating ---
    gate_threshold: float = 0.55      # WR vs previous best to pass gate (1)
    suite_threshold: float = 0.52     # Mean WR vs anchor suite to pass gate (2a)
    worst_threshold: float = 0.40     # Worst-case WR vs any anchor to pass gate (2b)
    suite_eval_games: int = 10        # Games per anchor opponent in suite evaluation
    gate_eval_games: int = 20         # Games vs best_model for gate check

    # --- Pool management ---
    max_pool_size: int = 30           # Maximum historical checkpoints in pool
    anchor_size: int = 6              # Number of anchors in evaluation suite
    anchor_refresh_interval: int = 5  # Re-select anchors every N iterations

    # --- Payoff matrix ---
    payoff_max_models: int = 30       # Max models tracked in payoff matrix

    # --- Exploiter ---
    exploiter_enabled: bool = False   # Whether to train an exploiter agent
    exploiter_train_games: int = 100  # Games to train exploiter per iteration
    exploiter_temp_boost: float = 1.5 # Temperature multiplier for exploiter exploration

    # --- Replay buffer recency bias ---
    recency_newest_frac: float = 0.70 # Fraction of training samples from newest 15% of buffer
    recency_newest_window: float = 0.15  # What fraction of buffer counts as "newest"

    @classmethod
    def from_config(cls, config: dict) -> "LeagueConfig":
        """Extract league config from full training config dict."""
        lc = config.get("league", {}) or {}
        kwargs = {}
        for f in cls.__dataclass_fields__:
            if f in lc:
                kwargs[f] = type(cls.__dataclass_fields__[f].default)(lc[f])
        return cls(**kwargs)


# =========================================================================
# PFSP Opponent Sampler
# =========================================================================

class PFSPSampler:
    """Prioritized Fictitious Self-Play opponent selection.

    Maintains estimated winrates and samples opponents with probability
    proportional to f(winrate), emphasizing matchups where the candidate
    is weakest. Includes a uniform exploration component.

    Winrate estimates use an exponential moving average (EMA) updated
    after each evaluation batch.
    """

    def __init__(self, alpha: float = 2.0, uniform_prob: float = 0.10, ema_decay: float = 0.3):
        self.alpha = alpha
        self.uniform_prob = max(0.0, min(1.0, uniform_prob))
        self.ema_decay = ema_decay
        # winrates[opponent_id] = estimated P(candidate beats opponent)
        self.winrates: Dict[str, float] = {}
        # counts[opponent_id] = number of games used to estimate winrate
        self.counts: Dict[str, int] = {}

    def update_winrate(self, opponent_id: str, wins: int, games: int) -> None:
        """Update estimated winrate vs opponent using EMA."""
        if games <= 0:
            return
        observed_wr = wins / games
        if opponent_id not in self.winrates:
            self.winrates[opponent_id] = observed_wr
            self.counts[opponent_id] = games
        else:
            old = self.winrates[opponent_id]
            # EMA: blend old estimate with new observation
            self.winrates[opponent_id] = (1 - self.ema_decay) * old + self.ema_decay * observed_wr
            self.counts[opponent_id] += games

    def set_winrate(self, opponent_id: str, winrate: float, count: int = 1) -> None:
        """Directly set winrate estimate (e.g., from payoff matrix)."""
        self.winrates[opponent_id] = winrate
        self.counts[opponent_id] = count

    def pfsp_weight(self, winrate: float) -> float:
        """Compute PFSP sampling weight. Higher weight = more likely to be sampled.

        Uses: weight = (1 - |0.5 - w|)^alpha
        This peaks at w=0.5 (hardest opponents) and falls off for easy/impossible ones.
        """
        deviation = abs(0.5 - winrate)
        return max(1e-6, (1.0 - deviation)) ** self.alpha

    def sample(self, pool_ids: List[str], rng: random.Random = None) -> str:
        """Sample an opponent from pool using PFSP weighting.

        With probability uniform_prob, samples uniformly (exploration).
        Otherwise, samples proportional to pfsp_weight(winrate).
        """
        if not pool_ids:
            raise ValueError("Cannot sample from empty pool")

        if rng is None:
            rng = random.Random()

        if len(pool_ids) == 1:
            return pool_ids[0]

        # Uniform exploration
        if rng.random() < self.uniform_prob:
            return rng.choice(pool_ids)

        # PFSP weighting
        weights = []
        for pid in pool_ids:
            wr = self.winrates.get(pid, 0.5)  # Default 50% for unknown
            weights.append(self.pfsp_weight(wr))

        total = sum(weights)
        if total <= 0:
            return rng.choice(pool_ids)

        probs = [w / total for w in weights]
        return rng.choices(pool_ids, weights=probs, k=1)[0]

    def sample_batch(self, pool_ids: List[str], n: int, rng: random.Random = None) -> List[str]:
        """Sample n opponents (with replacement) from pool."""
        if rng is None:
            rng = random.Random()
        return [self.sample(pool_ids, rng) for _ in range(n)]

    def get_state(self) -> dict:
        return {"winrates": dict(self.winrates), "counts": dict(self.counts)}

    def load_state(self, state: dict) -> None:
        self.winrates = {k: float(v) for k, v in state.get("winrates", {}).items()}
        self.counts = {k: int(v) for k, v in state.get("counts", {}).items()}


# =========================================================================
# Payoff Matrix
# =========================================================================

class PayoffMatrix:
    """Rolling winrate matrix over a pool of model checkpoints.

    Tracks pairwise winrates and provides non-transitivity diagnostics.
    """

    def __init__(self, max_models: int = 30):
        self.max_models = max_models
        # model_ids in order of addition
        self.model_ids: List[str] = []
        # wins[i][j] = number of wins of model_ids[i] vs model_ids[j]
        self.wins: Dict[str, Dict[str, int]] = {}
        # games[i][j] = total games between model_ids[i] and model_ids[j]
        self.games: Dict[str, Dict[str, int]] = {}

    def add_model(self, model_id: str) -> None:
        """Register a new model in the matrix."""
        if model_id in self.wins:
            return
        self.model_ids.append(model_id)
        self.wins[model_id] = {}
        self.games[model_id] = {}
        # Evict oldest if over capacity
        while len(self.model_ids) > self.max_models:
            old = self.model_ids.pop(0)
            self.wins.pop(old, None)
            self.games.pop(old, None)
            for mid in self.model_ids:
                self.wins.get(mid, {}).pop(old, None)
                self.games.get(mid, {}).pop(old, None)

    def record_result(self, model_a: str, model_b: str, a_wins: int, total_games: int) -> None:
        """Record match results: model_a won a_wins out of total_games vs model_b."""
        for m in (model_a, model_b):
            if m not in self.wins:
                self.add_model(m)

        self.wins.setdefault(model_a, {})
        self.games.setdefault(model_a, {})
        self.wins.setdefault(model_b, {})
        self.games.setdefault(model_b, {})

        self.wins[model_a][model_b] = self.wins[model_a].get(model_b, 0) + a_wins
        self.games[model_a][model_b] = self.games[model_a].get(model_b, 0) + total_games
        b_wins = total_games - a_wins
        self.wins[model_b][model_a] = self.wins[model_b].get(model_a, 0) + b_wins
        self.games[model_b][model_a] = self.games[model_b].get(model_a, 0) + total_games

    def get_winrate(self, model_a: str, model_b: str) -> Optional[float]:
        """Get P(model_a beats model_b). None if no data."""
        g = self.games.get(model_a, {}).get(model_b, 0)
        if g == 0:
            return None
        w = self.wins.get(model_a, {}).get(model_b, 0)
        return w / g

    def get_winrates_vs_all(self, model_id: str) -> Dict[str, float]:
        """Get winrates of model_id vs all opponents with data."""
        result = {}
        for opp in self.model_ids:
            if opp == model_id:
                continue
            wr = self.get_winrate(model_id, opp)
            if wr is not None:
                result[opp] = wr
        return result

    def get_mean_winrate(self, model_id: str, opponents: Optional[List[str]] = None) -> Optional[float]:
        """Mean winrate of model vs specified opponents (or all with data)."""
        if opponents is None:
            opponents = [m for m in self.model_ids if m != model_id]
        wrs = []
        for opp in opponents:
            wr = self.get_winrate(model_id, opp)
            if wr is not None:
                wrs.append(wr)
        return float(np.mean(wrs)) if wrs else None

    def get_worst_winrate(self, model_id: str, opponents: Optional[List[str]] = None) -> Tuple[Optional[float], Optional[str]]:
        """Worst winrate of model vs specified opponents. Returns (winrate, opponent_id)."""
        if opponents is None:
            opponents = [m for m in self.model_ids if m != model_id]
        worst_wr = None
        worst_opp = None
        for opp in opponents:
            wr = self.get_winrate(model_id, opp)
            if wr is not None:
                if worst_wr is None or wr < worst_wr:
                    worst_wr = wr
                    worst_opp = opp
        return worst_wr, worst_opp

    def count_cycles(self, sample_size: int = 200, rng: random.Random = None) -> int:
        """Count non-transitive cycles among sampled triples.

        A cycle is A > B > C > A where > means winrate > 0.5.
        Returns number of cycles found in sampled triples.
        """
        if rng is None:
            rng = random.Random(42)

        models = [m for m in self.model_ids if any(
            self.games.get(m, {}).get(o, 0) > 0 for o in self.model_ids if o != m
        )]
        if len(models) < 3:
            return 0

        cycles = 0
        for _ in range(sample_size):
            triple = rng.sample(models, 3)
            a, b, c = triple
            wr_ab = self.get_winrate(a, b)
            wr_bc = self.get_winrate(b, c)
            wr_ca = self.get_winrate(c, a)
            if wr_ab is None or wr_bc is None or wr_ca is None:
                continue
            # Check both directions: A>B>C>A or A<B<C<A
            if wr_ab > 0.5 and wr_bc > 0.5 and wr_ca > 0.5:
                cycles += 1
            elif wr_ab < 0.5 and wr_bc < 0.5 and wr_ca < 0.5:
                cycles += 1
        return cycles

    def exploitability_proxy(self) -> float:
        """Simple exploitability proxy: max regret over all models.

        For each model, compute the best response winrate against it
        (the opponent that beats it worst). The exploitability is the
        maximum such worst-case loss across all models.

        Returns 0.0 (no data) to 1.0 (maximally exploitable).
        """
        max_exploit = 0.0
        for model in self.model_ids:
            # Find the opponent that exploits this model the most
            worst = 1.0
            for opp in self.model_ids:
                if opp == model:
                    continue
                wr = self.get_winrate(model, opp)
                if wr is not None:
                    worst = min(worst, wr)
            if worst < 1.0:
                max_exploit = max(max_exploit, 1.0 - worst)
        return max_exploit

    def to_numpy(self) -> Tuple[np.ndarray, List[str]]:
        """Export as numpy matrix. matrix[i][j] = P(model_i beats model_j)."""
        n = len(self.model_ids)
        matrix = np.full((n, n), 0.5)
        for i, mi in enumerate(self.model_ids):
            for j, mj in enumerate(self.model_ids):
                if i == j:
                    continue
                wr = self.get_winrate(mi, mj)
                if wr is not None:
                    matrix[i, j] = wr
        return matrix, list(self.model_ids)

    def get_state(self) -> dict:
        return {
            "model_ids": list(self.model_ids),
            "wins": {k: dict(v) for k, v in self.wins.items()},
            "games": {k: dict(v) for k, v in self.games.items()},
        }

    def load_state(self, state: dict) -> None:
        self.model_ids = list(state.get("model_ids", []))
        self.wins = {k: dict(v) for k, v in state.get("wins", {}).items()}
        self.games = {k: dict(v) for k, v in state.get("games", {}).items()}


# =========================================================================
# Anchor Set Selection
# =========================================================================

def select_anchors(
    payoff: PayoffMatrix,
    pool_ids: List[str],
    best_id: str,
    prev_best_id: Optional[str],
    anchor_size: int = 6,
    rng: random.Random = None,
) -> List[str]:
    """Select diverse anchor set from pool for evaluation suite.

    Always includes:
      - best_model
      - previous_best (if available)
    Fills remaining slots using a greedy diversity heuristic:
      maximize minimum pairwise payoff difference among selected anchors.
    Falls back to: models that most exploit the current best.

    Args:
        payoff: PayoffMatrix with pairwise data.
        pool_ids: All available model IDs in pool.
        best_id: Current best model ID.
        prev_best_id: Previous best model ID (may be None).
        anchor_size: Target number of anchors.
        rng: Random number generator.

    Returns:
        List of model IDs forming the anchor set.
    """
    if rng is None:
        rng = random.Random(42)

    anchors = []
    # Always include best and previous best
    if best_id in pool_ids:
        anchors.append(best_id)
    if prev_best_id and prev_best_id in pool_ids and prev_best_id not in anchors:
        anchors.append(prev_best_id)

    candidates = [p for p in pool_ids if p not in anchors]

    while len(anchors) < anchor_size and candidates:
        # Greedy: pick candidate maximizing min pairwise difference to existing anchors
        best_cand = None
        best_score = -1.0

        for c in candidates:
            min_diff = float("inf")
            for a in anchors:
                wr_ca = payoff.get_winrate(c, a)
                wr_ac = payoff.get_winrate(a, c)
                if wr_ca is not None and wr_ac is not None:
                    # Pairwise "distance" = |wr_ca - 0.5| + |wr_ac - 0.5|
                    diff = abs(wr_ca - 0.5) + abs(wr_ac - 0.5)
                    min_diff = min(min_diff, diff)
                elif wr_ca is not None:
                    min_diff = min(min_diff, abs(wr_ca - 0.5))
                else:
                    # No data: treat as moderately different
                    min_diff = min(min_diff, 0.3)

            if min_diff > best_score:
                best_score = min_diff
                best_cand = c

        if best_cand is not None:
            anchors.append(best_cand)
            candidates.remove(best_cand)
        else:
            # Fallback: random selection
            pick = rng.choice(candidates)
            anchors.append(pick)
            candidates.remove(pick)

    return anchors


# =========================================================================
# Self-Play Schedule
# =========================================================================

@dataclass
class SelfPlaySchedule:
    """Determines opponent distribution for self-play games in an iteration."""
    vs_best: int = 0           # Games vs best_model
    vs_pfsp: Dict[str, int] = field(default_factory=dict)  # opponent_id -> num_games
    vs_uniform: Dict[str, int] = field(default_factory=dict)  # opponent_id -> num_games

    @property
    def total_games(self) -> int:
        return self.vs_best + sum(self.vs_pfsp.values()) + sum(self.vs_uniform.values())


def create_selfplay_schedule(
    total_games: int,
    best_id: str,
    pool_ids: List[str],
    pfsp_sampler: PFSPSampler,
    config: LeagueConfig,
    rng: random.Random = None,
) -> SelfPlaySchedule:
    """Create the self-play matchup schedule for one iteration.

    Distributes total_games across:
      - vs best_model (stability)
      - vs PFSP-selected opponents (coverage)
      - vs uniform random pool (exploration)
    """
    if rng is None:
        rng = random.Random()

    schedule = SelfPlaySchedule()

    # Compute game counts per category
    n_best = max(1, int(total_games * config.sp_frac_vs_best))
    n_uniform = max(0, int(total_games * config.sp_frac_vs_uniform))
    n_pfsp = total_games - n_best - n_uniform
    if n_pfsp < 0:
        n_pfsp = 0
        n_best = total_games - n_uniform

    schedule.vs_best = n_best

    # PFSP sampling: only from pool excluding best (best is already covered)
    pfsp_pool = [p for p in pool_ids if p != best_id]
    if pfsp_pool and n_pfsp > 0:
        opponents = pfsp_sampler.sample_batch(pfsp_pool, n_pfsp, rng)
        for opp in opponents:
            schedule.vs_pfsp[opp] = schedule.vs_pfsp.get(opp, 0) + 1
    else:
        # No pool available: all PFSP games go to best
        schedule.vs_best += n_pfsp

    # Uniform sampling from full pool
    if pool_ids and n_uniform > 0:
        for _ in range(n_uniform):
            opp = rng.choice(pool_ids)
            schedule.vs_uniform[opp] = schedule.vs_uniform.get(opp, 0) + 1
    else:
        schedule.vs_best += n_uniform

    return schedule


# =========================================================================
# Promotion Gate
# =========================================================================

@dataclass
class GateResult:
    """Result of promotion gating evaluation."""
    passed: bool = False
    vs_best_wr: float = 0.0
    suite_mean_wr: float = 0.0
    suite_worst_wr: float = 1.0
    suite_worst_opponent: str = ""
    regression_alerts: List[str] = field(default_factory=list)
    details: Dict = field(default_factory=dict)

    @property
    def reason(self) -> str:
        if self.passed:
            return "PASS"
        reasons = []
        if self.vs_best_wr < 0.55:
            reasons.append(f"vs_best={self.vs_best_wr:.1%}<55%")
        if self.suite_mean_wr < 0.52:
            reasons.append(f"suite_mean={self.suite_mean_wr:.1%}<52%")
        if self.suite_worst_wr < 0.40:
            reasons.append(f"suite_worst={self.suite_worst_wr:.1%}<40% vs {self.suite_worst_opponent}")
        if self.regression_alerts:
            reasons.append(f"regressions: {self.regression_alerts}")
        return "; ".join(reasons) if reasons else "UNKNOWN"


def evaluate_promotion(
    vs_best_wr: float,
    suite_winrates: Dict[str, float],
    config: LeagueConfig,
) -> GateResult:
    """Evaluate whether a candidate should be promoted to best.

    Two conditions must both pass:
      1) Candidate beats previous best by >= gate_threshold
      2) Candidate meets robustness thresholds vs anchor suite:
         a) mean winrate >= suite_threshold
         b) worst-case winrate >= worst_threshold

    Args:
        vs_best_wr: Win rate of candidate vs current best.
        suite_winrates: Dict of {anchor_id: winrate} from suite evaluation.
        config: League configuration.

    Returns:
        GateResult with pass/fail and diagnostics.
    """
    result = GateResult(vs_best_wr=vs_best_wr)

    # Condition 1: beats best
    gate_pass = vs_best_wr >= config.gate_threshold

    # Condition 2: robustness vs suite
    if suite_winrates:
        wrs = list(suite_winrates.values())
        result.suite_mean_wr = float(np.mean(wrs))
        worst_opp = min(suite_winrates, key=suite_winrates.get)
        result.suite_worst_wr = suite_winrates[worst_opp]
        result.suite_worst_opponent = worst_opp

        suite_pass = (
            result.suite_mean_wr >= config.suite_threshold
            and result.suite_worst_wr >= config.worst_threshold
        )

        # Regression alerts: any anchor where WR < 45%
        for opp, wr in suite_winrates.items():
            if wr < 0.45:
                result.regression_alerts.append(f"{opp}:{wr:.1%}")
    else:
        # No suite data: pass if we beat best (early iterations)
        suite_pass = True

    result.passed = gate_pass and suite_pass
    result.details = {
        "gate_threshold": config.gate_threshold,
        "suite_threshold": config.suite_threshold,
        "worst_threshold": config.worst_threshold,
        "gate_pass": gate_pass,
        "suite_pass": suite_pass,
    }
    return result


# =========================================================================
# League Manager
# =========================================================================

class LeagueManager:
    """Orchestrates the league-based training system.

    Manages:
      - Pool of historical checkpoints (model_id -> checkpoint_path)
      - Best model and main (candidate) model
      - Anchor set for evaluation suite
      - PFSP sampler for opponent selection
      - Payoff matrix for non-transitivity tracking
      - Optional exploiter snapshots
      - State persistence for crash recovery
    """

    def __init__(self, config: dict, run_root: Path):
        self.full_config = config
        self.config = LeagueConfig.from_config(config)
        self.run_root = Path(run_root)
        self.state_path = self.run_root / "league_state.json"

        # Pool: model_id -> checkpoint_path
        self.pool: Dict[str, str] = {}
        self.best_id: Optional[str] = None
        self.prev_best_id: Optional[str] = None
        self.main_id: Optional[str] = None

        # Anchors
        self.anchor_ids: List[str] = []
        self._last_anchor_refresh: int = -1

        # PFSP
        self.pfsp = PFSPSampler(
            alpha=self.config.pfsp_alpha,
            uniform_prob=self.config.pfsp_uniform_prob,
        )

        # Payoff matrix
        self.payoff = PayoffMatrix(max_models=self.config.payoff_max_models)

        # Exploiter
        self.exploiter_id: Optional[str] = None
        self.exploiter_path: Optional[str] = None

        # RNG
        self._rng = random.Random(config.get("seed", 42))

    # --- Pool management ---

    def model_id(self, iteration: int, tag: str = "main") -> str:
        """Generate a deterministic model ID."""
        return f"iter{iteration:03d}_{tag}"

    def add_to_pool(self, model_id: str, checkpoint_path: str) -> None:
        """Add a checkpoint to the pool."""
        self.pool[model_id] = str(checkpoint_path)
        self.payoff.add_model(model_id)

        # Evict oldest if over capacity
        if len(self.pool) > self.config.max_pool_size:
            pool_ids = list(self.pool.keys())
            # Never evict best, prev_best, or anchors
            protected = set(self.anchor_ids)
            if self.best_id:
                protected.add(self.best_id)
            if self.prev_best_id:
                protected.add(self.prev_best_id)

            for pid in pool_ids:
                if pid not in protected and len(self.pool) > self.config.max_pool_size:
                    del self.pool[pid]
                    logger.debug("Evicted %s from pool (over capacity)", pid)

    def get_pool_ids(self) -> List[str]:
        """All model IDs currently in the pool."""
        return list(self.pool.keys())

    def get_checkpoint_path(self, model_id: str) -> Optional[str]:
        """Get checkpoint path for a model ID.

        Resolves stale staging paths (pool stores paths from before commit) to
        committed paths, since staging is moved to committed during commit.
        """
        path = self.pool.get(model_id)
        if path and Path(path).exists():
            return path
        # Resolve from model_id (e.g. iter000_main -> committed/iter_000/iteration_000.pt)
        m = re.match(r"iter(\d+)(?:_\w+)?$", model_id)
        if m:
            iteration = int(m.group(1))
            committed_ckpt = committed_dir(self.run_root, iteration) / f"iteration_{iteration:03d}.pt"
            if committed_ckpt.exists():
                resolved = str(committed_ckpt)
                if model_id in self.pool:
                    self.pool[model_id] = resolved  # Fix stale path for future calls
                return resolved
        return path

    # --- Anchor management ---

    def refresh_anchors(self, iteration: int, force: bool = False) -> List[str]:
        """Recompute anchor set if due."""
        if (not force
                and self._last_anchor_refresh >= 0
                and (iteration - self._last_anchor_refresh) < self.config.anchor_refresh_interval):
            return self.anchor_ids

        pool_ids = self.get_pool_ids()
        if not pool_ids:
            self.anchor_ids = []
            return self.anchor_ids

        self.anchor_ids = select_anchors(
            payoff=self.payoff,
            pool_ids=pool_ids,
            best_id=self.best_id or "",
            prev_best_id=self.prev_best_id,
            anchor_size=min(self.config.anchor_size, len(pool_ids)),
            rng=self._rng,
        )
        self._last_anchor_refresh = iteration
        logger.info("Refreshed anchor set (%d anchors): %s", len(self.anchor_ids), self.anchor_ids)
        return self.anchor_ids

    # --- PFSP integration ---

    def update_pfsp_from_payoff(self, candidate_id: str) -> None:
        """Sync PFSP winrates from payoff matrix for current candidate."""
        wrs = self.payoff.get_winrates_vs_all(candidate_id)
        for opp, wr in wrs.items():
            self.pfsp.set_winrate(opp, wr, self.payoff.games.get(candidate_id, {}).get(opp, 1))

    def create_schedule(self, total_games: int) -> SelfPlaySchedule:
        """Create self-play schedule for current iteration."""
        pool_ids = self.get_pool_ids()
        return create_selfplay_schedule(
            total_games=total_games,
            best_id=self.best_id or "",
            pool_ids=pool_ids,
            pfsp_sampler=self.pfsp,
            config=self.config,
            rng=self._rng,
        )

    # --- Promotion ---

    def evaluate_candidate(
        self,
        candidate_id: str,
        vs_best_wr: float,
        suite_winrates: Dict[str, float],
    ) -> GateResult:
        """Run full promotion evaluation for a candidate."""
        return evaluate_promotion(vs_best_wr, suite_winrates, self.config)

    def promote(self, candidate_id: str, checkpoint_path: str) -> None:
        """Promote candidate to best model."""
        self.prev_best_id = self.best_id
        self.best_id = candidate_id
        self.add_to_pool(candidate_id, checkpoint_path)
        logger.info("PROMOTED %s to best (prev=%s)", candidate_id, self.prev_best_id)

    # --- Diagnostics ---

    def get_diagnostics(self, candidate_id: Optional[str] = None) -> Dict:
        """Compute league diagnostics for logging."""
        diag: Dict = {
            "pool_size": len(self.pool),
            "best_id": self.best_id,
            "anchor_ids": list(self.anchor_ids),
            "anchor_count": len(self.anchor_ids),
        }

        if candidate_id:
            wrs = self.payoff.get_winrates_vs_all(candidate_id)
            if wrs:
                diag["candidate_mean_wr"] = float(np.mean(list(wrs.values())))
                worst_opp = min(wrs, key=wrs.get)
                diag["candidate_worst_wr"] = wrs[worst_opp]
                diag["candidate_worst_opponent"] = worst_opp

            # vs best
            if self.best_id:
                wr_best = self.payoff.get_winrate(candidate_id, self.best_id)
                diag["candidate_vs_best_wr"] = wr_best

        # Non-transitivity
        diag["cycles_in_200_triples"] = self.payoff.count_cycles(200, self._rng)
        diag["exploitability_proxy"] = self.payoff.exploitability_proxy()

        return diag

    def log_diagnostics(self, iteration: int, candidate_id: Optional[str] = None) -> Dict:
        """Log diagnostics to logger and return dict for saving."""
        diag = self.get_diagnostics(candidate_id)

        logger.info("=" * 60)
        logger.info("  LEAGUE DIAGNOSTICS (iter %03d)", iteration)
        logger.info("=" * 60)
        logger.info("  Pool: %d models | Best: %s | Anchors: %s",
                     diag["pool_size"], diag["best_id"], diag["anchor_ids"])

        if candidate_id:
            if "candidate_vs_best_wr" in diag and diag["candidate_vs_best_wr"] is not None:
                logger.info("  Candidate vs best: %.1f%%", diag["candidate_vs_best_wr"] * 100)
            if "candidate_mean_wr" in diag:
                logger.info("  Candidate vs pool mean: %.1f%% | worst: %.1f%% (vs %s)",
                             diag.get("candidate_mean_wr", 0) * 100,
                             diag.get("candidate_worst_wr", 0) * 100,
                             diag.get("candidate_worst_opponent", "?"))

        logger.info("  Non-transitivity: %d cycles / 200 triples | exploitability=%.3f",
                     diag.get("cycles_in_200_triples", 0),
                     diag.get("exploitability_proxy", 0))
        logger.info("=" * 60)

        return diag

    # --- Reporting ---

    def save_payoff_csv(self, path: Path) -> None:
        """Save payoff matrix as CSV for analysis."""
        matrix, ids = self.payoff.to_numpy()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        lines = ["," + ",".join(ids)]
        for i, mid in enumerate(ids):
            row = [mid] + [f"{matrix[i, j]:.3f}" for j in range(len(ids))]
            lines.append(",".join(row))

        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        os.replace(tmp, path)

    def save_diagnostics_json(self, path: Path, iteration: int, candidate_id: Optional[str] = None) -> None:
        """Save diagnostics as JSON."""
        diag = self.get_diagnostics(candidate_id)
        diag["iteration"] = iteration

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Append to JSONL
        tmp = path.with_suffix(path.suffix + ".tmp")
        # Read existing if present
        existing = ""
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                existing = f.read()
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(existing)
            f.write(json.dumps(diag, default=_json_default) + "\n")
        os.replace(tmp, path)

    # --- State persistence ---

    def save_state(self) -> None:
        """Persist league state for crash recovery."""
        state = {
            "pool": dict(self.pool),
            "best_id": self.best_id,
            "prev_best_id": self.prev_best_id,
            "main_id": self.main_id,
            "anchor_ids": list(self.anchor_ids),
            "last_anchor_refresh": self._last_anchor_refresh,
            "pfsp": self.pfsp.get_state(),
            "payoff": self.payoff.get_state(),
            "exploiter_id": self.exploiter_id,
            "exploiter_path": self.exploiter_path,
        }
        tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, default=_json_default)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.state_path)

    def load_state(self) -> bool:
        """Restore league state from disk. Returns True on success."""
        if not self.state_path.exists():
            return False
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                state = json.load(f)

            self.pool = {k: str(v) for k, v in state.get("pool", {}).items()}
            self.best_id = state.get("best_id")
            self.prev_best_id = state.get("prev_best_id")
            self.main_id = state.get("main_id")
            self.anchor_ids = list(state.get("anchor_ids", []))
            self._last_anchor_refresh = int(state.get("last_anchor_refresh", -1))

            pfsp_state = state.get("pfsp")
            if pfsp_state:
                self.pfsp.load_state(pfsp_state)

            payoff_state = state.get("payoff")
            if payoff_state:
                self.payoff.load_state(payoff_state)

            self.exploiter_id = state.get("exploiter_id")
            self.exploiter_path = state.get("exploiter_path")

            # Validate pool paths exist; resolve stale staging paths to committed
            valid_pool = {}
            for mid, path in self.pool.items():
                if Path(path).exists():
                    valid_pool[mid] = path
                else:
                    # Resolve to committed path (staging is moved on commit)
                    m = re.match(r"iter(\d+)(?:_\w+)?$", mid)
                    if m:
                        iteration = int(m.group(1))
                        committed_ckpt = committed_dir(self.run_root, iteration) / f"iteration_{iteration:03d}.pt"
                        if committed_ckpt.exists():
                            valid_pool[mid] = str(committed_ckpt)
                            logger.debug("League pool: resolved %s to committed path", mid)
                            continue
                    logger.warning("League pool: missing checkpoint %s for %s, removing", path, mid)
            self.pool = valid_pool

            logger.info(
                "League state restored: pool=%d, best=%s, anchors=%d",
                len(self.pool), self.best_id, len(self.anchor_ids),
            )
            return True
        except Exception as e:
            logger.warning("Failed to restore league state: %s", e)
            return False


def _json_default(obj):
    """JSON serializer for non-standard types."""
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)
