"""
Evaluation System

Evaluates model strength through:
- Matches against baselines (pure MCTS, previous models)
- SPRT (Sequential Probability Ratio Test) for statistically rigorous gating
- Elo rating tracking
- Win rate and score-margin analysis

Uses in-process game engine (same rules as self-play).

SPRT Implementation (Stockfish/KataGo-style):
  Given two hypotheses:
    H0: model win rate = p0 (not better)
    H1: model win rate = p1 (better)
  We compute the log-likelihood ratio after each game and stop early
  when we can confidently accept or reject the model. This saves
  significant compute compared to fixed-game evaluation.
"""

import logging
import math
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.mcts.alphazero_mcts_optimized import OptimizedAlphaZeroMCTS, MCTSConfig, create_optimized_mcts
from src.network.encoder import ActionEncoder, StateEncoder, GoldV2StateEncoder
from src.game.patchwork_engine import (
    apply_action,
    apply_action_unchecked,
    current_player,
    current_player_fast,
    compute_score,
    compute_score_fast,
    get_winner,
    get_winner_fast,
    legal_actions_list,
    legal_actions_fast,
    new_game,
    terminal,
    terminal_fast,
    AT_PASS,
    GameState,
)

logger = logging.getLogger(__name__)


# =========================================================================
# SPRT (Sequential Probability Ratio Test)
# =========================================================================

class SPRTResult:
    """Result of an SPRT evaluation."""

    __slots__ = ("accept", "reject", "inconclusive", "llr", "lower_bound",
                 "upper_bound", "games_played", "wins", "losses", "win_rate")

    def __init__(self):
        self.accept: bool = False
        self.reject: bool = False
        self.inconclusive: bool = False
        self.llr: float = 0.0
        self.lower_bound: float = 0.0
        self.upper_bound: float = 0.0
        self.games_played: int = 0
        self.wins: int = 0
        self.losses: int = 0
        self.win_rate: float = 0.0


def sprt_llr_update(win: bool, p0: float, p1: float) -> float:
    """Compute log-likelihood ratio increment for a single Bernoulli observation.

    Args:
        win: True if model won this game.
        p0: Null hypothesis win probability (e.g. 0.50).
        p1: Alternative hypothesis win probability (e.g. 0.55).

    Returns:
        LLR increment (positive favours H1, negative favours H0).
    """
    # Clamp to avoid log(0)
    p0 = max(min(p0, 1.0 - 1e-9), 1e-9)
    p1 = max(min(p1, 1.0 - 1e-9), 1e-9)
    if win:
        return math.log(p1 / p0)
    else:
        return math.log((1.0 - p1) / (1.0 - p0))


def sprt_bounds(alpha: float = 0.05, beta: float = 0.05) -> Tuple[float, float]:
    """Compute SPRT decision boundaries (Wald).

    Args:
        alpha: False positive rate (accept H1 when H0 is true).
        beta: False negative rate (accept H0 when H1 is true).

    Returns:
        (lower_bound, upper_bound): Reject H1 when LLR <= lower, accept H1 when LLR >= upper.
    """
    lower = math.log(beta / (1.0 - alpha))
    upper = math.log((1.0 - beta) / alpha)
    return lower, upper


def build_eval_schedule(
    num_games: int, base_seed: int, paired_eval: bool, game_offset: int = 0
) -> List[Tuple[int, bool]]:
    """
    Build deterministic (seed, model_plays_first) schedule for evaluation games.
    game_offset: offset for seed computation (for batched evals, use offset=games_played).
    With offset=N, seeds are base_seed + (N+i)*1000 for pair i — no overlap with offset=0.
    Paired eval preserves 50/50 color balance per batch (each pair: (seed,True), (seed,False)).
    """
    schedule: List[Tuple[int, bool]] = []
    off = int(game_offset)
    if paired_eval:
        pair_idx = 0
        while len(schedule) < int(num_games):
            seed = int(base_seed) + (off + pair_idx) * 1000
            schedule.append((seed, True))
            if len(schedule) < int(num_games):
                schedule.append((seed, False))
            pair_idx += 1
    else:
        for game_idx in range(int(num_games)):
            seed = int(base_seed) + (off + game_idx) * 1000
            schedule.append((seed, game_idx % 2 == 0))
    return schedule


# =========================================================================
# Pure MCTS Evaluator (no neural network)
# =========================================================================

class PureMCTSEvaluator:
    """
    Pure MCTS for evaluation (no neural network).
    Uses UCB selection + random rollout.
    """

    def __init__(self, simulations: int = 200, exploration: float = 1.4, seed: int = 0):
        self.simulations = simulations
        self.exploration = exploration
        self.seed = seed

    def get_move(self, state: GameState, seed_offset: int = 0) -> tuple:
        """Get best move using pure MCTS."""
        rng = random.Random(self.seed + seed_offset)

        legal = legal_actions_fast(state)
        if not legal:
            return (0, 0, 0, 0, 0, 0)  # AT_PASS as engine tuple

        if len(legal) == 1:
            return legal[0]

        # Simple visit counting MCTS
        visit_counts = {a: 0 for a in legal}
        value_sums = {a: 0.0 for a in legal}

        root_player = current_player_fast(state)

        for _ in range(self.simulations):
            total_visits = sum(visit_counts.values()) + 1

            best_action = None
            best_ucb = float("-inf")

            for action in legal:
                n = visit_counts[action]
                if n == 0:
                    ucb = float("inf")
                else:
                    q = value_sums[action] / n
                    ucb = q + self.exploration * math.sqrt(math.log(total_visits) / n)

                if ucb > best_ucb:
                    best_ucb = ucb
                    best_action = action

            next_state = apply_action_unchecked(state, best_action)
            value = self._rollout(next_state, root_player, rng)

            visit_counts[best_action] += 1
            value_sums[best_action] += value

        return max(visit_counts.items(), key=lambda x: x[1])[0]

    def _rollout(
        self,
        state: GameState,
        root_player: int,
        rng: random.Random,
        max_steps: int = 200,
    ) -> float:
        """Random rollout to terminal."""
        s = state

        for _ in range(max_steps):
            if terminal_fast(s):
                break

            actions = legal_actions_fast(s)
            if not actions:
                break

            # Slight bias against pass
            if len(actions) > 1 and int(actions[0][0]) == AT_PASS and rng.random() < 0.7:
                action = rng.choice(actions[1:])
            else:
                action = rng.choice(actions)

            s = apply_action_unchecked(s, action)

        return self._terminal_value(s, root_player)

    def _terminal_value(self, state: GameState, perspective: int) -> float:
        """Compute value from perspective player's viewpoint."""
        winner = get_winner_fast(state)
        return 1.0 if winner == perspective else -1.0


# =========================================================================
# Evaluator
# =========================================================================

class Evaluator:
    """Evaluates model strength through competitive matches."""

    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device
        enc_ver = str((config.get("data", {}) or {}).get("encoding_version", ""))
        in_ch = int((config.get("network", {}) or {}).get("input_channels", 56))
        if enc_ver.lower() in ("gold_v2_32ch", "gold_v2_multimodal") or in_ch in (32, 56):
            self.state_encoder = GoldV2StateEncoder()
        else:
            self.state_encoder = StateEncoder()
        self.action_encoder = ActionEncoder()

    def evaluate_vs_checkpoint(
        self,
        candidate_path: str,
        opponent_path: str,
        num_games: int = 10,
        game_offset: int = 0,
    ) -> Dict:
        """Evaluate candidate model against a specific checkpoint.

        Lighter-weight than evaluate_vs_baseline: loads both models once,
        plays num_games, and returns stats. Used by LeagueManager for
        suite evaluation and payoff matrix updates.

        Args:
            candidate_path: Path to candidate model checkpoint.
            opponent_path: Path to opponent model checkpoint.
            num_games: Number of games to play.
            game_offset: Offset for seed schedule (for batched/top-up evals).

        Returns:
            Stats dict with win_rate, model_wins, total_games, etc.
        """
        return self.evaluate_vs_baseline(
            model_path=candidate_path,
            baseline_type="previous_best",
            baseline_path=opponent_path,
            num_games=num_games,
            game_offset=game_offset,
        )

    def evaluate_vs_suite(
        self,
        candidate_path: str,
        suite: Dict[str, str],
        games_per_opponent: int = 10,
    ) -> Dict[str, float]:
        """Evaluate candidate against an anchor suite.

        Args:
            candidate_path: Path to candidate model checkpoint.
            suite: Dict of {model_id: checkpoint_path} for each anchor.
            games_per_opponent: Games to play per anchor.

        Returns:
            Dict of {model_id: win_rate} for each anchor evaluated.
        """
        results = {}
        for model_id, opp_path in suite.items():
            if not Path(opp_path).exists():
                logger.warning("Suite opponent %s missing: %s", model_id, opp_path)
                continue
            try:
                stats = self.evaluate_vs_checkpoint(
                    candidate_path, opp_path, num_games=games_per_opponent
                )
                results[model_id] = float(stats.get("win_rate", 0.5))
                logger.info(
                    "  Suite eval vs %s: WR=%.1f%% (%d games)",
                    model_id, results[model_id] * 100, stats.get("total_games", 0),
                )
            except Exception as e:
                logger.error("Suite eval failed vs %s: %s", model_id, e)
                results[model_id] = 0.5  # Assume 50% on failure
        return results

    def evaluate_vs_baseline(
        self,
        model_path: str,
        baseline_type: str = "pure_mcts",
        baseline_path: Optional[str] = None,
        num_games: int = 100,
        game_offset: int = 0,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        progress_interval: Optional[int] = None,
    ) -> Dict:
        """Evaluate model against a baseline."""
        if progress_interval is None:
            progress_interval = int(self.config.get("evaluation", {}).get("eval_progress_interval", 10))
        # UI-friendly: show filenames and resolve what "previous_best" actually points to.
        candidate_name = Path(model_path).name
        extra = f" ({baseline_path})" if baseline_path else ""
        if baseline_type == "previous_best" and baseline_path:
            baseline_label = f"previous_best={Path(baseline_path).name}"
        else:
            baseline_label = baseline_type
        logger.debug(f"Evaluating {candidate_name} vs {baseline_label}{extra}")

        if int(num_games) <= 0:
            logger.info("num_games <= 0; skipping evaluation.")
            return {
                "model_wins": 0,
                "baseline_wins": 0,
                "ties": 0,
                "total_games": 0,
                "win_rate": 0.0,
                "avg_game_length": 0.0,
                "avg_score_diff": 0.0,
                "avg_model_score_margin": 0.0,
                "avg_win_margin": 0.0,
                "avg_loss_margin": 0.0,
                "results": [],
            }

        from src.network.model import create_network, load_model_checkpoint, get_state_dict_for_inference

        model = create_network(self.config)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        state_dict = get_state_dict_for_inference(checkpoint, self.config, for_selfplay=False)
        load_model_checkpoint(model, state_dict)
        model.to(self.device)
        model.eval()

        eval_config = self.config["evaluation"]["eval_mcts"]
        model_mcts = create_optimized_mcts(
            model, self.config, self.device, self.state_encoder, self.action_encoder
        )
        model_mcts.config.simulations = int(eval_config["simulations"])
        model_mcts.config.cpuct = float(eval_config["cpuct"])

        baseline_mcts: Optional[OptimizedAlphaZeroMCTS] = None
        pure_mcts_evaluator: Optional[PureMCTSEvaluator] = None

        if baseline_type == "pure_mcts":
            baselines = self.config.get("evaluation", {}).get("eval_baselines", []) or []
            entry = next((b for b in baselines if b.get("type") == "pure_mcts"), None)
            baseline_sims = int(entry.get("simulations", 200)) if entry else int(eval_config.get("simulations", 200))

            logger.debug(f"Baseline pure_mcts simulations: {baseline_sims}")

            pure_mcts_evaluator = PureMCTSEvaluator(
                simulations=baseline_sims,
                exploration=1.4,
                seed=self.config.get("seed", 42),
            )

        elif baseline_type == "previous_best":
            if baseline_path is None:
                raise ValueError("baseline_path required for previous_best")

            baseline_model = create_network(self.config)
            baseline_checkpoint = torch.load(baseline_path, map_location=self.device, weights_only=False)
            baseline_state_dict = get_state_dict_for_inference(baseline_checkpoint, self.config, for_selfplay=False)
            load_model_checkpoint(baseline_model, baseline_state_dict)
            baseline_model.to(self.device)
            baseline_model.eval()

            baseline_mcts = create_optimized_mcts(
                baseline_model, self.config, self.device, self.state_encoder, self.action_encoder
            )
            baseline_mcts.config.simulations = int(eval_config["simulations"])
            baseline_mcts.config.cpuct = float(eval_config["cpuct"])

        else:
            raise ValueError(f"Unknown baseline type: {baseline_type}")

        results: List[Dict] = []
        paired_eval = bool(self.config.get("evaluation", {}).get("paired_eval", True))
        base_seed = int(self.config.get("seed", 42))
        schedule = build_eval_schedule(int(num_games), base_seed, paired_eval, game_offset=int(game_offset))
        for game_idx, (seed, model_plays_first) in enumerate(schedule):
            if game_idx % 10 == 0:
                logger.debug(f"Game {game_idx}/{num_games}")
            try:
                result = self._play_match(
                    model_mcts,
                    baseline_mcts,
                    pure_mcts_evaluator,
                    model_plays_first,
                    seed,
                )
                if paired_eval:
                    result["pair_seed"] = seed
                results.append(result)

                # Progress callback (e.g. for incremental WR logging)
                if progress_callback and len(results) % progress_interval == 0:
                    wins = sum(1 for r in results if r["model_won"])
                    wr = wins / len(results) if results else 0.0
                    progress_callback(len(results), wins, wr)
            except Exception as e:
                logger.error(f"Error in game {game_idx}: {e}")
                continue

        stats = self._analyze_results(results)

        logger.debug(
            f"Results: Model wins={stats['model_wins']}, "
            f"Baseline wins={stats['baseline_wins']}, "
            f"Win rate={stats['win_rate']:.2%}"
        )

        # Free GPU memory used by evaluation models
        del model, model_mcts
        if baseline_mcts is not None:
            del baseline_mcts
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return stats

    def evaluate_sprt(
        self,
        model_path: str,
        baseline_path: str,
        sprt_cfg: dict,
    ) -> Dict:
        """Evaluate model vs baseline using SPRT for early stopping.

        Args:
            model_path: Path to candidate model checkpoint.
            baseline_path: Path to baseline model checkpoint.
            sprt_cfg: SPRT configuration dict with keys:
                p0, p1, alpha, beta, min_games, max_games.

        Returns:
            Stats dict with SPRT decision, win_rate, score margins, etc.
        """
        from src.network.model import create_network, load_model_checkpoint, get_state_dict_for_inference

        p0 = float(sprt_cfg.get("p0", 0.50))
        p1 = float(sprt_cfg.get("p1", 0.55))
        alpha = float(sprt_cfg.get("alpha", 0.05))
        beta = float(sprt_cfg.get("beta", 0.05))
        min_games = int(sprt_cfg.get("min_games", 10))
        max_games = int(sprt_cfg.get("max_games", 200))

        lower, upper = sprt_bounds(alpha, beta)
        logger.info(
            f"SPRT evaluation: H0=p<={p0:.3f}  H1=p>={p1:.3f}  "
            f"alpha={alpha}  beta={beta}  bounds=[{lower:.3f}, {upper:.3f}]  "
            f"games=[{min_games}, {max_games}]"
        )

        # Load candidate model
        model = create_network(self.config)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        state_dict = get_state_dict_for_inference(checkpoint, self.config, for_selfplay=False)
        load_model_checkpoint(model, state_dict)
        model.to(self.device)
        model.eval()

        eval_config = self.config["evaluation"]["eval_mcts"]
        model_mcts = create_optimized_mcts(
            model, self.config, self.device, self.state_encoder, self.action_encoder
        )
        model_mcts.config.simulations = int(eval_config["simulations"])
        model_mcts.config.cpuct = float(eval_config["cpuct"])

        # Load baseline model
        baseline_model = create_network(self.config)
        baseline_checkpoint = torch.load(baseline_path, map_location=self.device, weights_only=False)
        baseline_state_dict = get_state_dict_for_inference(baseline_checkpoint, self.config, for_selfplay=False)
        load_model_checkpoint(baseline_model, baseline_state_dict)
        baseline_model.to(self.device)
        baseline_model.eval()

        baseline_mcts = create_optimized_mcts(
            baseline_model, self.config, self.device, self.state_encoder, self.action_encoder
        )
        baseline_mcts.config.simulations = int(eval_config["simulations"])
        baseline_mcts.config.cpuct = float(eval_config["cpuct"])

        # Build paired eval schedule
        paired_eval = bool(self.config.get("evaluation", {}).get("paired_eval", True))
        base_seed = int(self.config.get("seed", 42))
        schedule = build_eval_schedule(max_games, base_seed, paired_eval)

        llr = 0.0
        results: List[Dict] = []
        sprt_result = SPRTResult()
        sprt_result.lower_bound = lower
        sprt_result.upper_bound = upper

        for game_idx, (seed, model_plays_first) in enumerate(schedule):
            if game_idx >= max_games:
                break

            try:
                result = self._play_match(
                    model_mcts, baseline_mcts, None, model_plays_first, seed
                )
                if paired_eval:
                    result["pair_seed"] = seed
                results.append(result)

                # Update SPRT
                llr += sprt_llr_update(result["model_won"], p0, p1)

                games_so_far = len(results)
                wins_so_far = sum(1 for r in results if r["model_won"])
                wr = wins_so_far / games_so_far if games_so_far > 0 else 0.0

                # Log progress every 10 games
                if games_so_far % 10 == 0:
                    logger.info(
                        f"SPRT progress: games={games_so_far}  wins={wins_so_far}  "
                        f"WR={wr:.3f}  LLR={llr:.3f}"
                    )

                # Check stopping conditions (only after min_games)
                if games_so_far >= min_games:
                    if llr >= upper:
                        logger.info(
                            f"SPRT ACCEPT at game {games_so_far}: LLR={llr:.3f} >= {upper:.3f}  "
                            f"WR={wr:.3f}  (model is stronger)"
                        )
                        sprt_result.accept = True
                        break
                    if llr <= lower:
                        logger.info(
                            f"SPRT REJECT at game {games_so_far}: LLR={llr:.3f} <= {lower:.3f}  "
                            f"WR={wr:.3f}  (model is not stronger)"
                        )
                        sprt_result.reject = True
                        break

                    # Futility cutoff (fishtest-style "hopeless" detection):
                    # Check if either SPRT bound can still be reached given
                    # remaining games.  This saves significant compute when a
                    # model is clearly weaker but the per-game LLR increments
                    # are small (common with tight p0/p1 hypotheses).
                    remaining = max_games - games_so_far
                    if remaining > 0:
                        win_inc = sprt_llr_update(True, p0, p1)
                        loss_inc = sprt_llr_update(False, p0, p1)
                        best_possible_llr = llr + remaining * win_inc
                        worst_possible_llr = llr + remaining * loss_inc

                        if best_possible_llr < upper and worst_possible_llr > lower:
                            # Neither bound reachable → guaranteed inconclusive
                            logger.info(
                                f"SPRT EARLY STOP at game {games_so_far}: neither bound reachable  "
                                f"LLR={llr:.3f}  possible=[{worst_possible_llr:.3f}, {best_possible_llr:.3f}]  "
                                f"bounds=[{lower:.3f}, {upper:.3f}]  WR={wr:.3f}"
                            )
                            break  # Leave as inconclusive → WR threshold decides

                        elif best_possible_llr < upper:
                            # Cannot reach ACCEPT even if all remaining games are wins
                            logger.info(
                                f"SPRT EARLY REJECT (futility) at game {games_so_far}: "
                                f"LLR={llr:.3f}  best_possible={best_possible_llr:.3f} < {upper:.3f}  "
                                f"WR={wr:.3f}  (cannot reach ACCEPT)"
                            )
                            sprt_result.reject = True
                            break

            except Exception as e:
                logger.error(f"Error in SPRT game {game_idx}: {e}")
                continue

        # If we exhausted max_games without a decision
        if not sprt_result.accept and not sprt_result.reject:
            sprt_result.inconclusive = True
            wr = sum(1 for r in results if r["model_won"]) / max(1, len(results))
            logger.info(
                f"SPRT INCONCLUSIVE after {len(results)} games: LLR={llr:.3f}  "
                f"WR={wr:.3f}  (defaulting to win-rate threshold)"
            )

        sprt_result.llr = llr
        sprt_result.games_played = len(results)
        sprt_result.wins = sum(1 for r in results if r["model_won"])
        sprt_result.losses = sprt_result.games_played - sprt_result.wins
        sprt_result.win_rate = sprt_result.wins / max(1, sprt_result.games_played)

        stats = self._analyze_results(results)
        stats["sprt"] = {
            "accept": sprt_result.accept,
            "reject": sprt_result.reject,
            "inconclusive": sprt_result.inconclusive,
            "llr": round(sprt_result.llr, 4),
            "lower_bound": round(lower, 4),
            "upper_bound": round(upper, 4),
            "p0": p0,
            "p1": p1,
            "alpha": alpha,
            "beta": beta,
            "games_played": sprt_result.games_played,
        }

        logger.debug(
            f"SPRT final: {'ACCEPT' if sprt_result.accept else 'REJECT' if sprt_result.reject else 'INCONCLUSIVE'}  "
            f"games={sprt_result.games_played}  WR={sprt_result.win_rate:.3f}  "
            f"LLR={sprt_result.llr:.3f}  avg_score_margin={stats['avg_model_score_margin']:.1f}"
        )

        # Free GPU memory used by evaluation models
        del model, model_mcts, baseline_model, baseline_mcts
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return stats

    def _play_match(
        self,
        model_mcts: OptimizedAlphaZeroMCTS,
        baseline_mcts: Optional[OptimizedAlphaZeroMCTS],
        pure_mcts_evaluator: Optional[PureMCTSEvaluator],
        model_plays_first: bool,
        seed: int,
    ) -> Dict:
        """Play a single match using in-process engine."""
        state = new_game(seed=seed)

        move_number = 0
        max_moves = int(self.config["selfplay"]["max_game_length"])

        while move_number < max_moves:
            if terminal_fast(state):
                break

            to_move = current_player_fast(state)

            is_model_turn = (to_move == 0 and model_plays_first) or (
                to_move == 1 and not model_plays_first
            )

            if is_model_turn:
                # Model's turn — neural MCTS takes numpy state directly
                visit_counts, *_ = model_mcts.search(state, to_move, move_number, add_noise=False)
                action = model_mcts.select_action(visit_counts, temperature=0.0, deterministic=True)
            else:
                if baseline_mcts is not None:
                    visit_counts, *_ = baseline_mcts.search(state, to_move, move_number, add_noise=False)
                    action = baseline_mcts.select_action(visit_counts, temperature=0.0, deterministic=True)
                else:
                    action = pure_mcts_evaluator.get_move(state, seed_offset=move_number)

            state = apply_action_unchecked(state, action)
            move_number += 1

        # Outcome — fast path (no upgrade_state)
        winner = get_winner_fast(state)
        p0_score = compute_score_fast(state, 0)
        p1_score = compute_score_fast(state, 1)
        score_diff = abs(p0_score - p1_score)

        # Score margin FROM THE MODEL'S PERSPECTIVE (positive = model scored more)
        if model_plays_first:
            model_score = p0_score
            baseline_score = p1_score
        else:
            model_score = p1_score
            baseline_score = p0_score
        model_score_margin = model_score - baseline_score

        if (winner == 0 and model_plays_first) or (winner == 1 and not model_plays_first):
            model_won = True
        else:
            model_won = False

        return {
            "model_won": model_won,
            "game_length": move_number,
            "score_diff": score_diff,
            "model_score_margin": model_score_margin,
            "model_plays_first": model_plays_first,
        }

    def _analyze_results(self, results: List[Dict]) -> Dict:
        """Analyze match results with score-margin tracking (KataGo-style)."""
        model_wins = sum(1 for r in results if r["model_won"] is True)
        baseline_wins = sum(1 for r in results if r["model_won"] is False)

        total_games = len(results)
        win_rate = (model_wins / total_games) if total_games > 0 else 0.0

        if total_games > 0:
            avg_game_length = float(np.mean([r["game_length"] for r in results]))
            avg_score_diff = float(np.mean([r["score_diff"] for r in results]))
            # KataGo-style: track how much the model wins/loses BY
            margins = [r.get("model_score_margin", 0) for r in results]
            avg_model_score_margin = float(np.mean(margins))
            win_margins = [r["model_score_margin"] for r in results if r["model_won"]]
            loss_margins = [r["model_score_margin"] for r in results if not r["model_won"]]
            avg_win_margin = float(np.mean(win_margins)) if win_margins else 0.0
            avg_loss_margin = float(np.mean(loss_margins)) if loss_margins else 0.0
        else:
            avg_game_length = 0.0
            avg_score_diff = 0.0
            avg_model_score_margin = 0.0
            avg_win_margin = 0.0
            avg_loss_margin = 0.0

        return {
            "model_wins": model_wins,
            "baseline_wins": baseline_wins,
            "ties": 0,
            "total_games": total_games,
            "win_rate": float(win_rate),
            "avg_game_length": avg_game_length,
            "avg_score_diff": avg_score_diff,
            "avg_model_score_margin": avg_model_score_margin,
            "avg_win_margin": avg_win_margin,
            "avg_loss_margin": avg_loss_margin,
            "results": results,
        }


# =========================================================================
# Elo Tracker
# =========================================================================

class EloTracker:
    """Track Elo ratings across iterations."""

    def __init__(self, initial_rating: int = 1500, k_factor: int = 32):
        self.initial_rating = int(initial_rating)
        self.k_factor = int(k_factor)
        self.ratings: Dict[str, float] = {}

    def update(self, player1: str, player2: str, result: float):
        """Update Elo ratings. result=1.0 if player1 wins, 0.5 tie, 0.0 loss."""
        if player1 not in self.ratings:
            self.ratings[player1] = float(self.initial_rating)
        if player2 not in self.ratings:
            self.ratings[player2] = float(self.initial_rating)

        expected1 = self._expected_score(self.ratings[player1], self.ratings[player2])
        expected2 = 1.0 - expected1

        self.ratings[player1] += self.k_factor * (float(result) - expected1)
        self.ratings[player2] += self.k_factor * ((1.0 - float(result)) - expected2)

    def _expected_score(self, rating1: float, rating2: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((rating2 - rating1) / 400.0))

    def get_rating(self, player: str) -> float:
        return float(self.ratings.get(player, float(self.initial_rating)))

    def save_state(self, path: Path) -> None:
        """Persist Elo ratings to disk for crash recovery."""
        import json, os
        state = {
            "initial_rating": self.initial_rating,
            "k_factor": self.k_factor,
            "ratings": {k: float(v) for k, v in self.ratings.items()},
        }
        tmp = Path(path).with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def load_state(self, path: Path) -> bool:
        """Restore Elo ratings from disk. Returns True on success."""
        import json
        path = Path(path)
        if not path.exists():
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)
            self.ratings = {k: float(v) for k, v in state.get("ratings", {}).items()}
            return True
        except Exception as e:
            logger.warning(f"Failed to restore Elo state from {path}: {e}")
            return False