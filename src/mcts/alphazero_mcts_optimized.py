"""
Optimized AlphaZero MCTS with In-Process Patchwork Engine

Tree transitions use engine actions: (AT_BUY, offset, piece_id, orient, top, left)
Neural evaluation uses slot-based encoding.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

from src.game.patchwork_engine import (
    apply_action,
    apply_action_unchecked,
    current_player,
    current_player_fast,
    empty_count_from_occ,
    legal_actions_list,
    legal_actions_fast,
    terminal,
    terminal_fast,
    compute_score,
    compute_score_fast,
    get_winner,
    get_winner_fast,
    AT_PASS,
    AT_PATCH,
    AT_BUY,
    CIRCLE_LEN,
    NEUTRAL,
    CIRCLE_START,
    P0_OCC0,
    P0_OCC1,
    P0_OCC2,
    P1_OCC0,
    P1_OCC1,
    P1_OCC2,
)

from src.utils.packing_metrics import fragmentation_from_occ_words

# Use the unified slot lookup from encoder
from src.network.encoder import encode_state_multimodal, get_slot_piece_id
from src.training.value_targets import terminal_value_from_scores


# ---------------------------------------------------------------------------
# Engine <-> Neural action conversion
# ---------------------------------------------------------------------------

def offset_to_slot_index(offset: int) -> int:
    return int(offset) - 1


def slot_index_to_offset(slot_index: int) -> int:
    return int(slot_index) + 1


def piece_id_in_slot(state, slot_index: int) -> int:
    """Get piece ID for a slot in the circle. Delegates to shared utility."""
    pid = get_slot_piece_id(state, slot_index)
    if pid is None:
        raise ValueError(f"Invalid slot_index: {slot_index}")
    return pid


def buy_slot_to_engine_action(state, slot_index: int, orient: int, top: int, left: int) -> tuple:
    """Convert neural buy_slot action to engine action."""
    piece_id = piece_id_in_slot(state, slot_index)
    offset = slot_index_to_offset(slot_index)
    return (AT_BUY, int(offset), int(piece_id), int(orient), int(top), int(left))


def engine_buy_to_slot_buy(action) -> tuple:
    """Convert engine buy action to neural slot action."""
    _, offset, _piece_id, orient, top, left = action
    return ("buy_slot", offset_to_slot_index(int(offset)), int(orient), int(top), int(left))


def engine_action_to_net_action(action) -> tuple:
    """Convert engine action to neural action format."""
    atype = int(action[0])
    if atype == AT_PASS:
        return ("pass",)
    if atype == AT_PATCH:
        return ("patch", int(action[1]))
    if atype == AT_BUY:
        return engine_buy_to_slot_buy(action)
    raise ValueError(f"Unknown action type: {action}")


def net_action_to_engine_action(net_action, state) -> tuple:
    """Convert neural action back to engine action (needs state for piece_id lookup)."""
    atype = net_action[0]
    if atype == "pass":
        return (AT_PASS, 0, 0, 0, 0, 0)
    if atype == "patch":
        return (AT_PATCH, int(net_action[1]), 0, 0, 0, 0)
    if atype == "buy_slot":
        slot_index = int(net_action[1])
        orient = int(net_action[2])
        top = int(net_action[3])
        left = int(net_action[4])
        return buy_slot_to_engine_action(state, slot_index, orient, top, left)
    raise ValueError(f"Unknown net action type: {atype}")


# ---------------------------------------------------------------------------
# Fast direct engine-action → flat-index encoding (skips string tuples)
# ---------------------------------------------------------------------------
_BOARD_SZ = 9
_NUM_ORIENTS = 8
_PASS_INDEX = 0
_PATCH_START = 1
_BUY_START = 1 + _BOARD_SZ * _BOARD_SZ  # 82


def engine_action_to_flat_index(action) -> int:
    """Encode engine action tuple directly to flat action index.

    Avoids creating intermediate string tuples (engine_action_to_net_action).
    ~3-5x faster per action than the two-step conversion.
    """
    atype = int(action[0])
    if atype == AT_PASS:
        return _PASS_INDEX
    if atype == AT_PATCH:
        return _PATCH_START + int(action[1])
    # AT_BUY: (AT_BUY, offset, piece_id, orient, top, left)
    slot_index = int(action[1]) - 1  # offset → slot_index
    orient = int(action[3])
    top = int(action[4])
    left = int(action[5])
    pos = top * _BOARD_SZ + left
    return _BUY_START + (slot_index * _NUM_ORIENTS + orient) * (_BOARD_SZ * _BOARD_SZ) + pos


def _apply_progressive_widening_order(
    node: "MCTSNode",
    pw_config: ProgressiveWideningConfig,
    is_root: bool,
    packing_config: Optional["PackingOrderingConfig"] = None,
) -> None:
    """
    Reorder node.legal_actions to [PASS, PATCH..., top BUY by prior (or rank_key if packing_ordering enabled)], set _pw_*.
    Call after _init_arrays and after priors are set. Mutates node.legal_actions and node._prior.
    """
    if not pw_config.enabled or not node.legal_actions or node._prior is None:
        return
    legal = node.legal_actions
    action_to_idx = node._action_to_idx
    pass_actions = [a for a in legal if int(a[0]) == AT_PASS] if pw_config.always_include_pass else []
    patch_actions = [a for a in legal if int(a[0]) == AT_PATCH] if pw_config.always_include_patch else []
    buy_actions = [a for a in legal if int(a[0]) == AT_BUY]
    if packing_config and packing_config.enabled and buy_actions and node.state is not None:
        from src.mcts.packing_heuristic import (
            packing_heuristic_scores_batch,
            occ_words_to_bitboard_for_node,
            cache_index,
        )
        alpha = packing_config.alpha
        use_log = packing_config.use_log_prior
        w = packing_config.weights
        w_adj = float(w.get("adj_edges", 1.0))
        w_corner = float(w.get("corner_bonus", 0.5))
        w_iso = float(w.get("iso_hole_penalty", 2.0))
        w_front = float(w.get("frontier_penalty", 0.25))
        w_area = float(w.get("area_bonus", 0.0))
        radius_index = min(max(0, packing_config.local_check_radius), 3)
        k0_buy = pw_config.k_root if is_root else pw_config.k0
        L_buy = len(buy_actions)
        M = min(L_buy, k0_buy + (16 if is_root else 8))

        if node._packing_filled_bb is None:
            pl = node.to_move
            if pl == 0:
                o0, o1, o2 = int(node.state[P0_OCC0]), int(node.state[P0_OCC1]), int(node.state[P0_OCC2])
            else:
                o0, o1, o2 = int(node.state[P1_OCC0]), int(node.state[P1_OCC1]), int(node.state[P1_OCC2])
            node._packing_filled_bb = occ_words_to_bitboard_for_node(o0, o1, o2)
        filled_bb = node._packing_filled_bb

        buy_with_prior = [(a, float(node._prior[action_to_idx[a]])) for a in buy_actions]
        buy_with_prior.sort(key=lambda x: -x[1])
        # Reuse preallocated int32 buffer to avoid per-call list + np.array allocation
        if node._packing_indices_buf is None or node._packing_indices_buf.shape[0] < M:
            node._packing_indices_buf = np.empty(max(M, 64), dtype=np.int32)
        buf = node._packing_indices_buf
        for i, (a, _) in enumerate(buy_with_prior[:M]):
            buf[i] = cache_index(int(a[2]), int(a[3]), int(a[4]) * 9 + int(a[5]))
        pack_scores = packing_heuristic_scores_batch(
            filled_bb, buf[:M], radius_index, w_adj, w_corner, w_iso, w_front, w_area
        )
        buy_with_rank = []
        for i, (a, prior) in enumerate(buy_with_prior):
            base = math.log(max(prior, 1e-10)) if use_log else max(prior, 1e-10)
            pack_score = pack_scores[i] if i < M else 0.0
            rank_key = base + alpha * pack_score
            buy_with_rank.append((a, rank_key, prior, pack_score))
        buy_with_rank.sort(key=lambda x: -x[1])
        buy_ordered = [a for a, _, _, _ in buy_with_rank]
        if is_root and buy_with_rank:
            node._packing_score_top1 = buy_with_rank[0][3]
        if is_root and logger.isEnabledFor(logging.DEBUG):
            top5_by_prior = sorted(buy_with_rank, key=lambda x: -x[2])[:5]
            top5_by_rank = buy_with_rank[:5]
            logger.debug(
                "[packing_ordering] root top5 by prior: %s",
                [(a, "prior=%.4f" % p, "pack=%.3f" % s) for a, _rk, p, s in top5_by_prior],
            )
            logger.debug(
                "[packing_ordering] root top5 by rank_key: %s (M=%d)",
                [(a, "rank=%.4f" % rk, "prior=%.4f" % p, "pack=%.3f" % s) for a, rk, p, s in top5_by_rank],
                M,
            )
    else:
        buy_with_prior = [(a, float(node._prior[action_to_idx[a]])) for a in buy_actions]
        buy_with_prior.sort(key=lambda x: -x[1])
        buy_ordered = [a for a, _ in buy_with_prior]
    ordered_legal = pass_actions + patch_actions + buy_ordered
    n_always = len(pass_actions) + len(patch_actions)
    L_buy = len(buy_actions)
    k0_buy = pw_config.k_root if is_root else pw_config.k0
    node.legal_actions = ordered_legal
    new_prior = np.array([node._prior[action_to_idx[a]] for a in ordered_legal], dtype=np.float64)
    node._prior = new_prior
    node._action_to_idx = {a: i for i, a in enumerate(ordered_legal)}
    node._pw_n_always = n_always
    node._pw_L_buy = L_buy
    node._pw_k0_buy = k0_buy
    n_total = len(ordered_legal)
    node._visit_count = np.zeros(n_total, dtype=np.int32)
    node._total_value = np.zeros(n_total, dtype=np.float64)
    node._virtual_loss = np.zeros(n_total, dtype=np.float64)
    node._value_sum = np.zeros(n_total, dtype=np.float64)
    node._score_sum = np.zeros(n_total, dtype=np.float64)


def encode_legal_actions_fast(legal_actions: list) -> tuple:
    """Encode list of engine actions directly to (indices_np, mask_np).

    Skips the engine→net→index double conversion.
    Returns (np.ndarray int64 indices, np.ndarray float32 mask).
    """
    n = len(legal_actions)
    indices = np.empty(n, dtype=np.int64)
    mask = np.zeros(2026, dtype=np.float32)
    for i, a in enumerate(legal_actions):
        idx = engine_action_to_flat_index(a)
        indices[i] = idx
        mask[idx] = 1.0
    return indices, mask


# ---------------------------------------------------------------------------
# MCTS Configuration
# ---------------------------------------------------------------------------

@dataclass
class ProgressiveWideningConfig:
    """Progressive widening (top-K BUY expansion) for MCTS. Reduces branching at root/internal nodes."""
    enabled: bool = False
    k_root: int = 64
    k0: int = 32
    k_sqrt_coef: float = 8.0  # With sims~208, sqrt(N)~14; coef=8 keeps expansion tighter for depth
    always_include_pass: bool = True
    always_include_patch: bool = True


def _parse_progressive_widening(cfg: dict) -> ProgressiveWideningConfig:
    """Parse progressive_widening block from MCTS config."""
    pw = cfg or {}
    return ProgressiveWideningConfig(
        enabled=bool(pw.get("enabled", False)),
        k_root=int(pw.get("k_root", 64)),
        k0=int(pw.get("k0", 32)),
        k_sqrt_coef=float(pw.get("k_sqrt_coef", 8.0)),
        always_include_pass=bool(pw.get("always_include_pass", True)),
        always_include_patch=bool(pw.get("always_include_patch", True)),
    )


@dataclass
class PackingOrderingConfig:
    """Packing placement-ordering for BUY actions (PW rank key = prior + alpha*heuristic). Default OFF."""
    enabled: bool = False
    alpha: float = 0.15
    use_log_prior: bool = True
    weights: dict = field(default_factory=lambda: {
        "adj_edges": 1.0,
        "corner_bonus": 0.5,
        "iso_hole_penalty": 2.0,
        "frontier_penalty": 0.25,
    })
    local_check_radius: int = 2


def _parse_packing_ordering(cfg: dict) -> PackingOrderingConfig:
    """Parse packing_ordering block from MCTS config."""
    po = cfg or {}
    w = po.get("weights") or {}
    return PackingOrderingConfig(
        enabled=bool(po.get("enabled", False)),
        alpha=float(po.get("alpha", 0.15)),
        use_log_prior=bool(po.get("use_log_prior", True)),
        weights={
            "adj_edges": float(w.get("adj_edges", 1.0)),
            "corner_bonus": float(w.get("corner_bonus", 0.5)),
            "iso_hole_penalty": float(w.get("iso_hole_penalty", 2.0)),
            "frontier_penalty": float(w.get("frontier_penalty", 0.25)),
            "area_bonus": float(w.get("area_bonus", 0.0)),
        },
        local_check_radius=int(po.get("local_check_radius", 2)),
    )


@dataclass
class PatchTiebreakConfig:
    """Root-only tie-break among PATCH actions when value is saturated / tied."""
    enabled: bool = False
    mode: str = "packing"  # "packing" | "score" | "hybrid"
    value_tie_eps: float = 0.01  # treat actions within eps of best Qv as tied
    win_prob_floor: float = 0.98  # only apply when best win prob is very high (Qv >= 2*floor-1)
    weights: dict = field(default_factory=lambda: {
        "empty_squares": 1.0,
        "empty_components": 2.0,
        "isolated_1x1": 3.0,
    })
    score_weight: float = 0.05  # small secondary preference for higher score_est_raw (points)


def _parse_patch_tiebreak(cfg: dict) -> PatchTiebreakConfig:
    pt = cfg or {}
    w = pt.get("weights") or {}
    mode = str(pt.get("mode", "packing")).lower()
    if mode not in ("packing", "score", "hybrid"):
        mode = "packing"
    value_tie_eps = max(0.0, min(0.5, float(pt.get("value_tie_eps", 0.01))))
    win_prob_floor = max(0.0, min(1.0, float(pt.get("win_prob_floor", 0.98))))
    score_weight = max(0.0, float(pt.get("score_weight", 0.05)))
    return PatchTiebreakConfig(
        enabled=bool(pt.get("enabled", False)),
        mode=mode,
        value_tie_eps=value_tie_eps,
        win_prob_floor=win_prob_floor,
        weights={
            "empty_squares": float(w.get("empty_squares", 1.0)),
            "empty_components": float(w.get("empty_components", 2.0)),
            "isolated_1x1": float(w.get("isolated_1x1", 3.0)),
        },
        score_weight=score_weight,
    )


@dataclass
class WinFirstConfig:
    """Win-first (Stockfish-like) root selection and DSUW gating. All fields validated at parse time."""
    enabled: bool = False
    value_delta_min: float = 0.01
    value_delta_max: float = 0.06
    value_delta_win_start: float = 0.60
    value_delta_win_full: float = 0.90
    gate_dsu_enabled: bool = True
    gate_dsu_win_start: float = 0.65
    gate_dsu_win_full: float = 0.90
    gate_dsu_power: float = 2.0
    gate_dsu_loss_start: float = -0.90
    gate_dsu_loss_full: float = -0.98
    tiebreak: str = "score_then_visits"  # "score_then_visits" | "visits_then_score"
    filter_before_sampling: bool = True
    # Validation only: log root selection once (best_Qv, delta, candidate count, top-3 by Qv, Qs_points, Nv)
    debug_log_one_game: bool = False


def _parse_and_validate_win_first(cfg: dict) -> WinFirstConfig:
    """Parse win_first block; clamp ranges and enforce start < full."""
    w = cfg or {}
    enabled = bool(w.get("enabled", False))
    v_min = max(0.0, min(0.5, float(w.get("value_delta_min", 0.01))))
    v_max = max(v_min, min(0.5, float(w.get("value_delta_max", 0.06))))
    win_start = max(-1.0, min(1.0, float(w.get("value_delta_win_start", 0.60))))
    win_full = max(win_start, min(1.0, float(w.get("value_delta_win_full", 0.90))))
    gate_enabled = bool(w.get("gate_dsu_enabled", True))
    g_win_start = max(-1.0, min(1.0, float(w.get("gate_dsu_win_start", 0.65))))
    g_win_full = max(g_win_start, min(1.0, float(w.get("gate_dsu_win_full", 0.90))))
    g_power = max(0.1, min(10.0, float(w.get("gate_dsu_power", 2.0))))
    g_loss_start = max(-1.0, min(1.0, float(w.get("gate_dsu_loss_start", -0.90))))
    g_loss_full = min(g_loss_start, min(1.0, float(w.get("gate_dsu_loss_full", -0.98))))
    if g_loss_full >= g_loss_start:
        g_loss_full = g_loss_start - 0.01  # ensure denominator (start - full) > 0
    tiebreak = str(w.get("tiebreak", "score_then_visits")).lower()
    if tiebreak not in ("score_then_visits", "visits_then_score"):
        tiebreak = "score_then_visits"
    filter_before_sampling = bool(w.get("filter_before_sampling", True))
    debug_log_one_game = bool(w.get("debug_log_one_game", False))
    return WinFirstConfig(
        enabled=enabled,
        value_delta_min=v_min,
        value_delta_max=v_max,
        value_delta_win_start=win_start,
        value_delta_win_full=win_full,
        gate_dsu_enabled=gate_enabled,
        gate_dsu_win_start=g_win_start,
        gate_dsu_win_full=g_win_full,
        gate_dsu_power=g_power,
        gate_dsu_loss_start=g_loss_start,
        gate_dsu_loss_full=g_loss_full,
        tiebreak=tiebreak,
        filter_before_sampling=filter_before_sampling,
        debug_log_one_game=debug_log_one_game,
    )


@dataclass
class MCTSConfig:
    simulations: int = 800
    parallel_leaves: int = 32
    # Tree reuse (permanent brain): when True, reuse subtree across searches. GUI/API only; training uses False.
    enable_tree_reuse: bool = False
    cpuct: float = 1.5
    temperature: float = 1.0
    temperature_threshold: int = 15
    root_dirichlet_alpha: float = 0.3
    root_noise_weight: float = 0.25
    virtual_loss: float = 1.0
    # FPU Reduction (KataGo-style): penalize unexplored children by reducing
    # their initial Q estimate. This encourages the search to prefer actions
    # the network already evaluated highly, rather than exploring randomly.
    # 0.0 = no reduction (legacy), 0.25 = KataGo default for value head.
    fpu_reduction: float = 0.25
    # KataGo Dual-Head (201-bin): score_estimate and _search_root_score are in POINT SPACE.
    # utility = value + score_utility (score_utility computed on GPU from distribution; or locally from score_logits).
    # Dynamic centering: root's mean_points; DSU gate scales effective weights for WIN_FIRST.
    static_score_utility_weight: float = 0.0   # Absolute score lead (disabled in self-play)
    dynamic_score_utility_weight: float = 0.3   # Relative score gain vs. root prediction
    score_utility_scale: float = 30.0           # sat() scale; matches value_targets
    score_min: int = -100
    score_max: int = 100
    win_first: WinFirstConfig = field(default_factory=WinFirstConfig)
    progressive_widening: ProgressiveWideningConfig = field(default_factory=ProgressiveWideningConfig)
    packing_ordering: PackingOrderingConfig = field(default_factory=PackingOrderingConfig)
    patch_tiebreak: PatchTiebreakConfig = field(default_factory=PatchTiebreakConfig)


# ---------------------------------------------------------------------------
# MCTS Node
# ---------------------------------------------------------------------------

class MCTSNode:
    """MCTS tree node.

    Performance: uses numpy arrays indexed by action position in legal_actions
    instead of Python dicts. This reduces per-simulation overhead ~3-5x for
    the UCB computation and update operations.
    """

    __slots__ = (
        "state", "to_move", "parent", "children",
        "_visit_count", "_total_value", "_prior", "_virtual_loss",
        "_value_sum", "_score_sum",  # Win-first: per-child value and score (root-perspective at backup)
        "legal_actions", "_action_to_idx", "terminal", "terminal_value", "n_total",
        "score_estimate",  # raw tanh-normalised score from last network evaluation; used for dynamic centering on tree reuse
        "_pw_n_always", "_pw_L_buy", "_pw_k0_buy",  # Progressive widening: always-include count, num BUY, K0 for BUY
        "_packing_filled_bb",  # Cached: current-player occupancy bitboard for packing heuristic (once per node)
        "_packing_indices_buf",  # Reusable int32 buffer for packing_heuristic_scores_batch (no per-call alloc)
        "_packing_score_top1",  # Packing score of top-ranked BUY after ordering (root only, for logging)
    )

    def __init__(self, state, to_move: int, parent: Optional["MCTSNode"] = None):
        self.state = state
        self.to_move = int(to_move)
        self.parent = parent
        self.children: Dict[tuple, Optional["MCTSNode"]] = {}
        # Arrays allocated lazily when legal_actions are set
        self._visit_count: Optional[np.ndarray] = None
        self._total_value: Optional[np.ndarray] = None
        self._prior: Optional[np.ndarray] = None
        self._virtual_loss: Optional[np.ndarray] = None
        self._value_sum: Optional[np.ndarray] = None
        self._score_sum: Optional[np.ndarray] = None
        self.legal_actions: Optional[List[tuple]] = None
        self._action_to_idx: Optional[Dict[tuple, int]] = None
        self.terminal: bool = False
        self.terminal_value: Optional[float] = None
        self.n_total = 0
        self.score_estimate: float = 0.0  # network score prediction; 0.0 = uninitialised / neutral
        self._pw_n_always: Optional[int] = None  # Progressive widening: count of always-included (PASS + PATCH)
        self._pw_L_buy: Optional[int] = None      # number of legal BUY actions
        self._pw_k0_buy: Optional[int] = None   # K0 or K_root for BUY widening formula
        self._packing_filled_bb: Optional[int] = None  # Cached occupancy bitboard for packing (lazy)
        self._packing_indices_buf: Optional[np.ndarray] = None  # int32 buffer for packing batch (lazy)
        self._packing_score_top1: Optional[float] = None  # top-1 BUY packing score at root (logging only)

    def _init_arrays(self) -> None:
        """Allocate arrays once legal_actions are known."""
        n = len(self.legal_actions)
        self._visit_count = np.zeros(n, dtype=np.int32)
        self._total_value = np.zeros(n, dtype=np.float64)
        self._prior = np.zeros(n, dtype=np.float64)
        self._virtual_loss = np.zeros(n, dtype=np.float64)
        self._value_sum = np.zeros(n, dtype=np.float64)
        self._score_sum = np.zeros(n, dtype=np.float64)
        self._action_to_idx = {a: i for i, a in enumerate(self.legal_actions)}

    def is_expanded(self) -> bool:
        return self.legal_actions is not None

    def get_expanded_count(self, pw_config: Optional[ProgressiveWideningConfig]) -> int:
        """Number of actions that are selectable (expanded). When PW disabled, all legal.

        CRITICAL: Widening uses NODE VISITS N (self.n_total), not number of legal actions.
        N = total simulations that passed through this node (sum of child visit counts).
        Formula: K_buy = min(L_buy, k0_buy + floor(k_sqrt_coef * sqrt(N))); expanded = n_always + K_buy.
        """
        if self.legal_actions is None or self._visit_count is None:
            return 0
        L = len(self.legal_actions)
        if pw_config is None or not pw_config.enabled or self._pw_n_always is None:
            return L
        n_always = self._pw_n_always
        L_buy = self._pw_L_buy
        k0_buy = self._pw_k0_buy
        if L_buy is None or k0_buy is None or L_buy <= 0:
            return L
        # N = node total visit count (updated in update() on backup), NOT len(legal_actions)
        N = max(0, self.n_total)
        K_buy = min(L_buy, k0_buy + int(math.floor(pw_config.k_sqrt_coef * math.sqrt(N))))
        return min(L, n_always + K_buy)

    def get_K_buy(self, pw_config: Optional[ProgressiveWideningConfig]) -> Optional[int]:
        """Current K_buy for this node (for reporting). Returns None if PW disabled or no BUY."""
        if pw_config is None or not pw_config.enabled or self._pw_L_buy is None or self._pw_k0_buy is None:
            return None
        if self._pw_L_buy <= 0:
            return 0
        N = max(0, self.n_total)
        return min(self._pw_L_buy, self._pw_k0_buy + int(math.floor(pw_config.k_sqrt_coef * math.sqrt(N))))

    @property
    def visit_count(self) -> Dict[tuple, int]:
        """Compatibility: return dict view of visit counts."""
        if self._visit_count is None:
            return {}
        return {a: int(self._visit_count[i]) for i, a in enumerate(self.legal_actions)}

    @property
    def prior(self) -> Dict[tuple, float]:
        """Compatibility: return dict view of priors."""
        if self._prior is None:
            return {}
        return {a: float(self._prior[i]) for i, a in enumerate(self.legal_actions)}

    def set_prior(self, action: tuple, value: float) -> None:
        """Set prior for a specific action."""
        if self._action_to_idx is not None:
            idx = self._action_to_idx.get(action)
            if idx is not None:
                self._prior[idx] = value

    def set_priors_bulk(self, actions: list, values) -> None:
        """Set priors for multiple actions at once."""
        if self._action_to_idx is not None:
            for action, v in zip(actions, values):
                idx = self._action_to_idx.get(action)
                if idx is not None:
                    self._prior[idx] = float(v)

    def normalize_priors(self) -> None:
        """Normalize priors to sum to 1."""
        if self._prior is not None:
            total = self._prior.sum()
            if total > 0:
                self._prior /= total
            elif self.legal_actions:
                self._prior[:] = 1.0 / len(self.legal_actions)

    def get_ucb(self, action: tuple, cpuct: float, fpu_reduction: float = 0.0) -> float:
        idx = self._action_to_idx[action]
        n = self._visit_count[idx]
        p = self._prior[idx]
        vl = self._virtual_loss[idx]
        effective_n = n + vl
        if effective_n == 0:
            q_value = -fpu_reduction if self.parent else 0.0
        else:
            q_value = (self._total_value[idx] - vl) / effective_n
        exploration = cpuct * p * math.sqrt(self.n_total + 1) / (1 + n)
        return q_value + exploration

    def select_action(
        self,
        cpuct: float,
        fpu_reduction: float = 0.0,
        pw_config: Optional[ProgressiveWideningConfig] = None,
    ) -> tuple:
        """Vectorized UCB action selection. When progressive widening is enabled, only over expanded (top-K) actions."""
        cap = self.get_expanded_count(pw_config)
        n = self._visit_count[:cap].astype(np.float64)
        vl = self._virtual_loss[:cap]
        prior_slice = self._prior[:cap]
        total_value_slice = self._total_value[:cap]
        effective_n = n + vl
        # Q values
        with np.errstate(divide='ignore', invalid='ignore'):
            q = np.where(effective_n > 0,
                          (total_value_slice - vl) / effective_n,
                          -fpu_reduction if self.parent else 0.0)
        # Exploration
        sqrt_total = math.sqrt(self.n_total + 1)
        exploration = cpuct * prior_slice * sqrt_total / (1.0 + n)
        ucb = q + exploration
        best_idx = int(np.argmax(ucb))
        return self.legal_actions[best_idx]

    def add_virtual_loss(self, action: tuple, loss: float = 1.0) -> None:
        idx = self._action_to_idx[action]
        self._virtual_loss[idx] += loss

    def remove_virtual_loss(self, action: tuple, loss: float = 1.0) -> None:
        idx = self._action_to_idx[action]
        self._virtual_loss[idx] = max(0.0, self._virtual_loss[idx] - loss)

    def update(self, action: tuple, value: float, score: float = 0.0, utility: Optional[float] = None) -> None:
        """Update with (value, score, utility) from leaf. If utility is None, utility = value (backward compat)."""
        idx = self._action_to_idx[action]
        self._visit_count[idx] += 1
        u = utility if utility is not None else value
        self._total_value[idx] += u
        if self._value_sum is not None:
            self._value_sum[idx] += value
        if self._score_sum is not None:
            self._score_sum[idx] += score
        self.n_total += 1


# ---------------------------------------------------------------------------
# Main MCTS class
# ---------------------------------------------------------------------------

class OptimizedAlphaZeroMCTS:
    """AlphaZero MCTS with neural network guidance."""

    def __init__(
        self,
        network,
        config: MCTSConfig,
        device: torch.device,
        state_encoder,
        action_encoder,
        eval_client=None,
        inference_settings: Optional[dict] = None,
        full_config: Optional[dict] = None,
    ):
        self.network = network
        self.config = config
        self.device = device
        self.state_encoder = state_encoder
        self.action_encoder = action_encoder

        inf = inference_settings or {}
        self.use_amp = bool(inf.get("use_amp", True))
        self.amp_dtype = str(inf.get("amp_dtype", "float16")).lower()
        self.allow_tf32 = bool(inf.get("allow_tf32", True))

        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = self.allow_tf32
            torch.backends.cudnn.allow_tf32 = self.allow_tf32
            deterministic = bool(inf.get("deterministic", True))
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = deterministic
            # NOTE: torch.use_deterministic_algorithms(True) disabled - CUBLAS compatibility
            # if deterministic:
            #     torch.use_deterministic_algorithms(True)

        self.eval_client = eval_client
        self.total_simulations = 0
        self.total_time = 0.0
        self._root: Optional[MCTSNode] = None
        # Zero-copy IPC: set by worker setup when SharedMemory is available.
        # When both are set, _batch_expand_and_evaluate writes encoded state directly
        # into SHM slots instead of pickling arrays through the queue.
        self._shm_buf = None          # WorkerSharedBuffer | None
        self._gold_v2_encoder = None  # GoldV2StateEncoder | None
        # When using eval_client, network is None; derive from full_config.
        # 56ch networks always need gold_v2 multimodal encoding, regardless of use_film.
        if network is not None:
            use_film = getattr(network, "use_film", False)
            in_ch = 56
            conv = getattr(network, "conv_input", None)
            if conv is not None:
                in_ch = getattr(conv, "in_channels", 56)
            self._use_multimodal = use_film or (in_ch in (56, 60))
        elif full_config:
            data_cfg = full_config.get("data", {}) or {}
            enc = str(data_cfg.get("encoding_version", ""))
            if enc and enc.lower().startswith("gold_v2"):
                self._use_multimodal = True
            else:
                in_ch = int(full_config.get("network", {}).get("input_channels", 56))
                self._use_multimodal = in_ch in (56, 60)
        else:
            self._use_multimodal = False

        # FIX (M4): seeded RNG for Dirichlet noise — reproducible searches.
        # Initialised with a default; callers can reset via set_noise_seed().
        self._noise_rng = np.random.default_rng(42)

        # KataGo dynamic score centering: root's mean_points (POINT SPACE) at the start of each search.
        self._search_root_score: float = 0.0
        # Win-first: root's value estimate (from initial NN eval) for DSUW gating.
        self._search_root_value: float = 0.0
        # Effective score-utility weights (DSU-gated) for this search; set after root is evaluated.
        self._search_effective_static_w: float = 0.0
        self._search_effective_dynamic_w: float = 0.3
        # 201-bin local eval: bins and scale for distributional score_utility (mirror server).
        self._score_bins_t = torch.arange(
            config.score_min, config.score_max + 1,
            device=self.device, dtype=torch.float32,
        )
        # WIN_FIRST validation: log root selection once when debug_log_one_game is True
        self._win_first_debug_logged = False
        self._score_utility_scale = float(config.score_utility_scale)

    def set_noise_seed(self, seed: int) -> None:
        """Reset the Dirichlet noise RNG. Call before each game for reproducibility."""
        self._noise_rng = np.random.default_rng(seed)

    def clear_tree(self) -> None:
        """Clear the MCTS tree (e.g. on new game). Used by GUI/API when starting a fresh game."""
        self._root = None

    def get_root_legal_count(self) -> int:
        """Number of legal actions at root (after last search). For reporting / A-B tests."""
        if self._root is None or self._root.legal_actions is None:
            return 0
        return len(self._root.legal_actions)

    def get_root_expanded_count(self) -> int:
        """Number of expanded (selectable) actions at root. For reporting / A-B tests."""
        if self._root is None:
            return 0
        pw = self.config.progressive_widening if self.config.progressive_widening.enabled else None
        return self._root.get_expanded_count(pw)

    def get_root_n_total(self) -> int:
        """Total node visits N at root (for PW reporting)."""
        if self._root is None:
            return 0
        return int(self._root.n_total)

    def get_root_K_buy(self) -> Optional[int]:
        """Current K_buy at root (for PW reporting). None if PW disabled."""
        if self._root is None:
            return None
        pw = self.config.progressive_widening if self.config.progressive_widening.enabled else None
        return self._root.get_K_buy(pw)

    def get_root_packing_score_top1(self) -> Optional[float]:
        """Packing score of top-ranked BUY at root (for logging). None if packing ordering didn't run."""
        if self._root is None:
            return None
        return self._root._packing_score_top1

    def advance_tree(self, action: tuple) -> None:
        """Move root to the child for the given action (GUI tree reuse). Next search continues from that node."""
        if self._root is not None and self._root.children and action in self._root.children:
            child = self._root.children[action]
            if child is not None:
                self._root = child
                return
        self._root = None

    # ------------------------------------------------------------------
    # Score utility (KataGo dual-head)
    # ------------------------------------------------------------------

    def _compute_dsu_gate(self) -> float:
        """Win-first: gate in [0,1]. g=0 => DSUW off (fight to win); g=1 => DSUW on (convert or defend margin)."""
        wf = self.config.win_first
        if not wf.enabled or not wf.gate_dsu_enabled:
            return 1.0
        v = self._search_root_value
        # Win gate: ramp from gate_dsu_win_start to gate_dsu_win_full
        t_win = (v - wf.gate_dsu_win_start) / max(1e-9, wf.gate_dsu_win_full - wf.gate_dsu_win_start)
        t_win = max(0.0, min(1.0, t_win))
        g_win = t_win ** wf.gate_dsu_power
        # Forced-loss gate: ramp when v is very negative (gate_dsu_loss_start to gate_dsu_loss_full)
        t_loss = (wf.gate_dsu_loss_start - v) / max(1e-9, wf.gate_dsu_loss_start - wf.gate_dsu_loss_full)
        t_loss = max(0.0, min(1.0, t_loss))
        g_loss = t_loss ** wf.gate_dsu_power
        return max(g_win, g_loss)

    def _compute_score_utility(self, score: float, is_root_player: bool = True) -> float:
        """
        Legacy: score utility from scalar (tanh) score. Used only when server returns precomputed score_utility.
        When using 201-bin protocol, non-terminal leaves get score_utility from GPU; terminals use _compute_terminal_score_utility.
        """
        g = self._compute_dsu_gate()
        static_w = self.config.static_score_utility_weight * g
        dynamic_w = self.config.dynamic_score_utility_weight * g
        static = static_w * score
        root_s = self._search_root_score if is_root_player else -self._search_root_score
        dynamic = dynamic_w * (score - root_s)
        return static + dynamic

    def _compute_terminal_score_utility(self, raw_margin_points: float, is_root_player: bool) -> float:
        """Terminal score utility in point space: sat((margin - center)/scale) with effective weights."""
        scale = self._score_utility_scale
        center = self._search_root_score if is_root_player else -self._search_root_score

        def sat(x: float) -> float:
            return (2.0 / math.pi) * math.atan(x)

        static_util = sat(raw_margin_points / scale)
        dynamic_util = sat((raw_margin_points - center) / scale)
        return self._search_effective_static_w * static_util + self._search_effective_dynamic_w * dynamic_util

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        state: np.ndarray,
        to_move: int,
        move_number: int = 0,
        add_noise: bool = True,
        root_legal_actions: Optional[list] = None,
    ) -> Tuple[Dict[tuple, int], float, float]:
        """
        Run MCTS search and return visit counts + root Q-value.
        Supports tree reuse (permanent brain): when the next state matches a child
        of the current root, that child becomes the new root for continued search.

        Args:
            state: Engine state (np.ndarray int32).
            to_move: Current player (0 or 1).
            move_number: Move number in the game (unused currently).
            add_noise: Whether to add Dirichlet noise at root.
            root_legal_actions: Pre-computed legal actions for root (avoids
                duplicate legal_actions_fast call when caller already has them).

        Returns:
            visit_counts: Dict[engine_action_tuple, int]
            search_time: float
            root_q: float — average Q-value at root (from to_move perspective)
        """
        start_time = time.time()
        # Effective weights for score utility (root request uses these; then we apply DSU gate after root eval).
        self._search_effective_static_w = self.config.static_score_utility_weight
        self._search_effective_dynamic_w = self.config.dynamic_score_utility_weight

        # GUI/API inference is intentionally stateless per solve to avoid stale-tree carryover
        # after undo, refresh, or accidental inputs. Always start from a fresh root.
        root = MCTSNode(state, int(to_move))
        self._root = root  # Ensure root is available for is_root_player checks
        self._expand_and_evaluate(root, precomputed_legal=root_legal_actions)
        if add_noise and root.legal_actions and len(root.legal_actions) > 0:
            self._add_dirichlet_noise(root)
        # Apply progressive widening order at root AFTER noise (so top-K BUY uses noisy priors in self-play)
        if self.config.progressive_widening.enabled and root.legal_actions:
            _apply_progressive_widening_order(
                root, self.config.progressive_widening,
                is_root=True,
                packing_config=self.config.packing_ordering,
            )

        # Reused root may need expansion if it was a leaf (shouldn't happen, but be safe)
        if not root.is_expanded() and not root.terminal:
            self._expand_and_evaluate(root, precomputed_legal=root_legal_actions)

        # KataGo dynamic score centering (POINT SPACE) and WIN_FIRST effective weights.
        # Root was just evaluated; score_estimate is mean_points from server or local 201-bin.
        self._search_root_score = root.score_estimate
        g = self._compute_dsu_gate()
        self._search_effective_static_w = self.config.static_score_utility_weight * g
        self._search_effective_dynamic_w = self.config.dynamic_score_utility_weight * g

        # Use batched MCTS if parallel_leaves > 1 (both local network and eval_client modes)
        if self.config.parallel_leaves > 1 and (self.network is not None or self.eval_client is not None):
            self._simulate_batched(root)
        else:
            for _ in range(self.config.simulations):
                self._simulate(root)

        search_time = time.time() - start_time
        self.total_simulations += self.config.simulations
        self.total_time += search_time

        self._root = root

        # Progressive widening debug: ensure we use NODE VISITS N, not legal count; log root stats when PW enabled
        if self.config.progressive_widening.enabled and root is not None and root.legal_actions is not None:
            N = int(root.n_total)
            pw = self.config.progressive_widening
            expanded = root.get_expanded_count(pw)
            K_buy = root.get_K_buy(pw)
            root_legal = len(root.legal_actions)
            logger.debug(
                "[PW root] root_legal_count=%d root_expanded_count=%d N=%d K_buy=%s",
                root_legal, expanded, N, K_buy,
            )
            assert expanded <= root_legal, "expanded_count must not exceed legal_count"

        # Build visit counts from the array-backed node
        if root.legal_actions and root._visit_count is not None:
            visit_counts = {a: int(root._visit_count[i]) for i, a in enumerate(root.legal_actions)}
        else:
            visit_counts = {}

        # Root Q-value: weighted average of backed-up values across all children.
        # This is the MCTS estimate of position value from to_move's perspective.
        # Used for Q-value training targets (Oracle Part 4 / Leela Chess Zero).
        root_q = 0.0
        if root.n_total > 0 and root._total_value is not None and root._visit_count is not None:
            total_visits = root._visit_count.sum()
            if total_visits > 0:
                root_q = float(root._total_value.sum() / total_visits)

        return visit_counts, search_time, root_q

    def _select_action_win_first(self, temperature: float, deterministic: bool) -> tuple:
        """Win-first root move selection using root node's value_sum, score_sum, visit_count.
        Only considers children with Nv>0 for best_Qv, candidate set C, and tiebreak; safe fallback if all Nv==0.
        """
        wf = self.config.win_first
        root = self._root
        if root is None or not root.legal_actions or root._visit_count is None or root._value_sum is None or root._score_sum is None:
            vc = {a: int(root._visit_count[i]) for i, a in enumerate(root.legal_actions)} if root and root._visit_count is not None else {}
            return max(vc.items(), key=lambda x: x[1])[0] if vc else root.legal_actions[0]

        n = len(root.legal_actions)
        visits = root._visit_count.astype(np.float64)
        has_visits = visits > 0
        if not np.any(has_visits):
            if wf.debug_log_one_game and not self._win_first_debug_logged:
                self._win_first_debug_logged = True
                logger.info("[WIN_FIRST debug] best_Qv=N/A (no visits yet) delta=N/A candidate_count=0")
            return root.legal_actions[0]

        # Qv, Qs only meaningful where Nv>0; use 0 for zero-visit children so they are excluded from best_Qv and C
        safe_visits = np.where(has_visits, visits, 1.0)
        qv = np.where(has_visits, root._value_sum / safe_visits, -np.inf)
        qs = np.where(has_visits, root._score_sum / safe_visits, -np.inf)
        best_qv = float(np.max(qv))
        if not np.isfinite(best_qv):
            return root.legal_actions[0]

        # Adaptive delta: widen when win secured
        if best_qv <= wf.value_delta_win_start:
            delta = wf.value_delta_min
        elif best_qv >= wf.value_delta_win_full:
            delta = wf.value_delta_max
        else:
            t = (best_qv - wf.value_delta_win_start) / max(1e-9, wf.value_delta_win_full - wf.value_delta_win_start)
            delta = wf.value_delta_min + t * (wf.value_delta_max - wf.value_delta_min)
        # Candidate set: Qv >= best_Qv - delta, and Nv>0 only
        mask = has_visits & (qv >= (best_qv - delta))
        candidate_indices = np.nonzero(mask)[0]
        if len(candidate_indices) == 0:
            candidate_indices = np.nonzero(has_visits)[0]
        if len(candidate_indices) == 0:
            return root.legal_actions[0]

        # Optional one-time debug log (validation only; do not enable in production)
        if wf.debug_log_one_game and not self._win_first_debug_logged:
            self._win_first_debug_logged = True
            qv_c = qv[candidate_indices]
            qs_c = qs[candidate_indices]
            nv_c = visits[candidate_indices]
            order = np.lexsort((-nv_c, -qs_c, -qv_c))  # tiebreak: Qv desc, then Qs desc, then Nv desc
            top3 = order[: min(3, len(order))]
            lines = [
                f"[WIN_FIRST debug] best_Qv={best_qv:.4f} delta={delta:.4f} candidate_count={len(candidate_indices)}",
            ]
            for i, idx in enumerate(top3):
                lines.append(
                    f"  top-{i+1}: Qv={qv_c[idx]:.4f} Qs_points={qs_c[idx]:.2f} Nv={int(nv_c[idx])}"
                )
            logger.info("\n".join(lines))

        if temperature == 0 or deterministic:
            if wf.tiebreak == "visits_then_score":
                best_idx = candidate_indices[
                    np.argmax(visits[candidate_indices] * 1e6 + qs[candidate_indices])
                ]
            else:
                best_idx = candidate_indices[
                    np.argmax(qs[candidate_indices] * 1e6 + visits[candidate_indices])
                ]
            return root.legal_actions[int(best_idx)]
        visits_c = visits[candidate_indices] ** (1.0 / max(1e-9, temperature))
        probs = visits_c / visits_c.sum()
        idx = int(self._noise_rng.choice(len(candidate_indices), p=probs))
        return root.legal_actions[int(candidate_indices[idx])]

    def select_action(
        self,
        visit_counts: Dict[tuple, int],
        temperature: float = 1.0,
        deterministic: bool = False,
        *,
        patch_tiebreak_override: Optional[bool] = None,
    ) -> tuple:
        """Select action from visit counts. When win_first.enabled, uses root node for strict win-first selection.

        If patch_tiebreak is enabled (or overridden), may apply a root-only deterministic tie-break among
        near-equal PATCH actions in sure-winning positions. This does NOT affect tree search.
        """
        if not visit_counts:
            raise ValueError("No visit counts")

        use_pt = self.config.patch_tiebreak.enabled if patch_tiebreak_override is None else bool(patch_tiebreak_override)

        if self.config.win_first.enabled and self._root is not None and self._root._value_sum is not None:
            action = self._select_action_win_first(temperature, deterministic)
            if use_pt:
                action = self._maybe_apply_patch_tiebreak(selected_action=action)
            return action

        if temperature == 0 or deterministic:
            action = max(visit_counts.items(), key=lambda x: x[1])[0]
            if use_pt:
                action = self._maybe_apply_patch_tiebreak(selected_action=action)
            return action

        actions = list(visit_counts.keys())
        visits = np.array([visit_counts[a] for a in actions], dtype=np.float64)
        if visits.sum() == 0:
            action = actions[int(self._noise_rng.integers(0, len(actions)))]
            if use_pt:
                action = self._maybe_apply_patch_tiebreak(selected_action=action)
            return action
        visits_temp = visits ** (1.0 / temperature)
        probs = visits_temp / visits_temp.sum()
        action = actions[int(self._noise_rng.choice(len(actions), p=probs))]
        if use_pt:
            action = self._maybe_apply_patch_tiebreak(selected_action=action)
        return action

    def _maybe_apply_patch_tiebreak(self, selected_action: tuple) -> tuple:
        """Root-only tie-break among PATCH actions when value saturates / is tied.

        Only breaks ties among actions within value_tie_eps of best Qv (value head).
        Never overrides a clearly better-Q action.
        """
        cfg = self.config.patch_tiebreak
        root = self._root
        if root is None or not root.legal_actions:
            return selected_action
        if root._visit_count is None or root._value_sum is None or root._score_sum is None:
            return selected_action

        # We only ever consider PATCH actions at the root.
        legal = root.legal_actions
        patch_idxs = [i for i, a in enumerate(legal) if int(a[0]) == AT_PATCH]
        if not patch_idxs:
            return selected_action

        visits = root._visit_count.astype(np.float64)
        has_visits = visits > 0
        if not np.any(has_visits):
            return selected_action

        safe_visits = np.where(has_visits, visits, 1.0)
        qv = np.where(has_visits, root._value_sum / safe_visits, -np.inf)  # value head [-1,1]
        qs_points = np.where(has_visits, root._score_sum / safe_visits, -np.inf)  # points (margin) estimate

        best_qv = float(np.max(qv))
        if not math.isfinite(best_qv):
            return selected_action

        # Trigger condition: sure-win OR there exist ties within eps.
        best_win_prob = 0.5 * (best_qv + 1.0)
        within_eps = np.where(qv >= (best_qv - float(cfg.value_tie_eps)))[0]
        if within_eps.size <= 1 and best_win_prob < float(cfg.win_prob_floor):
            return selected_action

        # Candidate set for tie-break: PATCH actions that are within eps of best value.
        cand = [i for i in patch_idxs if qv[i] >= (best_qv - float(cfg.value_tie_eps))]
        if len(cand) <= 1:
            return selected_action

        # Compute pre-move packing metrics once for root player (the one placing the patch).
        to_move = int(root.to_move)
        if to_move == 0:
            base_occ = (int(root.state[P0_OCC0]), int(root.state[P0_OCC1]), int(root.state[P0_OCC2]))
        else:
            base_occ = (int(root.state[P1_OCC0]), int(root.state[P1_OCC1]), int(root.state[P1_OCC2]))
        base_empty = int(empty_count_from_occ(*base_occ))
        base_comp, base_iso = fragmentation_from_occ_words(*base_occ)

        w_empty = float(cfg.weights.get("empty_squares", 1.0))
        w_comp = float(cfg.weights.get("empty_components", 2.0))
        w_iso = float(cfg.weights.get("isolated_1x1", 3.0))
        score_w = float(cfg.score_weight)

        best_i = cand[0]
        best_pack = None
        best_score = None
        best_combined = None

        for i in cand:
            a = legal[i]
            # Apply patch to a copy; compute metrics for the SAME player (root.to_move).
            ns = apply_action_unchecked(root.state, a)
            if to_move == 0:
                occ = (int(ns[P0_OCC0]), int(ns[P0_OCC1]), int(ns[P0_OCC2]))
            else:
                occ = (int(ns[P1_OCC0]), int(ns[P1_OCC1]), int(ns[P1_OCC2]))
            empty = int(empty_count_from_occ(*occ))
            comp, iso = fragmentation_from_occ_words(*occ)

            pack_cost = (
                w_empty * float(empty) +
                w_comp * float(comp) +
                w_iso * float(iso)
            )
            # Use backed-up mean points estimate as "score_est_raw" proxy.
            score_raw = float(qs_points[i]) if math.isfinite(float(qs_points[i])) else 0.0

            if cfg.mode == "packing":
                key = (pack_cost, -score_raw, i)  # deterministic final tie-break by index
            elif cfg.mode == "score":
                key = (-score_raw, pack_cost, i)
            else:  # hybrid: lexicographic packing then score, plus a tiny combined guard
                combined = pack_cost - score_w * score_raw
                key = (pack_cost, -score_raw, combined, i)

            if best_combined is None or key < best_combined:
                best_combined = key
                best_i = i
                best_pack = pack_cost
                best_score = score_raw

        # Only override if it changes the selected action AND the selected action is in the tied set.
        selected_idx = root._action_to_idx.get(selected_action) if root._action_to_idx is not None else None
        if selected_idx is None:
            return legal[int(best_i)]
        if selected_idx in cand:
            return legal[int(best_i)]
        return selected_action

    # ------------------------------------------------------------------
    # Tree traversal
    # ------------------------------------------------------------------

    def _select_leaf(self, root: MCTSNode):
        node = root
        search_path: List[Tuple[MCTSNode, tuple]] = []

        pw = self.config.progressive_widening if self.config.progressive_widening.enabled else None
        while node.is_expanded() and not node.terminal:
            action = node.select_action(
                self.config.cpuct, self.config.fpu_reduction, pw_config=pw
            )
            search_path.append((node, action))
            node.add_virtual_loss(action, self.config.virtual_loss)

            if action not in node.children or node.children[action] is None:
                # Use unchecked apply — action is from legal_actions_list
                next_state = apply_action_unchecked(node.state, action)
                next_to_move = current_player_fast(next_state)
                node.children[action] = MCTSNode(next_state, next_to_move, parent=node)

            node = node.children[action]

        return node, search_path

    # ------------------------------------------------------------------
    # Value backup
    # ------------------------------------------------------------------

    def _backup_path(
        self,
        search_path: List[Tuple[MCTSNode, tuple]],
        leaf_v: float,
        leaf_s: float,
        leaf_u: float,
        leaf_to_move: int,
    ) -> None:
        """
        Backup (value, score, utility) through the search path. All from leaf's to_move perspective;
        at each parent we convert to parent's perspective (flip sign when parent.to_move != leaf_to_move).
        """
        for parent_node, action_taken in reversed(search_path):
            parent_node.remove_virtual_loss(action_taken, self.config.virtual_loss)
            if parent_node.to_move == leaf_to_move:
                vp, sp, up = leaf_v, leaf_s, leaf_u
            else:
                vp, sp, up = -leaf_v, -leaf_s, -leaf_u
            parent_node.update(action_taken, vp, sp, up)

    # ------------------------------------------------------------------
    # Single simulation
    # ------------------------------------------------------------------

    def _simulate(self, root: MCTSNode):
        node = root
        search_path: List[Tuple[MCTSNode, tuple]] = []

        pw = self.config.progressive_widening if self.config.progressive_widening.enabled else None
        while node.is_expanded() and not node.terminal:
            action = node.select_action(
                self.config.cpuct, self.config.fpu_reduction, pw_config=pw
            )
            search_path.append((node, action))
            node.add_virtual_loss(action, self.config.virtual_loss)

            if action not in node.children or node.children[action] is None:
                next_state = apply_action_unchecked(node.state, action)
                next_to_move = current_player_fast(next_state)
                node.children[action] = MCTSNode(next_state, next_to_move, parent=node)

            node = node.children[action]

        if node.terminal:
            v, s, u = self._get_terminal_v_s_u(node.state, node.to_move)
        else:
            v, s, u = self._expand_and_evaluate(node)
        self._backup_path(search_path, v, s, u, node.to_move)

    # ------------------------------------------------------------------
    # Batched simulation
    # ------------------------------------------------------------------

    def _simulate_batched(self, root: MCTSNode) -> None:
        sims_remaining = int(self.config.simulations)
        batch_n = max(1, int(self.config.parallel_leaves))
        # Never use more SHM slots than allocated (schedule may request more than buffer size)
        if self._shm_buf is not None:
            batch_n = min(batch_n, self._shm_buf.n_slots)

        while sims_remaining > 0:
            k = min(batch_n, sims_remaining)

            leaves: List[MCTSNode] = []
            paths: List[List[Tuple[MCTSNode, tuple]]] = []
            terminal_data: List[Tuple[List[Tuple[MCTSNode, tuple]], float, int]] = []

            for _ in range(k):
                leaf, path = self._select_leaf(root)
                if leaf.terminal or terminal_fast(leaf.state):
                    v, s, u = self._get_terminal_v_s_u(leaf.state, leaf.to_move)
                    terminal_data.append((path, v, s, u, leaf.to_move))
                else:
                    leaves.append(leaf)
                    paths.append(path)

            values: List[Tuple[float, float, float]] = []
            if leaves:
                values = self._batch_expand_and_evaluate(leaves)

            for path, (v, s, u), leaf_to_move in zip(paths, values, [l.to_move for l in leaves]):
                self._backup_path(path, v, s, u, leaf_to_move)
            for path, v, s, u, leaf_to_move in terminal_data:
                self._backup_path(path, v, s, u, leaf_to_move)

            sims_remaining -= k

    # ------------------------------------------------------------------
    # Batch expand + evaluate
    # ------------------------------------------------------------------

    def _batch_expand_and_evaluate(self, nodes: List[MCTSNode]) -> List[Tuple[float, float, float]]:
        states_np: List[np.ndarray] = []
        masks_np: List[np.ndarray] = []
        mm_extras: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        legal_actions_per_node: List[List[tuple]] = []
        legal_indices_list: List[np.ndarray] = []
        terminal_out: List[Optional[float]] = [None] * len(nodes)
        # SHM path: (slot, n_legal) pairs for non-terminal leaves encoded via encode_into.
        shm_slot_list: List[Tuple[int, int]] = []
        n_nonterminal = 0  # total non-terminal leaves (SHM + pickle paths)

        # SHM is active when both _shm_buf and _gold_v2_encoder are set AND
        # we are in GPU server mode with gold_v2 encoding.
        _use_shm = (
            self._shm_buf is not None
            and self._gold_v2_encoder is not None
            and self.eval_client is not None
            and self._use_multimodal
        )

        for i, node in enumerate(nodes):
            if terminal_fast(node.state):
                node.terminal = True
                v = self._get_terminal_value(node.state, node.to_move)
                node.terminal_value = v
                terminal_out[i] = v
                continue

            node.legal_actions = legal_actions_fast(node.state)
            if not node.legal_actions:
                node.terminal = True
                v = self._get_terminal_value(node.state, node.to_move)
                node.terminal_value = v
                terminal_out[i] = v
                continue

            node._init_arrays()

            legal_action_indices, action_mask = encode_legal_actions_fast(node.legal_actions)
            masks_np.append(action_mask)
            legal_actions_per_node.append(node.legal_actions)
            legal_indices_list.append(legal_action_indices)

            if _use_shm:
                # Zero-copy path: encode directly into shared memory slot n_nonterminal.
                slot = n_nonterminal
                n_lj = len(legal_action_indices)
                worker_id = getattr(self.eval_client, "worker_id", 0) if self.eval_client else 0
                self._shm_buf.check_slot_write_bounds(
                    slot, n_lj,
                    worker_id=worker_id,
                    expected_n_slots=None,
                    derived_n_slots=self._shm_buf.n_slots,
                )
                assert slot < self._shm_buf.n_slots, (
                    f"SHM slot {slot} >= n_slots {self._shm_buf.n_slots}"
                )
                enc = self._gold_v2_encoder
                enc.encode_into(
                    node.state, node.to_move,
                    self._shm_buf.spatial_view(slot),
                    self._shm_buf.global_view(slot),
                    self._shm_buf.track_view(slot),
                    self._shm_buf.shopids_view(slot),
                    self._shm_buf.shopfeats_view(slot),
                )
                self._shm_buf.mask_view(slot)[:] = action_mask
                li32 = legal_action_indices.astype(np.int32, copy=False)
                self._shm_buf.legalidxs_view(slot, n_lj)[:] = li32
                self._shm_buf.write_nlegal(slot, n_lj)
                shm_slot_list.append((slot, n_lj))
            elif self._use_multimodal:
                x_spatial, x_global, x_track, shop_ids, shop_feats = self.state_encoder.encode_state_multimodal(node.state, node.to_move)
                states_np.append(x_spatial.astype(np.float32, copy=False))
                mm_extras.append((x_global.astype(np.float32), x_track.astype(np.float32), shop_ids, shop_feats.astype(np.float32)))
            else:
                state_np = self.state_encoder.encode_state(node.state, node.to_move).astype(np.float32, copy=False)
                states_np.append(state_np)

            n_nonterminal += 1

        if n_nonterminal == 0:
            return [self._get_terminal_v_s_u(nodes[i].state, nodes[i].to_move) for i in range(len(nodes))]

        # GPU server mode: 201-bin protocol returns (priors, value, mean_points, score_utility).
        if self.eval_client is not None:
            nonterminal_nodes = [n for i, n in enumerate(nodes) if terminal_out[i] is None]
            root_to_move = int(self._root.to_move) if self._root is not None else -1
            req_rids = []
            if shm_slot_list:
                for idx, (slot, n_lj) in enumerate(shm_slot_list):
                    node = nonterminal_nodes[idx]
                    is_root = (int(node.to_move) == root_to_move) if root_to_move >= 0 else True
                    center = self._search_root_score if is_root else -self._search_root_score
                    rid = self.eval_client.submit_shm(
                        slot, n_lj,
                        center, self._search_effective_static_w, self._search_effective_dynamic_w,
                    )
                    req_rids.append(rid)
            elif self._use_multimodal and mm_extras:
                for j in range(len(states_np)):
                    node = nonterminal_nodes[j]
                    is_root = (int(node.to_move) == root_to_move) if root_to_move >= 0 else True
                    center = self._search_root_score if is_root else -self._search_root_score
                    xg, xt, si, sf = mm_extras[j]
                    legal_i32 = legal_indices_list[j].astype(np.int32, copy=False)
                    rid = self.eval_client.submit_multimodal(
                        states_np[j], xg, xt, si, sf, masks_np[j], legal_i32,
                        center, self._search_effective_static_w, self._search_effective_dynamic_w,
                    )
                    req_rids.append(rid)
            else:
                for j in range(len(states_np)):
                    node = nonterminal_nodes[j]
                    is_root = (int(node.to_move) == root_to_move) if root_to_move >= 0 else True
                    center = self._search_root_score if is_root else -self._search_root_score
                    legal_i32 = legal_indices_list[j].astype(np.int32, copy=False)
                    rid = self.eval_client.submit_legacy(
                        states_np[j], masks_np[j], legal_i32,
                        center, self._search_effective_static_w, self._search_effective_dynamic_w,
                    )
                    req_rids.append(rid)

            resp_priors = []
            resp_values = []
            resp_mean_pts = []
            resp_score_util = []
            for rid in req_rids:
                priors_legal, value, mean_points, score_utility = self.eval_client.receive(rid)
                resp_priors.append(priors_legal)
                resp_values.append(float(value))
                resp_mean_pts.append(float(mean_points))
                resp_score_util.append(float(score_utility))

            result: List[Tuple[float, float, float]] = [None] * len(nodes)  # type: ignore[list-item]
            for i in range(len(nodes)):
                if terminal_out[i] is not None:
                    result[i] = self._get_terminal_v_s_u(nodes[i].state, nodes[i].to_move)
            nonterm_idx = 0
            for i, node in enumerate(nodes):
                if terminal_out[i] is not None:
                    continue
                priors_legal = resp_priors[nonterm_idx]
                node._prior[:] = priors_legal[:len(node.legal_actions)]
                if self.config.progressive_widening.enabled:
                    _apply_progressive_widening_order(
                        node, self.config.progressive_widening,
                        is_root=False,
                        packing_config=self.config.packing_ordering,
                    )
                node.normalize_priors()
                v = resp_values[nonterm_idx]
                s_points = resp_mean_pts[nonterm_idx]
                su = resp_score_util[nonterm_idx]
                node.score_estimate = s_points
                u = v + su
                result[i] = (v, s_points, u)
                nonterm_idx += 1
            return result

        # Local network mode: true batched evaluation
        if self.network is None:
            raise RuntimeError("MCTS requires either a network or eval_client")

        st = torch.from_numpy(np.stack(states_np, axis=0))
        am = torch.from_numpy(np.stack(masks_np, axis=0))

        non_block = self.device.type == "cuda"
        st = st.to(self.device, non_blocking=non_block)
        am = am.to(self.device, non_blocking=non_block)

        x_global_t = x_track_t = shop_ids_t = shop_feats_t = None
        if self._use_multimodal and mm_extras:
            xg_list, xt_list, si_list, sf_list = zip(*mm_extras)
            x_global_t = torch.from_numpy(np.stack(xg_list, axis=0).astype(np.float32)).to(self.device, non_blocking=non_block)
            x_track_t = torch.from_numpy(np.stack(xt_list, axis=0).astype(np.float32)).to(self.device, non_blocking=non_block)
            shop_ids_t = torch.from_numpy(np.stack(si_list, axis=0).astype(np.int64)).to(self.device, non_blocking=non_block)
            shop_feats_t = torch.from_numpy(np.stack(sf_list, axis=0).astype(np.float32)).to(self.device, non_blocking=non_block)

        with torch.inference_mode():
            if self.device.type == "cuda" and self.use_amp:
                dtype = torch.float16 if self.amp_dtype == "float16" else torch.bfloat16
                with torch.autocast(device_type="cuda", dtype=dtype):
                    policy_logits, value_t, score_logits = self.network(
                        st, am,
                        x_global=x_global_t,
                        x_track=x_track_t,
                        shop_ids=shop_ids_t,
                        shop_feats=shop_feats_t,
                    )
            else:
                policy_logits, value_t, score_logits = self.network(
                    st, am,
                    x_global=x_global_t,
                    x_track=x_track_t,
                    shop_ids=shop_ids_t,
                    shop_feats=shop_feats_t,
                )

        value_t = value_t.squeeze(-1)
        # 201-bin: compute mean_points and score_utility on GPU (mirror server).
        score_logits_f = score_logits.float()
        p = torch.softmax(score_logits_f, dim=-1)
        bins = self._score_bins_t
        mean_points_t = (p * bins).sum(dim=-1)
        is_root_flags = [
            (int(nodes[i].to_move) == int(self._root.to_move)) if self._root is not None else True
            for i in range(len(nodes)) if terminal_out[i] is None
        ]
        centers_t = torch.tensor(
            [self._search_root_score if is_r else -self._search_root_score for is_r in is_root_flags],
            device=self.device, dtype=torch.float32,
        )
        scale = self._score_utility_scale
        sat = lambda x: (2.0 / math.pi) * torch.atan(x)
        static_util_t = (p * sat(bins.unsqueeze(0) / scale)).sum(dim=-1)
        dynamic_util_t = (p * sat((bins.unsqueeze(0) - centers_t.unsqueeze(1)) / scale)).sum(dim=-1)
        score_utility_t = self._search_effective_static_w * static_util_t + self._search_effective_dynamic_w * dynamic_util_t

        value_cpu = value_t.float().cpu().numpy()
        mean_points_cpu = mean_points_t.float().cpu().numpy()
        score_utility_cpu = score_utility_t.float().cpu().numpy()

        result_local: List[Tuple[float, float, float]] = [None] * len(nodes)  # type: ignore[list-item]
        for i in range(len(nodes)):
            if terminal_out[i] is not None:
                result_local[i] = self._get_terminal_v_s_u(nodes[i].state, nodes[i].to_move)
        nonterm_idx = 0
        for i, node in enumerate(nodes):
            if terminal_out[i] is not None:
                continue
            legal_idxs = legal_indices_list[nonterm_idx]
            idx_t = torch.from_numpy(legal_idxs).to(self.device)
            legal_logits = policy_logits[nonterm_idx].index_select(0, idx_t)
            priors = torch.softmax(legal_logits, dim=0)
            priors_cpu = priors.float().cpu().numpy()
            node._prior[:] = priors_cpu[:len(node.legal_actions)]
            if self.config.progressive_widening.enabled:
                _apply_progressive_widening_order(
                    node, self.config.progressive_widening,
                    is_root=False,
                    packing_config=self.config.packing_ordering,
                )
            node.normalize_priors()
            v = float(value_cpu[nonterm_idx])
            s_points = float(mean_points_cpu[nonterm_idx])
            su = float(score_utility_cpu[nonterm_idx])
            node.score_estimate = s_points
            u = v + su
            result_local[i] = (v, s_points, u)
            nonterm_idx += 1
        return result_local

    # ------------------------------------------------------------------
    # Single expand + evaluate
    # ------------------------------------------------------------------

    def _expand_and_evaluate(self, node: MCTSNode, precomputed_legal: Optional[list] = None) -> Tuple[float, float, float]:
        """Returns (value, score, utility) from node's to_move perspective. Sets _search_root_value when node is root."""
        if terminal_fast(node.state):
            node.terminal = True
            v, s, u = self._get_terminal_v_s_u(node.state, node.to_move)
            node.terminal_value = u
            return (v, s, u)

        node.legal_actions = precomputed_legal if precomputed_legal is not None else legal_actions_fast(node.state)
        if len(node.legal_actions) == 0:
            node.terminal = True
            v, s, u = self._get_terminal_v_s_u(node.state, node.to_move)
            node.terminal_value = u
            return (v, s, u)

        # Initialize arrays now that legal_actions are known
        node._init_arrays()

        if self._use_multimodal:
            x_spatial, x_global, x_track, shop_ids, shop_feats = self.state_encoder.encode_state_multimodal(node.state, node.to_move)
            state_np = x_spatial.astype(np.float32, copy=False)
            mm = (x_global, x_track, shop_ids, shop_feats)
        else:
            state_np = self.state_encoder.encode_state(node.state, node.to_move).astype(np.float32, copy=False)
            mm = None
        legal_action_indices, action_mask = encode_legal_actions_fast(node.legal_actions)

        if self.eval_client is not None:
            legal_idxs_np = np.asarray(legal_action_indices, dtype=np.int32)
            is_root = (int(node.to_move) == int(self._root.to_move)) if self._root is not None else True
            center = 0.0 if (self._root is not None and node is self._root) else (
                self._search_root_score if is_root else -self._search_root_score
            )
            if self._use_multimodal and mm is not None:
                xg, xt, si, sf = mm
                priors_legal, v, mean_points, score_utility = self.eval_client.evaluate_multimodal(
                    state_np, xg, xt, si, sf, action_mask, legal_idxs_np,
                    center, self._search_effective_static_w, self._search_effective_dynamic_w,
                )
            else:
                priors_legal, v, mean_points, score_utility = self.eval_client.evaluate(
                    state_np, action_mask, legal_idxs_np,
                    center, self._search_effective_static_w, self._search_effective_dynamic_w,
                )
            v, s_points, su = float(v), float(mean_points), float(score_utility)
            node.score_estimate = s_points
            if self._root is not None and node is self._root:
                self._search_root_value = v
            u = v + su
            node._prior[:] = priors_legal[:len(node.legal_actions)]
            # Root: PW order applied in search() after Dirichlet noise. Non-root: apply here.
            if self.config.progressive_widening.enabled and not (self._root is not None and node is self._root):
                _apply_progressive_widening_order(
                    node, self.config.progressive_widening,
                    is_root=False,
                )
        else:
            if self.network is None:
                raise RuntimeError("MCTS requires either a network or eval_client")

            state_tensor = torch.from_numpy(state_np).unsqueeze(0)
            action_mask_t = torch.from_numpy(action_mask).unsqueeze(0)

            non_block = self.device.type == "cuda"
            state_tensor = state_tensor.to(self.device, non_blocking=non_block)
            action_mask_t = action_mask_t.to(self.device, non_blocking=non_block)

            x_global_t = x_track_t = shop_ids_t = shop_feats_t = None
            if mm is not None:
                xg, xt, si, sf = mm
                x_global_t = torch.from_numpy(xg.astype(np.float32)).unsqueeze(0).to(self.device, non_blocking=non_block)
                x_track_t = torch.from_numpy(xt.astype(np.float32)).unsqueeze(0).to(self.device, non_blocking=non_block)
                shop_ids_t = torch.from_numpy(np.asarray(si, dtype=np.int64)).unsqueeze(0).to(self.device, non_blocking=non_block)
                shop_feats_t = torch.from_numpy(sf.astype(np.float32)).unsqueeze(0).to(self.device, non_blocking=non_block)

            with torch.inference_mode():
                if self.device.type == "cuda" and self.use_amp:
                    dtype = torch.float16 if self.amp_dtype == "float16" else torch.bfloat16
                    with torch.autocast(device_type="cuda", dtype=dtype):
                        policy_logits, value_t, score_logits = self.network(
                            state_tensor, action_mask_t,
                            x_global=x_global_t, x_track=x_track_t,
                            shop_ids=shop_ids_t, shop_feats=shop_feats_t,
                        )
                else:
                    policy_logits, value_t, score_logits = self.network(
                        state_tensor, action_mask_t,
                        x_global=x_global_t, x_track=x_track_t,
                        shop_ids=shop_ids_t, shop_feats=shop_feats_t,
                    )

                v = float(value_t.squeeze(-1).item())
                # 201-bin: mean_points and score_utility from logits (single sample).
                p = torch.softmax(score_logits.float().squeeze(0), dim=-1)
                bins = self._score_bins_t
                s_points = float((p * bins).sum().item())
                is_root = (int(node.to_move) == int(self._root.to_move)) if self._root is not None else True
                center = self._search_root_score if is_root else -self._search_root_score
                scale = self._score_utility_scale
                sat = lambda x: (2.0 / math.pi) * torch.atan(x)
                static_util = float((p * sat(bins / scale)).sum().item())
                dynamic_util = float((p * sat((bins - center) / scale)).sum().item())
                su = self._search_effective_static_w * static_util + self._search_effective_dynamic_w * dynamic_util
                node.score_estimate = s_points
                if self._root is not None and node is self._root:
                    self._search_root_value = v
                u = v + su

                idx_t = torch.as_tensor(legal_action_indices, device=self.device, dtype=torch.long)
                legal_logits = policy_logits.squeeze(0).index_select(0, idx_t)
                priors = torch.softmax(legal_logits, dim=0)

            priors_cpu = priors.float().cpu().numpy()
            node._prior[:] = priors_cpu[:len(node.legal_actions)]
            # Root: PW order applied in search() after Dirichlet noise. Non-root: apply here.
            if self.config.progressive_widening.enabled and not (self._root is not None and node is self._root):
                _apply_progressive_widening_order(
                    node, self.config.progressive_widening,
                    is_root=False,
                    packing_config=self.config.packing_ordering,
                )

        node.normalize_priors()
        return (v, s_points, u)

    # ------------------------------------------------------------------
    # Noise (FIX M4: use seeded RNG)
    # ------------------------------------------------------------------

    def _add_dirichlet_noise(self, node: MCTSNode) -> None:
        if not node.legal_actions or node._prior is None:
            return

        alpha = float(self.config.root_dirichlet_alpha)
        epsilon = float(self.config.root_noise_weight)

        noise = self._noise_rng.dirichlet([alpha] * len(node.legal_actions))

        # Vectorized noise mixing
        node._prior[:] = (1 - epsilon) * node._prior + epsilon * noise
        node.normalize_priors()

    # ------------------------------------------------------------------
    # Terminal value
    # ------------------------------------------------------------------

    def _get_terminal_v_s_u(self, state, to_move: int) -> Tuple[float, float, float]:
        """Terminal (value, score_points, utility) from to_move perspective.
        value ±1/0; score_points = raw margin in points (for WIN_FIRST score_sum); utility = value + score_utility.
        """
        score0 = compute_score_fast(state, 0)
        score1 = compute_score_fast(state, 1)
        winner = int(get_winner_fast(state))
        value = terminal_value_from_scores(
            score0=score0, score1=score1, winner=winner, to_move=int(to_move),
        )
        raw_margin_points = float(int(score0) - int(score1)) if int(to_move) == 0 else float(int(score1) - int(score0))
        is_root = (int(to_move) == int(self._root.to_move)) if self._root is not None else True
        su = self._compute_terminal_score_utility(raw_margin_points, is_root)
        utility = value + su
        return (value, raw_margin_points, utility)

    def _get_terminal_value(self, state, to_move: int) -> float:
        """Terminal utility only (for backward compat). Prefer _get_terminal_v_s_u for backup."""
        _, _, u = self._get_terminal_v_s_u(state, to_move)
        return u


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_optimized_mcts(
    network,
    config: dict,
    device: torch.device,
    state_encoder,
    action_encoder,
    eval_client=None,
    *,
    enable_tree_reuse: Optional[bool] = None,
) -> OptimizedAlphaZeroMCTS:
    """Factory function to create MCTS from config."""
    mcts_cfg = config.get("selfplay", {}).get("mcts", {}) or {}
    win_first = _parse_and_validate_win_first(mcts_cfg.get("win_first"))
    progressive_widening = _parse_progressive_widening(mcts_cfg.get("progressive_widening"))
    packing_ordering = _parse_packing_ordering(mcts_cfg.get("packing_ordering"))
    patch_tiebreak = _parse_patch_tiebreak(mcts_cfg.get("patch_tiebreak"))
    mcts_config = MCTSConfig(
        simulations=int(mcts_cfg.get("simulations", 800)),
        parallel_leaves=int(mcts_cfg.get("parallel_leaves", 32)),
        cpuct=float(mcts_cfg.get("cpuct", 1.5)),
        temperature=float(mcts_cfg.get("temperature", 1.0)),
        temperature_threshold=int(mcts_cfg.get("temperature_threshold", 15)),
        root_dirichlet_alpha=float(mcts_cfg.get("root_dirichlet_alpha", 0.3)),
        root_noise_weight=float(mcts_cfg.get("root_noise_weight", 0.25)),
        virtual_loss=float(mcts_cfg.get("virtual_loss", 1.0)),
        fpu_reduction=float(mcts_cfg.get("fpu_reduction", 0.25)),  # KataGo default
        # KataGo dual-head score utility (self-play defaults: static=0.0, dynamic=0.3)
        # Backward-compat: if only legacy score_utility_weight is set, treat it as static.
        static_score_utility_weight=float(mcts_cfg.get(
            "static_score_utility_weight",
            mcts_cfg.get("score_utility_weight", 0.0),  # legacy key → static (0.0 if absent)
        )),
        dynamic_score_utility_weight=float(mcts_cfg.get("dynamic_score_utility_weight", 0.3)),
        score_utility_scale=float(mcts_cfg.get("score_utility_scale", 30.0)),
        score_min=int(mcts_cfg.get("score_min", -100)),
        score_max=int(mcts_cfg.get("score_max", 100)),
        enable_tree_reuse=enable_tree_reuse if enable_tree_reuse is not None else bool(mcts_cfg.get("enable_tree_reuse", False)),
        win_first=win_first,
        progressive_widening=progressive_widening,
        packing_ordering=packing_ordering,
        patch_tiebreak=patch_tiebreak,
    )

    # [C1 FIX] Safety clamp: parallel_leaves must be < simulations so the tree
    # develops depth.  At least 2 batches ensures some re-visiting and deepening.
    if mcts_config.simulations > 1:
        max_leaves = max(1, mcts_config.simulations // 2)
        if mcts_config.parallel_leaves > max_leaves:
            mcts_config.parallel_leaves = max_leaves

    return OptimizedAlphaZeroMCTS(
        network,
        mcts_config,
        device,
        state_encoder,
        action_encoder,
        eval_client=eval_client,
        inference_settings=(config.get("inference", {}) or {}),
        full_config=config,
    )