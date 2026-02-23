"""
Optimized AlphaZero MCTS with In-Process Patchwork Engine

Tree transitions use engine actions: (AT_BUY, offset, piece_id, orient, top, left)
Neural evaluation uses slot-based encoding.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.game.patchwork_engine import (
    apply_action,
    apply_action_unchecked,
    current_player,
    current_player_fast,
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
)

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
    # KataGo Dual-Head: blend value [-1,1] and raw score margin into MCTS utility.
    # utility = value + score_utility_weight * score
    # Small weight because score magnitudes are much larger than value.
    score_utility_weight: float = 0.02


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
        "legal_actions", "_action_to_idx", "terminal", "terminal_value", "n_total",
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
        self.legal_actions: Optional[List[tuple]] = None
        self._action_to_idx: Optional[Dict[tuple, int]] = None
        self.terminal: bool = False
        self.terminal_value: Optional[float] = None
        self.n_total = 0

    def _init_arrays(self) -> None:
        """Allocate arrays once legal_actions are known."""
        n = len(self.legal_actions)
        self._visit_count = np.zeros(n, dtype=np.int32)
        self._total_value = np.zeros(n, dtype=np.float64)
        self._prior = np.zeros(n, dtype=np.float64)
        self._virtual_loss = np.zeros(n, dtype=np.float64)
        self._action_to_idx = {a: i for i, a in enumerate(self.legal_actions)}

    def is_expanded(self) -> bool:
        return self.legal_actions is not None

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

    def select_action(self, cpuct: float, fpu_reduction: float = 0.0) -> tuple:
        """Vectorized UCB action selection."""
        n = self._visit_count.astype(np.float64)
        vl = self._virtual_loss
        effective_n = n + vl
        # Q values
        with np.errstate(divide='ignore', invalid='ignore'):
            q = np.where(effective_n > 0,
                          (self._total_value - vl) / effective_n,
                          -fpu_reduction if self.parent else 0.0)
        # Exploration
        sqrt_total = math.sqrt(self.n_total + 1)
        exploration = cpuct * self._prior * sqrt_total / (1.0 + n)
        ucb = q + exploration
        best_idx = int(np.argmax(ucb))
        return self.legal_actions[best_idx]

    def add_virtual_loss(self, action: tuple, loss: float = 1.0) -> None:
        idx = self._action_to_idx[action]
        self._virtual_loss[idx] += loss

    def remove_virtual_loss(self, action: tuple, loss: float = 1.0) -> None:
        idx = self._action_to_idx[action]
        self._virtual_loss[idx] = max(0.0, self._virtual_loss[idx] - loss)

    def update(self, action: tuple, value: float) -> None:
        idx = self._action_to_idx[action]
        self._visit_count[idx] += 1
        self._total_value[idx] += value
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
            self._use_multimodal = use_film or (in_ch == 56)
        elif full_config:
            data_cfg = full_config.get("data", {}) or {}
            enc = str(data_cfg.get("encoding_version", ""))
            if enc and enc.lower() in ("gold_v2_32ch", "gold_v2_multimodal"):
                self._use_multimodal = True
            else:
                in_ch = int(full_config.get("network", {}).get("input_channels", 56))
                self._use_multimodal = in_ch == 56
        else:
            self._use_multimodal = False

        # FIX (M4): seeded RNG for Dirichlet noise — reproducible searches.
        # Initialised with a default; callers can reset via set_noise_seed().
        self._noise_rng = np.random.default_rng(42)

    def set_noise_seed(self, seed: int) -> None:
        """Reset the Dirichlet noise RNG. Call before each game for reproducibility."""
        self._noise_rng = np.random.default_rng(seed)

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

        root: Optional[MCTSNode] = None
        if self._root is not None and self._root.children:
            for _action, child in self._root.children.items():
                if child is not None and np.array_equal(child.state, state):
                    self._root = child
                    root = child
                    break
        if root is None:
            root = MCTSNode(state, int(to_move))
            self._expand_and_evaluate(root, precomputed_legal=root_legal_actions)
            if add_noise and root.legal_actions and len(root.legal_actions) > 0:
                self._add_dirichlet_noise(root)
        else:
            self._root = root

        # Reused root may need expansion if it was a leaf (shouldn't happen, but be safe)
        if not root.is_expanded() and not root.terminal:
            self._expand_and_evaluate(root, precomputed_legal=root_legal_actions)

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

    def select_action(
        self,
        visit_counts: Dict[tuple, int],
        temperature: float = 1.0,
        deterministic: bool = False,
    ) -> tuple:
        """Select action from visit counts."""
        if not visit_counts:
            raise ValueError("No visit counts")

        if temperature == 0 or deterministic:
            return max(visit_counts.items(), key=lambda x: x[1])[0]

        actions = list(visit_counts.keys())
        visits = np.array([visit_counts[a] for a in actions], dtype=np.float64)

        if visits.sum() == 0:
            return actions[int(self._noise_rng.integers(0, len(actions)))]

        visits_temp = visits ** (1.0 / temperature)
        probs = visits_temp / visits_temp.sum()
        return actions[int(self._noise_rng.choice(len(actions), p=probs))]

    # ------------------------------------------------------------------
    # Tree traversal
    # ------------------------------------------------------------------

    def _select_leaf(self, root: MCTSNode):
        node = root
        search_path: List[Tuple[MCTSNode, tuple]] = []

        while node.is_expanded() and not node.terminal:
            action = node.select_action(self.config.cpuct, self.config.fpu_reduction)
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
        leaf_value: float,
        leaf_to_move: int,
    ) -> None:
        """
        Backup value through the search path.

        Convention: leaf_value is from the LEAF node's to_move perspective.
        At each parent, we determine the value from the PARENT's perspective:
          - If parent.to_move == leaf_to_move => same sign
          - If parent.to_move != leaf_to_move => negate
        """
        for parent_node, action_taken in reversed(search_path):
            parent_node.remove_virtual_loss(action_taken, self.config.virtual_loss)

            # Value from parent's perspective
            if parent_node.to_move == leaf_to_move:
                value_for_parent = leaf_value
            else:
                value_for_parent = -leaf_value

            parent_node.update(action_taken, value_for_parent)

    # ------------------------------------------------------------------
    # Single simulation
    # ------------------------------------------------------------------

    def _simulate(self, root: MCTSNode):
        node = root
        search_path: List[Tuple[MCTSNode, tuple]] = []

        while node.is_expanded() and not node.terminal:
            action = node.select_action(self.config.cpuct, self.config.fpu_reduction)
            search_path.append((node, action))
            node.add_virtual_loss(action, self.config.virtual_loss)

            if action not in node.children or node.children[action] is None:
                next_state = apply_action_unchecked(node.state, action)
                next_to_move = current_player_fast(next_state)
                node.children[action] = MCTSNode(next_state, next_to_move, parent=node)

            node = node.children[action]

        if node.terminal:
            value = self._get_terminal_value(node.state, node.to_move)
        else:
            value = self._expand_and_evaluate(node)

        self._backup_path(search_path, value, node.to_move)

    # ------------------------------------------------------------------
    # Batched simulation
    # ------------------------------------------------------------------

    def _simulate_batched(self, root: MCTSNode) -> None:
        sims_remaining = int(self.config.simulations)
        batch_n = max(1, int(self.config.parallel_leaves))

        while sims_remaining > 0:
            k = min(batch_n, sims_remaining)

            leaves: List[MCTSNode] = []
            paths: List[List[Tuple[MCTSNode, tuple]]] = []
            terminal_data: List[Tuple[List[Tuple[MCTSNode, tuple]], float, int]] = []

            for _ in range(k):
                leaf, path = self._select_leaf(root)
                if leaf.terminal or terminal_fast(leaf.state):
                    v = self._get_terminal_value(leaf.state, leaf.to_move)
                    terminal_data.append((path, v, leaf.to_move))
                else:
                    leaves.append(leaf)
                    paths.append(path)

            values: List[float] = []
            if leaves:
                values = self._batch_expand_and_evaluate(leaves)

            for path, v, leaf_to_move in zip(paths, values, [l.to_move for l in leaves]):
                self._backup_path(path, v, leaf_to_move)
            for path, v, leaf_to_move in terminal_data:
                self._backup_path(path, v, leaf_to_move)

            sims_remaining -= k

    # ------------------------------------------------------------------
    # Batch expand + evaluate
    # ------------------------------------------------------------------

    def _batch_expand_and_evaluate(self, nodes: List[MCTSNode]) -> List[float]:
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
                n_lj = len(legal_action_indices)
                li32 = legal_action_indices.astype(np.int32, copy=False)
                self._shm_buf.legalidxs_view(slot, n_lj)[:] = li32
                self._shm_buf.write_nlegal(slot, n_lj)
                shm_slot_list.append((slot, n_lj))
            elif self._use_multimodal:
                x_spatial, x_global, x_track, shop_ids, shop_feats = encode_state_multimodal(node.state, node.to_move)
                states_np.append(x_spatial.astype(np.float32, copy=False))
                mm_extras.append((x_global.astype(np.float32), x_track.astype(np.float32), shop_ids, shop_feats.astype(np.float32)))
            else:
                state_np = self.state_encoder.encode_state(node.state, node.to_move).astype(np.float32, copy=False)
                states_np.append(state_np)

            n_nonterminal += 1

        if n_nonterminal == 0:
            return [float(v) for v in terminal_out]

        # GPU server mode
        if self.eval_client is not None:
            req_rids = []
            if shm_slot_list:
                # SHM path: data already in shared memory — submit tiny metadata dicts.
                for slot, n_lj in shm_slot_list:
                    rid = self.eval_client.submit_shm(slot, n_lj)
                    req_rids.append(rid)
            elif self._use_multimodal and mm_extras:
                for j in range(len(states_np)):
                    xg, xt, si, sf = mm_extras[j]
                    legal_i32 = legal_indices_list[j].astype(np.int32, copy=False)
                    rid = self.eval_client.submit_multimodal(
                        states_np[j], xg, xt, si, sf, masks_np[j], legal_i32
                    )
                    req_rids.append(rid)
            else:
                for j in range(len(states_np)):
                    legal_i32 = legal_indices_list[j].astype(np.int32, copy=False)
                    rid = self.eval_client.submit_legacy(states_np[j], masks_np[j], legal_i32)
                    req_rids.append(rid)

            resp_priors = []
            resp_values = []
            for rid in req_rids:
                priors_legal, value, score = self.eval_client.receive(rid)
                resp_priors.append(priors_legal)
                w = float(self.config.score_utility_weight)
                utility = float(value) + w * float(score)
                resp_values.append(utility)

            out_values = []
            nonterm_idx = 0
            for i, node in enumerate(nodes):
                if terminal_out[i] is not None:
                    out_values.append(float(terminal_out[i]))
                    continue

                priors_legal = resp_priors[nonterm_idx]
                node._prior[:] = priors_legal[:len(node.legal_actions)]
                node.normalize_priors()

                out_values.append(resp_values[nonterm_idx])
                nonterm_idx += 1

            return out_values

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
                    policy_logits, value_t, score_t = self.network(
                        st, am,
                        x_global=x_global_t,
                        x_track=x_track_t,
                        shop_ids=shop_ids_t,
                        shop_feats=shop_feats_t,
                    )
            else:
                policy_logits, value_t, score_t = self.network(
                    st, am,
                    x_global=x_global_t,
                    x_track=x_track_t,
                    shop_ids=shop_ids_t,
                    shop_feats=shop_feats_t,
                )

        value_t = value_t.squeeze(-1)
        score_t = score_t.squeeze(-1)
        w = float(self.config.score_utility_weight)
        utility_t = value_t + w * score_t

        out_values: List[float] = []
        nonterm_idx = 0

        for i, node in enumerate(nodes):
            if terminal_out[i] is not None:
                out_values.append(float(terminal_out[i]))
                continue

            legal_idxs = legal_indices_list[nonterm_idx]
            idx_t = torch.from_numpy(legal_idxs).to(self.device)

            legal_logits = policy_logits[nonterm_idx].index_select(0, idx_t)
            priors = torch.softmax(legal_logits, dim=0)
            priors_cpu = priors.float().cpu().numpy()

            node._prior[:] = priors_cpu[:len(node.legal_actions)]
            node.normalize_priors()

            out_values.append(float(utility_t[nonterm_idx].item()))
            nonterm_idx += 1

        return out_values

    # ------------------------------------------------------------------
    # Single expand + evaluate
    # ------------------------------------------------------------------

    def _expand_and_evaluate(self, node: MCTSNode, precomputed_legal: Optional[list] = None) -> float:
        if terminal_fast(node.state):
            node.terminal = True
            value = self._get_terminal_value(node.state, node.to_move)
            node.terminal_value = value
            return value

        node.legal_actions = precomputed_legal if precomputed_legal is not None else legal_actions_fast(node.state)
        if len(node.legal_actions) == 0:
            node.terminal = True
            value = self._get_terminal_value(node.state, node.to_move)
            node.terminal_value = value
            return value

        # Initialize arrays now that legal_actions are known
        node._init_arrays()

        if self._use_multimodal:
            x_spatial, x_global, x_track, shop_ids, shop_feats = encode_state_multimodal(node.state, node.to_move)
            state_np = x_spatial.astype(np.float32, copy=False)
            mm = (x_global, x_track, shop_ids, shop_feats)
        else:
            state_np = self.state_encoder.encode_state(node.state, node.to_move).astype(np.float32, copy=False)
            mm = None
        legal_action_indices, action_mask = encode_legal_actions_fast(node.legal_actions)

        if self.eval_client is not None:
            legal_idxs_np = np.asarray(legal_action_indices, dtype=np.int32)
            if self._use_multimodal and mm is not None:
                xg, xt, si, sf = mm
                priors_legal, v, s = self.eval_client.evaluate_multimodal(
                    state_np, xg, xt, si, sf, action_mask, legal_idxs_np
                )
            else:
                priors_legal, v, s = self.eval_client.evaluate(state_np, action_mask, legal_idxs_np)
            # Utility blend for MCTS backup
            w = float(self.config.score_utility_weight)
            value = float(v) + w * float(s)
            # Bulk set priors directly into array
            node._prior[:] = priors_legal[:len(node.legal_actions)]
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
                        policy_logits, value_t, score_t = self.network(
                            state_tensor, action_mask_t,
                            x_global=x_global_t, x_track=x_track_t,
                            shop_ids=shop_ids_t, shop_feats=shop_feats_t,
                        )
                else:
                    policy_logits, value_t, score_t = self.network(
                        state_tensor, action_mask_t,
                        x_global=x_global_t, x_track=x_track_t,
                        shop_ids=shop_ids_t, shop_feats=shop_feats_t,
                    )

                v = float(value_t.item())
                s = float(score_t.item())
                w = float(self.config.score_utility_weight)
                value = v + w * s

                idx_t = torch.as_tensor(legal_action_indices, device=self.device, dtype=torch.long)
                legal_logits = policy_logits.squeeze(0).index_select(0, idx_t)
                priors = torch.softmax(legal_logits, dim=0)

            priors_cpu = priors.float().cpu().numpy()
            # Bulk set priors directly into array
            node._prior[:] = priors_cpu[:len(node.legal_actions)]

        # Normalize priors
        node.normalize_priors()

        return value

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

    def _get_terminal_value(self, state, to_move: int) -> float:
        """
        Terminal utility from perspective of to_move.
        Value is strictly -1/0/1; for terminals we use value as utility
        (score_utility_weight only applies to non-terminal network evaluation).
        """
        score0 = compute_score_fast(state, 0)
        score1 = compute_score_fast(state, 1)
        winner = int(get_winner_fast(state))
        return terminal_value_from_scores(
            score0=score0,
            score1=score1,
            winner=winner,
            to_move=int(to_move),
        )


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
        score_utility_weight=float(mcts_cfg.get("score_utility_weight", 0.02)),
        enable_tree_reuse=enable_tree_reuse if enable_tree_reuse is not None else bool(mcts_cfg.get("enable_tree_reuse", False)),
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