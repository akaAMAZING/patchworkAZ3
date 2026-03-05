"""
Optimized Self-Play Worker — In-Process Engine

Uses the flat int32 numpy array game engine directly.
Generates training data: (encoded_state, policy_target, value_target) per position.

Bootstrap iteration uses pure MCTS (no neural network).
Later iterations use AlphaZero MCTS with neural guidance.

Data augmentation: each position is optionally stored with a vertical and/or
horizontal flip (controlled by config ``selfplay.augment_flips``). Because buy-action
orient remapping requires knowing which piece sits in each slot, augmentation is
performed here where we have the full game state — not in the training dataloader.

KataGo features:
- Playout cap randomization: randomly reduce MCTS sims for some moves
  to increase data diversity without increasing compute budget.
  (see: KataGo paper, Section 3.5 "Playout Cap Randomization")

"""

import logging
import math
import random
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)
import torch

# Absolute imports for multiprocessing safety
from src.game.patchwork_engine import (
    apply_action,
    apply_action_unchecked,
    compute_score,
    compute_score_fast,
    current_player,
    current_player_fast,
    empty_count_from_occ,
    get_winner,
    get_winner_fast,
    legal_actions_list,
    legal_actions_fast,
    new_game,
    terminal,
    terminal_fast,
    NEUTRAL,
    BONUS_OWNER,
    P0_BUTTONS,
    P0_OCC0, P0_OCC1, P0_OCC2,
    P1_BUTTONS,
    P1_OCC0, P1_OCC1, P1_OCC2,
    TIE_PLAYER,
)

from src.mcts.alphazero_mcts_optimized import create_optimized_mcts, encode_legal_actions_fast, engine_action_to_flat_index
from src.network.encoder import StateEncoder, ActionEncoder, get_slot_piece_id, encode_state_multimodal
from src.network.model import create_network, load_model_checkpoint, get_state_dict_for_inference
from src.mcts.gpu_eval_client import GPUEvalClient
from src.network.d4_augmentation import (
    apply_d4_augment, apply_ownership_transform,
    get_d4_transform_tag, get_d4_transform_idx, D4_COUNT,
)
from src.training.value_targets import value_and_score_from_scores


# =========================================================================
# Pure MCTS (for bootstrap — no neural network)
# =========================================================================

class PureMCTSNode:
    """Node for pure MCTS (no neural network)."""

    __slots__ = ("state", "to_move", "parent", "children", "untried_actions", "visits", "value_sum")

    def __init__(self, state, to_move: int, parent=None):
        self.state = state
        self.to_move = to_move
        self.parent = parent
        self.children: Dict[tuple, "PureMCTSNode"] = {}
        self.untried_actions: Optional[List[tuple]] = None
        self.visits = 0
        self.value_sum = 0.0

    def is_fully_expanded(self) -> bool:
        if self.untried_actions is None:
            self.untried_actions = legal_actions_fast(self.state)
        return len(self.untried_actions) == 0

    def best_child(self, c: float = 1.4):
        best_score = -1e9
        best = None
        for child in self.children.values():
            if child.visits == 0:
                score = 1e9
            else:
                # child.value_sum is from child.to_move perspective;
                # negate when child is the opponent so we maximize
                # from the PARENT's perspective.
                q = child.value_sum / child.visits
                if child.to_move != self.to_move:
                    q = -q
                explore = c * math.sqrt(math.log(self.visits + 1) / (child.visits))
                score = q + explore
            if score > best_score:
                best_score = score
                best = child
        return best

    def expand(self):
        if self.untried_actions is None:
            self.untried_actions = legal_actions_fast(self.state)
        if not self.untried_actions:
            return None
        a = self.untried_actions.pop()
        ns = apply_action_unchecked(self.state, a)
        child = PureMCTSNode(ns, current_player_fast(ns), parent=self)
        self.children[a] = child
        return child

    def backprop(self, value: float):
        """Backpropagate value up the tree.

        CRITICAL: Patchwork doesn't alternate turns strictly (same player can move multiple times).
        Only negate value when player identity changes between child and parent.
        """
        self.visits += 1
        self.value_sum += value
        if self.parent is not None:
            # Negate only if player changed (not strict alternation like chess)
            if self.to_move != self.parent.to_move:
                self.parent.backprop(-value)
            else:
                # Same player moved again (common in Patchwork)
                self.parent.backprop(value)


def pure_mcts_search(state, sims: int, rng: random.Random) -> Dict[tuple, int]:
    """Pure MCTS (no network). Returns visit counts keyed by engine actions."""
    root = PureMCTSNode(state, current_player_fast(state))
    sims = int(sims)
    for _ in range(sims):
        node = root
        st = state

        # Selection
        while node.is_fully_expanded() and node.children:
            node = node.best_child()
            st = node.state

        # Expansion
        if not terminal_fast(st):
            node = node.expand() or node
            st = node.state

        # Rollout
        rollout_state = st
        while not terminal_fast(rollout_state):
            actions = legal_actions_fast(rollout_state)
            rollout_state = apply_action_unchecked(rollout_state, rng.choice(actions))

        # Rollout value from LEAF to-move perspective (negamax)
        to_move = current_player_fast(st)
        winner = get_winner_fast(rollout_state)
        leaf_value = 1.0 if int(winner) == to_move else -1.0

        node.backprop(leaf_value)

    return {a: int(ch.visits) for a, ch in root.children.items()}


def _select_from_visits(visit_counts: Dict[tuple, int], temperature: float, rng: random.Random) -> tuple:
    if not visit_counts:
        raise ValueError("empty visit_counts")
    if temperature <= 0:
        return max(visit_counts.items(), key=lambda kv: kv[1])[0]
    actions = list(visit_counts.keys())
    visits = np.asarray([visit_counts[a] for a in actions], dtype=np.float64)
    s = float(visits.sum())
    if s <= 0:
        return rng.choice(actions)
    visits = visits ** (1.0 / float(temperature))
    probs = visits / float(visits.sum())
    return actions[int(rng.choices(range(len(actions)), weights=probs, k=1)[0])]


# =========================================================================
# Optimized SelfPlay Worker
# =========================================================================

class OptimizedSelfPlayWorker:
    """
    Self-play worker that can:
      - Run pure MCTS (bootstrap) or AlphaZero MCTS (network-guided)
      - Optionally do action/state augmentation via flips
      - Use either local GPU inference or a shared GPU inference server (req/resp queues)
    """

    def __init__(self, network_path: Optional[str], config: dict, device: str = "cpu", req_q=None, resp_q=None, worker_id: int = 0):
        self.config = config
        self.device_str = device
        self.req_q = req_q
        self.resp_q = resp_q
        self.worker_id = worker_id
        self.network_path = network_path

        self.state_encoder = StateEncoder()
        self.action_encoder = ActionEncoder()
        self._use_multimodal = self._is_gold_v2_config(config)

        sp = (config.get("selfplay", {}) or {})
        mcfg = (sp.get("mcts", {}) or {})

        if "augmentation" in sp:
            self.augmentation = str(sp["augmentation"])
        elif "augment_flips" in sp:
            self.augmentation = "flip" if sp.get("augment_flips", False) else "none"
        else:
            self.augmentation = "flip"
        if self.augmentation not in ("none", "flip", "d4"):
            self.augmentation = "flip"
        self.augment_flips = self.augmentation == "flip"
        # Dynamic D4: store 1 canonical position (not 8x); augmentation applied at training time
        self.store_canonical_only = bool(sp.get("store_canonical_only", True))
        # FIX: Changed "mcts_sims" to "mcts_simulations" to match config key
        self.bootstrap_sims = int((sp.get("bootstrap", {}) or {}).get("mcts_simulations", 32))

        self.temperature = float(mcfg.get("temperature", 1.0))
        self.temperature_threshold = int(mcfg.get("temperature_threshold", 15))
        self.policy_target_mode = str(sp.get("policy_target_mode", "visits")).lower()
        if self.policy_target_mode not in ("visits", "visits_temperature_shaped"):
            self.policy_target_mode = "visits"

        # MAJOR FIX: Enforce max_game_length from config
        self.max_game_length = int(sp.get("max_game_length", 200))

        # KataGo Dual-Head: value is strictly -1/0/1; score is raw margin

        # Q-value mixing (Oracle Part 4 / Leela Chess Zero):
        # Mix root MCTS Q-value with game outcome for value targets.
        # q_weight=0.0 → pure game outcome (standard AlphaZero)
        # q_weight=0.5 → 50/50 mix (faster convergence, risk of feedback loops)
        self.q_value_weight = float(sp.get("q_value_weight", 0.0))

        # KataGo Playout Cap Randomization: randomly use fewer sims for some moves
        # to increase training data diversity without increasing compute.
        pcr = sp.get("playout_cap_randomization", {}) or {}
        self.pcr_enabled = bool(pcr.get("enabled", False))
        self.pcr_fraction = float(pcr.get("cap_fraction", 0.25))  # fraction of sims for "fast" moves
        self.pcr_probability = float(pcr.get("fast_probability", 0.25))  # probability of using fast sims

        self.base_simulations = int(mcfg.get("simulations", 800))

        self._rng = random.Random(int(config.get("seed", 42)) + self.worker_id * 1000)
        self.mcts = None
        assert hasattr(self, "_rng"), "OptimizedSelfPlayWorker must have _rng initialized"

        backend = (sp.get("inference_backend") or "local").lower()
        if (req_q is not None and resp_q is not None):
            backend = "server"

        eval_client = None
        network = None
        device_t = torch.device("cpu")

        if network_path:
            if backend == "server":
                eval_client = GPUEvalClient(
                    req_q,
                    resp_q,
                    worker_id=self.worker_id,
                    timeout_s=float(sp.get("api_timeout", 60.0)),
                    retry_attempts=int(sp.get("api_retry_attempts", 3)),
                )
                device_t = torch.device("cpu")
            else:
                hw = (config.get("hardware", {}) or {}).get("device", "cpu")
                if hw == "cuda" and torch.cuda.is_available():
                    device_t = torch.device("cuda")
                network = create_network(config)
                ckpt = torch.load(network_path, map_location=device_t, weights_only=False)
                state_dict = get_state_dict_for_inference(ckpt, config, for_selfplay=True)
                load_model_checkpoint(network, state_dict)
                network.eval()
                network.to(device_t)

            self.mcts = create_optimized_mcts(
                network=network,
                config=config,
                device=device_t,
                state_encoder=self.state_encoder,
                action_encoder=self.action_encoder,
                eval_client=eval_client,
            )

    def _is_gold_v2_config(self, config: dict) -> bool:
        enc = str((config.get("data", {}) or {}).get("encoding_version", ""))
        if enc:
            return enc.lower() in ("gold_v2_32ch", "gold_v2_multimodal")
        return int((config.get("network", {}) or {}).get("input_channels", 56)) in (32, 56)

    # ----------------------------
    # Utilities for augmentation
    # ----------------------------

    def _maybe_augment(self, state_enc: np.ndarray, action_mask: np.ndarray, policy: np.ndarray, value: float, raw_state) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, float, str]]:
        """
        Return augmented variants based on self.augmentation:
          - none: 1 item (identity)
          - flip: 4 items (id, v, h, vh) — D4 transforms 0, 6, 4, 2
          - d4: 8 items (all D4 transforms)

        Each tuple includes a transform tag for ownership mapping.
        """
        items = []
        slot_piece_ids = [get_slot_piece_id(raw_state, i) for i in range(3)]

        if self.augmentation == "none":
            items.append((state_enc.copy(), action_mask.copy(), policy.copy(), value, "id"))
            return items

        if self.augmentation == "flip":
            transform_indices = [0, 6, 4, 2]
        else:
            transform_indices = list(range(D4_COUNT))

        for ti in transform_indices:
            s2, p2, m2 = apply_d4_augment(state_enc, policy, action_mask, slot_piece_ids, ti)
            tag = get_d4_transform_tag(ti)
            items.append((s2, m2, p2, value, tag))

        return items

    # ----------------------------
    # Main game loop
    # ----------------------------

    def play_game(self, game_idx: int, iteration: int, seed: Optional[int]) -> Dict:
        """
        Play a single game, returning dict with:
          - states: list[np.ndarray]
          - action_masks: list[np.ndarray]
          - policies: list[np.ndarray]
          - values: list[float]
          - game_length, winner, final_score_diff

        The returned lists already include augmentation if enabled.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed % (2**32 - 1))
            torch.manual_seed(seed)

        # (S4) Seed a separate RNG for MCTS noise (so different workers diverge even if OS RNG aligns)
        self._rng.seed((seed or 0) + 1337)
        if self.mcts is not None:
            # Ensure Dirichlet noise is per-game reproducible and varies across games
            self.mcts.set_noise_seed((seed or 0) + 9001)
        # CRITICAL FIX: Pass seed to new_game for deterministic piece circle shuffle
        st = new_game(seed=seed)

        states: List[np.ndarray] = []
        masks: List[np.ndarray] = []
        policies: List[np.ndarray] = []
        values: List[float] = []
        score_margins: List[float] = []
        stored_players: List[int] = []
        stored_flip_types: List[str] = []
        stored_slot_piece_ids: List[Tuple[int, ...]] = []
        stored_global_states: List[np.ndarray] = []
        stored_track_states: List[np.ndarray] = []
        stored_shop_ids: List[np.ndarray] = []
        stored_shop_feats: List[np.ndarray] = []

        # Policy collapse canary: track entropy and sharpness per move
        move_entropies: List[float] = []
        move_top1_probs: List[float] = []
        move_num_legal: List[int] = []

        # Root Q-values per move (for Q-value mixing in value targets)
        move_root_qs: List[float] = []

        # Position redundancy tracking (Oracle Part 2): hash board states
        seen_state_hashes: set = set()

        move_count = 0

        while (not terminal_fast(st)) and (move_count < self.max_game_length):
            to_move = current_player_fast(st)
            legal = legal_actions_fast(st)
            if not legal:
                break

            if self._use_multimodal:
                x_spatial, x_global, x_track, shop_ids, shop_feats = encode_state_multimodal(st, to_move)
                enc = x_spatial.astype(np.float32, copy=False)
            else:
                enc = self.state_encoder.encode_state(st, to_move).astype(np.float32, copy=False)

            # Fast direct encoding: skip string tuple intermediaries
            _, mask = encode_legal_actions_fast(legal)

            if self.network_path and self.mcts is not None:
                # KataGo Playout Cap Randomization: use fewer sims for some moves
                # to diversify training data. "Fast" moves still contribute to the
                # policy target but with less search depth.
                use_full_sims = True
                if self.pcr_enabled and self._rng.random() < self.pcr_probability:
                    reduced_sims = max(1, int(self.base_simulations * self.pcr_fraction))
                    self.mcts.config.simulations = reduced_sims
                    use_full_sims = False
                else:
                    self.mcts.config.simulations = self.base_simulations

                visit_counts, _, root_q = self.mcts.search(state=st, to_move=to_move, move_number=move_count, add_noise=True, root_legal_actions=legal)
                move_root_qs.append(root_q)
                # ACTION SELECTION: use temperature schedule (greedy after threshold)
                action_temp = self.temperature if move_count < self.temperature_threshold else 0.0
                action = self.mcts.select_action(visit_counts, temperature=action_temp, deterministic=(action_temp <= 0))
                # POLICY TARGET: visits (normalized counts) or visits_temperature_shaped (legacy)
                if self.policy_target_mode == "visits":
                    pi = self.action_encoder.create_target_policy(visit_counts, mode="visits")
                else:
                    policy_temp = max(self.temperature, 1e-6)
                    pi = self.action_encoder.create_target_policy(
                        visit_counts, temperature=policy_temp, mode="visits_temperature_shaped"
                    )
                assert np.isfinite(pi).all() and abs(float(pi.sum()) - 1.0) < 1e-5, (
                    f"policy target invalid: sum={pi.sum()} finite={np.isfinite(pi).all()}"
                )
                if move_count == 0:
                    logger.debug(
                        "[POLICY_TARGET] iter=%d mode=%s action_selection_temp=%.2f policy_target_temp=%s",
                        iteration, self.policy_target_mode, action_temp,
                        "N/A" if self.policy_target_mode == "visits" else f"{policy_temp:.2f}",
                    )
            else:
                visit_counts = pure_mcts_search(st, sims=self.bootstrap_sims, rng=self._rng)
                temp_bootstrap = 1.0  # Bootstrap action selection
                action = _select_from_visits(visit_counts, temperature=temp_bootstrap, rng=self._rng)
                # Policy target: same mode as AlphaZero path
                if self.policy_target_mode == "visits":
                    pi = self.action_encoder.create_target_policy(visit_counts, mode="visits")
                else:
                    policy_temp = max(self.temperature, 1e-6)
                    pi = self.action_encoder.create_target_policy(
                        visit_counts, temperature=policy_temp, mode="visits_temperature_shaped"
                    )
                assert np.isfinite(pi).all() and abs(float(pi.sum()) - 1.0) < 1e-5, (
                    f"policy target invalid (bootstrap): sum={pi.sum()} finite={np.isfinite(pi).all()}"
                )
                move_root_qs.append(0.0)  # No Q-value for pure MCTS bootstrap

            # enforce legal mask + renorm (prevents illegal leakage)
            pi = (pi * mask).astype(np.float32, copy=False)
            s = float(pi.sum())
            if s > 0:
                pi /= s
            else:
                li = np.nonzero(mask > 0)[0]
                if len(li) > 0:
                    pi[li] = 1.0 / float(len(li))

            # Policy collapse canary: track entropy and sharpness
            nonzero = pi[pi > 0]
            move_entropies.append(float(-np.sum(nonzero * np.log(nonzero))))
            move_top1_probs.append(float(np.max(pi)))
            move_num_legal.append(int(np.sum(mask > 0)))

            # Position redundancy: hash the encoded state (original only, not augments)
            seen_state_hashes.add(hash(enc.tobytes()))

            # Defensive invariant: chosen action index must be legal in current mask.
            selected_idx = int(engine_action_to_flat_index(action))
            if mask[selected_idx] <= 0:
                raise RuntimeError("Selected action is not legal under current action mask")

            # Store training target for this position (value filled after game)
            # DYNAMIC D4: Store only 1 canonical perspective when store_canonical_only is True.
            # Slot piece IDs enable correct D4 transforms at training time (in Dataset __getitem__).
            slot_piece_ids = [get_slot_piece_id(st, i) for i in range(3)]
            slot_piece_ids_arr = [(p if p is not None else -1) for p in slot_piece_ids]

            if self.store_canonical_only:
                states.append(enc.copy())
                if self._use_multimodal:
                    stored_global_states.append(x_global.copy())
                    stored_track_states.append(x_track.copy())
                    stored_shop_ids.append(shop_ids.copy())
                    stored_shop_feats.append(shop_feats.copy())
                masks.append(mask.copy())
                policies.append(pi.copy())
                values.append(0.0)
                score_margins.append(0.0)
                stored_players.append(to_move)
                stored_flip_types.append("id")
                stored_slot_piece_ids.append(tuple(slot_piece_ids_arr))
            else:
                # Legacy: pre-multiply by D4 at self-play time (spatial only for multimodal)
                aug_items = self._maybe_augment(enc, mask, pi, 0.0, st)
                for a_enc, a_mask, a_pi, _, flip_type in aug_items:
                    states.append(a_enc)
                    if self._use_multimodal:
                        stored_global_states.append(x_global.copy())
                        stored_track_states.append(x_track.copy())
                        stored_shop_ids.append(shop_ids.copy())
                        stored_shop_feats.append(shop_feats.copy())
                    masks.append(a_mask)
                    policies.append(a_pi)
                    values.append(0.0)
                    score_margins.append(0.0)
                    stored_players.append(to_move)
                    stored_flip_types.append(flip_type)

            # Apply action (unchecked — action comes from legal_actions)
            st = apply_action_unchecked(st, action)
            move_count += 1

            # Safety to avoid runaway loops in case of bugs (should never trigger now)
            if move_count > 5000:
                break

        # Terminal evaluation (score-based) — fast path (no upgrade_state)
        score0 = compute_score_fast(st, 0)
        score1 = compute_score_fast(st, 1)
        final_score_diff = float(score0 - score1)
        terminal_winner = int(get_winner_fast(st))

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
                    # Clamp Q to [-1, 1] before mixing into the value target.
                    # MCTS utility (value + score_utility) can exceed ±1 at terminal/near-terminal
                    # positions; the value head is bounded by tanh to [-1,1], so training on
                    # targets outside this range causes gradient saturation.  Score information
                    # already enters training via the dedicated score head target (z_score).
                    q_clamped = max(-1.0, min(1.0, q))
                    values[i] = (1.0 - q_w) * z_value + q_w * q_clamped
                else:
                    values[i] = z_value
            else:
                values[i] = z_value

        # ---------------------------------------------------------------
        # Ownership targets (KataGo-style auxiliary signal)
        #
        # Patchwork has *separate* 9x9 boards per player, so we predict
        # two binary occupancy maps (2, 9, 9) from the current player's
        # perspective:
        #   Channel 0: current player's board (1=filled, 0=empty at end)
        #   Channel 1: opponent's board (1=filled, 0=empty at end)
        #
        # This provides 2 × 81 = 162 dense spatial training signals that
        # help the network learn packing efficiency and dead-area recognition.
        # Flipped variants get the spatially-flipped ownership map.
        # ---------------------------------------------------------------
        p0_board = self.state_encoder._decode_occ_words(
            int(st[P0_OCC0]), int(st[P0_OCC1]), int(st[P0_OCC2])
        )
        p1_board = self.state_encoder._decode_occ_words(
            int(st[P1_OCC0]), int(st[P1_OCC1]), int(st[P1_OCC2])
        )

        ownerships: List[np.ndarray] = []
        for i, (p, transform_tag) in enumerate(zip(stored_players, stored_flip_types)):
            own = np.zeros((2, 9, 9), dtype=np.float32)
            if int(p) == 0:
                own[0] = p0_board
                own[1] = p1_board
            else:
                own[0] = p1_board
                own[1] = p0_board
            ti = get_d4_transform_idx(transform_tag)
            own = apply_ownership_transform(own, ti)
            ownerships.append(own)

        winner = terminal_winner

        # Audit: raw components for scoring verification (buttons, empty, bonus, true scores)
        _empty0 = empty_count_from_occ(int(st[P0_OCC0]), int(st[P0_OCC1]), int(st[P0_OCC2]))
        _empty1 = empty_count_from_occ(int(st[P1_OCC0]), int(st[P1_OCC1]), int(st[P1_OCC2]))
        _bonus_owner = int(st[BONUS_OWNER])
        out_audit = {
            "final_buttons_p0": int(st[P0_BUTTONS]),
            "final_buttons_p1": int(st[P1_BUTTONS]),
            "empty_squares_p0": _empty0,
            "empty_squares_p1": _empty1,
            "bonus7x7_p0": 7 if _bonus_owner == 0 else 0,
            "bonus7x7_p1": 7 if _bonus_owner == 1 else 0,
            "final_score_p0": int(score0),
            "final_score_p1": int(score1),
            "winner": winner,
            "tie_break_tie_player": int(st[TIE_PLAYER]),
            "raw_margin_p0_perspective": float(score0 - score1),
            "stored_score_margin_tanh_p0": value_and_score_from_scores(int(score0), int(score1), winner, 0)[1],
            "stored_score_margin_tanh_p1": value_and_score_from_scores(int(score0), int(score1), winner, 1)[1],
        }

        # Position redundancy: 1 - unique/total (0 = all unique, 1 = all duplicates)
        redundancy = 1.0 - len(seen_state_hashes) / max(1, move_count)

        out: Dict = {
            "states": states,
            "action_masks": masks,
            "stored_flip_types": stored_flip_types,
            "policies": policies,
            "values": values,
            "score_margins": score_margins,
            "ownerships": ownerships,
            "game_length": int(move_count),
            "winner": int(winner),
            "final_score_diff": float(final_score_diff),
            "audit": out_audit,
            # Policy collapse canary
            "avg_policy_entropy": float(np.mean(move_entropies)) if move_entropies else 0.0,
            "avg_top1_prob": float(np.mean(move_top1_probs)) if move_top1_probs else 0.0,
            "avg_num_legal": float(np.mean(move_num_legal)) if move_num_legal else 0.0,
            # Position redundancy (Oracle Part 2): 0 = fully unique, 1 = all duplicates
            "redundancy": float(redundancy),
            "unique_positions": int(len(seen_state_hashes)),
            # Root Q stats
            "avg_root_q": float(np.mean(move_root_qs)) if move_root_qs else 0.0,
        }
        if self.store_canonical_only and stored_slot_piece_ids:
            out["slot_piece_ids"] = np.array(stored_slot_piece_ids, dtype=np.int16)  # (N, 3), -1 = empty
        if self._use_multimodal and stored_global_states:
            out["spatial_states"] = np.array(states, dtype=np.float32)
            out["global_states"] = np.array(stored_global_states, dtype=np.float32)
            out["track_states"] = np.array(stored_track_states, dtype=np.float32)
            out["shop_ids"] = np.array(stored_shop_ids, dtype=np.int16)
            out["shop_feats"] = np.array(stored_shop_feats, dtype=np.float32)
            out["states"] = out["spatial_states"]
        return out


# =========================================================================
# Multiprocessing entrypoints
# =========================================================================

_WORKER: Optional[OptimizedSelfPlayWorker] = None

# Controls what the multiprocessing worker returns.
# - 'dict': return full game data (legacy; can cause MemoryError on Windows when large)
# - 'shard': write arrays to a temporary .npz shard file and return only metadata + path
_RETURN_MODE: str = "dict"
_SHARD_DIR: Optional[str] = None

# Threaded shard writer: writes NPZ files in a background thread so the GPU/worker
# process doesn't block waiting for disk I/O. Bounded to 4 pending writes to avoid
# memory buildup on 16GB systems.
_SHARD_WRITER: Optional[ThreadPoolExecutor] = None
_SHARD_WRITE_FUTURES: List = []  # Track pending futures for cleanup
_SHARD_WRITE_LOCK = threading.Lock()


def _write_shard_async(
    shard_path: str,
    states: np.ndarray,
    masks: np.ndarray,
    policies: np.ndarray,
    values: np.ndarray,
    ownerships: np.ndarray,
    score_margins: np.ndarray,
    slot_piece_ids: Optional[np.ndarray] = None,
    spatial_states: Optional[np.ndarray] = None,
    global_states: Optional[np.ndarray] = None,
    track_states: Optional[np.ndarray] = None,
    shop_ids_arr: Optional[np.ndarray] = None,
    shop_feats: Optional[np.ndarray] = None,
) -> None:
    """Write NPZ shard to disk (runs in background thread)."""
    kwargs = {
        "states": states, "action_masks": masks, "policies": policies,
        "values": values, "ownerships": ownerships, "score_margins": score_margins,
    }
    if slot_piece_ids is not None:
        kwargs["slot_piece_ids"] = slot_piece_ids
    if spatial_states is not None:
        kwargs["spatial_states"] = spatial_states
        kwargs["global_states"] = global_states
        kwargs["track_states"] = track_states
        kwargs["shop_ids"] = shop_ids_arr
        kwargs["shop_feats"] = shop_feats
    np.savez(shard_path, **kwargs)


def _flush_shard_writes() -> None:
    """Wait for all pending shard writes to complete."""
    global _SHARD_WRITE_FUTURES
    with _SHARD_WRITE_LOCK:
        for fut in _SHARD_WRITE_FUTURES:
            try:
                fut.result(timeout=30)
            except Exception:
                pass
        _SHARD_WRITE_FUTURES.clear()


def init_optimized_worker(
    network_path: Optional[str],
    config: dict,
    req_q=None,
    resp_qs=None,
    return_mode: str = "dict",
    shard_dir: Optional[str] = None,
    worker_shm_names: Optional[dict] = None,
    expected_n_slots: Optional[int] = None,
):
    """Initialize worker (called once per process). expected_n_slots used for SHM attach validation."""
    global _WORKER, _RETURN_MODE, _SHARD_DIR, _SHARD_WRITER

    # Limit PyTorch threads per worker to avoid oversubscription (many workers × default threads)
    mcts_cfg = (config.get("selfplay", {}) or {}).get("mcts", {}) or {}
    n_threads = max(1, int(mcts_cfg.get("num_threads", 1)))
    torch.set_num_threads(n_threads)
    torch.set_num_interop_threads(n_threads)

    _RETURN_MODE = return_mode or "dict"
    _SHARD_DIR = shard_dir
    if _RETURN_MODE == "shard":
        if not _SHARD_DIR:
            raise RuntimeError("return_mode='shard' requires shard_dir")
        import os
        os.makedirs(_SHARD_DIR, exist_ok=True)
        # Async shard writer is safe: the pending queue is bounded to 8 entries
        # (see play_game_optimized), so peak memory is ~8 × ~7MB = ~56MB regardless
        # of channel count. The old "disable for 56ch" check incorrectly keyed on
        # input_channels (model trunk = 56) while the actual stored spatial is only
        # C_SPATIAL_ENC=32 channels, wasting time on synchronous disk I/O.
        if _SHARD_WRITER is None:
            _SHARD_WRITER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="shard_writer")

    # Extract worker_id from multiprocessing pool identity
    worker_id = 0
    resp_q = None
    if resp_qs is not None:
        import multiprocessing as mp
        proc = mp.current_process()
        ident = getattr(proc, "_identity", None) or ()
        # Pool workers have identity (1,), (2,), (3,), ... but IDs increment across pool reuses
        # Use modulo to wrap into valid range [0, num_workers-1]
        if len(ident) > 0:
            worker_id = (int(ident[0]) - 1) % len(resp_qs)
        else:
            worker_id = 0
        resp_q = resp_qs[worker_id]

    backend = (config.get("selfplay", {}) or {}).get("inference_backend")
    if backend is None and (req_q is not None and resp_q is not None):
        backend = "server"
    if backend is None:
        backend = "local"

    device = "cpu"
    if backend == "local":
        hw = (config.get("hardware", {}) or {}).get("device", "cpu")
        if hw == "cuda" and torch.cuda.is_available():
            device = "cuda"

    _WORKER = OptimizedSelfPlayWorker(network_path, config, device=device, req_q=req_q, resp_q=resp_q, worker_id=worker_id)

    # Attach shared memory buffer if available (zero-copy IPC path; expected_n_slots for validation)
    if worker_shm_names is not None and _WORKER.mcts is not None:
        shm_name = worker_shm_names.get(worker_id)
        if shm_name is not None:
            try:
                from src.mcts.shared_state_buffer import WorkerSharedBuffer
                from src.network.encoder import GoldV2StateEncoder
                _WORKER.mcts._shm_buf = WorkerSharedBuffer(
                    n_slots=None, worker_id=worker_id, create=False, name=shm_name,
                    expected_n_slots=expected_n_slots,
                )
                _WORKER.mcts._gold_v2_encoder = GoldV2StateEncoder()
            except Exception as _shm_err:
                import logging as _log
                _log.getLogger(__name__).warning("SHM attach failed for worker %d: %s", worker_id, _shm_err)


def play_game_optimized(args: Tuple[int, int, Optional[int]]) -> Optional[Dict]:
    """Worker function for multiprocessing pool.

    IMPORTANT (Windows): returning large nested Python objects through multiprocessing pipes
    can raise MemoryError in the parent during unpickling. In 'shard' mode, the worker
    writes arrays to disk and returns only a small metadata dict with the shard path.
    """
    game_idx, iteration, seed = args

    if _WORKER is None:
        raise RuntimeError("Worker not initialized")

    try:
        data = _WORKER.play_game(game_idx, iteration, seed)
        if data is None:
            return None

        # Shard mode: write heavy arrays to a temporary NPZ file and return only metadata.
        if _RETURN_MODE == "shard":
            if not _SHARD_DIR:
                raise RuntimeError("Shard mode enabled but _SHARD_DIR is not set")

            states = np.asarray(data["states"], dtype=np.float32)
            masks = np.asarray(data["action_masks"], dtype=np.float32)
            policies = np.asarray(data["policies"], dtype=np.float32)
            values = np.asarray(data["values"], dtype=np.float32)
            ownerships = np.asarray(data["ownerships"], dtype=np.float32)
            score_margins_arr = np.asarray(data["score_margins"], dtype=np.float32)
            slot_piece_ids = data.get("slot_piece_ids")
            spatial = data.get("spatial_states")
            global_s = data.get("global_states")
            track_s = data.get("track_states")
            shop_i = data.get("shop_ids")
            shop_f = data.get("shop_feats")

            import os
            pid = os.getpid()
            shard_name = f"sp_iter{iteration:03d}_g{game_idx:06d}_p{pid}.npz"
            shard_path = os.path.join(_SHARD_DIR, shard_name)

            # Non-blocking write: submit to background thread so the worker
            # can immediately start the next game without waiting for disk I/O.
            if _SHARD_WRITER is not None:
                # Make copies of arrays before submitting (numpy arrays are mutable)
                slot_cp = slot_piece_ids.copy() if slot_piece_ids is not None else None
                sp_cp = spatial.copy() if spatial is not None else None
                gb_cp = global_s.copy() if global_s is not None else None
                tr_cp = track_s.copy() if track_s is not None else None
                si_cp = shop_i.copy() if shop_i is not None else None
                sf_cp = shop_f.copy() if shop_f is not None else None
                fut = _SHARD_WRITER.submit(
                    _write_shard_async, shard_path,
                    states.copy(), masks.copy(), policies.copy(), values.copy(),
                    ownerships.copy(), score_margins_arr.copy(), slot_cp,
                    sp_cp, gb_cp, tr_cp, si_cp, sf_cp,
                )
                with _SHARD_WRITE_LOCK:
                    # Prune completed futures to avoid unbounded growth
                    _SHARD_WRITE_FUTURES[:] = [f for f in _SHARD_WRITE_FUTURES if not f.done()]
                    _SHARD_WRITE_FUTURES.append(fut)
                    # If too many pending writes (>8), wait for the oldest to complete
                    # This bounds memory usage on 16GB systems
                    while len(_SHARD_WRITE_FUTURES) > 8:
                        oldest = _SHARD_WRITE_FUTURES.pop(0)
                        oldest.result(timeout=30)
            else:
                # Fallback: synchronous write
                kwargs = {
                    "states": states, "action_masks": masks, "policies": policies,
                    "values": values, "ownerships": ownerships, "score_margins": score_margins_arr,
                }
                if slot_piece_ids is not None:
                    kwargs["slot_piece_ids"] = slot_piece_ids
                if spatial is not None:
                    kwargs["spatial_states"] = spatial
                    kwargs["global_states"] = global_s
                    kwargs["track_states"] = track_s
                    kwargs["shop_ids"] = shop_i
                    kwargs["shop_feats"] = shop_f
                np.savez(shard_path, **kwargs)

            return {
                "shard_path": shard_path,
                "game_length": int(data.get("game_length", 0)),
                "winner": int(data.get("winner", -1)),
                "final_score_diff": float(data.get("final_score_diff", 0.0)),
                "num_positions": int(states.shape[0]),
                # Policy collapse canary (flows to TensorBoard)
                "avg_policy_entropy": float(data.get("avg_policy_entropy", 0.0)),
                "avg_top1_prob": float(data.get("avg_top1_prob", 0.0)),
                "avg_num_legal": float(data.get("avg_num_legal", 0.0)),
                # Position redundancy & Q stats
                "redundancy": float(data.get("redundancy", 0.0)),
                "unique_positions": int(data.get("unique_positions", 0)),
                "avg_root_q": float(data.get("avg_root_q", 0.0)),
            }

        # Legacy: return full game data.
        return data
    except Exception as e:
        import logging
        logging.error(f"Game {game_idx} failed: {e}", exc_info=True)
        return None


def create_optimized_worker(network_path: Optional[str], config: dict, req_q=None, resp_q=None):
    """Create an optimized worker instance (non-multiprocessing)."""
    backend = (config.get("selfplay", {}) or {}).get("inference_backend")
    if backend is None and (req_q is not None and resp_q is not None):
        backend = "server"
    if backend is None:
        backend = "local"

    device = "cpu"
    if backend == "local":
        hw = (config.get("hardware", {}) or {}).get("device", "cpu")
        if hw == "cuda" and torch.cuda.is_available():
            device = "cuda"

    return OptimizedSelfPlayWorker(network_path, config, device=device, req_q=req_q, resp_q=resp_q)