"""
Patchwork AlphaZero Neural Network Architecture

Input:  14 channels × 9×9 board
Policy: 2026 action logits
Value:  scalar in [-1, 1]
Score:  categorical distribution over 201 integer bins [-100, +100] (logits)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


DEFAULT_INPUT_CHANNELS = 14
DEFAULT_MAX_ACTIONS = 2026

# ---------------------------------------------------------------------------
# Fixed-weight piece mask construction (called once at model init)
# ---------------------------------------------------------------------------

def _build_piece_masks_5x5() -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (N_PIECES, MAX_ORIENTS, 5, 5) float32 piece masks for all orientations.

    Uses MASK_W0/W1/W2 from the engine (already computed at engine import).
    Invalid orientations (orient >= ORIENT_COUNT[pid]) stay all-zeros (area=0),
    so the legality check never marks them as legal.

    Returns:
        masks : (N_PIECES, MAX_ORIENTS, 5, 5) float32
        areas : (N_PIECES, MAX_ORIENTS) float32 -- filled-cell count per (pid, orient)
    """
    from src.game.patchwork_engine import N_PIECES, MAX_ORIENTS, ORIENT_COUNT, MASK_W0, MASK_W1

    masks = np.zeros((N_PIECES, MAX_ORIENTS, 5, 5), dtype=np.float32)

    for pid in range(N_PIECES):
        n_orient = int(ORIENT_COUNT[pid])
        for o in range(n_orient):
            # Mask at TL = board position 0 = (row 0, col 0)
            m0 = int(MASK_W0[pid, o, 0])
            m1 = int(MASK_W1[pid, o, 0])
            # All Patchwork pieces fit within a 5x5 bounding box.
            # Board positions: row in [0,4], col in [0,4] -> flat index in [0,40].
            # Word 0 covers flat indices 0-31, word 1 covers 32-63.
            for board_idx in range(41):
                r, c = board_idx // 9, board_idx % 9
                if r >= 5 or c >= 5:
                    continue
                word = board_idx >> 5
                bit = board_idx & 31
                if word == 0 and (m0 >> bit) & 1:
                    masks[pid, o, r, c] = 1.0
                elif word == 1 and (m1 >> bit) & 1:
                    masks[pid, o, r, c] = 1.0

    areas = masks.sum(axis=(-1, -2))  # (N_PIECES, MAX_ORIENTS) float32
    return masks, areas


# ---------------------------------------------------------------------------
# DeterministicLegalityModule
# ---------------------------------------------------------------------------

class DeterministicLegalityModule(nn.Module):
    """
    Non-trainable GPU module: computes placement legality maps for ALL pieces
    x orientations via a single fixed-weight F.conv2d call.

    Replaces:
      * 24x CPU legalTL plane loops in GoldV2StateEncoder  (channels 32-55)
      * 33x _count_legal_placements_clip20() calls for shop features 8-9

    Conv math (cross-correlation with padding=4):
        Input  (B, 1, 9, 9) + padding 4  -> padded (B, 1, 17, 17)
        Filter (264, 1, 5, 5)
        Output (B, 264, 13, 13)   [because (9 + 2*4 - 5 + 1) = 13]
        Crop output[:, :, 4:13, 4:13] -> (B, 264, 9, 9)
        output[b, n, r, c] = number of free cells under piece n at TL (r, c)
        legal[b, n, r, c]  = 1 iff output[b,n,r,c] == piece_area[n]

    Throughput on RTX 3080: ~0.1 ms/batch vs O(33 * 648) Python iters per leaf.
    """

    def __init__(self, piece_masks: np.ndarray, piece_areas: np.ndarray) -> None:
        super().__init__()
        N_P, N_O, H, W = piece_masks.shape
        self.N_PIECES = N_P
        self.N_ORIENTS = N_O
        self.N_FILTERS = N_P * N_O  # 264 = 33 pieces * 8 orientations

        filters = torch.from_numpy(
            piece_masks.reshape(N_P * N_O, 1, H, W).copy()
        ).float()
        areas_flat = torch.from_numpy(
            piece_areas.reshape(N_P * N_O).copy()
        ).float()

        self.register_buffer("filters", filters)      # (264, 1, 5, 5) -- non-trainable
        self.register_buffer("areas", areas_flat)     # (264,)
        valid = (areas_flat > 0.5).float()
        self.register_buffer("valid_orient", valid)   # (264,)

    def parameters(self, recurse: bool = True):
        return iter([])  # zero trainable parameters

    def forward(self, board_free: torch.Tensor) -> torch.Tensor:
        """
        Args:
            board_free : (B, 1, 9, 9) float32  -- 1 = empty, 0 = occupied
        Returns:
            legal : (B, 264, 9, 9) float32  -- 1 = legal TL placement
        """
        # Float32 ensures exact integer arithmetic for the equality check.
        corr = F.conv2d(board_free.float(), self.filters.float(), padding=4)
        corr = corr[:, :, 4:13, 4:13]  # crop to (B, 264, 9, 9)

        areas = self.areas.view(1, -1, 1, 1)
        valid = self.valid_orient.view(1, -1, 1, 1)
        # Tolerance 0.5: values are always exact integers, but guard against
        # any residual float imprecision.
        return (corr >= areas - 0.5).float() * valid

    def extract_vis_legalTL(
        self,
        legal: torch.Tensor,          # (B, 264, 9, 9)
        vis_piece_ids: torch.Tensor,   # (B, 3) int64 -- piece IDs for 3 visible slots
        affordability: torch.Tensor,   # (B, 3) float -- 1 if affordable
    ) -> torch.Tensor:
        """
        Extract and return 24-channel legalTL planes (3 slots x 8 orientations),
        masked by affordability.  Replaces encoder channels 32-55.

        Returns: (B, 24, 9, 9)
        """
        B = legal.shape[0]
        N_O = self.N_ORIENTS

        valid = (vis_piece_ids >= 0).float()
        safe_ids = vis_piece_ids.clamp(min=0)

        orient_offsets = torch.arange(N_O, device=legal.device, dtype=torch.long)
        gather_idxs = (safe_ids.unsqueeze(-1) * N_O + orient_offsets).reshape(B, 24)

        b_idx = torch.arange(B, device=legal.device).unsqueeze(1).expand(B, 24)
        legalTL = legal[b_idx, gather_idxs]  # (B, 24, 9, 9)

        # Expand (B,3) -> (B,24): each slot covers 8 consecutive orient planes
        valid_24 = valid.unsqueeze(-1).expand(B, 3, N_O).reshape(B, 24, 1, 1)
        afford_24 = affordability.unsqueeze(-1).expand(B, 3, N_O).reshape(B, 24, 1, 1)
        return legalTL * valid_24 * afford_24

    def compute_num_legal_normalized(
        self,
        legal: torch.Tensor,    # (B, 264, 9, 9)
        shop_ids: torch.Tensor,  # (B, NMAX) int64 -- piece IDs, -1 = empty
    ) -> torch.Tensor:
        """
        Count total legal positions per shop piece across all orientations,
        normalize to [0,1] by clamping at 20.

        Returns: (B, NMAX) float
        """
        B = legal.shape[0]
        NMAX = shop_ids.shape[1]
        N_O = self.N_ORIENTS

        valid = (shop_ids >= 0).float()
        safe_ids = shop_ids.clamp(min=0)

        orient_offsets = torch.arange(N_O, device=legal.device, dtype=torch.long)
        gather_idxs = (safe_ids.unsqueeze(-1) * N_O + orient_offsets).reshape(B, NMAX * N_O)

        b_idx = torch.arange(B, device=legal.device).unsqueeze(1).expand(B, NMAX * N_O)
        gathered = legal[b_idx, gather_idxs].reshape(B, NMAX, N_O, 9, 9)
        num_legal = gathered.sum(dim=(-1, -2, -3)) * valid
        return (num_legal.clamp(max=20.0) / 20.0)


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block."""

    def __init__(self, channels: int, se_ratio: float = 0.0625):
        super().__init__()
        # se_ratio determines the reduction: reduced = channels * se_ratio
        reduced = max(int(channels * se_ratio), 1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        squeeze = x.mean(dim=(2, 3))
        excitation = self.fc(squeeze).view(b, c, 1, 1)
        return x * excitation


class ResidualBlock(nn.Module):
    """Residual block with optional SE attention."""

    def __init__(self, channels: int, use_batch_norm: bool = True, se_ratio: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=not use_batch_norm)
        self.bn1 = nn.BatchNorm2d(channels) if use_batch_norm else nn.Identity()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=not use_batch_norm)
        self.bn2 = nn.BatchNorm2d(channels) if use_batch_norm else nn.Identity()
        self.se = SqueezeExcitation(channels, se_ratio=se_ratio) if se_ratio > 0 else None

    def forward(self, x: torch.Tensor, gamma: Optional[torch.Tensor] = None, beta: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        out = self.bn1(self.conv1(x))
        if gamma is not None and beta is not None:
            out = out * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        out = F.relu(out, inplace=True)
        out = self.bn2(self.conv2(out))
        if self.se is not None:
            out = self.se(out)
        return F.relu(out + residual, inplace=True)


# Structured action space constants (2026 indexing)
PASS_INDEX = 0
PATCH_START = 1
BUY_START = 82
NUM_SLOTS = 3
NUM_ORIENTS = 8
BOARD_SIZE = 9
PATCH_SIZE = BOARD_SIZE * BOARD_SIZE  # 81
BUY_SIZE = NUM_SLOTS * NUM_ORIENTS * PATCH_SIZE  # 1944


class StructuredConvPolicyHead(nn.Module):
    """Structured conv policy head with exact 2026 indexing (pass + patch map + buy maps).

    Output order (exact):
    - 0: pass
    - 1..81: patch placement (pos=row*9+col, index=1+pos)
    - 82..2025: buy (index=82 + (slot*8+orient)*81 + pos)

    global_inject_dim: if >0, concatenate a projected global-feature vector into the
    pass-logit computation.  Pass and buy/patch are spatially very different actions;
    the pass decision is heavily position/button-driven (scalar), so direct global
    injection here is more targeted than relying on FiLM alone.
    """

    def __init__(
        self,
        input_channels: int,
        policy_hidden: int = 48,
        use_batch_norm: bool = True,
        global_inject_dim: int = 0,
    ):
        super().__init__()
        self.p_conv = nn.Conv2d(input_channels, policy_hidden, 1, bias=not use_batch_norm)
        self.p_bn = nn.BatchNorm2d(policy_hidden) if use_batch_norm else nn.Identity()
        self.buy_conv = nn.Conv2d(policy_hidden, NUM_SLOTS * NUM_ORIENTS, 1)  # 24 channels
        self.patch_conv = nn.Conv2d(policy_hidden, 1, 1)
        self.global_inject_dim = global_inject_dim
        self.pass_linear = nn.Linear(policy_hidden + global_inject_dim, 1)
        # Global bias broadcast over buy/patch spatial maps (KataGo-style spatial global context).
        # g_to_buy_bias: for each of 24 (slot, orient) channels, a learned scalar bias from
        # global state (buttons, affordability, position) independent of board location.
        # g_to_patch_bias: scalar bias over the entire patch placement map.
        # Zero-initialized: neutral at start, learns to modulate based on game state.
        self.g_to_buy_bias: Optional[nn.Linear] = None
        self.g_to_patch_bias: Optional[nn.Linear] = None
        if global_inject_dim > 0:
            self.g_to_buy_bias = nn.Linear(global_inject_dim, NUM_SLOTS * NUM_ORIENTS)
            self.g_to_patch_bias = nn.Linear(global_inject_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        return_maps: bool = False,
    ) -> torch.Tensor:
        # x: (B, 128, 9, 9)
        p = F.relu(self.p_bn(self.p_conv(x)), inplace=True)  # (B, 48, 9, 9)
        buy_map = self.buy_conv(p)   # (B, 24, 9, 9)
        patch_map = self.patch_conv(p)  # (B, 1, 9, 9)

        # Global bias: per-(slot,orient) scalar from game state, broadcast over all board positions.
        # Lets the network learn "given current buttons/position, how desirable is slot 0 orient 3?"
        # independently of spatial placement quality.
        if g is not None and self.g_to_buy_bias is not None:
            buy_map = buy_map + self.g_to_buy_bias(g).unsqueeze(-1).unsqueeze(-1)
            patch_map = patch_map + self.g_to_patch_bias(g).unsqueeze(-1).unsqueeze(-1)

        # pass: global mean pool -> linear (+ optional global injection)
        p_mean = p.mean(dim=(2, 3))   # (B, 48)
        if g is not None and self.global_inject_dim > 0:
            pass_input = torch.cat([p_mean, g], dim=1)  # (B, 48 + global_inject_dim)
        else:
            pass_input = p_mean
        pass_vec = self.pass_linear(pass_input)  # (B, 1)

        # flatten row-major: (r,c) -> r*9+c
        patch_vec = patch_map.flatten(start_dim=1)  # (B, 81)
        buy_vec = buy_map.flatten(start_dim=1)      # (B, 1944)

        logits = torch.cat([pass_vec, patch_vec, buy_vec], dim=1)  # (B, 2026)
        if return_maps:
            return logits, patch_map, buy_map
        return logits


class HierarchicalPolicyHead(nn.Module):
    """Hierarchical factored policy head for Patchwork.

    Decomposes the flat 2026-action space into a probability hierarchy:
      Level 1: Type       — P(PASS), P(PATCH), P(BUY)           (3-way softmax)
      Level 2a: Patch pos — P(pos | PATCH)                      (81-way spatial)
      Level 2b: Buy slot  — P(slot | BUY)                       (3-way)
      Level 3: Buy orient — P(orient | slot, BUY)               (8-way per slot)
      Level 4: Buy pos    — P(pos | slot, orient, BUY)          (81-way spatial)

    Joint: P(buy, s, o, p) = P(BUY) × P(s|BUY) × P(o|s) × P(p|s,o)

    Returns (B, 2026) log-probabilities.  Since log_softmax is idempotent on
    already-normalized log-probs, downstream code that applies log_softmax()
    or softmax() produces correct results without any interface change.

    Key benefit: each softmax operates over a small set (3/8/81-way) instead
    of one 2026-way.  Learning "slot 0 is best" adjusts 1 logit in a 3-way
    softmax rather than pushing 648 logits in a 2026-way softmax.
    """

    def __init__(
        self,
        input_channels: int,
        policy_hidden: int = 48,
        use_batch_norm: bool = True,
        global_inject_dim: int = 0,
    ):
        super().__init__()
        # Shared spatial backbone
        self.p_conv = nn.Conv2d(input_channels, policy_hidden, 1, bias=not use_batch_norm)
        self.p_bn = nn.BatchNorm2d(policy_hidden) if use_batch_norm else nn.Identity()

        self.global_inject_dim = global_inject_dim
        g_dim = policy_hidden + global_inject_dim

        # Level 1: type selection (PASS / PATCH / BUY)
        self.type_linear = nn.Linear(g_dim, 3)

        # Level 2a: patch position (81-way spatial)
        self.patch_conv = nn.Conv2d(policy_hidden, 1, 1)

        # Level 2b: buy slot (3-way from scalar features)
        self.slot_linear = nn.Linear(g_dim, NUM_SLOTS)

        # Level 3: buy orient per slot (3 × 8 = 24 outputs)
        self.orient_linear = nn.Linear(g_dim, NUM_SLOTS * NUM_ORIENTS)

        # Level 4: buy position per (slot, orient) — 24 spatial maps
        self.pos_conv = nn.Conv2d(policy_hidden, NUM_SLOTS * NUM_ORIENTS, 1)

        # Global biases (same role as StructuredConvPolicyHead)
        self.g_to_pos_bias: Optional[nn.Linear] = None
        self.g_to_patch_bias: Optional[nn.Linear] = None
        if global_inject_dim > 0:
            self.g_to_pos_bias = nn.Linear(global_inject_dim, NUM_SLOTS * NUM_ORIENTS)
            self.g_to_patch_bias = nn.Linear(global_inject_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        g: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns (B, 2026) log-probabilities with level-wise masking."""
        B = x.shape[0]

        # --- Shared features ---
        p = F.relu(self.p_bn(self.p_conv(x)), inplace=True)  # (B, H, 9, 9)
        p_mean = p.mean(dim=(2, 3))  # (B, H)
        if g is not None and self.global_inject_dim > 0:
            scalar_in = torch.cat([p_mean, g], dim=1)
        else:
            scalar_in = p_mean

        # --- Derive level-wise masks from flat action mask ---
        if action_mask is not None:
            patch_mask = action_mask[:, PATCH_START:PATCH_START + PATCH_SIZE]        # (B, 81)
            buy_mask = action_mask[:, BUY_START:BUY_START + BUY_SIZE]               # (B, 1944)
            buy_4d = buy_mask.view(B, NUM_SLOTS, NUM_ORIENTS, PATCH_SIZE)           # (B, 3, 8, 81)
            type_mask = torch.stack([
                action_mask[:, PASS_INDEX] > 0,
                patch_mask.any(dim=-1),
                buy_mask.any(dim=-1),
            ], dim=-1)                                                               # (B, 3) bool
            slot_mask = buy_4d.any(dim=-1).any(dim=-1)                               # (B, 3)
            orient_mask = buy_4d.any(dim=-1)                                         # (B, 3, 8)
            pos_mask = buy_4d                                                        # (B, 3, 8, 81)
        else:
            type_mask = slot_mask = orient_mask = pos_mask = patch_mask = None

        _NEG = -1e9  # mask fill value (finite to avoid NaN in log_softmax)

        # === Level 1: Type ===
        type_logits = self.type_linear(scalar_in)                       # (B, 3)
        if type_mask is not None:
            type_logits = type_logits.masked_fill(~type_mask, _NEG)
        type_lp = F.log_softmax(type_logits, dim=-1).clamp(min=-100.0) # (B, 3)

        # === Level 2a: Patch position ===
        patch_logits = self.patch_conv(p).view(B, PATCH_SIZE)           # (B, 81)
        if g is not None and self.g_to_patch_bias is not None:
            patch_logits = patch_logits + self.g_to_patch_bias(g)
        if patch_mask is not None:
            patch_logits = patch_logits.masked_fill(patch_mask < 0.5, _NEG)
        patch_lp = F.log_softmax(patch_logits, dim=-1).clamp(min=-100.0)  # (B, 81)

        # === Level 2b: Buy slot ===
        slot_logits = self.slot_linear(scalar_in)                       # (B, 3)
        if slot_mask is not None:
            slot_logits = slot_logits.masked_fill(~slot_mask, _NEG)
        slot_lp = F.log_softmax(slot_logits, dim=-1).clamp(min=-100.0) # (B, 3)

        # === Level 3: Buy orient (per slot) ===
        orient_logits = self.orient_linear(scalar_in).view(B, NUM_SLOTS, NUM_ORIENTS)
        if orient_mask is not None:
            orient_logits = orient_logits.masked_fill(~orient_mask, _NEG)
        orient_lp = F.log_softmax(orient_logits, dim=-1).clamp(min=-100.0)  # (B, 3, 8)

        # === Level 4: Buy position (spatial per slot×orient) ===
        pos_logits = self.pos_conv(p).view(B, NUM_SLOTS, NUM_ORIENTS, PATCH_SIZE)
        if g is not None and self.g_to_pos_bias is not None:
            pos_logits = pos_logits + self.g_to_pos_bias(g).view(B, NUM_SLOTS, NUM_ORIENTS, 1)
        if pos_mask is not None:
            pos_logits = pos_logits.masked_fill(pos_mask < 0.5, _NEG)
        pos_lp = F.log_softmax(pos_logits, dim=-1).clamp(min=-100.0)  # (B, 3, 8, 81)

        # === Compose joint log-probabilities ===
        pass_lp = type_lp[:, 0:1]                                      # (B, 1)
        patch_joint = type_lp[:, 1:2] + patch_lp                       # (B, 81)
        buy_joint = (
            type_lp[:, 2:3].unsqueeze(-1).unsqueeze(-1)                # (B, 1, 1, 1)
            + slot_lp.unsqueeze(-1).unsqueeze(-1)                      # (B, 3, 1, 1)
            + orient_lp.unsqueeze(-1)                                  # (B, 3, 8, 1)
            + pos_lp                                                   # (B, 3, 8, 81)
        ).reshape(B, BUY_SIZE)                                         # (B, 1944)

        return torch.cat([pass_lp, patch_joint, buy_joint], dim=1)     # (B, 2026)


class PolicyHead(nn.Module):
    """Legacy policy head: single fc or factorized fc1+fc2.

    use_factorized_policy_head=false => this head (legacy single fc).
    use_factorized_policy_head=true => StructuredConvPolicyHead instead.
    """

    def __init__(
        self,
        input_channels: int,
        policy_channels: int,
        max_actions: int,
        policy_hidden: int,
        use_factorized: bool = False,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        policy_in = policy_channels * 9 * 9
        self.conv = nn.Conv2d(input_channels, policy_channels, 1, bias=not use_batch_norm)
        self.bn = nn.BatchNorm2d(policy_channels) if use_batch_norm else nn.Identity()
        self._use_factorized = use_factorized
        if use_factorized:
            self.fc1 = nn.Linear(policy_in, policy_hidden)
            self.fc2 = nn.Linear(policy_hidden, max_actions)
        else:
            self.fc = nn.Linear(policy_in, max_actions)

    def forward(self, x: torch.Tensor, g: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Legacy head ignores g; kept for call-site compatibility with StructuredConvPolicyHead."""
        x = F.relu(self.bn(self.conv(x)), inplace=True)
        p = x.flatten(start_dim=1)
        if self._use_factorized:
            p = F.relu(self.fc1(p), inplace=True)
            return self.fc2(p)
        return self.fc(p)


class ValueHead(nn.Module):
    """Value head outputting scalar value in [-1, 1]. Optional score head outputs
    201-bin logits over integer margin bins (KataGo-style distributional score).

    global_inject_dim: if >0, a projected global-feature vector is concatenated with
    the spatial conv output before fc1.  This gives the value head direct access to
    scalar game state (positions, buttons, income) instead of relying purely on FiLM
    to propagate that information through the spatial trunk.  This is the KataGo
    'global pooling' pattern adapted for multimodal input.
    """

    SCORE_MIN = -100
    SCORE_MAX = 100
    SCORE_BINS = SCORE_MAX - SCORE_MIN + 1  # 201

    def __init__(
        self,
        input_channels: int,
        value_channels: int,
        value_hidden: int,
        use_batch_norm: bool = True,
        with_score_head: bool = True,
        global_inject_dim: int = 0,
    ):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, value_channels, 1, bias=not use_batch_norm)
        self.bn = nn.BatchNorm2d(value_channels) if use_batch_norm else nn.Identity()
        self.global_inject_dim = global_inject_dim
        self.fc1 = nn.Linear(value_channels * 9 * 9 + global_inject_dim, value_hidden)
        self.fc2 = nn.Linear(value_hidden, 1)
        # 201-bin score head: categorical logits over integer margins [SCORE_MIN, SCORE_MAX].
        # MCTS uses distribution on GPU to compute mean_points and score_utility (no IPC of logits).
        self.score_head = nn.Linear(value_hidden, self.SCORE_BINS) if with_score_head else None

    def forward(
        self,
        x: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.bn(self.conv(x)), inplace=True)
        flat = x.flatten(start_dim=1)  # (B, value_channels * 81)
        if g is not None and self.global_inject_dim > 0:
            flat = torch.cat([flat, g], dim=1)  # (B, value_channels*81 + global_inject_dim)
        hidden = F.relu(self.fc1(flat), inplace=True)
        value = torch.tanh(self.fc2(hidden))
        if self.score_head is not None:
            score_logits = self.score_head(hidden)  # (B, 201)
            return value, score_logits
        return value


class ShopEncoder(nn.Module):
    """Encode shop (piece_ids + feats) into a pooled vector. Pad=-1 masked."""

    def __init__(self, num_piece_ids: int = 34, feat_dim: int = 10, embed_dim: int = 32, out_dim: int = 128, n_layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(num_piece_ids + 1, embed_dim, padding_idx=0)
        self.feat_proj = nn.Linear(feat_dim, embed_dim)
        self.d_model = embed_dim * 2
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=4,
            dim_feedforward=64,
            dropout=0.1,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_proj = nn.Linear(self.d_model, out_dim)

    def forward(self, shop_ids: torch.Tensor, shop_feats: torch.Tensor) -> torch.Tensor:
        """shop_ids: (B, N) int64, pad=-1. shop_feats: (B, N, F). Returns (B, out_dim)."""
        ids = (shop_ids.clamp(min=-1).long() + 1)
        emb = self.embed(ids)
        feats = self.feat_proj(shop_feats)
        combined = torch.cat([emb, feats], dim=-1)
        pad_mask = (shop_ids == -1)
        out = self.transformer(combined, src_key_padding_mask=pad_mask)
        cnt = (~pad_mask).float().sum(dim=1, keepdim=True).clamp(min=1)
        pooled = out.masked_fill(pad_mask.unsqueeze(-1), 0.0).sum(dim=1) / cnt
        return self.out_proj(pooled)


class OwnershipHead(nn.Module):
    """Auxiliary ownership head predicting per-cell board occupancy at game end.

    Patchwork has *separate* 9x9 boards per player (no shared territory), so
    we predict two binary maps instead of a 3-class shared map:
      Channel 0: P(current player's cell is filled at game end)
      Channel 1: P(opponent's cell is filled at game end)

    This gives 2 × 81 = 162 dense spatial training signals that help the
    network learn spatial packing efficiency and dead-area recognition,
    similar to KataGo's ownership prediction adapted for Patchwork's
    dual-board structure.
    """

    def __init__(self, input_channels: int, ownership_channels: int, use_batch_norm: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, ownership_channels, 1, bias=not use_batch_norm)
        self.bn = nn.BatchNorm2d(ownership_channels) if use_batch_norm else nn.Identity()
        self.conv_out = nn.Conv2d(ownership_channels, 2, 1)  # 2 binary channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn(self.conv(x)), inplace=True)
        return self.conv_out(x)  # (B, 2, 9, 9) logits


class AuxScalarHead(nn.Module):
    """Lightweight auxiliary head for scalar/binary game-end predictions.

    Conv1×1 → BN → ReLU → flatten → linear → n_outputs raw logits.
    Use BCE-with-logits for binary targets, sigmoid+MSE for bounded regression.

    Two heads are instantiated when aux_channels > 0:
      bonus_head  (n_outputs=2): [P(me gets 7×7 bonus), P(opp gets 7×7 bonus)]
      threat_head (n_outputs=1): opponent's final empty-square fraction [0, 1]
    """

    def __init__(
        self,
        input_channels: int,
        aux_channels: int,
        n_outputs: int,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, aux_channels, 1, bias=not use_batch_norm)
        self.bn = nn.BatchNorm2d(aux_channels) if use_batch_norm else nn.Identity()
        self.fc = nn.Linear(aux_channels * 9 * 9, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn(self.conv(x)), inplace=True)
        return self.fc(x.flatten(start_dim=1))  # (B, n_outputs) raw logits


class PatchworkNetwork(nn.Module):
    """Complete AlphaZero network for Patchwork."""

    def __init__(
        self,
        input_channels: int = DEFAULT_INPUT_CHANNELS,
        num_res_blocks: int = 10,
        channels: int = 128,
        policy_channels: int = 32,
        policy_hidden: int = 256,
        use_factorized_policy_head: bool = True,
        value_channels: int = 32,
        value_hidden: int = 128,
        max_actions: int = DEFAULT_MAX_ACTIONS,
        use_batch_norm: bool = True,
        se_ratio: float = 0.0,
        dropout: float = 0.0,
        ownership_channels: int = 0,
        aux_channels: int = 0,
        use_film: bool = False,
        film_hidden: int = 256,
        film_input_plane_indices: Optional[List[int]] = None,
        film_use_plane_mean: bool = True,
        film_per_block: bool = True,
        film_global_dim: int = 0,
        film_track_dim: int = 0,
        film_shop_dim: int = 0,
        use_gpu_legality: bool = True,
        film_track_use_conv: bool = False,
        film_global_inject_dim: int = 0,
        use_hierarchical_policy: bool = False,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.max_actions = max_actions
        self.use_film = use_film
        self.film_input_plane_indices = film_input_plane_indices or []
        self.film_use_plane_mean = film_use_plane_mean
        self.film_per_block = film_per_block
        self.film_global_dim = film_global_dim
        self.film_track_dim = film_track_dim
        self.film_shop_dim = film_shop_dim
        self.num_res_blocks = num_res_blocks
        self.channels = channels
        self.film_track_use_conv = film_track_use_conv
        self.film_global_inject_dim = film_global_inject_dim

        # GPU legality module: computes legalTL and num_legal on the GPU.
        # Encoder outputs 36ch (gold_v2_36ch) or 32ch (gold_v2_32ch legacy) states.
        # _apply_gpu_legality auto-detects new (32/36ch) vs legacy (56/60ch) states.
        # New: 36ch + 24 legalTL → 60ch trunk. Legacy: overwrites last-24 channels.
        self.det_legality: Optional[DeterministicLegalityModule] = None
        if use_gpu_legality and input_channels in (56, 60):
            try:
                masks_np, areas_np = _build_piece_masks_5x5()
                self.det_legality = DeterministicLegalityModule(masks_np, areas_np)
                logger.debug(
                    "DeterministicLegalityModule initialised (%d filters, %dx%d kernels).",
                    self.det_legality.N_FILTERS, 5, 5,
                )
            except Exception as e:  # pragma: no cover
                logger.warning("DeterministicLegalityModule init failed (%s); falling back to CPU.", e)

        self.conv_input = nn.Conv2d(input_channels, channels, 3, padding=1, bias=not use_batch_norm)
        self.bn_input = nn.BatchNorm2d(channels) if use_batch_norm else nn.Identity()
        self.res_blocks = nn.ModuleList([ResidualBlock(channels, use_batch_norm, se_ratio) for _ in range(num_res_blocks)])
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

        self.film_mlp: Optional[nn.Sequential] = None
        self.film_global_mlp: Optional[nn.Sequential] = None
        self.film_track_mlp: Optional[nn.Sequential] = None
        self.film_track_conv: Optional[nn.Sequential] = None
        self.film_track_pool: Optional[nn.Linear] = None
        self.shop_encoder: Optional[ShopEncoder] = None
        self.global_to_heads: Optional[nn.Sequential] = None
        self.trunk_to_heads: Optional[nn.Linear] = None
        if use_film:
            film_out_dim = 2 * channels * num_res_blocks
            if film_global_dim > 0:
                self.film_global_mlp = nn.Sequential(nn.Linear(film_global_dim, film_hidden), nn.ReLU(inplace=True))
                if film_track_use_conv:
                    # 1D conv preserving sequence structure: (B, 8, 54) -> (B, 32, 54) -> mean -> (B, film_hidden)
                    self.film_track_conv = nn.Sequential(
                        nn.Conv1d(8, 32, kernel_size=5, padding=2),
                        nn.ReLU(inplace=True),
                        nn.Conv1d(32, 32, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                    )
                    self.film_track_pool = nn.Linear(32, film_hidden)
                else:
                    self.film_track_mlp = nn.Sequential(nn.Linear(film_track_dim, film_hidden), nn.ReLU(inplace=True))
                self.shop_encoder = ShopEncoder(num_piece_ids=34, feat_dim=10, embed_dim=32, out_dim=film_shop_dim)
                self.film_mlp = nn.Sequential(
                    nn.Linear(film_hidden * 2 + film_shop_dim, film_hidden),
                    nn.ReLU(inplace=True),
                    nn.Linear(film_hidden, film_out_dim),
                )
                if film_global_inject_dim > 0:
                    # Direct global feature injection into both heads (KataGo global pooling pattern)
                    self.global_to_heads = nn.Sequential(
                        nn.Linear(film_global_dim, film_global_inject_dim),
                        nn.ReLU(inplace=True),
                    )
                    # Trunk spatial pooling: pools the *learned* spatial trunk features (mean over 9×9)
                    # and adds them additively to g_heads. Zero-initialized → identity at start.
                    # Gives heads access to the network's trained representation, not just raw scalars.
                    self.trunk_to_heads = nn.Linear(channels, film_global_inject_dim)
            elif self.film_input_plane_indices:
                self.film_mlp = nn.Sequential(
                    nn.Linear(len(self.film_input_plane_indices), film_hidden),
                    nn.ReLU(inplace=True),
                    nn.Linear(film_hidden, film_out_dim),
                )
            else:
                raise ValueError("use_film=true requires film_global_dim>0 or film_input_plane_indices")

        self._use_structured_policy_head = use_factorized_policy_head
        self._use_hierarchical_policy = use_hierarchical_policy
        if use_hierarchical_policy:
            self.policy_head = HierarchicalPolicyHead(
                channels, policy_hidden=48, use_batch_norm=use_batch_norm,
                global_inject_dim=film_global_inject_dim,
            )
        elif use_factorized_policy_head:
            self.policy_head = StructuredConvPolicyHead(
                channels, policy_hidden=48, use_batch_norm=use_batch_norm,
                global_inject_dim=film_global_inject_dim,
            )
        else:
            self.policy_head = PolicyHead(
                channels, policy_channels, max_actions, policy_hidden,
                use_factorized=False, use_batch_norm=use_batch_norm
            )
        self.value_head = ValueHead(
            channels, value_channels, value_hidden, use_batch_norm, with_score_head=True,
            global_inject_dim=film_global_inject_dim,
        )

        # Auxiliary ownership head (KataGo-style) — optional
        self.ownership_head: Optional[OwnershipHead] = None
        if ownership_channels > 0:
            self.ownership_head = OwnershipHead(channels, ownership_channels, use_batch_norm)

        # Auxiliary scalar heads — optional (enabled when aux_channels > 0)
        # bonus_head:  2 outputs — [P(me gets 7×7 bonus), P(opp gets 7×7 bonus)]  (BCE)
        # threat_head: 1 output  — opponent final empty-square fraction [0,1]       (sigmoid+MSE)
        self.bonus_head: Optional[AuxScalarHead] = None
        self.threat_head: Optional[AuxScalarHead] = None
        if aux_channels > 0:
            self.bonus_head = AuxScalarHead(channels, aux_channels, n_outputs=2, use_batch_norm=use_batch_norm)
            self.threat_head = AuxScalarHead(channels, aux_channels, n_outputs=1, use_batch_norm=use_batch_norm)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # AlphaZero trick: zero-init final layers feeding tanh/softmax so network
        # starts near-neutral (value≈0, policy≈uniform), avoiding saturation.
        nn.init.zeros_(self.value_head.fc2.weight)
        nn.init.zeros_(self.value_head.fc2.bias)
        if self.value_head.score_head is not None:
            nn.init.zeros_(self.value_head.score_head.weight)
            nn.init.zeros_(self.value_head.score_head.bias)
        # Policy head: zero-init output layers so network starts near-uniform policy
        if self._use_hierarchical_policy:
            nn.init.zeros_(self.policy_head.type_linear.weight)
            nn.init.zeros_(self.policy_head.type_linear.bias)
            nn.init.zeros_(self.policy_head.patch_conv.weight)
            if self.policy_head.patch_conv.bias is not None:
                nn.init.zeros_(self.policy_head.patch_conv.bias)
            nn.init.zeros_(self.policy_head.slot_linear.weight)
            nn.init.zeros_(self.policy_head.slot_linear.bias)
            nn.init.zeros_(self.policy_head.orient_linear.weight)
            nn.init.zeros_(self.policy_head.orient_linear.bias)
            nn.init.zeros_(self.policy_head.pos_conv.weight)
            if self.policy_head.pos_conv.bias is not None:
                nn.init.zeros_(self.policy_head.pos_conv.bias)
        elif self._use_structured_policy_head:
            nn.init.zeros_(self.policy_head.buy_conv.weight)
            if self.policy_head.buy_conv.bias is not None:
                nn.init.zeros_(self.policy_head.buy_conv.bias)
            nn.init.zeros_(self.policy_head.patch_conv.weight)
            if self.policy_head.patch_conv.bias is not None:
                nn.init.zeros_(self.policy_head.patch_conv.bias)
            nn.init.zeros_(self.policy_head.pass_linear.weight)
            nn.init.zeros_(self.policy_head.pass_linear.bias)
        elif getattr(self.policy_head, "_use_factorized", False):
            nn.init.zeros_(self.policy_head.fc2.weight)
            nn.init.zeros_(self.policy_head.fc2.bias)
        else:
            nn.init.zeros_(self.policy_head.fc.weight)
            nn.init.zeros_(self.policy_head.fc.bias)
        # FiLM: zero-init final layer so gamma=0, beta=0 → identity at start
        if self.film_mlp is not None:
            nn.init.zeros_(self.film_mlp[-1].weight)
            nn.init.zeros_(self.film_mlp[-1].bias)
        # Global bias layers: zero-init so they start neutral (no distortion at init)
        if self._use_hierarchical_policy:
            if self.policy_head.g_to_pos_bias is not None:
                nn.init.zeros_(self.policy_head.g_to_pos_bias.weight)
                nn.init.zeros_(self.policy_head.g_to_pos_bias.bias)
            if self.policy_head.g_to_patch_bias is not None:
                nn.init.zeros_(self.policy_head.g_to_patch_bias.weight)
                nn.init.zeros_(self.policy_head.g_to_patch_bias.bias)
        elif self._use_structured_policy_head:
            if self.policy_head.g_to_buy_bias is not None:
                nn.init.zeros_(self.policy_head.g_to_buy_bias.weight)
                nn.init.zeros_(self.policy_head.g_to_buy_bias.bias)
            if self.policy_head.g_to_patch_bias is not None:
                nn.init.zeros_(self.policy_head.g_to_patch_bias.weight)
                nn.init.zeros_(self.policy_head.g_to_patch_bias.bias)
        # Trunk pooling projection: zero-init so g_heads starts identical to global_to_heads(x_global)
        if self.trunk_to_heads is not None:
            nn.init.zeros_(self.trunk_to_heads.weight)
            nn.init.zeros_(self.trunk_to_heads.bias)

    def _trunk_forward(
        self,
        state: torch.Tensor,
        x_global: Optional[torch.Tensor] = None,
        x_track: Optional[torch.Tensor] = None,
        shop_ids: Optional[torch.Tensor] = None,
        shop_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Shared residual trunk — used by all heads."""
        gamma_list: Optional[List[torch.Tensor]] = None
        beta_list: Optional[List[torch.Tensor]] = None

        if self.use_film and self.film_mlp is not None:
            if self.film_global_mlp is not None and x_global is not None:
                g_enc = self.film_global_mlp(x_global)
                if self.film_track_conv is not None:
                    t_out = self.film_track_conv(x_track)       # (B, 32, 54)
                    t_enc = self.film_track_pool(t_out.mean(dim=2))  # (B, film_hidden)
                else:
                    t_enc = self.film_track_mlp(x_track.flatten(1))
                s_enc = self.shop_encoder(shop_ids, shop_feats)
                g = torch.cat([g_enc, t_enc, s_enc], dim=1)
            else:
                if not self.film_input_plane_indices:
                    raise ValueError("use_film=true but no multimodal inputs and film_input_plane_indices empty")
                max_idx = max(self.film_input_plane_indices)
                if max_idx >= state.shape[1]:
                    raise ValueError(f"film_input_plane_indices {max_idx} >= state channels {state.shape[1]}")
                feats: List[torch.Tensor] = []
                for idx in self.film_input_plane_indices:
                    if self.film_use_plane_mean:
                        feats.append(state[:, idx, :, :].mean(dim=(1, 2)))
                    else:
                        feats.append(state[:, idx, 0, 0])
                g = torch.stack(feats, dim=1)
            film_out = self.film_mlp(g)
            film = film_out.view(film_out.shape[0], self.num_res_blocks, 2, self.channels)
            gamma_list = [film[:, i, 0, :] for i in range(self.num_res_blocks)]
            beta_list = [film[:, i, 1, :] for i in range(self.num_res_blocks)]

        x = F.relu(self.bn_input(self.conv_input(state)), inplace=True)
        if gamma_list is not None and beta_list is not None:
            for i, block in enumerate(self.res_blocks):
                x = block(x, gamma_list[i], beta_list[i])
        else:
            for block in self.res_blocks:
                x = block(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x

    def _apply_gpu_legality(
        self,
        state: torch.Tensor,
        shop_ids: Optional[torch.Tensor],
        shop_feats: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Run DeterministicLegalityModule to compute legalTL and update shop_feats.

        Accepts two state formats:
          - (B, 36, 9, 9): gold_v2_36ch new encoder output (channels 0-35 only).
            Concatenates legalTL_24 → (B, 60, 9, 9) for the trunk.
          - (B, 32, 9, 9): gold_v2_32ch legacy encoder output.
            Concatenates legalTL_24 → (B, 56, 9, 9) for the trunk.
          - (B, 60, 9, 9) or (B, 56, 9, 9): already-full trunk states.
            Overwrites the last 24 channels with fresh GPU values.

        Returns: (state_trunk_ch, shop_feats_with_num_legal)
        """
        if self.det_legality is None or shop_ids is None or shop_feats is None:
            return state, shop_feats

        B = state.shape[0]
        n_ch = state.shape[1]  # 36 (new 36ch) / 32 (legacy 32ch) / 60 or 56 (full trunk)

        # board_free: 1 = empty cell, 0 = occupied.  Channel 0 = current player.
        cur_free = (1.0 - state[:, 0:1, :, :]).float()  # (B, 1, 9, 9)
        opp_free = (1.0 - state[:, 1:2, :, :]).float()  # (B, 1, 9, 9)

        # Stack into (2B, 1, 9, 9) so we pay for one conv call instead of two.
        both_free = torch.cat([cur_free, opp_free], dim=0)   # (2B, 1, 9, 9)
        legal_both = self.det_legality(both_free)             # (2B, 264, 9, 9)
        legal_cur = legal_both[:B]                            # (B, 264, 9, 9)
        legal_opp = legal_both[B:]                            # (B, 264, 9, 9)

        # --- legalTL for visible slots → appended as last 24 channels ---
        vis_ids = shop_ids[:, :3].long()          # (B, 3)
        afford_vis = shop_feats[:, :3, 6].float() # (B, 3) -- slot affordability flag
        legalTL_24 = self.det_legality.extract_vis_legalTL(legal_cur, vis_ids, afford_vis)

        if n_ch in (32, 36):
            # New path: cat encoded channels + 24 GPU legalTL → full trunk
            state = torch.cat([state, legalTL_24], dim=1)  # (B, 56 or 60, 9, 9)
        else:
            # Legacy path: state already has full trunk channels; overwrite legalTL portion
            state = torch.cat([state[:, :n_ch - 24], legalTL_24], dim=1)  # (B, same n_ch)

        # --- num_legal for all 33 shop pieces (features 8 and 9) ---
        shop_ids_long = shop_ids.long()
        nl_cur = self.det_legality.compute_num_legal_normalized(legal_cur, shop_ids_long)
        nl_opp = self.det_legality.compute_num_legal_normalized(legal_opp, shop_ids_long)

        # Rebuild shop_feats without in-place modification
        shop_feats = torch.cat(
            [shop_feats[:, :, :8], nl_cur.unsqueeze(-1), nl_opp.unsqueeze(-1)],
            dim=-1,
        )  # (B, 33, 10)

        return state, shop_feats

    def forward(
        self,
        state: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        x_global: Optional[torch.Tensor] = None,
        x_track: Optional[torch.Tensor] = None,
        shop_ids: Optional[torch.Tensor] = None,
        shop_feats: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning (policy_logits, value, score).

        state: x_spatial (B, C_SPATIAL_ENC, 9, 9) for gold_v2 (36ch new / 32ch legacy)
        x_global, x_track, shop_ids, shop_feats: optional multimodal inputs for FiLM
        """
        state, shop_feats = self._apply_gpu_legality(state, shop_ids, shop_feats)
        trunk = self._trunk_forward(state, x_global, x_track, shop_ids, shop_feats)
        g_heads = None
        if self.global_to_heads is not None and x_global is not None:
            g_heads = self.global_to_heads(x_global)
            if self.trunk_to_heads is not None:
                # Add trunk spatial pooling: mean over 9×9 → projected to same dim → additive residual
                g_heads = g_heads + self.trunk_to_heads(trunk.mean(dim=(2, 3)))
        if self._use_hierarchical_policy:
            # Hierarchical head returns log-probabilities; masking is internal.
            # softmax(log_probs) = probs and log_softmax(log_probs) = log_probs,
            # so downstream code works unchanged.
            policy_logits = self.policy_head(trunk, action_mask=action_mask, g=g_heads)
        else:
            policy_logits = self.policy_head(trunk, g=g_heads)
            if action_mask is not None:
                policy_logits = policy_logits.masked_fill(action_mask == 0, float("-inf"))
        value_out = self.value_head(trunk, g=g_heads)
        if isinstance(value_out, tuple):
            value, score_logits = value_out[0], value_out[1]
        else:
            value = value_out
            B = value.shape[0]
            score_logits = torch.zeros(B, ValueHead.SCORE_BINS, device=value.device, dtype=value.dtype)
        return policy_logits, value, score_logits

    def predict(
        self, state: torch.Tensor, action_mask: Optional[torch.Tensor] = None, temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inference helper. Returns (policy_probs, value, score_logits)."""
        self.eval()  # Ensure BatchNorm/Dropout in eval mode
        with torch.no_grad():
            policy_logits, value, score_logits = self.forward(state, action_mask)
            if temperature > 0:
                policy_probs = F.softmax(policy_logits / temperature, dim=-1)
            else:
                policy_probs = torch.zeros_like(policy_logits)
                policy_probs.scatter_(1, policy_logits.argmax(dim=-1, keepdim=True), 1.0)
            return policy_probs, value, score_logits

    def get_loss(
        self,
        state: torch.Tensor,
        action_mask: torch.Tensor,
        target_policy: torch.Tensor,
        target_value: torch.Tensor,
        policy_weight: float = 1.0,
        value_weight: float = 1.0,
        target_score: Optional[torch.Tensor] = None,
        target_score_dist: Optional[torch.Tensor] = None,
        score_loss_weight: float = 0.02,
        target_ownership: Optional[torch.Tensor] = None,
        ownership_weight: float = 0.0,
        ownership_valid_mask: Optional[torch.Tensor] = None,
        ownership_pos_weight: float = 1.0,
        target_bonus: Optional[torch.Tensor] = None,
        bonus_weight: float = 0.0,
        target_threat: Optional[torch.Tensor] = None,
        threat_weight: float = 0.0,
        policy_entropy_bonus_weight: float = 0.0,
        value_label_smoothing: float = 0.0,
        x_global: Optional[torch.Tensor] = None,
        x_track: Optional[torch.Tensor] = None,
        shop_ids: Optional[torch.Tensor] = None,
        shop_feats: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute training loss.

        Args:
            target_ownership: (B, 2, 9, 9) float32 binary targets.
                              Channel 0: current player's board (0=empty, 1=filled).
                              Channel 1: opponent's board (0=empty, 1=filled).
                              Only used when ownership_head exists and weight > 0.
            ownership_weight: Loss weight for the auxiliary ownership head.
            ownership_valid_mask: (B,) bool tensor. When provided, ownership loss
                              is computed only on samples where mask is True.
                              Allows mixed old/new data batches to still train
                              the ownership head on valid samples.
        """
        state, shop_feats = self._apply_gpu_legality(state, shop_ids, shop_feats)
        trunk = self._trunk_forward(state, x_global, x_track, shop_ids, shop_feats)
        g_heads = None
        if self.global_to_heads is not None and x_global is not None:
            g_heads = self.global_to_heads(x_global)
            if self.trunk_to_heads is not None:
                # Add trunk spatial pooling: mean over 9×9 → projected to same dim → additive residual
                g_heads = g_heads + self.trunk_to_heads(trunk.mean(dim=(2, 3)))
        if self._use_hierarchical_policy:
            policy_logits = self.policy_head(trunk, action_mask=action_mask, g=g_heads)
        else:
            policy_logits = self.policy_head(trunk, g=g_heads)
            if action_mask is not None:
                policy_logits = policy_logits.masked_fill(action_mask == 0, float("-inf"))
        value_out = self.value_head(trunk, g=g_heads)
        if isinstance(value_out, tuple):
            value, score_logits = value_out[0], value_out[1]
        else:
            value = value_out
            B = value.shape[0]
            score_logits = torch.zeros(B, ValueHead.SCORE_BINS, device=value.device, dtype=value.dtype)

        # Policy loss (cross-entropy)
        target_policy_sum = target_policy.sum(dim=-1, keepdim=True)
        target_policy_norm = target_policy / torch.clamp(target_policy_sum, min=1e-8)
        log_probs = F.log_softmax(policy_logits, dim=-1).clamp(min=-100.0)
        policy_loss = -(target_policy_norm * log_probs).sum(dim=-1).mean()
        # CRITICAL FIX: Raise on NaN instead of silently zeroing (hides bugs)
        if torch.isnan(policy_loss):
            if self.training:
                raise RuntimeError("policy_loss is NaN - check masks/logits/targets")
            policy_loss = torch.tensor(0.0, device=policy_loss.device)  # Inference: skip

        # Policy entropy (in-graph so entropy regularisation receives gradients).
        # Reuses log_probs already computed above — no extra forward pass.
        policy_entropy_live = -(log_probs.exp() * log_probs).sum(dim=-1).mean()

        # Value loss (MSE on pure binary target: +1/-1/0)
        # Value head predicts P(win) via tanh; score head independently handles margin.
        # Label smoothing: clamp targets away from ±1 to prevent infinite-logit pressure.
        # Targets are in {-1, 0, +1}; smoothed range becomes [-(1-ε), +(1-ε)].
        if value_label_smoothing > 0.0:
            target_value = target_value.clamp(
                -(1.0 - value_label_smoothing), 1.0 - value_label_smoothing
            )
        value_loss = F.mse_loss(value, target_value)

        # Score loss: distributional CE over 201 bins (soft target from replay score_margins).
        score_loss = torch.tensor(0.0, device=state.device)
        if score_loss_weight > 0 and self.value_head.score_head is not None:
            if target_score_dist is not None:
                # (B, 201) soft labels; normalize and CE
                tgt = target_score_dist.float() / target_score_dist.float().sum(dim=-1, keepdim=True).clamp(min=1e-12)
                logp = F.log_softmax(score_logits.float(), dim=-1).clamp(min=-100.0)
                score_loss = -(tgt * logp).sum(dim=-1).mean()
            elif target_score is not None:
                # Legacy scalar target: no-op (trainer should pass target_score_dist for 201-bin).
                pass

        # Ownership loss (auxiliary, KataGo-style adapted for dual-board)
        # ownership_pos_weight > 1: upweights empty cells (target=0) to fix class imbalance.
        # ~83% of cells are filled at game end; without weighting, the model learns to predict
        # "all filled" (empty-cell recall ~30%). Formula: cell_weight = 1 + (w-1)*(1-target),
        # so empty cells (target=0) get weight=w, filled cells (target=1) get weight=1.
        ownership_loss = torch.tensor(0.0, device=state.device)
        ownership_logits = None
        if (
            self.ownership_head is not None
            and target_ownership is not None
            and ownership_weight > 0
        ):
            ownership_logits = self.ownership_head(trunk)  # (B, 2, 9, 9)
            if ownership_valid_mask is not None and not ownership_valid_mask.all():
                # Per-sample masking: only compute loss on valid samples
                valid_logits = ownership_logits[ownership_valid_mask]
                valid_targets = target_ownership[ownership_valid_mask]
                if valid_logits.shape[0] > 0:
                    if ownership_pos_weight != 1.0:
                        per_cell = F.binary_cross_entropy_with_logits(
                            valid_logits, valid_targets, reduction='none'
                        )
                        cell_w = 1.0 + (ownership_pos_weight - 1.0) * (1.0 - valid_targets)
                        ownership_loss = (per_cell * cell_w).mean()
                    else:
                        ownership_loss = F.binary_cross_entropy_with_logits(
                            valid_logits, valid_targets
                        )
            else:
                # All samples valid (or no mask) — standard path
                if ownership_pos_weight != 1.0:
                    per_cell = F.binary_cross_entropy_with_logits(
                        ownership_logits, target_ownership, reduction='none'
                    )
                    cell_w = 1.0 + (ownership_pos_weight - 1.0) * (1.0 - target_ownership)
                    ownership_loss = (per_cell * cell_w).mean()
                else:
                    ownership_loss = F.binary_cross_entropy_with_logits(
                        ownership_logits, target_ownership
                    )

        # 7×7 bonus prediction (binary BCE with logits, two outputs: me / opponent)
        # target_bonus: (B, 2) float32 — [1.0 if current player got bonus else 0.0,
        #                                  1.0 if opponent got bonus else 0.0]
        bonus_loss = torch.tensor(0.0, device=state.device)
        if self.bonus_head is not None and target_bonus is not None and bonus_weight > 0:
            bonus_logits = self.bonus_head(trunk)  # (B, 2) raw logits
            bonus_loss = F.binary_cross_entropy_with_logits(bonus_logits, target_bonus.float())

        # Opponent board threat (sigmoid MSE regression on normalized empty fraction)
        # target_threat: (B,) float32 — opponent's final empty squares / 81  ∈ [0, 1]
        threat_loss = torch.tensor(0.0, device=state.device)
        if self.threat_head is not None and target_threat is not None and threat_weight > 0:
            threat_logits = self.threat_head(trunk).squeeze(-1)  # (B,)
            threat_pred = torch.sigmoid(threat_logits)           # bound to [0, 1]
            threat_loss = F.mse_loss(threat_pred, target_threat.float())

        total_loss = (
            policy_weight * policy_loss
            + value_weight * value_loss
            + score_loss_weight * score_loss
            + ownership_weight * ownership_loss
            + bonus_weight * bonus_loss
            + threat_weight * threat_loss
            - policy_entropy_bonus_weight * policy_entropy_live
        )

        # Guard against NaN in any loss component (not just policy)
        if torch.isnan(total_loss):
            if self.training:
                raise RuntimeError(
                    f"total_loss is NaN — policy={policy_loss.item()}, "
                    f"value={value_loss.item()}, ownership={ownership_loss.item()}, "
                    f"bonus={bonus_loss.item()}, threat={threat_loss.item()}. "
                    f"Check inputs for corruption."
                )
            total_loss = torch.tensor(0.0, device=total_loss.device)

        with torch.no_grad():
            pred_actions = policy_logits.argmax(dim=-1)
            target_actions = target_policy.argmax(dim=-1)
            policy_accuracy = (pred_actions == target_actions).float().mean()
            _, top5_pred = policy_logits.topk(5, dim=-1)
            policy_top5_accuracy = (top5_pred == target_actions.unsqueeze(1)).any(dim=1).float().mean()

            # Training diagnostics: policy entropy (reuse in-graph tensor — already computed above)
            policy_entropy = policy_entropy_live

            target_log = target_policy_norm.clamp(min=1e-8).log()
            target_entropy = -(target_policy_norm * target_log).sum(dim=-1).mean()

            # KL(MCTS_target || network_policy) - measures how far net is from MCTS
            kl_div = (target_policy_norm * (target_log - log_probs)).sum(dim=-1).mean()

            policy_cross_entropy = policy_loss
            ce_minus_policy_entropy = policy_cross_entropy - policy_entropy

            # Per-sample identity check: CE(pi,p) = H(pi) + KL(pi||p) per sample, then mean(|err|)
            ce_i = -(target_policy_norm * log_probs).sum(dim=-1)
            h_i = -(target_policy_norm * target_log).sum(dim=-1)
            kl_i = (target_policy_norm * (target_log - log_probs)).sum(dim=-1)
            approx_identity_error = (h_i + kl_i - ce_i).abs().mean()

            # Ownership accuracy (binary, threshold at 0.5) — only over valid samples
            ownership_accuracy = 0.0
            # Class-imbalance / metric-lie diagnostics (E): empty-cell recall/precision, balanced acc, MAE empty count
            ownership_empty_recall = 0.0
            ownership_empty_precision = 0.0
            ownership_balanced_accuracy = 0.0
            ownership_mae_empty_count = 0.0
            ownership_filled_fraction_mean = 0.0
            ownership_accuracy_all_filled_baseline = 0.0
            if ownership_logits is not None:
                ownership_pred = (ownership_logits > 0).float()  # sigmoid > 0.5
                if ownership_valid_mask is not None and not ownership_valid_mask.all():
                    valid_pred = ownership_pred[ownership_valid_mask]
                    valid_targets = target_ownership[ownership_valid_mask]
                    if valid_pred.shape[0] > 0:
                        ownership_accuracy = float(
                            (valid_pred == valid_targets).float().mean().item()
                        )
                        # E: filled fraction (mean target) = accuracy if we predicted "all filled"
                        ownership_filled_fraction_mean = float(valid_targets.mean().item())
                        ownership_accuracy_all_filled_baseline = ownership_filled_fraction_mean
                        # Empty-cell: target==0
                        n_empty_true = (valid_targets < 0.5).float().sum().item()
                        n_empty_pred = (valid_pred < 0.5).float().sum().item()
                        true_empty = (valid_targets < 0.5)
                        pred_empty = (valid_pred < 0.5)
                        if n_empty_true > 0:
                            ownership_empty_recall = float((true_empty & pred_empty).float().sum().item() / n_empty_true)
                        if n_empty_pred > 0:
                            ownership_empty_precision = float((true_empty & pred_empty).float().sum().item() / n_empty_pred)
                        # Filled recall (sensitivity): among target==1, how many pred==1
                        n_filled_true = (valid_targets >= 0.5).float().sum().item()
                        if n_filled_true > 0:
                            filled_recall = float(((valid_targets >= 0.5) & (valid_pred >= 0.5)).float().sum().item() / n_filled_true)
                        else:
                            filled_recall = 0.0
                        ownership_balanced_accuracy = 0.5 * (ownership_empty_recall + filled_recall) if n_filled_true > 0 else ownership_empty_recall
                        # MAE empty count: per-sample over valid; need to aggregate per (B,2,9,9) sample then mean
                        # valid_pred/targets are flattened; we don't have sample dim here. Skip MAE per-sample or compute on full tensors.
                else:
                    ownership_accuracy = float(
                        (ownership_pred == target_ownership).float().mean().item()
                    )
                    ownership_filled_fraction_mean = float(target_ownership.mean().item())
                    ownership_accuracy_all_filled_baseline = ownership_filled_fraction_mean
                    true_empty = (target_ownership < 0.5)
                    pred_empty = (ownership_pred < 0.5)
                    n_empty_true = true_empty.float().sum().item()
                    n_empty_pred = pred_empty.float().sum().item()
                    if n_empty_true > 0:
                        ownership_empty_recall = float((true_empty & pred_empty).float().sum().item() / n_empty_true)
                    if n_empty_pred > 0:
                        ownership_empty_precision = float((true_empty & pred_empty).float().sum().item() / n_empty_pred)
                    n_filled_true = (target_ownership >= 0.5).float().sum().item()
                    if n_filled_true > 0:
                        filled_recall = float(((target_ownership >= 0.5) & (ownership_pred >= 0.5)).float().sum().item() / n_filled_true)
                        ownership_balanced_accuracy = 0.5 * (ownership_empty_recall + filled_recall)
                    else:
                        ownership_balanced_accuracy = ownership_empty_recall
                    # MAE empty count: per sample (B, 2, 9, 9) -> empty count per sample
                    pred_empty_per_sample = (ownership_pred < 0.5).float().view(ownership_pred.shape[0], -1).sum(dim=1)
                    target_empty_per_sample = (target_ownership < 0.5).float().view(target_ownership.shape[0], -1).sum(dim=1)
                    ownership_mae_empty_count = float((pred_empty_per_sample - target_empty_per_sample).abs().mean().item())

                # When valid_mask used, compute MAE on full tensors (flattened valid only would need indexing)
                if ownership_valid_mask is None or ownership_valid_mask.all():
                    pred_empty_per_sample = (ownership_pred < 0.5).float().view(ownership_pred.shape[0], -1).sum(dim=1)
                    target_empty_per_sample = (target_ownership < 0.5).float().view(target_ownership.shape[0], -1).sum(dim=1)
                    ownership_mae_empty_count = float((pred_empty_per_sample - target_empty_per_sample).abs().mean().item())

        metrics = {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "score_loss": float(score_loss.item()),
            "ownership_loss": float(ownership_loss.item()),
            "total_loss": float(total_loss.item()),
            "policy_accuracy": float(policy_accuracy.item()),
            "policy_top5_accuracy": float(policy_top5_accuracy.item()),
            "value_mse": float(value_loss.item()),
            "policy_entropy": float(policy_entropy.item()),
            "target_entropy": float(target_entropy.item()),
            "policy_cross_entropy": float(policy_cross_entropy.item()),
            "ce_minus_policy_entropy": float(ce_minus_policy_entropy.item()),
            "kl_divergence": float(kl_div.item()),
            "approx_identity_check": float(approx_identity_error.item()),
            "ownership_accuracy": float(ownership_accuracy),
            "ownership_filled_fraction_mean": float(ownership_filled_fraction_mean),
            "ownership_accuracy_all_filled_baseline": float(ownership_accuracy_all_filled_baseline),
            "ownership_empty_recall": float(ownership_empty_recall),
            "ownership_empty_precision": float(ownership_empty_precision),
            "ownership_balanced_accuracy": float(ownership_balanced_accuracy),
            "ownership_mae_empty_count": float(ownership_mae_empty_count),
            "bonus_loss": float(bonus_loss.item()),
            "threat_loss": float(threat_loss.item()),
        }
        return total_loss, metrics

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_network(config: dict) -> PatchworkNetwork:
    """Factory function to create network from config."""
    net_config = config["network"]
    film_indices = net_config.get("film_input_plane_indices", [])
    if isinstance(film_indices, (list, tuple)):
        film_indices = list(film_indices)
    else:
        film_indices = []
    return PatchworkNetwork(
        input_channels=int(net_config.get("input_channels", DEFAULT_INPUT_CHANNELS)),
        num_res_blocks=int(net_config["num_res_blocks"]),
        channels=int(net_config["channels"]),
        policy_channels=int(net_config["policy_channels"]),
        policy_hidden=int(net_config.get("policy_hidden", 256)),
        use_factorized_policy_head=bool(net_config.get("use_factorized_policy_head", True)),
        value_channels=int(net_config["value_channels"]),
        value_hidden=int(net_config["value_hidden"]),
        max_actions=int(net_config.get("max_actions", DEFAULT_MAX_ACTIONS)),
        use_batch_norm=bool(net_config["use_batch_norm"]),
        se_ratio=float(net_config["se_ratio"]),
        dropout=float(net_config["dropout"]),
        ownership_channels=int(net_config.get("ownership_channels", 0)),
        aux_channels=int(net_config.get("aux_channels", 0)),
        use_film=bool(net_config.get("use_film", False)),
        film_hidden=int(net_config.get("film_hidden", 256)),
        film_input_plane_indices=film_indices,
        film_use_plane_mean=bool(net_config.get("film_use_plane_mean", True)),
        film_per_block=bool(net_config.get("film_per_block", True)),
        film_global_dim=int(net_config.get("film_global_dim", 0)),
        film_track_dim=int(net_config.get("film_track_dim", 0)),
        film_shop_dim=int(net_config.get("film_shop_dim", 0)),
        use_gpu_legality=bool(net_config.get("use_gpu_legality", True)),
        film_track_use_conv=bool(net_config.get("film_track_use_conv", False)),
        film_global_inject_dim=int(net_config.get("film_global_inject_dim", 0)),
        use_hierarchical_policy=bool(net_config.get("use_hierarchical_policy", False)),
    )


def load_model_checkpoint(
    network: PatchworkNetwork, state_dict: dict
) -> None:
    """Load model state dict with backward compatibility for architectural changes.

    - Old checkpoints without ownership_head.*: load with strict=False, init head from scratch.
    - Policy head format mismatch (fc vs fc1/fc2): load with strict=False; policy head re-inits.
    - Old checkpoints without film_mlp.*: load with strict=False; FiLM init from scratch.
      (Use use_factorized_policy_head: false in config to compare legacy vs factorized.)
    """
    # Widen conv_input when loading old 14/16 channel checkpoints into 56ch (gold_v2) model
    if "conv_input.weight" in state_dict:
        old_w = state_dict["conv_input.weight"]
        new_w = network.conv_input.weight
        old_in = old_w.shape[1]
        new_in = new_w.shape[1]
        if old_in != new_in:
            if old_in in (14, 16) and new_in in (56, 60, 61):
                widened = torch.zeros(new_w.shape, dtype=old_w.dtype, device=old_w.device)
                widened[:, :old_in, :, :] = old_w
                state_dict["conv_input.weight"] = widened
                logger.warning("Checkpoint conv_input widened %d->%d channels (new channels zero-init).", old_in, new_in)
            elif old_in != new_in:
                raise ValueError(f"Checkpoint conv_input has {old_in} channels, model expects {new_in}. Incompatible.")

    # Drop incompatible score_head weights (old scalar 1-dim vs new 201-bin)
    for shk in list(state_dict.keys()):
        if "score_head" in shk and shk in network.state_dict():
            if state_dict[shk].shape != network.state_dict()[shk].shape:
                logger.warning(
                    "Dropping incompatible score_head key %s (shape %s vs %s); score head will reinitialise.",
                    shk, state_dict[shk].shape, network.state_dict()[shk].shape,
                )
                del state_dict[shk]

    # Widen FiLM MLP first layer when loading old checkpoints (11->20 inputs)
    if network.film_mlp is not None and "film_mlp.0.weight" in state_dict:
        old_film_w = state_dict["film_mlp.0.weight"]
        new_film_w = network.film_mlp[0].weight
        old_in_film = old_film_w.shape[1]
        new_in_film = new_film_w.shape[1]
        if old_in_film < new_in_film:
            out_film = old_film_w.shape[0]
            widened_film = torch.zeros(out_film, new_in_film, dtype=old_film_w.dtype, device=old_film_w.device)
            widened_film[:, :old_in_film] = old_film_w
            state_dict["film_mlp.0.weight"] = widened_film
            logger.warning(
                "Checkpoint film_mlp.0 widened %d->%d inputs (new columns zero-init).",
                old_in_film, new_in_film,
            )

    current_keys = set(network.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    missing = current_keys - ckpt_keys
    unexpected = ckpt_keys - current_keys

    ckpt_has_legacy = "policy_head.fc.weight" in ckpt_keys and "policy_head.fc1.weight" not in ckpt_keys
    ckpt_has_factorized = "policy_head.fc1.weight" in ckpt_keys
    ckpt_has_structured = "policy_head.p_conv.weight" in ckpt_keys and "policy_head.type_linear.weight" not in ckpt_keys
    ckpt_has_hierarchical = "policy_head.type_linear.weight" in ckpt_keys

    policy_structured_missing = {k for k in missing if any(
        p in k for p in ("policy_head.p_", "policy_head.buy_", "policy_head.patch_", "policy_head.pass_",
                         "policy_head.type_", "policy_head.slot_", "policy_head.orient_",
                         "policy_head.pos_", "policy_head.g_to_pos_", "policy_head.g_to_patch_",
                         "policy_head.g_to_buy_")
    )}
    policy_fc_factorized_missing = {
        k for k in missing if k.startswith("policy_head.fc1") or k.startswith("policy_head.fc2")
    }
    policy_fc_legacy_missing = {k for k in missing if k in ("policy_head.fc.weight", "policy_head.fc.bias")}
    policy_legacy_conv = {k for k in missing if k.startswith("policy_head.conv") or k.startswith("policy_head.bn")}

    allowable_missing = (
        {k for k in missing if "ownership_head" in k}
        | {k for k in missing if "score_head" in k}
        | {k for k in missing if "film_mlp" in k}
        | {k for k in missing if "det_legality" in k}  # new non-trainable module
        | {k for k in missing if "bonus_head" in k}    # aux head — absent in pre-aux checkpoints
        | {k for k in missing if "threat_head" in k}   # aux head — absent in pre-aux checkpoints
        | policy_structured_missing
        | policy_fc_factorized_missing
        | policy_fc_legacy_missing
        | policy_legacy_conv
    )
    other_missing = missing - allowable_missing
    if other_missing:
        raise ValueError(
            f"Checkpoint has incompatible keys. Missing in model: {other_missing}. "
            "Only trunk, value, ownership, FiLM, and policy head params may differ."
        )

    if not missing and not unexpected:
        network.load_state_dict(state_dict)
        return

    to_load = {k: v for k, v in state_dict.items() if k in current_keys}
    msgs = []
    if {k for k in missing if "ownership_head" in k}:
        msgs.append("ownership head — init from scratch")
    if policy_structured_missing and (ckpt_has_legacy or ckpt_has_factorized):
        msgs.append("policy head (ckpt has legacy/factorized) — structured/hierarchical head init from scratch")
    if policy_structured_missing and ckpt_has_structured:
        msgs.append("policy head (ckpt has structured) — hierarchical head init from scratch")
    if policy_structured_missing and ckpt_has_hierarchical:
        msgs.append("policy head (ckpt has hierarchical) — structured head init from scratch")
    if (policy_fc_factorized_missing or policy_fc_legacy_missing or policy_legacy_conv) and (ckpt_has_structured or ckpt_has_hierarchical):
        msgs.append("policy head (ckpt has structured/hierarchical) — legacy head init from scratch")
    if policy_fc_factorized_missing and ckpt_has_legacy:
        msgs.append("policy head (ckpt has legacy fc) — factorized head init from scratch")
    if policy_fc_legacy_missing and ckpt_has_factorized:
        msgs.append("policy head (ckpt has factorized fc1/fc2) — legacy head init from scratch")
    if {k for k in missing if "film_mlp" in k}:
        msgs.append("FiLM MLP — init from scratch")
    if unexpected and any("policy_head" in k for k in unexpected):
        msgs.append("policy head (ckpt has different format) — skipping unexpected")
    if msgs:
        logger.warning(
            "Checkpoint incompatible with model: %s. Loading with strict=False.",
            "; ".join(msgs),
        )
    network.load_state_dict(to_load, strict=False)


def get_state_dict_for_inference(ckpt: dict, config: dict, for_selfplay: bool = False) -> dict:
    """Return appropriate state dict for inference based on config."""
    use_ema = False
    if "ema" in config.get("training", {}):
        ema_cfg = config["training"]["ema"]
        if ema_cfg.get("enabled", False):
            if for_selfplay and ema_cfg.get("use_for_selfplay", True):
                use_ema = True
            elif not for_selfplay and ema_cfg.get("use_for_eval", True):
                use_ema = True
                
    if use_ema and "ema_state_dict" in ckpt:
        return ckpt["ema_state_dict"]
    return ckpt.get("model_state_dict", ckpt)