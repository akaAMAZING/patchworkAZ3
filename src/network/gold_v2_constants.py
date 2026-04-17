"""
Gold v2 Multimodal — Shared Constants (Dimension Locking)

Single source of truth for encoding dimensions and policy indexing.
All encoders, D4 transforms, HDF5 schema, and tests MUST use these.
"""

from __future__ import annotations

# Spatial / global / track / shop dimensions
C_SPATIAL = 60      # Model trunk input: 36 encoded + 24 GPU-computed legalTL
C_SPATIAL_ENC = 36  # Encoder output / replay buffer / IPC shape (channels 0-35 only)
C_TRACK = 8
F_GLOBAL = 65
F_SHOP = 10
NMAX = 33  # from patchwork_engine.MAX_CIRCLE
TRACK_LEN = 54  # positions 0..53

# Policy indexing (MUST match encoder.ActionEncoder and model)
PASS_INDEX = 0
PATCH_START = 1
PATCH_SIZE = 81  # 9*9
BUY_START = 82
NUM_SLOTS = 3
NUM_ORIENTS = 8
BUY_SIZE = NUM_SLOTS * NUM_ORIENTS * PATCH_SIZE  # 1944
MAX_ACTIONS = BUY_START + BUY_SIZE  # 2026

# HDF5 / IPC schema version — bump this to reject stale 32ch/56ch replay data
ENCODING_VERSION = "gold_v2_36ch"

# Spatial plane indices (encoder output: channels 0-35 only)
BOARD_CHANNELS = [0, 1]      # cur_occ, opp_occ
COORD_CHANNELS = [2, 3]      # coord_row, coord_col
FRONTIER_CHANNELS = [4, 5]
VALID_7X7_CHANNELS = [6, 7]
SLOT_ORIENT_SHAPE_BASE = 8   # indices 8..31: slot{s}_orient{o}_shape
# Packing quality channels (new in gold_v2_36ch):
ISO_HOLE_CHANNELS = [32, 33]      # ch 32=cur isolated holes, ch 33=opp isolated holes
CONSTRAINED_CHANNELS = [34, 35]  # ch 34=cur constrained empty (≤1 empty 4-nbr), ch 35=opp
# Note: legalTL (was 32..55, now 36..59) is computed on-GPU by DeterministicLegalityModule.
# SLOT_ORIENT_LEGAL_BASE kept for backward compatibility of D4 CPU code.
SLOT_ORIENT_LEGAL_BASE = 36
