"""
Gold v2 Multimodal — Shared Constants (Dimension Locking)

Single source of truth for encoding dimensions and policy indexing.
All encoders, D4 transforms, HDF5 schema, and tests MUST use these.
"""

from __future__ import annotations

# Spatial / global / track / shop dimensions
C_SPATIAL = 56      # Model trunk input: 32 encoded + 24 GPU-computed legalTL
C_SPATIAL_ENC = 32  # Encoder output / replay buffer / IPC shape (channels 0-31 only)
C_TRACK = 8
F_GLOBAL = 61
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

# HDF5 / IPC schema version — bump this to reject stale 56ch replay data
ENCODING_VERSION = "gold_v2_32ch"

# Spatial plane indices (encoder output: channels 0-31 only)
BOARD_CHANNELS = [0, 1]      # cur_occ, opp_occ
COORD_CHANNELS = [2, 3]      # coord_row, coord_col
FRONTIER_CHANNELS = [4, 5]
VALID_7X7_CHANNELS = [6, 7]
SLOT_ORIENT_SHAPE_BASE = 8   # indices 8..31: slot{s}_orient{o}_shape
# Note: legalTL (was 32..55) is now computed on-GPU by DeterministicLegalityModule.
# SLOT_ORIENT_LEGAL_BASE kept for backward compatibility of D4 CPU code.
SLOT_ORIENT_LEGAL_BASE = 32
