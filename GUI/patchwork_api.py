#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import math
import os
import random
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

# =========================
# AlphaZero NN integration (lazy-loaded)
# =========================
_NN_AGENT = None  # Will hold PatchworkAgent if model is loaded
_NN_SIMULATIONS = 800
_NN_DEVICE = "cuda"
_NN_MODEL_PATH: Optional[str] = None
_NN_CONFIG_PATH: Optional[str] = None

logger = logging.getLogger(__name__)

# =========================
# Constants
# =========================
BOARD_SIZE = 9
BOARD_CELLS = BOARD_SIZE * BOARD_SIZE  # 81
BOARD_MASK = (1 << BOARD_CELLS) - 1    # bits 0..80 only

SCOREBOARD_LENGTH = 53
BUTTONS_AFTER = [5, 11, 17, 23, 29, 35, 41, 47, 53]

# Special patch positions (revised edition only; original is obsolete).
EDITION_PATCHES = {
    "revised":  [26, 32, 38, 44, 50],
}
DEFAULT_EDITION = "revised"

# =========================
# Pieces (33 circle pieces)
# shape: 0 empty, 1 fabric, 2 fabric+income
# =========================
PIECES: List[dict] = [
    {"id": 0, "cost_buttons": 5, "cost_time": 5, "shape": [[2, 2, 1], [0, 1, 0], [0, 1, 0]]},
    {"id": 1, "cost_buttons": 7, "cost_time": 2, "shape": [[2, 0, 0, 0], [2, 1, 1, 1], [1, 0, 0, 0]]},
    {"id": 2, "cost_buttons": 1, "cost_time": 2, "shape": [[0, 0, 0, 1], [1, 1, 1, 1], [1, 0, 0, 0]]},
    {"id": 3, "cost_buttons": 2, "cost_time": 1, "shape": [[0, 1, 0, 0], [1, 1, 1, 1], [0, 0, 1, 0]]},
    {"id": 4, "cost_buttons": 5, "cost_time": 4, "shape": [[0, 2, 0], [1, 2, 1], [0, 1, 0]]},
    {"id": 5, "cost_buttons": 0, "cost_time": 3, "shape": [[0, 1, 0, 0], [1, 2, 1, 1], [0, 1, 0, 0]]},
    {"id": 6, "cost_buttons": 1, "cost_time": 4, "shape": [[0, 0, 1, 0, 0], [1, 1, 2, 1, 1], [0, 0, 1, 0, 0]]},
    {"id": 7, "cost_buttons": 8, "cost_time": 6, "shape": [[2, 2, 0], [2, 1, 1], [0, 0, 1]]},
    {"id": 8, "cost_buttons": 2, "cost_time": 3, "shape": [[1, 0, 1], [1, 1, 1], [1, 0, 1]]},
    {"id": 9, "cost_buttons": 10, "cost_time": 4, "shape": [[2, 2, 0], [0, 2, 1], [0, 0, 1]]},
    {"id": 10, "cost_buttons": 3, "cost_time": 6, "shape": [[0, 2, 0], [2, 1, 1], [1, 0, 1]]},
    {"id": 11, "cost_buttons": 6, "cost_time": 5, "shape": [[2, 2], [1, 1]]},
    {"id": 12, "cost_buttons": 7, "cost_time": 1, "shape": [[2, 1, 1, 1, 1]]},
    {"id": 13, "cost_buttons": 3, "cost_time": 3, "shape": [[2, 1, 1, 1]]},
    {"id": 14, "cost_buttons": 2, "cost_time": 2, "shape": [[1, 1, 1]]},
    {"id": 15, "cost_buttons": 1, "cost_time": 5, "shape": [[2, 1, 1, 1], [1, 0, 0, 1]]},
    {"id": 16, "cost_buttons": 3, "cost_time": 4, "shape": [[0, 0, 2, 0], [1, 1, 1, 1]]},
    {"id": 17, "cost_buttons": 7, "cost_time": 4, "shape": [[0, 2, 2, 0], [1, 1, 1, 1]]},
    {"id": 18, "cost_buttons": 2, "cost_time": 2, "shape": [[0, 1, 0], [1, 1, 1]]},
    {"id": 19, "cost_buttons": 2, "cost_time": 2, "shape": [[1, 1, 0], [1, 1, 1]]},
    {"id": 20, "cost_buttons": 10, "cost_time": 3, "shape": [[2, 0, 0, 0], [2, 1, 1, 1]]},
    {"id": 21, "cost_buttons": 10, "cost_time": 5, "shape": [[2, 2, 0, 0], [2, 1, 1, 1]]},
    {"id": 22, "cost_buttons": 3, "cost_time": 2, "shape": [[2, 1, 0], [0, 1, 1]]},
    {"id": 23, "cost_buttons": 7, "cost_time": 6, "shape": [[2, 2, 0], [0, 2, 1]]},
    {"id": 24, "cost_buttons": 4, "cost_time": 2, "shape": [[1, 1, 1, 0], [0, 1, 1, 1]]},
    {"id": 25, "cost_buttons": 2, "cost_time": 3, "shape": [[2, 1, 1, 0], [0, 0, 1, 1]]},
    {"id": 26, "cost_buttons": 1, "cost_time": 2, "shape": [[1, 1, 1], [1, 0, 1]]},
    {"id": 27, "cost_buttons": 4, "cost_time": 6, "shape": [[2, 0, 0], [2, 1, 1]]},
    {"id": 28, "cost_buttons": 4, "cost_time": 2, "shape": [[2, 0, 0], [1, 1, 1]]},
    {"id": 29, "cost_buttons": 1, "cost_time": 3, "shape": [[1, 0], [1, 1]]},
    {"id": 30, "cost_buttons": 3, "cost_time": 1, "shape": [[1, 0], [1, 1]]},
    {"id": 31, "cost_buttons": 5, "cost_time": 3, "shape": [[0, 2, 1, 0], [1, 1, 1, 1], [0, 1, 1, 0]]},
    {"id": 32, "cost_buttons": 2, "cost_time": 1, "shape": [[1, 1]]},
]
PIECE_BY_ID = {p["id"]: p for p in PIECES}

# =========================
# Circle setup helpers
# =========================
def _make_shuffled_circle(seed: Optional[int] = None) -> Tuple[Tuple[int, ...], int]:
    """
    Create a randomized circle order for a fresh game.

    Per Patchwork rules, the neutral pawn starts immediately clockwise of piece id 32 (the 2x1 tile).
    With this engine's representation, that means `neutral` is the index of piece 32 in `circle`.
    """
    ids = [p["id"] for p in PIECES]
    rng: random.Random = random.Random(seed) if seed is not None else random.SystemRandom()
    rng.shuffle(ids)
    neutral = ids.index(32) if 32 in ids else 0
    return tuple(ids), int(neutral)

def _is_fresh_player(p: "PlayerState") -> bool:
    return p.pos == 0 and p.occ == 0 and p.income == 0

def _looks_like_fresh_game(p0: "PlayerState", p1: "PlayerState", d: dict, circle_in) -> bool:
    """
    Heuristic used for backwards compatibility: if a client sends the default ordered circle for what
    looks like a brand-new game, auto-randomize it.
    """
    try:
        ordered = list(map(int, circle_in)) == list(range(len(PIECES)))
    except Exception:
        ordered = False

    return (
        ordered
        and _is_fresh_player(p0)
        and _is_fresh_player(p1)
        and int(d.get("pending_patches", 0)) == 0
        and int(d.get("bonus_owner", -1)) == -1
    )

# =========================
# Bitboard helpers
# =========================
def rc_to_i(r: int, c: int) -> int:
    return r * BOARD_SIZE + c

SEVEN_BY_SEVEN_MASKS: List[int] = []
for top in range(0, 3):
    for left in range(0, 3):
        m = 0
        for r in range(top, top + 7):
            for c in range(left, left + 7):
                m |= 1 << rc_to_i(r, c)
        SEVEN_BY_SEVEN_MASKS.append(m)

def has_seven_by_seven(occ: int) -> bool:
    return any((occ & m) == m for m in SEVEN_BY_SEVEN_MASKS)

def empty_count(occ: int) -> int:
    return BOARD_CELLS - (occ & BOARD_MASK).bit_count()

def empty_indices(occ: int) -> List[int]:
    occ = occ & BOARD_MASK
    return [i for i in range(BOARD_CELLS) if ((occ >> i) & 1) == 0]

# =========================
# Shape transforms
# =========================
def rotate_ccw(shape: List[List[int]]) -> List[List[int]]:
    rows, cols = len(shape), len(shape[0])
    out = []
    for c in range(cols - 1, -1, -1):
        out.append([shape[r][c] for r in range(rows)])
    return out

def flip_vertical(shape: List[List[int]]) -> List[List[int]]:
    return list(reversed([row[:] for row in shape]))

def canonical_shape(shape: List[List[int]]) -> Tuple[Tuple[int, ...], ...]:
    return tuple(tuple(r) for r in shape)

@dataclass(frozen=True, slots=True)
class Orientation:
    h: int
    w: int
    offsets: Tuple[int, ...]
    income_offsets: Tuple[int, ...]
    income: int
    size: int

@dataclass(frozen=True, slots=True)
class Placement:
    top: int
    left: int
    mask: int

@dataclass(frozen=True, slots=True)
class PieceGeom:
    piece_id: int
    cost_buttons: int
    cost_time: int
    orientations: Tuple[Orientation, ...]
    placements: Tuple[Tuple[Placement, ...], ...]

def build_piece_geom(piece: dict) -> PieceGeom:
    base = piece["shape"]
    seen = set()
    shapes: List[List[List[int]]] = []

    cur = base
    for _ in range(4):
        for f in (False, True):
            s = flip_vertical(cur) if f else [row[:] for row in cur]
            key = canonical_shape(s)
            if key not in seen:
                seen.add(key)
                shapes.append(s)
        cur = rotate_ccw(cur)

    orients: List[Orientation] = []
    placements: List[Tuple[Placement, ...]] = []

    for s in shapes:
        h, w = len(s), len(s[0])
        offs: List[int] = []
        inc_offs: List[int] = []
        inc = 0
        for r in range(h):
            for c in range(w):
                v = s[r][c]
                if v != 0:
                    off = r * BOARD_SIZE + c
                    offs.append(off)
                    if v == 2:
                        inc += 1
                        inc_offs.append(off)

        o = Orientation(
            h=h, w=w,
            offsets=tuple(offs),
            income_offsets=tuple(inc_offs),
            income=inc,
            size=len(offs),
        )
        orients.append(o)

        plist: List[Placement] = []
        for top in range(0, BOARD_SIZE - h + 1):
            for left in range(0, BOARD_SIZE - w + 1):
                delta = top * BOARD_SIZE + left
                m = 0
                for off in o.offsets:
                    m |= 1 << (off + delta)
                plist.append(Placement(top=top, left=left, mask=m))
        placements.append(tuple(plist))

    return PieceGeom(
        piece_id=piece["id"],
        cost_buttons=piece["cost_buttons"],
        cost_time=piece["cost_time"],
        orientations=tuple(orients),
        placements=tuple(placements),
    )

PIECE_GEOMS: Dict[int, PieceGeom] = {p["id"]: build_piece_geom(p) for p in PIECES}

# =========================
# Game state
# =========================
@dataclass(frozen=True, slots=True)
class PlayerState:
    pos: int
    buttons: int
    occ: int
    income: int

@dataclass(frozen=True, slots=True)
class GameState:
    p0: PlayerState
    p1: PlayerState
    circle: Tuple[int, ...]
    neutral: int
    bonus_owner: int
    pending_patches: int
    pending_owner: int
    tie_player: int
    edition: str
    patch_markers: Tuple[int, ...]


    def player(self, idx: int) -> PlayerState:
        return self.p0 if idx == 0 else self.p1

def clamp_pos(pos: int) -> int:
    return SCOREBOARD_LENGTH if pos > SCOREBOARD_LENGTH else pos

def current_player(state: GameState) -> int:
    if state.pending_patches > 0:
        return state.pending_owner
    if state.p0.pos < state.p1.pos:
        return 0
    if state.p1.pos < state.p0.pos:
        return 1
    return state.tie_player

def terminal(state: GameState) -> bool:
    return (
        state.p0.pos >= SCOREBOARD_LENGTH
        and state.p1.pos >= SCOREBOARD_LENGTH
        and state.pending_patches == 0
    )

def final_score(player: PlayerState, bonus_owner: int, player_idx: int) -> int:
    bonus = 7 if bonus_owner == player_idx else 0
    return player.buttons - 2 * empty_count(player.occ) + bonus

def score_diff_p0_minus_p1(state: GameState) -> int:
    return final_score(state.p0, state.bonus_owner, 0) - final_score(state.p1, state.bonus_owner, 1)


# =========================
# State validation (paranoid API hardening)
# =========================
VALID_PIECE_IDS = set(PIECE_BY_ID.keys())

# Safe upper bound: total income buttons across all pieces (no single board can exceed this).
_MAX_POSSIBLE_INCOME = sum(
    sum(1 for row in p["shape"] for v in row if v == 2) for p in PIECES
)

def validate_state(state: GameState) -> None:
    """Raise ValueError if any state invariant is violated."""
    for label, p in [("p0", state.p0), ("p1", state.p1)]:
        if not (0 <= p.pos <= SCOREBOARD_LENGTH):
            raise ValueError(f"{label}.pos={p.pos} out of range [0, {SCOREBOARD_LENGTH}]")
        if p.buttons < 0:
            raise ValueError(f"{label}.buttons={p.buttons} is negative")
        if p.occ & ~BOARD_MASK:
            raise ValueError(f"{label}.occ has bits set outside the 9x9 board")
        if not (0 <= p.income <= _MAX_POSSIBLE_INCOME):
            raise ValueError(f"{label}.income={p.income} out of range [0, {_MAX_POSSIBLE_INCOME}]")
        occ_bits = (p.occ & BOARD_MASK).bit_count()
        if p.income > occ_bits:
            raise ValueError(f"{label}.income={p.income} exceeds occupied squares {occ_bits}")
    if state.bonus_owner not in (-1, 0, 1):
        raise ValueError(f"bonus_owner={state.bonus_owner} must be -1, 0, or 1")
    if state.tie_player not in (0, 1):
        raise ValueError(f"tie_player={state.tie_player} must be 0 or 1")
    if state.pending_patches < 0:
        raise ValueError(f"pending_patches={state.pending_patches} is negative")
    if state.pending_owner not in (-1, 0, 1):
        raise ValueError(f"pending_owner={state.pending_owner} must be -1, 0, or 1")
    if state.pending_patches > 0 and state.pending_owner not in (0, 1):
        raise ValueError("pending_patches > 0 but pending_owner is not 0 or 1")
    if state.pending_patches == 0 and state.pending_owner != -1:
        raise ValueError("pending_patches == 0 but pending_owner != -1")
    if state.edition not in EDITION_PATCHES:
        raise ValueError(f"Unknown edition {state.edition!r}")
    # Circle validation
    if len(set(state.circle)) != len(state.circle):
        raise ValueError("circle contains duplicate piece IDs")
    for pid in state.circle:
        if pid not in VALID_PIECE_IDS:
            raise ValueError(f"circle contains unknown piece ID {pid}")
    if state.circle and not (0 <= state.neutral < len(state.circle)):
        raise ValueError(f"neutral={state.neutral} out of range for circle length {len(state.circle)}")

# =========================
# Parse UI JSON
# =========================
def parse_board(board_in) -> Tuple[int, int]:
    if isinstance(board_in, list) and len(board_in) == BOARD_SIZE and all(isinstance(r, str) for r in board_in):
        occ = 0
        income = 0
        for r, row in enumerate(board_in):
            if len(row) != BOARD_SIZE:
                raise ValueError("Each board row must be length 9")
            for c, ch in enumerate(row):
                if ch in (".", "0", " "):
                    continue
                if ch in ("1", "#", "X"):
                    occ |= 1 << rc_to_i(r, c)
                elif ch == "2":
                    occ |= 1 << rc_to_i(r, c)
                    income += 1
                else:
                    raise ValueError(f"Unknown board cell {ch!r}")
        return occ, income
    raise ValueError("Board must be list[str] with 9 rows.")

def state_from_dict(d: dict) -> GameState:
    players = d.get("players", [])
    if len(players) != 2:
        raise ValueError("state.players must have length 2")

    def parse_player(x) -> PlayerState:
        occ, board_income = parse_board(x.get("board"))
        # Use explicit income field if present, otherwise fall back to board parsing
        income = int(x.get("income", board_income))
        return PlayerState(
            pos=int(x.get("position", 0)),
            buttons=int(x.get("buttons", 0)),
            occ=occ,
            income=income,
        )

    p0 = parse_player(players[0])
    p1 = parse_player(players[1])
    circle_in = d.get("circle", None)
    randomize_circle = d.get("randomize_circle", None)  # default: True when omitted
    seed = d.get("seed", None)

    if circle_in is None or (isinstance(circle_in, (list, tuple)) and len(circle_in) == 0):
        # If the caller doesn't provide a circle, initialize a fresh randomized one.
        circle, neutral = _make_shuffled_circle(seed=seed)
    else:
        # Backwards-compat: if a client sends the default ordered circle for a fresh game,
        # randomize it unless they explicitly disable randomization.
        if (randomize_circle is None or bool(randomize_circle)) and _looks_like_fresh_game(p0, p1, d, circle_in):
            circle, neutral = _make_shuffled_circle(seed=seed)
        else:
            circle = tuple(int(x) for x in circle_in)
            neutral = int(d.get("neutral", 0))
            neutral = 0 if not circle else max(0, min(neutral, len(circle) - 1))

    bonus_owner = int(d.get("bonus_owner", -1))
    if bonus_owner not in (-1, 0, 1):
        raise ValueError("bonus_owner must be -1, 0, or 1")

    pending_patches = int(d.get("pending_patches", 0))
    pending_owner = int(d.get("pending_owner", -1))
    if pending_patches < 0:
        raise ValueError("pending_patches cannot be negative")
    if pending_patches > 0 and pending_owner not in (0, 1):
        raise ValueError("pending_patches > 0 but pending_owner is not 0 or 1")
    if pending_patches == 0 and pending_owner != -1:
        raise ValueError("pending_patches == 0 but pending_owner != -1")

    tie_player = int(d.get("tie_player", 0))
    if tie_player not in (0, 1):
        raise ValueError("tie_player must be 0 or 1")

    edition = str(d.get("edition", DEFAULT_EDITION))
    if edition not in EDITION_PATCHES:
        raise ValueError(f"Unknown edition {edition!r}. Valid: {sorted(EDITION_PATCHES.keys())}")
    patch_markers = tuple(EDITION_PATCHES[edition])

    state = GameState(
        p0=p0,
        p1=p1,
        circle=circle,
        neutral=neutral,
        bonus_owner=bonus_owner,
        pending_patches=pending_patches,
        pending_owner=pending_owner,
        tie_player=tie_player,
        edition=edition,
        patch_markers=patch_markers,
    )
    validate_state(state)
    return state

# =========================
# Rules
# =========================
def button_marks_crossed(prev_pos: int, new_pos: int) -> int:
    return sum(1 for m in BUTTONS_AFTER if prev_pos < m <= new_pos)

def patches_crossed(prev_pos: int, new_pos: int, opponent_pos: int, patch_markers: Tuple[int, ...]) -> int:
    return sum(1 for m in patch_markers if prev_pos < m <= new_pos and opponent_pos < m)

def legal_actions_raw(state: GameState) -> List[Tuple]:
    """Raw legal actions for current player: pass, patch idx, or buy placement."""
    if terminal(state):
        return []
    pl = current_player(state)
    player = state.player(pl)

    if state.pending_patches > 0:
        return [("patch", idx) for idx in empty_indices(player.occ)]

    acts: List[Tuple] = [("pass",)] if player.pos < SCOREBOARD_LENGTH else []
    n = len(state.circle)
    if n == 0:
        return acts

    max_pick = min(3, n)
    for offset in range(1, max_pick + 1):
        idx = (state.neutral + offset) % n
        piece_id = state.circle[idx]
        geom = PIECE_GEOMS[piece_id]
        if player.buttons < geom.cost_buttons:
            continue

        for orient_idx, orient in enumerate(geom.orientations):
            for plc in geom.placements[orient_idx]:
                if plc.mask & player.occ:
                    continue
                acts.append(("buy", offset, piece_id, orient_idx, plc.top, plc.left))
    return acts

def apply_action(state: GameState, action: Tuple) -> GameState:
    if terminal(state):
        return state

    pl = current_player(state)
    player = state.player(pl)
    opp = state.player(1 - pl)

    def maybe_claim_bonus(new_occ: int, bonus_owner: int) -> int:
        if bonus_owner != -1:
            return bonus_owner
        return pl if has_seven_by_seven(new_occ) else bonus_owner

    t = action[0]

    if t == "patch":
        if state.pending_patches <= 0:
            raise ValueError("No pending patches to place")
        if state.pending_owner != pl:
            raise ValueError("Not the pending patch owner")

        idx = int(action[1])
        if not (0 <= idx < BOARD_CELLS):
            raise ValueError(f"Patch idx={idx} out of range [0, {BOARD_CELLS})")
        if (player.occ >> idx) & 1:
            raise ValueError("Patch on occupied square")

        new_occ = (player.occ | (1 << idx)) & BOARD_MASK
        new_player = PlayerState(pos=player.pos, buttons=player.buttons, occ=new_occ, income=player.income)
        new_bonus_owner = maybe_claim_bonus(new_occ, state.bonus_owner)

        new_pending = state.pending_patches - 1
        if new_pending < 0:
            raise ValueError("pending_patches would become negative")
        new_pending_owner = state.pending_owner if new_pending > 0 else -1

        return GameState(
            p0=new_player if pl == 0 else state.p0,
            p1=new_player if pl == 1 else state.p1,
            circle=state.circle,
            neutral=state.neutral,
            bonus_owner=new_bonus_owner,
            pending_patches=new_pending,
            pending_owner=new_pending_owner,
            tie_player=state.tie_player,
            edition=state.edition,
            patch_markers=state.patch_markers,
        )

    if state.pending_patches > 0:
        raise ValueError("Must place patches first")

    if t == "pass":
        prev = player.pos
        new_pos = clamp_pos(opp.pos + 1)
        steps = new_pos - prev

        new_buttons = player.buttons + steps
        incomes = button_marks_crossed(prev, new_pos)
        if incomes:
            new_buttons += incomes * player.income

        gained_p = patches_crossed(prev, new_pos, opp.pos, state.patch_markers)

        tie_player = state.tie_player
        if new_pos == opp.pos:
            tie_player = pl

        new_player = PlayerState(pos=new_pos, buttons=new_buttons, occ=player.occ, income=player.income)
        return GameState(
            p0=new_player if pl == 0 else state.p0,
            p1=new_player if pl == 1 else state.p1,
            circle=state.circle,
            neutral=state.neutral,
            bonus_owner=state.bonus_owner,
            pending_patches=gained_p,
            pending_owner=pl if gained_p > 0 else -1,
            tie_player=tie_player,
            edition=state.edition,
            patch_markers=state.patch_markers,
        )

    if t == "buy":
        _, offset, piece_id, orient_idx, top, left = action
        n = len(state.circle)
        if n == 0:
            raise ValueError("No pieces left to buy")

        max_pick = min(3, n)
        if not (1 <= int(offset) <= max_pick):
            raise ValueError(f"Buy offset={offset} out of range [1, {max_pick}]")

        idx = (state.neutral + int(offset)) % n
        if state.circle[idx] != piece_id:
            raise ValueError("Buy mismatch")

        if piece_id not in PIECE_GEOMS:
            raise ValueError(f"Unknown piece_id={piece_id}")

        geom = PIECE_GEOMS[piece_id]
        if player.buttons < geom.cost_buttons:
            raise ValueError("Not enough buttons")

        if not (0 <= int(orient_idx) < len(geom.orientations)):
            raise ValueError(f"orient_idx={orient_idx} out of range")

        orient = geom.orientations[int(orient_idx)]

        top = int(top)
        left = int(left)
        if not (0 <= top <= BOARD_SIZE - orient.h):
            raise ValueError("top out of range for orientation")
        if not (0 <= left <= BOARD_SIZE - orient.w):
            raise ValueError("left out of range for orientation")

        delta = top * BOARD_SIZE + left
        mask = 0
        for off in orient.offsets:
            mask |= 1 << (off + delta)

        if mask & ~BOARD_MASK:
            raise ValueError("Placement extends outside the 9x9 board")
        if mask & player.occ:
            raise ValueError("Overlap")

        occ2 = (player.occ | mask) & BOARD_MASK
        income2 = player.income + orient.income

        circle_list = list(state.circle)
        circle_list.pop(idx)
        new_neutral = (idx - 1) % len(circle_list) if circle_list else 0

        prev = player.pos
        new_pos = clamp_pos(prev + geom.cost_time)

        new_buttons = player.buttons - geom.cost_buttons
        incomes = button_marks_crossed(prev, new_pos)
        if incomes:
            new_buttons += incomes * income2

        gained_p = patches_crossed(prev, new_pos, opp.pos, state.patch_markers)

        new_bonus_owner = state.bonus_owner
        if new_bonus_owner == -1 and has_seven_by_seven(occ2):
            new_bonus_owner = pl

        tie_player = state.tie_player
        if new_pos == opp.pos:
            tie_player = pl

        new_player = PlayerState(pos=new_pos, buttons=new_buttons, occ=occ2, income=income2)
        return GameState(
            p0=new_player if pl == 0 else state.p0,
            p1=new_player if pl == 1 else state.p1,
            circle=tuple(circle_list),
            neutral=new_neutral,
            bonus_owner=new_bonus_owner,
            pending_patches=gained_p,
            pending_owner=pl if gained_p > 0 else -1,
            tie_player=tie_player,
            edition=state.edition,
            patch_markers=state.patch_markers,
        )

    raise ValueError(f"Unknown action {action}")

# =========================
# Action <-> UI object
# =========================
def _buy_cells(piece_id: int, orient_idx: int, top: int, left: int) -> List[dict]:
    geom = PIECE_GEOMS[piece_id]
    orient = geom.orientations[orient_idx]
    delta = top * BOARD_SIZE + left
    income_set = set((off + delta) for off in orient.income_offsets)
    cells = []
    for off in orient.offsets:
        idx = off + delta
        r, c = divmod(idx, BOARD_SIZE)
        val = 2 if idx in income_set else 1
        cells.append({"r": r, "c": c, "val": val})
    return cells

def action_to_obj(a: Tuple) -> dict:
    t = a[0]
    if t == "pass":
        return {"type": "pass"}
    if t == "patch":
        idx = int(a[1])
        r, c = divmod(idx, BOARD_SIZE)
        return {"type": "patch", "idx": idx, "row": r, "col": c, "cells": [{"r": r, "c": c, "val": 1}]}
    if t == "buy":
        _, offset, pid, oi, top, left = a
        pid = int(pid); oi = int(oi); top = int(top); left = int(left)
        p = PIECE_BY_ID[pid]
        return {
            "type": "buy",
            "offset": int(offset),
            "piece_id": pid,
            "orient": oi,
            "top": top,
            "left": left,
            "cost_buttons": p["cost_buttons"],
            "cost_time": p["cost_time"],
            "cells": _buy_cells(pid, oi, top, left),
        }
    return {"type": "unknown", "raw": repr(a)}

def pretty_action(a: Tuple) -> str:
    t = a[0]
    if t == "pass":
        return "PASS"
    if t == "patch":
        idx = int(a[1])
        r, c = divmod(idx, BOARD_SIZE)
        return f"PLACE_PATCH ({r+1},{c+1})"
    if t == "buy":
        _, offset, pid, oi, top, left = a
        p = PIECE_BY_ID[int(pid)]
        return f"BUY piece {pid} (offset {int(offset)}) cost {p['cost_buttons']}b/{p['cost_time']}t at row {int(top)+1}, col {int(left)+1}"
    return repr(a)

# =========================
# Rollout + MCTS
# =========================
def heuristic_eval(state: GameState, root_player: int) -> float:
    # value in [-1,1] from root_player perspective
    diff = score_diff_p0_minus_p1(state)
    if root_player == 1:
        diff = -diff
    p = state.player(root_player)
    o = state.player(1 - root_player)
    raw = diff + 0.6 * (p.income - o.income) + 0.04 * (o.pos - p.pos)
    return math.tanh(raw / 15.0)

def rollout(state: GameState, rng: random.Random, root_player: int, max_steps: int = 800) -> Tuple[float, int]:
    s = state
    for _ in range(max_steps):
        if terminal(s):
            # Terminal scoring per Patchwork rules.
            d0 = score_diff_p0_minus_p1(s)

            # Tie-breaker: if scores are tied, the player who reached the final space first wins.
            # At terminal both tokens are on the final space; the token on top (state.tie_player) arrived last.
            if d0 == 0 and s.p0.pos >= SCOREBOARD_LENGTH and s.p1.pos >= SCOREBOARD_LENGTH:
                first_finisher = 1 - s.tie_player
                d0 = 1 if first_finisher == 0 else -1

            d = d0 if root_player == 0 else -d0
            return math.tanh(d / 15.0), d

        acts = legal_actions_raw(s)
        if not acts:
            return heuristic_eval(s, root_player), 0

        # light bias away from PASS if buys exist
        if acts and acts[0][0] == "pass" and len(acts) > 1 and rng.random() < 0.85:
            a = rng.choice(acts[1:min(len(acts), 25)])
        else:
            a = rng.choice(acts[: min(len(acts), 25)])

        s = apply_action(s, a)

    d = score_diff_p0_minus_p1(s)
    if root_player == 1:
        d = -d
    return heuristic_eval(s, root_player), d

# Sentinel to distinguish 'not initialized' from 'initialized but empty'
_UNTRIED_NOT_INIT = object()

@dataclass
class Node:
    state: GameState
    parent: Optional["Node"]
    action_from_parent: Optional[Tuple]
    player_to_move: int
    root_player: int
    visits: int = 0
    value_sum: float = 0.0
    win_sum: float = 0.0
    score_sum: float = 0.0
    children: Dict[Tuple, "Node"] = None
    untried: object = None  # _UNTRIED_NOT_INIT or List[Tuple]

    def __post_init__(self):
        self.children = {}
        self.untried = _UNTRIED_NOT_INIT

class MCTS:
    def __init__(self, root_state: GameState, exploration: float = 1.4, seed: int = 0):
        self.root_player = current_player(root_state)
        self.root = Node(
            state=root_state,
            parent=None,
            action_from_parent=None,
            player_to_move=current_player(root_state),
            root_player=self.root_player,
        )
        self.c = exploration
        self.rng = random.Random(seed)

    def _init_untried(self, node: Node) -> None:
        if node.untried is not _UNTRIED_NOT_INIT:
            return
        node.untried = legal_actions_raw(node.state)
        self.rng.shuffle(node.untried)

    def _uct_score(self, parent: Node, child: Node) -> float:
        if child.visits == 0:
            return float("inf") if parent.player_to_move == parent.root_player else float("-inf")
        mean = child.value_sum / child.visits
        expl = self.c * math.sqrt(math.log(parent.visits + 1) / child.visits)
        return mean + expl if parent.player_to_move == parent.root_player else mean - expl

    def _select_child(self, node: Node) -> Node:
        best = None
        best_score = None
        for ch in node.children.values():
            s = self._uct_score(node, ch)
            if best is None:
                best, best_score = ch, s
            else:
                if node.player_to_move == node.root_player:
                    if s > best_score:
                        best, best_score = ch, s
                else:
                    if s < best_score:
                        best, best_score = ch, s
        return best

    def iterate(self) -> None:
        node = self.root
        path = [node]

        while True:
            if terminal(node.state):
                break
            self._init_untried(node)
            if node.untried:
                break
            if not node.children:
                break
            node = self._select_child(node)
            path.append(node)

        if not terminal(node.state):
            self._init_untried(node)
            if node.untried:
                a = node.untried.pop()
                s2 = apply_action(node.state, a)
                child = Node(
                    state=s2,
                    parent=node,
                    action_from_parent=a,
                    player_to_move=current_player(s2),
                    root_player=self.root_player,
                )
                node.children[a] = child
                node = child
                path.append(node)

        v, d = rollout(node.state, self.rng, self.root_player)
        win = 1.0 if d > 0 else 0.0 if d < 0 else 0.5

        for n in path:
            n.visits += 1
            n.value_sum += v
            n.win_sum += win
            n.score_sum += float(d)

    def search(self, iterations: int) -> None:
        for _ in range(iterations):
            self.iterate()

# =========================
# Parallel worker + warm pool
# =========================
def _mcts_worker(state_dict: dict, iterations: int, seed: int, exploration: float):
    s = state_from_dict(state_dict)
    m = MCTS(s, exploration=exploration, seed=seed)
    m.search(iterations)
    out = []
    for a, node in m.root.children.items():
        out.append((a, node.visits, node.win_sum, node.score_sum, node.value_sum))
    return out

_EXECUTOR: Optional[ProcessPoolExecutor] = None
_EXECUTOR_MAX = max(1, os.cpu_count() or 1)

def _ensure_executor():
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = ProcessPoolExecutor(max_workers=_EXECUTOR_MAX)

def solve_parallel(state_dict: dict, iterations: int, seed: int, exploration: float, workers: int) -> dict:
    state = state_from_dict(state_dict)

    if terminal(state):
        return {
            "terminal": True,
            "to_move": current_player(state),
            "score_p0": final_score(state.p0, state.bonus_owner, 0),
            "score_p1": final_score(state.p1, state.bonus_owner, 1),
        }

    workers = max(1, min(int(workers), _EXECUTOR_MAX))
    iterations = max(500, min(int(iterations), 5_000_000))

    it_each = max(1, iterations // workers)
    extra = iterations - it_each * workers

    # Always use the process pool so /solve doesn't block the event loop thread and so
    # concurrent requests can utilize multiple CPU cores even when workers=1.
    _ensure_executor()
    futs = []
    for w in range(workers):
        its = it_each + (1 if w < extra else 0)
        futs.append(_EXECUTOR.submit(_mcts_worker, state_dict, its, seed + 10007 * w, exploration))

    all_results = [f.result() for f in futs]

    agg: Dict[Tuple, List[float]] = {}
    for results in all_results:
        for a, v, ws, ss, vs in results:
            if a not in agg:
                agg[a] = [0.0, 0.0, 0.0, 0.0]  # visits, win_sum, score_sum, value_sum
            agg[a][0] += float(v)
            agg[a][1] += float(ws)
            agg[a][2] += float(ss)
            agg[a][3] += float(vs)

    if not agg:
        acts = legal_actions_raw(state)
        a = acts[0] if acts else ("pass",)
        return {
            "terminal": False,
            "to_move": current_player(state),
            "total_sims": iterations,
            "best": {"pretty": pretty_action(a), "action": action_to_obj(a), "winProb": 0.5, "scoreDiff": 0.0, "visits": 0},
            "top": [],
        }

    items = []
    for a, (vis, win_sum, score_sum, _value_sum) in agg.items():
        vis_d = max(1.0, vis)
        items.append((a, int(vis), float(win_sum) / vis_d, float(score_sum) / vis_d))
    # IMPORTANT: choose best by winProb, then scoreDiff, then visits
    items.sort(key=lambda x: (x[2], x[3], x[1]), reverse=True)

    best = items[0]
    top_list = items[: min(10, len(items))]

    return {
        "terminal": False,
        "to_move": current_player(state),
        "total_sims": iterations,
        "best": {
            "pretty": pretty_action(best[0]),
            "action": action_to_obj(best[0]),
            "winProb": best[2],
            "scoreDiff": best[3],
            "visits": best[1],
        },
        "top": [
            {"pretty": pretty_action(a), "action": action_to_obj(a), "winProb": winp, "scoreDiff": scored, "visits": vis}
            for (a, vis, winp, scored) in top_list
        ],
    }

# =========================
# API helpers
# =========================
def state_to_dict(state: GameState) -> dict:
    # Export occupied as "1"; UI stamps income squares (2) from action.cells when applying.
    def occ_to_rows(occ: int) -> List[str]:
        occ = occ & BOARD_MASK
        rows = []
        for r in range(BOARD_SIZE):
            s = []
            for c in range(BOARD_SIZE):
                idx = rc_to_i(r, c)
                s.append("1" if ((occ >> idx) & 1) else ".")
            rows.append("".join(s))
        return rows

    return {
        "edition": state.edition,
        "players": [
            {
                "position": int(state.p0.pos),
                "buttons": int(state.p0.buttons),
                "board": occ_to_rows(state.p0.occ),
                "income": int(state.p0.income),
            },
            {
                "position": int(state.p1.pos),
                "buttons": int(state.p1.buttons),
                "board": occ_to_rows(state.p1.occ),
                "income": int(state.p1.income),
            },
        ],
        "circle": list(state.circle),
        "neutral": int(state.neutral),
        "bonus_owner": int(state.bonus_owner),
        "pending_patches": int(state.pending_patches),
        "pending_owner": int(state.pending_owner),
        "tie_player": int(state.tie_player),
    }

def legal_moves_for_ui(state: GameState) -> dict:
    if terminal(state):
        return {"terminal": True, "to_move": current_player(state), "mode": "terminal", "actions": []}

    pl = current_player(state)
    acts = legal_actions_raw(state)

    # For UI friendliness, group buys by (offset, piece_id) and include placements
    if state.pending_patches > 0:
        # patch placement required
        empties = []
        for a in acts:
            obj = action_to_obj(a)
            empties.append(obj)
        return {
            "terminal": False,
            "to_move": pl,
            "mode": "patch",
            "pending_patches": int(state.pending_patches),
            "actions": empties,  # each is a patch action with idx,row,col,cells
        }

    pass_allowed = any(a[0] == "pass" for a in acts)
    buy_map: Dict[Tuple[int, int], List[dict]] = {}
    for a in acts:
        if a[0] == "buy":
            obj = action_to_obj(a)
            key = (obj["offset"], obj["piece_id"])
            buy_map.setdefault(key, []).append(obj)

    buy_groups = []
    for (offset, pid), placements in buy_map.items():
        p = PIECE_BY_ID[pid]
        buy_groups.append({
            "offset": offset,
            "piece_id": pid,
            "cost_buttons": p["cost_buttons"],
            "cost_time": p["cost_time"],
            "placements": placements,  # each is a full buy action obj (orient/top/left/cells)
        })

    buy_groups.sort(key=lambda g: g["offset"])

    return {
        "terminal": False,
        "to_move": pl,
        "mode": "normal",
        "pass_allowed": pass_allowed,
        "buy_groups": buy_groups,
    }

# =========================
# FastAPI app
# =========================

@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Startup: warm pool so repeated solves are MUCH faster on Windows
    _ensure_executor()
    yield
    # Shutdown
    global _EXECUTOR
    if _EXECUTOR is not None:
        _EXECUTOR.shutdown(wait=False, cancel_futures=True)
        _EXECUTOR = None


app = FastAPI(title="Patchwork Solver API", version="2.1", lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {
        "ok": True,
        "executor_max": _EXECUTOR_MAX,
        "default_edition": DEFAULT_EDITION,
        "editions": sorted(EDITION_PATCHES.keys()),
        "nn_loaded": _NN_AGENT is not None,
    }


@app.get("/nn/status")
def nn_status():
    return {
        "nn_loaded": _NN_AGENT is not None,
        "model_path": _NN_MODEL_PATH,
        "config_path": _NN_CONFIG_PATH,
        "device": _NN_DEVICE,
        "simulations": _NN_SIMULATIONS,
        "checkpoints": _discover_checkpoints(),
    }


@app.post("/nn/load")
async def nn_load(request: Request):
    """Load a checkpoint at runtime without restarting API."""
    global _NN_SIMULATIONS
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    model_path = str(payload.get("model_path", "")).strip()
    if not model_path:
        return JSONResponse({"error": "model_path is required"}, status_code=400)

    config_path = str(payload.get("config_path", "configs/config_overnight.yaml")).strip()
    device = str(payload.get("device", _NN_DEVICE)).strip() or _NN_DEVICE
    simulations = int(payload.get("simulations", _NN_SIMULATIONS))
    simulations = max(50, min(simulations, 20000))

    try:
        _load_nn_agent(model_path=model_path, config_path=config_path, device=device)
        _NN_SIMULATIONS = simulations
        return {
            "ok": True,
            "nn_loaded": True,
            "model_path": _NN_MODEL_PATH,
            "config_path": _NN_CONFIG_PATH,
            "device": _NN_DEVICE,
            "simulations": _NN_SIMULATIONS,
        }
    except Exception as e:
        logger.exception("Failed to load NN model")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/nn/unload")
def nn_unload():
    _unload_nn_agent()
    return {
        "ok": True,
        "nn_loaded": False,
    }

@app.get("/new")
def new_game(
    seed: Optional[int] = None,
    starting_buttons: int = 5,
    starting_player: int = 0,
    edition: str = DEFAULT_EDITION,
):
    """
    Create a fresh game state with a randomized circle.

    Per Patchwork rules, the neutral pawn starts immediately clockwise of piece id 32 (the 2x1 tile).

    starting_player is caller-chosen (rulebook: "the player who last used a needle begins").
    edition selects special (1x1) patch marker positions.
    """
    if edition not in EDITION_PATCHES:
        return JSONResponse(
            {"error": f"Unknown edition {edition!r}. Valid: {sorted(EDITION_PATCHES.keys())}"},
            status_code=400,
        )
    if starting_player not in (0, 1):
        starting_player = 0

    circle, neutral = _make_shuffled_circle(seed=seed)
    st = GameState(
        p0=PlayerState(pos=0, buttons=int(starting_buttons), occ=0, income=0),
        p1=PlayerState(pos=0, buttons=int(starting_buttons), occ=0, income=0),
        circle=circle,
        neutral=neutral,
        bonus_owner=-1,
        pending_patches=0,
        pending_owner=-1,
        tie_player=int(starting_player),
        edition=str(edition),
        patch_markers=tuple(EDITION_PATCHES[str(edition)]),
    )
    if _NN_AGENT is not None:
        _NN_AGENT.mcts.clear_tree()
    return {"state": state_to_dict(st)}

@app.get("/pieces")
def pieces():
    out = []
    for p in PIECES:
        inc = sum(1 for row in p["shape"] for v in row if v == 2)
        size = sum(1 for row in p["shape"] for v in row if v != 0)
        out.append({**p, "income": inc, "size": size})
    return {"count": len(out), "pieces": out}

@app.post("/legal")
async def legal(request: Request):
    try:
        payload = await request.json()
        state_dict = payload["state"] if isinstance(payload, dict) and "state" in payload else payload
        st = state_from_dict(state_dict)
        return legal_moves_for_ui(st)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

@app.post("/apply")
async def apply(request: Request):
    try:
        payload = await request.json()
        state_dict = payload["state"]
        action_obj = payload["action"]
    except Exception:
        return JSONResponse({"error": "Expected JSON with {state, action}."}, status_code=400)

    try:
        st = state_from_dict(state_dict)

        t = action_obj.get("type")
        if t == "pass":
            a = ("pass",)
        elif t == "patch":
            idx = int(action_obj["idx"])
            a = ("patch", idx)
        elif t == "buy":
            a = ("buy",
                 int(action_obj["offset"]),
                 int(action_obj["piece_id"]),
                 int(action_obj["orient"]),
                 int(action_obj["top"]),
                 int(action_obj["left"]))
        else:
            return JSONResponse({"error": f"Unknown action type: {t}"}, status_code=400)

        st2 = apply_action(st, a)
        return {
            "terminal": terminal(st2),
            "to_move": current_player(st2),
            "state": state_to_dict(st2),
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

@app.post("/solve")
async def solve(request: Request):
    """
    Human-vs-AI: by default, ONLY recommends for Player 0.
    If it's Player 1's turn, returns needs_opponent_move=True.
    """
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    state_dict = payload["state"] if isinstance(payload, dict) and "state" in payload else payload
    iterations = int(payload.get("iterations", 80000)) if isinstance(payload, dict) else 80000
    seed = int(payload.get("seed", 0)) if isinstance(payload, dict) else 0
    exploration = float(payload.get("exploration", 1.4)) if isinstance(payload, dict) else 1.4
    workers = int(payload.get("workers", 1)) if isinstance(payload, dict) else 1
    only_player0 = bool(payload.get("only_player0", True)) if isinstance(payload, dict) else True

    try:
        st = state_from_dict(state_dict)
        to_move = current_player(st)

        if only_player0 and to_move != 0:
            # Don't "play" for the opponent; tell UI to ask user for opponent move instead.
            return {
                "terminal": terminal(st),
                "to_move": to_move,
                "needs_opponent_move": True,
                "message": "Opponent turn. Use /legal to select Player 1 move, then /apply, then solve again."
            }

        return await run_in_threadpool(solve_parallel, state_dict, iterations, seed, exploration, workers)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# =========================
# AlphaZero NN solve
# =========================

def _load_nn_agent(model_path: str, config_path: str, device: str = "cuda"):
    """Load the AlphaZero model for NN-powered solving."""
    global _NN_AGENT, _NN_DEVICE, _NN_MODEL_PATH, _NN_CONFIG_PATH

    # Add project root to path so imports work when running from GUI/
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from inference import PatchworkAgent
    _NN_DEVICE = device
    _NN_AGENT = PatchworkAgent(model_path, config_path, device)
    _NN_MODEL_PATH = model_path
    _NN_CONFIG_PATH = config_path
    logger.info(f"AlphaZero model loaded from {model_path}")


def _unload_nn_agent():
    """Unload currently loaded AlphaZero model."""
    global _NN_AGENT, _NN_MODEL_PATH, _NN_CONFIG_PATH
    _NN_AGENT = None
    _NN_MODEL_PATH = None
    _NN_CONFIG_PATH = None


def _discover_checkpoints(limit: int = 200) -> List[str]:
    """Return recent checkpoint candidates for UI convenience."""
    root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    candidates: List[Path] = []
    for rel in ("checkpoints", "models"):
        d = root / rel
        if d.exists() and d.is_dir():
            candidates.extend(d.glob("*.pt"))
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p.resolve()) for p in candidates[:limit]]


def _engine_action_to_api_tuple(action) -> Tuple:
    """Convert AlphaZero engine action to API action tuple.

    Engine format:  (AT_PASS=0, 0, 0, 0, 0, 0)
                    (AT_PATCH=1, idx, 0, 0, 0, 0)
                    (AT_BUY=2, offset, piece_id, orient, top, left)
    API format:     ("pass",) / ("patch", idx) / ("buy", offset, pid, orient, top, left)
    """
    atype = int(action[0])
    if atype == 0:  # AT_PASS
        return ("pass",)
    if atype == 1:  # AT_PATCH
        return ("patch", int(action[1]))
    if atype == 2:  # AT_BUY
        return ("buy", int(action[1]), int(action[2]), int(action[3]), int(action[4]), int(action[5]))
    raise ValueError(f"Unknown engine action type: {atype}")


def _solve_nn(state_dict: dict, simulations: int, temperature: float) -> dict:
    """Run AlphaZero MCTS on the given state and return results in /solve format."""
    from src.game.patchwork_engine import (
        state_from_dict as engine_state_from_dict,
        current_player as engine_current_player,
        terminal as engine_terminal,
    )

    engine_state = engine_state_from_dict(state_dict)
    to_move = int(engine_current_player(engine_state))

    if engine_terminal(engine_state):
        return {"terminal": True, "to_move": to_move}

    # Override simulations for this search
    old_sims = _NN_AGENT.mcts.config.simulations
    _NN_AGENT.mcts.config.simulations = simulations

    visit_counts, search_time, root_q = _NN_AGENT.mcts.search(
        engine_state, to_move, add_noise=False
    )

    _NN_AGENT.mcts.config.simulations = old_sims

    if not visit_counts:
        return {"terminal": False, "to_move": to_move, "total_sims": simulations,
                "best": {"pretty": "PASS", "action": {"type": "pass"}, "winProb": 0.5, "visits": 0}, "top": []}

    total_visits = sum(visit_counts.values())
    # root_q is backed-up value from tree search ([-1,1] from to_move's perspective)
    # Convert to win prob for to_move: (root_q + 1) / 2; frontend converts to P0 for display
    root_q_val = float(root_q) if root_q is not None else 0.0
    win_prob = max(0.0, min(1.0, (root_q_val + 1.0) / 2.0))

    items = []
    for engine_action, visits in visit_counts.items():
        api_tuple = _engine_action_to_api_tuple(engine_action)
        items.append((api_tuple, visits, visits / max(1, total_visits)))

    items.sort(key=lambda x: x[1], reverse=True)

    selected_engine_action = _NN_AGENT.mcts.select_action(visit_counts, temperature=temperature)
    _NN_AGENT.mcts.advance_tree(selected_engine_action)  # GUI permanent brain: reuse subtree
    selected_api_tuple = _engine_action_to_api_tuple(selected_engine_action)

    top_list = items[:min(10, len(items))]

    return {
        "terminal": False,
        "to_move": to_move,
        "total_sims": simulations,
        "search_time": round(search_time, 3),
        "best": {
            "pretty": pretty_action(selected_api_tuple),
            "action": action_to_obj(selected_api_tuple),
            "winProb": win_prob,
            "visits": items[0][1],
        },
        "top": [
            {
                "pretty": pretty_action(a),
                "action": action_to_obj(a),
                "winProb": win_prob,
                "visits": vis,
            }
            for (a, vis, _) in top_list
        ],
    }


@app.post("/solve_nn")
async def solve_nn(request: Request):
    """Solve using AlphaZero neural network + MCTS."""
    if _NN_AGENT is None:
        return JSONResponse(
            {"error": "No AlphaZero model loaded. Start server with --model <path>."},
            status_code=503,
        )

    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    state_dict = payload["state"] if isinstance(payload, dict) and "state" in payload else payload
    simulations = int(payload.get("simulations", _NN_SIMULATIONS)) if isinstance(payload, dict) else _NN_SIMULATIONS
    temperature = float(payload.get("temperature", 0.0)) if isinstance(payload, dict) else 0.0

    try:
        return await run_in_threadpool(_solve_nn, state_dict, simulations, temperature)
    except Exception as e:
        logger.exception("solve_nn failed")
        return JSONResponse({"error": str(e)}, status_code=500)


# =========================
# CLI entry point
# =========================
if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Patchwork API Server")
    parser.add_argument("--model", type=str, default=None, help="Path to AlphaZero checkpoint (enables /solve_nn)")
    parser.add_argument("--config", type=str, default="configs/config_best.yaml", help="Config for NN architecture")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--simulations", type=int, default=800, help="Default MCTS simulations for /solve_nn")

    args = parser.parse_args()

    if args.model:
        _NN_SIMULATIONS = args.simulations
        _load_nn_agent(args.model, args.config, args.device)

    uvicorn.run(app, host=args.host, port=args.port)
