"""Unit tests for win-first MCTS: root filter, delta widening, DSU gate, sampling over candidates."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.mcts.alphazero_mcts_optimized import (
    MCTSNode,
    MCTSConfig,
    WinFirstConfig,
    OptimizedAlphaZeroMCTS,
    _parse_and_validate_win_first,
)


def _make_mcts(win_first: WinFirstConfig) -> OptimizedAlphaZeroMCTS:
    """Minimal MCTS instance for testing win-first logic (no network, no encoders)."""
    config = MCTSConfig(win_first=win_first)
    return OptimizedAlphaZeroMCTS(
        network=None,
        config=config,
        device=torch.device("cpu"),
        state_encoder=None,
        action_encoder=None,
        full_config={},
    )


def _make_root_with_children(
    qv_list: list,
    qs_list: list,
    nv_list: list,
) -> MCTSNode:
    """Build a root node with legal_actions and arrays set so Qv=qv_list, Qs=qs_list, Nv=nv_list."""
    n = len(qv_list)
    assert n == len(qs_list) == len(nv_list)
    root = MCTSNode(state=None, to_move=0)
    root.legal_actions = [(i,) for i in range(n)]
    root._init_arrays()
    root._visit_count[:] = nv_list
    root._value_sum[:] = [qv_list[i] * nv_list[i] for i in range(n)]
    root._score_sum[:] = [qs_list[i] * nv_list[i] for i in range(n)]
    root._total_value[:] = [qv_list[i] * nv_list[i] for i in range(n)]  # dummy utility
    root.n_total = int(sum(nv_list))
    return root


# A) Root filter excludes value drop even if score high
def test_root_filter_excludes_value_drop_even_if_score_high():
    # Use win_start=0.85 so best_Qv=0.80 is below start => delta=value_delta_min=0.02 => only A in candidate set.
    wf = WinFirstConfig(
        enabled=True,
        value_delta_min=0.02,
        value_delta_max=0.06,
        value_delta_win_start=0.85,
        value_delta_win_full=0.95,
    )
    mcts = _make_mcts(wf)
    # A: Qv=0.80, Qs=0.10, Nv=10; B: Qv=0.77, Qs=0.90, Nv=100
    root = _make_root_with_children(
        qv_list=[0.80, 0.77],
        qs_list=[0.10, 0.90],
        nv_list=[10, 100],
    )
    mcts._root = root
    action = mcts._select_action_win_first(temperature=0.0, deterministic=True)
    # best_Qv=0.80 < 0.85 => delta=0.02 => candidates Qv >= 0.78 => only A (0.80). Must pick A.
    assert action == (0,)


# B) Delta widens when secured
def test_delta_widens_when_secured():
    wf = WinFirstConfig(
        enabled=True,
        value_delta_min=0.02,
        value_delta_max=0.10,
        value_delta_win_start=0.60,
        value_delta_win_full=0.90,
    )
    mcts = _make_mcts(wf)
    # We test the logic by checking candidate set width. best_Qv=0.50 => delta=min=0.02
    root_lo = _make_root_with_children([0.50], [0.0], [10])
    mcts._root = root_lo
    mcts._select_action_win_first(temperature=0.0, deterministic=True)
    # With one child, delta doesn't matter for selection. Test via two children.
    root_mid = _make_root_with_children([0.75, 0.70], [0.2, 0.5], [10, 10])
    mcts._root = root_mid
    # best_Qv=0.75, between start and full => delta in (0.02, 0.10). Both in candidate set if delta >= 0.05.
    action = mcts._select_action_win_first(temperature=0.0, deterministic=True)
    assert action in ((0,), (1,))
    root_hi = _make_root_with_children([0.95, 0.90], [0.1, 0.9], [10, 10])
    mcts._root = root_hi
    # best_Qv=0.95 >= 0.90 => delta=max=0.10 => both candidates; tiebreak picks higher Qs => (1,)
    action_hi = mcts._select_action_win_first(temperature=0.0, deterministic=True)
    assert action_hi == (1,)


# C) DSU gate off when unclear
def test_dsu_gate_off_when_unclear():
    wf = WinFirstConfig(
        enabled=True,
        gate_dsu_enabled=True,
        gate_dsu_win_start=0.65,
        gate_dsu_win_full=0.90,
        gate_dsu_loss_start=-0.90,
        gate_dsu_loss_full=-0.98,
    )
    mcts = _make_mcts(wf)
    mcts._search_root_value = 0.2
    g = mcts._compute_dsu_gate()
    assert g == 0.0


# D) DSU gate on when winning
def test_dsu_gate_on_when_winning():
    wf = WinFirstConfig(
        enabled=True,
        gate_dsu_enabled=True,
        gate_dsu_win_start=0.65,
        gate_dsu_win_full=0.90,
        gate_dsu_power=2.0,
    )
    mcts = _make_mcts(wf)
    mcts._search_root_value = 0.92
    g = mcts._compute_dsu_gate()
    assert g >= 0.95


# E) DSU gate on only when forced losing
def test_dsu_gate_on_only_when_forced_losing():
    wf = WinFirstConfig(
        enabled=True,
        gate_dsu_enabled=True,
        gate_dsu_win_start=0.65,
        gate_dsu_win_full=0.90,
        gate_dsu_loss_start=-0.90,
        gate_dsu_loss_full=-0.98,
    )
    mcts = _make_mcts(wf)
    mcts._search_root_value = -0.50
    g_fight = mcts._compute_dsu_gate()
    assert g_fight == 0.0
    mcts._search_root_value = -0.97
    g_defend = mcts._compute_dsu_gate()
    assert g_defend > 0


# F) Sampling filters candidates
def test_sampling_filters_candidates():
    wf = WinFirstConfig(
        enabled=True,
        value_delta_min=0.02,
        value_delta_max=0.06,
        value_delta_win_start=0.60,
        value_delta_win_full=0.90,
        filter_before_sampling=True,
    )
    mcts = _make_mcts(wf)
    # Root with 3 actions: A Qv=0.85, B Qv=0.82, C Qv=0.50. delta=0.02 => C excluded.
    root = _make_root_with_children(
        qv_list=[0.85, 0.82, 0.50],
        qs_list=[0.1, 0.2, 0.9],
        nv_list=[100, 50, 30],
    )
    mcts._root = root
    # Sample many times; we should never get action (2,) because it's outside candidate set.
    chosen = []
    for _ in range(100):
        a = mcts._select_action_win_first(temperature=1.0, deterministic=False)
        chosen.append(a)
    assert all(a in ((0,), (1,)) for a in chosen)


# Zero-visit children: never divide by zero, never select a zero-visit child when others have visits
def test_win_first_ignores_zero_visit_children():
    wf = WinFirstConfig(
        enabled=True,
        value_delta_min=0.02,
        value_delta_max=0.06,
        value_delta_win_start=0.85,
        value_delta_win_full=0.95,
    )
    mcts = _make_mcts(wf)
    # Root with 3 children: A Nv=10 Qv=0.8, B Nv=0, C Nv=5 Qv=0.7. Only A and C have visits.
    root = _make_root_with_children(
        qv_list=[0.80, 0.0, 0.70],   # B's Qv unused (Nv=0)
        qs_list=[0.1, 0.0, 0.2],
        nv_list=[10, 0, 5],
    )
    mcts._root = root
    # Must not divide by zero; must select A or C, never B (index 1)
    for _ in range(30):
        action = mcts._select_action_win_first(temperature=0.0, deterministic=True)
        assert action in ((0,), (2,)), "must not select zero-visit child (1,)"
    for _ in range(30):
        action = mcts._select_action_win_first(temperature=1.0, deterministic=False)
        assert action in ((0,), (2,)), "must not select zero-visit child when sampling"

    # All visits zero: fallback to first action, no div-by-zero
    root_all_zero = _make_root_with_children(
        qv_list=[0.0, 0.0],
        qs_list=[0.0, 0.0],
        nv_list=[0, 0],
    )
    mcts._root = root_all_zero
    action = mcts._select_action_win_first(temperature=0.0, deterministic=True)
    assert action in ((0,), (1,))


# Parse and validation
def test_parse_win_first_clamps_and_enforces_start_full():
    cfg = {
        "enabled": True,
        "value_delta_min": -0.1,
        "value_delta_max": 0.5,
        "value_delta_win_start": 0.6,
        "value_delta_win_full": 0.9,
        "gate_dsu_win_start": 0.65,
        "gate_dsu_win_full": 0.90,
        "gate_dsu_loss_start": -0.90,
        "gate_dsu_loss_full": -0.98,
    }
    w = _parse_and_validate_win_first(cfg)
    assert w.value_delta_min >= 0
    assert w.value_delta_max >= w.value_delta_min
    assert w.value_delta_win_full >= w.value_delta_win_start
    assert w.gate_dsu_win_full >= w.gate_dsu_win_start
    assert w.gate_dsu_loss_full <= w.gate_dsu_loss_start  # more negative
