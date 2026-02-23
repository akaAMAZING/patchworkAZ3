"""
Unit tests for ActionEncoder.create_target_policy temperature scaling.
"""
import numpy as np
import pytest

from src.network.encoder import ActionEncoder


def _make_encoder():
    return ActionEncoder()


def test_create_target_policy_visits_mode():
    """mode=visits gives normalized counts pi(a)=N(a)/sum(N)."""
    enc = _make_encoder()
    visit_counts = {("pass",): 10, ("patch", 5): 20}
    pi = enc.create_target_policy(visit_counts, mode="visits")
    assert pi.shape == (2026,)
    assert np.isfinite(pi).all()
    assert abs(pi.sum() - 1.0) < 1e-5
    pass_idx = 0
    patch5_idx = 1 + 5
    assert abs(pi[pass_idx] - 10 / 30) < 1e-5
    assert abs(pi[patch5_idx] - 20 / 30) < 1e-5


def test_create_target_policy_temperature_proportional():
    """temperature=1.0 (legacy mode) gives distribution proportional to visit counts."""
    enc = _make_encoder()
    visit_counts = {("pass",): 10, ("patch", 5): 20}
    pi = enc.create_target_policy(visit_counts, temperature=1.0, mode="visits_temperature_shaped")
    assert pi.shape == (2026,)
    assert np.isfinite(pi).all()
    assert abs(pi.sum() - 1.0) < 1e-5
    pass_idx = 0
    patch5_idx = 1 + 5
    assert abs(pi[pass_idx] - 10 / 30) < 1e-5
    assert abs(pi[patch5_idx] - 20 / 30) < 1e-5


def test_create_target_policy_temperature_sharper():
    """temperature=0.5 produces sharper distribution than temperature=1.0 (legacy mode)."""
    enc = _make_encoder()
    visit_counts = {("pass",): 10, ("patch", 3): 20}
    pi_1 = enc.create_target_policy(visit_counts, temperature=1.0, mode="visits_temperature_shaped")
    pi_05 = enc.create_target_policy(visit_counts, temperature=0.5, mode="visits_temperature_shaped")
    pass_idx = 0
    patch3_idx = 1 + 3
    # T=0.5: count^2, so 100 vs 400 → pass=0.2, patch=0.8
    # T=1: proportional → pass=1/3, patch=2/3
    # Sharper = higher max prob
    assert pi_05[patch3_idx] > pi_1[patch3_idx]
    assert pi_05[pass_idx] < pi_1[pass_idx]
    assert np.isfinite(pi_05).all()
    assert abs(pi_05.sum() - 1.0) < 1e-5


def test_create_target_policy_temperature_flatter():
    """temperature=2.0 produces flatter distribution than temperature=1.0 (legacy mode)."""
    enc = _make_encoder()
    visit_counts = {("pass",): 10, ("patch", 7): 20}
    pi_1 = enc.create_target_policy(visit_counts, temperature=1.0, mode="visits_temperature_shaped")
    pi_2 = enc.create_target_policy(visit_counts, temperature=2.0, mode="visits_temperature_shaped")
    pass_idx = 0
    patch7_idx = 1 + 7
    # T=2: count^0.5 flattens → less extreme than proportional
    assert pi_2[patch7_idx] < pi_1[patch7_idx]
    assert pi_2[pass_idx] > pi_1[pass_idx]
    assert np.isfinite(pi_2).all()
    assert abs(pi_2.sum() - 1.0) < 1e-5


def test_create_target_policy_temperature_very_small():
    """Very small T returns finite distribution, sums to 1, preserves support, sharper than T=1.0 (legacy)."""
    enc = _make_encoder()
    visit_counts = {("pass",): 10, ("patch", 1): 20}
    pi_small = enc.create_target_policy(visit_counts, temperature=0.05, mode="visits_temperature_shaped")
    pi_1 = enc.create_target_policy(visit_counts, temperature=1.0, mode="visits_temperature_shaped")

    assert np.isfinite(pi_small).all()
    assert abs(pi_small.sum() - 1.0) < 1e-5
    # Preserve support: both actions with count>0 have positive probability
    pass_idx = 0
    patch1_idx = 1 + 1
    assert pi_small[pass_idx] > 0
    assert pi_small[patch1_idx] > 0
    # Sharper than T=1.0: higher max, lower min among support
    assert pi_small[patch1_idx] > pi_1[patch1_idx]
    assert pi_small[pass_idx] < pi_1[pass_idx]


def test_create_target_policy_temperature_1e6_finite():
    """T=1e-6 (caller clamp) returns finite distribution that sums to 1 (legacy mode)."""
    enc = _make_encoder()
    visit_counts = {("pass",): 10, ("patch", 1): 20}
    pi = enc.create_target_policy(visit_counts, temperature=1e-6, mode="visits_temperature_shaped")
    assert np.isfinite(pi).all()
    assert abs(pi.sum() - 1.0) < 1e-5


def test_create_target_policy_temperature_zero_raises():
    """temperature <= 0 raises ValueError in legacy shaped mode only."""
    enc = _make_encoder()
    visit_counts = {("pass",): 1, ("patch", 0): 1}
    with pytest.raises(ValueError, match="temperature must be positive"):
        enc.create_target_policy(visit_counts, temperature=0, mode="visits_temperature_shaped")
    with pytest.raises(ValueError, match="temperature must be positive"):
        enc.create_target_policy(visit_counts, temperature=-0.1, mode="visits_temperature_shaped")


def test_create_target_policy_visits_ignores_temperature():
    """mode=visits ignores temperature; result is same for any T."""
    enc = _make_encoder()
    visit_counts = {("pass",): 5, ("patch", 2): 15}
    pi_a = enc.create_target_policy(visit_counts, temperature=0.5, mode="visits")
    pi_b = enc.create_target_policy(visit_counts, temperature=2.0, mode="visits")
    np.testing.assert_allclose(pi_a, pi_b, rtol=1e-6)
    assert abs(pi_a.sum() - 1.0) < 1e-5


def test_create_target_policy_legacy_matches_visits_at_t1():
    """At T=1.0, legacy shaped mode matches visits mode."""
    enc = _make_encoder()
    visit_counts = {("pass",): 7, ("patch", 4): 13}
    pi_visits = enc.create_target_policy(visit_counts, mode="visits")
    pi_legacy = enc.create_target_policy(visit_counts, temperature=1.0, mode="visits_temperature_shaped")
    np.testing.assert_allclose(pi_visits, pi_legacy, rtol=1e-5)
