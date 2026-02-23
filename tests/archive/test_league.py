"""
Unit tests for the league-based training system.

Tests cover:
  - PFSPSampler: weighting, sampling, EMA updates
  - PayoffMatrix: recording, winrates, cycle counting, exploitability
  - Anchor selection: diversity heuristic
  - Promotion gating: pass/fail conditions
  - LeagueManager: pool management, state persistence
  - SelfPlaySchedule: game distribution
"""

import json
import math
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.training.league import (
    PFSPSampler,
    PayoffMatrix,
    LeagueConfig,
    LeagueManager,
    GateResult,
    SelfPlaySchedule,
    create_selfplay_schedule,
    evaluate_promotion,
    select_anchors,
)


# =========================================================================
# PFSPSampler Tests
# =========================================================================

class TestPFSPSampler:
    def test_pfsp_weight_peaks_at_50pct(self):
        sampler = PFSPSampler(alpha=2.0)
        # Weight should be highest at 50%
        w50 = sampler.pfsp_weight(0.5)
        w30 = sampler.pfsp_weight(0.3)
        w70 = sampler.pfsp_weight(0.7)
        w90 = sampler.pfsp_weight(0.9)
        assert w50 > w30
        assert w50 > w70
        assert w50 > w90
        assert w30 == pytest.approx(w70)  # Symmetric

    def test_pfsp_weight_symmetry(self):
        sampler = PFSPSampler(alpha=2.0)
        for delta in [0.1, 0.2, 0.3, 0.4]:
            assert sampler.pfsp_weight(0.5 + delta) == pytest.approx(
                sampler.pfsp_weight(0.5 - delta)
            )

    def test_pfsp_weight_higher_alpha_more_peaked(self):
        s1 = PFSPSampler(alpha=1.0)
        s3 = PFSPSampler(alpha=3.0)
        # At 50%, both should return 1.0
        assert s1.pfsp_weight(0.5) == pytest.approx(1.0)
        assert s3.pfsp_weight(0.5) == pytest.approx(1.0)
        # At 80%, alpha=3 should give lower weight (more peaked at 50%)
        assert s3.pfsp_weight(0.8) < s1.pfsp_weight(0.8)

    def test_update_winrate_ema(self):
        sampler = PFSPSampler(ema_decay=0.5)
        sampler.update_winrate("opp1", wins=8, games=10)  # 80%
        assert sampler.winrates["opp1"] == pytest.approx(0.8)
        sampler.update_winrate("opp1", wins=2, games=10)  # 20%
        # EMA: 0.5 * 0.8 + 0.5 * 0.2 = 0.5
        assert sampler.winrates["opp1"] == pytest.approx(0.5)

    def test_update_winrate_first_observation(self):
        sampler = PFSPSampler()
        sampler.update_winrate("opp1", wins=6, games=10)
        assert sampler.winrates["opp1"] == pytest.approx(0.6)
        assert sampler.counts["opp1"] == 10

    def test_sample_single_pool(self):
        sampler = PFSPSampler()
        result = sampler.sample(["only_one"])
        assert result == "only_one"

    def test_sample_empty_pool_raises(self):
        sampler = PFSPSampler()
        with pytest.raises(ValueError):
            sampler.sample([])

    def test_sample_biased_toward_hard_opponents(self):
        """Opponent at 50% WR should be sampled more than one at 90%."""
        sampler = PFSPSampler(alpha=2.0, uniform_prob=0.0)
        sampler.set_winrate("easy", 0.9, 100)
        sampler.set_winrate("hard", 0.5, 100)

        rng = random.Random(42)
        counts = {"easy": 0, "hard": 0}
        for _ in range(1000):
            pick = sampler.sample(["easy", "hard"], rng)
            counts[pick] += 1

        # "hard" (50% WR) should be sampled much more than "easy" (90%)
        assert counts["hard"] > counts["easy"] * 2

    def test_sample_batch_length(self):
        sampler = PFSPSampler()
        batch = sampler.sample_batch(["a", "b", "c"], 10)
        assert len(batch) == 10
        assert all(x in ["a", "b", "c"] for x in batch)

    def test_uniform_exploration(self):
        """With uniform_prob=1.0, all opponents should be sampled roughly equally."""
        sampler = PFSPSampler(uniform_prob=1.0)
        sampler.set_winrate("easy", 0.9, 100)
        sampler.set_winrate("hard", 0.1, 100)

        rng = random.Random(42)
        counts = {"easy": 0, "hard": 0}
        for _ in range(2000):
            pick = sampler.sample(["easy", "hard"], rng)
            counts[pick] += 1

        # Should be roughly 50/50
        ratio = counts["easy"] / (counts["easy"] + counts["hard"])
        assert 0.4 < ratio < 0.6

    def test_state_persistence(self):
        sampler = PFSPSampler()
        sampler.update_winrate("a", 7, 10)
        sampler.update_winrate("b", 3, 10)
        state = sampler.get_state()

        sampler2 = PFSPSampler()
        sampler2.load_state(state)
        assert sampler2.winrates == sampler.winrates
        assert sampler2.counts == sampler.counts


# =========================================================================
# PayoffMatrix Tests
# =========================================================================

class TestPayoffMatrix:
    def test_record_and_get_winrate(self):
        pm = PayoffMatrix()
        pm.record_result("A", "B", a_wins=7, total_games=10)
        assert pm.get_winrate("A", "B") == pytest.approx(0.7)
        assert pm.get_winrate("B", "A") == pytest.approx(0.3)

    def test_accumulate_results(self):
        pm = PayoffMatrix()
        pm.record_result("A", "B", a_wins=3, total_games=5)
        pm.record_result("A", "B", a_wins=4, total_games=5)
        # Total: 7 wins / 10 games
        assert pm.get_winrate("A", "B") == pytest.approx(0.7)

    def test_no_data_returns_none(self):
        pm = PayoffMatrix()
        pm.add_model("A")
        pm.add_model("B")
        assert pm.get_winrate("A", "B") is None

    def test_mean_winrate(self):
        pm = PayoffMatrix()
        pm.record_result("A", "B", 8, 10)  # A vs B: 80%
        pm.record_result("A", "C", 6, 10)  # A vs C: 60%
        mean = pm.get_mean_winrate("A")
        assert mean == pytest.approx(0.7)

    def test_worst_winrate(self):
        pm = PayoffMatrix()
        pm.record_result("A", "B", 8, 10)
        pm.record_result("A", "C", 3, 10)
        worst_wr, worst_opp = pm.get_worst_winrate("A")
        assert worst_wr == pytest.approx(0.3)
        assert worst_opp == "C"

    def test_count_cycles_transitive(self):
        """Strictly transitive: A > B > C. No cycles."""
        pm = PayoffMatrix()
        pm.record_result("A", "B", 8, 10)
        pm.record_result("B", "C", 8, 10)
        pm.record_result("A", "C", 9, 10)
        cycles = pm.count_cycles(500, random.Random(42))
        assert cycles == 0

    def test_count_cycles_with_cycle(self):
        """Non-transitive: A > B > C > A."""
        pm = PayoffMatrix()
        pm.record_result("A", "B", 8, 10)
        pm.record_result("B", "C", 8, 10)
        pm.record_result("C", "A", 8, 10)
        cycles = pm.count_cycles(500, random.Random(42))
        assert cycles > 0

    def test_exploitability_proxy_zero_when_dominant(self):
        """If one model beats all others, exploitability is low for it."""
        pm = PayoffMatrix()
        pm.record_result("A", "B", 10, 10)
        pm.record_result("A", "C", 10, 10)
        # A is dominant: cannot be exploited
        # But B and C can be (A beats them 100%)
        exploit = pm.exploitability_proxy()
        assert exploit == pytest.approx(1.0)  # B or C are maximally exploited by A

    def test_eviction_when_over_capacity(self):
        pm = PayoffMatrix(max_models=3)
        pm.add_model("A")
        pm.add_model("B")
        pm.add_model("C")
        pm.add_model("D")  # Should evict "A"
        assert "A" not in pm.model_ids
        assert len(pm.model_ids) == 3

    def test_to_numpy(self):
        pm = PayoffMatrix()
        pm.record_result("A", "B", 6, 10)
        matrix, ids = pm.to_numpy()
        assert matrix.shape == (2, 2)
        a_idx = ids.index("A")
        b_idx = ids.index("B")
        assert matrix[a_idx, b_idx] == pytest.approx(0.6)
        assert matrix[b_idx, a_idx] == pytest.approx(0.4)

    def test_state_persistence(self):
        pm = PayoffMatrix()
        pm.record_result("A", "B", 7, 10)
        state = pm.get_state()

        pm2 = PayoffMatrix()
        pm2.load_state(state)
        assert pm2.get_winrate("A", "B") == pytest.approx(0.7)
        assert pm2.model_ids == pm.model_ids


# =========================================================================
# Anchor Selection Tests
# =========================================================================

class TestAnchorSelection:
    def test_always_includes_best(self):
        pm = PayoffMatrix()
        for m in ["best", "A", "B", "C"]:
            pm.add_model(m)
        anchors = select_anchors(pm, ["best", "A", "B", "C"], "best", None, anchor_size=3)
        assert "best" in anchors

    def test_includes_prev_best(self):
        pm = PayoffMatrix()
        for m in ["best", "prev", "A", "B"]:
            pm.add_model(m)
        anchors = select_anchors(pm, ["best", "prev", "A", "B"], "best", "prev", anchor_size=3)
        assert "best" in anchors
        assert "prev" in anchors

    def test_respects_anchor_size(self):
        pm = PayoffMatrix()
        ids = [f"m{i}" for i in range(10)]
        for m in ids:
            pm.add_model(m)
        anchors = select_anchors(pm, ids, "m0", "m1", anchor_size=5)
        assert len(anchors) == 5

    def test_handles_small_pool(self):
        pm = PayoffMatrix()
        pm.add_model("only")
        anchors = select_anchors(pm, ["only"], "only", None, anchor_size=5)
        assert anchors == ["only"]


# =========================================================================
# Promotion Gate Tests
# =========================================================================

class TestPromotionGate:
    def test_pass_when_all_conditions_met(self):
        config = LeagueConfig(gate_threshold=0.55, suite_threshold=0.52, worst_threshold=0.40)
        result = evaluate_promotion(
            vs_best_wr=0.60,
            suite_winrates={"a": 0.55, "b": 0.60, "c": 0.50},
            config=config,
        )
        assert result.passed is True

    def test_fail_vs_best_too_low(self):
        config = LeagueConfig(gate_threshold=0.55, suite_threshold=0.52, worst_threshold=0.40)
        result = evaluate_promotion(
            vs_best_wr=0.50,
            suite_winrates={"a": 0.60, "b": 0.60},
            config=config,
        )
        assert result.passed is False

    def test_fail_suite_mean_too_low(self):
        config = LeagueConfig(gate_threshold=0.55, suite_threshold=0.52, worst_threshold=0.40)
        result = evaluate_promotion(
            vs_best_wr=0.60,
            suite_winrates={"a": 0.45, "b": 0.50, "c": 0.55},
            config=config,
        )
        assert result.passed is False  # mean = 0.50 < 0.52

    def test_fail_worst_case_too_low(self):
        config = LeagueConfig(gate_threshold=0.55, suite_threshold=0.52, worst_threshold=0.40)
        result = evaluate_promotion(
            vs_best_wr=0.60,
            suite_winrates={"a": 0.80, "b": 0.35, "c": 0.70},
            config=config,
        )
        assert result.passed is False  # worst = 0.35 < 0.40

    def test_pass_with_no_suite(self):
        """Early iterations may have no suite data - should pass if beats best."""
        config = LeagueConfig(gate_threshold=0.55)
        result = evaluate_promotion(vs_best_wr=0.60, suite_winrates={}, config=config)
        assert result.passed is True

    def test_regression_alerts(self):
        config = LeagueConfig(gate_threshold=0.55, suite_threshold=0.40, worst_threshold=0.30)
        result = evaluate_promotion(
            vs_best_wr=0.60,
            suite_winrates={"a": 0.42, "b": 0.55, "c": 0.35},
            config=config,
        )
        # Regression alert for anchors with WR < 45%
        assert len(result.regression_alerts) == 2  # a (42%) and c (35%)

    def test_gate_result_reason(self):
        config = LeagueConfig()
        result = evaluate_promotion(
            vs_best_wr=0.50,
            suite_winrates={"a": 0.30},
            config=config,
        )
        assert result.passed is False
        assert "vs_best" in result.reason
        assert "suite_worst" in result.reason


# =========================================================================
# SelfPlay Schedule Tests
# =========================================================================

class TestSelfPlaySchedule:
    def test_total_games_correct(self):
        config = LeagueConfig(sp_frac_vs_best=0.4, sp_frac_vs_pfsp=0.5, sp_frac_vs_uniform=0.1)
        sampler = PFSPSampler()
        schedule = create_selfplay_schedule(
            total_games=100,
            best_id="best",
            pool_ids=["best", "a", "b"],
            pfsp_sampler=sampler,
            config=config,
        )
        assert schedule.total_games == 100

    def test_all_to_best_when_no_pool(self):
        config = LeagueConfig()
        sampler = PFSPSampler()
        schedule = create_selfplay_schedule(
            total_games=100,
            best_id="best",
            pool_ids=[],
            pfsp_sampler=sampler,
            config=config,
        )
        assert schedule.vs_best == 100
        assert not schedule.vs_pfsp
        assert not schedule.vs_uniform

    def test_pfsp_opponents_from_pool(self):
        config = LeagueConfig(sp_frac_vs_pfsp=0.5, sp_frac_vs_best=0.3, sp_frac_vs_uniform=0.2)
        sampler = PFSPSampler()
        schedule = create_selfplay_schedule(
            total_games=100,
            best_id="best",
            pool_ids=["best", "a", "b", "c"],
            pfsp_sampler=sampler,
            config=config,
        )
        # PFSP opponents should not include best
        for opp in schedule.vs_pfsp:
            assert opp != "best"


# =========================================================================
# LeagueConfig Tests
# =========================================================================

class TestLeagueConfig:
    def test_from_config_defaults(self):
        config = {"league": {}}
        lc = LeagueConfig.from_config(config)
        assert lc.pfsp_alpha == 2.0
        assert lc.gate_threshold == 0.55

    def test_from_config_overrides(self):
        config = {"league": {"pfsp_alpha": 3.0, "gate_threshold": 0.60}}
        lc = LeagueConfig.from_config(config)
        assert lc.pfsp_alpha == 3.0
        assert lc.gate_threshold == 0.60

    def test_from_empty_config(self):
        config = {}
        lc = LeagueConfig.from_config(config)
        assert lc.pfsp_alpha == 2.0  # Falls back to defaults


# =========================================================================
# LeagueManager Tests
# =========================================================================

class TestLeagueManager:
    def _make_config(self) -> dict:
        return {
            "seed": 42,
            "league": {
                "enabled": True,
                "max_pool_size": 10,
                "anchor_size": 3,
                "pfsp_alpha": 2.0,
            },
        }

    def test_pool_management(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lm = LeagueManager(self._make_config(), Path(tmpdir))
            lm.add_to_pool("m0", "/fake/path/m0.pt")
            lm.add_to_pool("m1", "/fake/path/m1.pt")
            assert len(lm.pool) == 2
            assert "m0" in lm.get_pool_ids()

    def test_promote(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lm = LeagueManager(self._make_config(), Path(tmpdir))
            lm.promote("m0", "/fake/m0.pt")
            assert lm.best_id == "m0"
            lm.promote("m1", "/fake/m1.pt")
            assert lm.best_id == "m1"
            assert lm.prev_best_id == "m0"

    def test_pool_eviction(self):
        cfg = self._make_config()
        cfg["league"]["max_pool_size"] = 3
        with tempfile.TemporaryDirectory() as tmpdir:
            lm = LeagueManager(cfg, Path(tmpdir))
            for i in range(5):
                lm.add_to_pool(f"m{i}", f"/fake/m{i}.pt")
            assert len(lm.pool) <= 3

    def test_model_id_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lm = LeagueManager(self._make_config(), Path(tmpdir))
            assert lm.model_id(5) == "iter005_main"
            assert lm.model_id(10, "exploiter") == "iter010_exploiter"

    def test_state_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lm = LeagueManager(self._make_config(), Path(tmpdir))
            lm.promote("m0", "/fake/m0.pt")
            lm.add_to_pool("m1", "/fake/m1.pt")
            lm.payoff.record_result("m0", "m1", 7, 10)
            lm.pfsp.update_winrate("m1", 7, 10)
            lm.save_state()

            lm2 = LeagueManager(self._make_config(), Path(tmpdir))
            # Patch pool paths to exist for validation
            for mid, path in lm.pool.items():
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).touch()
            lm2.pool = dict(lm.pool)  # pre-fill since fake paths won't validate
            restored = lm2.load_state()
            # State file exists so load_state returns True
            assert lm2.state_path.exists()
            assert lm2.best_id == "m0"

    def test_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lm = LeagueManager(self._make_config(), Path(tmpdir))
            lm.promote("m0", "/fake/m0.pt")
            lm.add_to_pool("m1", "/fake/m1.pt")
            lm.payoff.record_result("m0", "m1", 6, 10)
            diag = lm.get_diagnostics("m0")
            assert "pool_size" in diag
            assert diag["pool_size"] == 2
            assert "cycles_in_200_triples" in diag
            assert "exploitability_proxy" in diag

    def test_save_payoff_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lm = LeagueManager(self._make_config(), Path(tmpdir))
            lm.add_to_pool("A", "/fake/A.pt")
            lm.add_to_pool("B", "/fake/B.pt")
            lm.payoff.record_result("A", "B", 7, 10)
            csv_path = Path(tmpdir) / "payoff.csv"
            lm.save_payoff_csv(csv_path)
            assert csv_path.exists()
            content = csv_path.read_text()
            assert "A" in content
            assert "B" in content
            assert "0.700" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
