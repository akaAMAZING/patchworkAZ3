"""Unit tests for iteration-based schedule lookup semantics and schedule wiring."""

import pytest
import tempfile
from pathlib import Path

import yaml

from src.training.main import (
    _step_schedule_lookup,
    _get_window_iterations_for_iteration,
    _get_num_games_for_iteration,
    _apply_q_value_weight_and_cpuct_schedules,
)
from src.training.selfplay_optimized_integration import create_selfplay_generator


class TestStepScheduleLookup:
    """Step schedule: select last entry with entry.iteration <= i."""

    def test_empty_schedule_returns_base(self):
        assert _step_schedule_lookup([], 0, "val", 1.0) == 1.0
        assert _step_schedule_lookup([], 100, "val", 0.5) == 0.5

    def test_single_entry(self):
        s = [{"iteration": 0, "val": 2.0}]
        assert _step_schedule_lookup(s, 0, "val", 1.0) == 2.0
        assert _step_schedule_lookup(s, 50, "val", 1.0) == 2.0

    def test_multiple_entries_step_semantics(self):
        s = [
            {"iteration": 0, "val": 1.0},
            {"iteration": 50, "val": 2.0},
            {"iteration": 100, "val": 3.0},
        ]
        assert _step_schedule_lookup(s, 0, "val", 0.0) == 1.0
        assert _step_schedule_lookup(s, 49, "val", 0.0) == 1.0
        assert _step_schedule_lookup(s, 50, "val", 0.0) == 2.0
        assert _step_schedule_lookup(s, 99, "val", 0.0) == 2.0
        assert _step_schedule_lookup(s, 100, "val", 0.0) == 3.0
        assert _step_schedule_lookup(s, 200, "val", 0.0) == 3.0

    def test_unsorted_schedule_order_independent(self):
        s = [
            {"iteration": 100, "val": 3.0},
            {"iteration": 0, "val": 1.0},
            {"iteration": 50, "val": 2.0},
        ]
        assert _step_schedule_lookup(s, 75, "val", 0.0) == 2.0


class TestGetWindowIterationsForIteration:
    """window_iterations_schedule: step schedule for replay buffer window size."""

    def test_no_schedule_uses_base_from_replay_buffer(self):
        config = {"replay_buffer": {"window_iterations": 8}, "iteration": {}}
        assert _get_window_iterations_for_iteration(config, 0) == 8
        assert _get_window_iterations_for_iteration(config, 500) == 8

    def test_schedule_overrides_base(self):
        config = {
            "replay_buffer": {"window_iterations": 5},
            "iteration": {
                "window_iterations_schedule": [
                    {"iteration": 0, "window_iterations": 8},
                    {"iteration": 60, "window_iterations": 10},
                    {"iteration": 200, "window_iterations": 12},
                    {"iteration": 400, "window_iterations": 15},
                ]
            },
        }
        assert _get_window_iterations_for_iteration(config, 0) == 8
        assert _get_window_iterations_for_iteration(config, 59) == 8
        assert _get_window_iterations_for_iteration(config, 60) == 10
        assert _get_window_iterations_for_iteration(config, 199) == 10
        assert _get_window_iterations_for_iteration(config, 200) == 12
        assert _get_window_iterations_for_iteration(config, 399) == 12
        assert _get_window_iterations_for_iteration(config, 400) == 15
        assert _get_window_iterations_for_iteration(config, 500) == 15


class TestApplyQValueWeightAndCpuctSchedules:
    """Integration of schedule application and config modification."""

    def test_absent_schedules_use_base_values(self):
        config = {
            "selfplay": {"q_value_weight": 0.4, "mcts": {"cpuct": 1.75}},
            "evaluation": {"lock_eval_cpuct_to_selfplay": True, "eval_mcts": {"cpuct": 1.75}},
            "iteration": {},
        }
        qvw, cpuct = _apply_q_value_weight_and_cpuct_schedules(config, 0)
        assert qvw == 0.4
        assert cpuct == 1.75
        assert config["selfplay"]["q_value_weight"] == 0.4
        assert config["selfplay"]["mcts"]["cpuct"] == 1.75
        assert config["evaluation"]["eval_mcts"]["cpuct"] == 1.75

    def test_schedules_override_base(self):
        config = {
            "selfplay": {"q_value_weight": 0.4, "mcts": {"cpuct": 1.75}},
            "evaluation": {"lock_eval_cpuct_to_selfplay": True, "eval_mcts": {"cpuct": 1.75}},
            "iteration": {
                "q_value_weight_schedule": [{"iteration": 0, "q_value_weight": 0.13}],
                "cpuct_schedule": [{"iteration": 0, "cpuct": 1.47}],
            },
        }
        qvw, cpuct = _apply_q_value_weight_and_cpuct_schedules(config, 0)
        assert qvw == 0.13
        assert cpuct == 1.47
        assert config["selfplay"]["q_value_weight"] == 0.13
        assert config["selfplay"]["mcts"]["cpuct"] == 1.47
        assert config["evaluation"]["eval_mcts"]["cpuct"] == 1.47

    def test_lock_eval_false_keeps_eval_cpuct_unchanged(self):
        config = {
            "selfplay": {"q_value_weight": 0.4, "mcts": {"cpuct": 1.75}},
            "evaluation": {"lock_eval_cpuct_to_selfplay": False, "eval_mcts": {"cpuct": 2.0}},
            "iteration": {"cpuct_schedule": [{"iteration": 0, "cpuct": 1.5}]},
        }
        _apply_q_value_weight_and_cpuct_schedules(config, 0)
        assert config["selfplay"]["mcts"]["cpuct"] == 1.5
        assert config["evaluation"]["eval_mcts"]["cpuct"] == 2.0

    def test_clamping(self):
        config = {
            "selfplay": {"q_value_weight": 0.5, "mcts": {"cpuct": 1.5}},
            "evaluation": {"lock_eval_cpuct_to_selfplay": True, "eval_mcts": {"cpuct": 1.5}},
            "iteration": {
                "q_value_weight_schedule": [{"iteration": 0, "q_value_weight": 1.5}],
                "cpuct_schedule": [{"iteration": 0, "cpuct": 10.0}],
            },
        }
        qvw, cpuct = _apply_q_value_weight_and_cpuct_schedules(config, 0)
        assert qvw == 1.0
        assert cpuct == 5.0


class TestScheduleWiring:
    """Verify ALL config schedules are wired correctly end-to-end."""

    @pytest.fixture
    def schedule_config(self):
        """Config with all schedules (matches config_best.yaml structure)."""
        return {
            "seed": 42,
            "hardware": {"device": "cpu"},
            "paths": {"checkpoints_dir": "ckpt", "logs_dir": "logs", "data_dir": "data"},
            "selfplay": {
                "games_per_iteration": 400,
                "q_value_weight": 0.40,
                "bootstrap": {"games": 200},
                "mcts": {
                    "simulations": 192,
                    "cpuct": 1.75,
                    "temperature": 1.0,
                    "root_dirichlet_alpha": 0.08,
                    "root_noise_weight": 0.25,
                },
            },
            "replay_buffer": {"window_iterations": 8, "max_size": 500000},
            "training": {"learning_rate": 0.0016, "batch_size": 1024},
            "evaluation": {
                "lock_eval_cpuct_to_selfplay": True,
                "eval_mcts": {"cpuct": 1.75, "simulations": 192},
                "elo": {"enabled": False, "initial_rating": 1500, "k_factor": 32},
            },
            "network": {"input_channels": 16, "channels": 64, "num_res_blocks": 4},
            "iteration": {
                "max_iterations": 400,
                "lr_schedule": [{"iteration": 0, "lr": 0.0016}, {"iteration": 80, "lr": 0.0008}],
                "temperature_schedule": [
                    {"iteration": 0, "temperature": 1.0},
                    {"iteration": 30, "temperature": 0.8},
                    {"iteration": 80, "temperature": 0.5},
                ],
                "mcts_schedule": [
                    {"iteration": 0, "simulations": 192},
                    {"iteration": 40, "simulations": 256},
                ],
                "dirichlet_alpha_schedule": [
                    {"iteration": 0, "alpha": 0.10},
                    {"iteration": 50, "alpha": 0.08},
                ],
                "noise_weight_schedule": [
                    {"iteration": 0, "weight": 0.25},
                    {"iteration": 80, "weight": 0.18},
                ],
                "q_value_weight_schedule": [
                    {"iteration": 0, "q_value_weight": 0.13},
                    {"iteration": 50, "q_value_weight": 0.20},
                ],
                "cpuct_schedule": [
                    {"iteration": 0, "cpuct": 1.47},
                    {"iteration": 30, "cpuct": 1.52},
                ],
                "games_schedule": [
                    {"iteration": 0, "games": 400},
                    {"iteration": 80, "games": 320},
                ],
                "window_iterations_schedule": [
                    {"iteration": 0, "window_iterations": 8},
                    {"iteration": 80, "window_iterations": 10},
                ],
            },
        }

    def test_selfplay_schedules_match_generator(self, schedule_config):
        """Selfplay generator's _apply_iteration_schedules returns correct temp/sims/alpha/noise."""
        gen = create_selfplay_generator(schedule_config)
        # iter 32: temp=0.8 (>=30), sims=192 (<40), alpha=0.10 (<50), noise=0.25 (<80)
        sp32 = gen._apply_iteration_schedules(32, quiet=True)
        mcts = sp32["selfplay"]["mcts"]
        assert mcts["temperature"] == pytest.approx(0.8)
        assert mcts["simulations"] == 192
        assert mcts["root_dirichlet_alpha"] == pytest.approx(0.10)
        assert mcts["root_noise_weight"] == pytest.approx(0.25)

        # iter 50: alpha=0.08 (>=50), noise still 0.25 (<80)
        sp50 = gen._apply_iteration_schedules(50, quiet=True)
        assert sp50["selfplay"]["mcts"]["root_dirichlet_alpha"] == pytest.approx(0.08)
        assert sp50["selfplay"]["mcts"]["root_noise_weight"] == pytest.approx(0.25)

        # iter 80: temp=0.5 (>=80), noise=0.18 (>=80)
        sp80 = gen._apply_iteration_schedules(80, quiet=True)
        assert sp80["selfplay"]["mcts"]["temperature"] == pytest.approx(0.5)
        assert sp80["selfplay"]["mcts"]["root_noise_weight"] == pytest.approx(0.18)

    def test_collect_applied_settings_matches_selfplay_at_iter32(self, schedule_config):
        """_collect_applied_settings (main) returns same selfplay values as generator would use."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "cfg.yaml"
            cfg_path.write_text(yaml.safe_dump(schedule_config, sort_keys=False), encoding="utf-8")
            from src.training.main import AlphaZeroTrainer

            trainer = AlphaZeroTrainer(str(cfg_path), cli_run_dir=str(Path(tmp) / "runs" / "schedule_test"))
            trainer._apply_iteration_schedules(32)
            applied = trainer._collect_applied_settings(32)

            # Compare with selfplay generator's schedule-applied config
            sp_config = trainer.selfplay_generator._apply_iteration_schedules(32, quiet=True)
            sp = sp_config["selfplay"]
            mcts = sp["mcts"]

            assert applied["selfplay"]["temperature"] == pytest.approx(mcts["temperature"])
            assert applied["selfplay"]["simulations"] == mcts["simulations"]
            assert applied["selfplay"]["dirichlet_alpha"] == pytest.approx(mcts["root_dirichlet_alpha"])
            assert applied["selfplay"]["noise_weight"] == pytest.approx(mcts["root_noise_weight"])
            assert applied["selfplay"]["cpuct"] == pytest.approx(mcts["cpuct"])
            assert applied["selfplay"]["q_value_weight"] == pytest.approx(sp["q_value_weight"])

    def test_games_schedule_consistent(self, schedule_config):
        """main and selfplay use same games logic; _get_num_games_for_iteration matches."""
        gen = create_selfplay_generator(schedule_config)
        for it in [0, 32, 79, 80, 100]:
            main_games = _get_num_games_for_iteration(schedule_config, it)
            sp_games = gen._get_num_games(it)
            assert main_games == sp_games, f"iter {it}: main={main_games} selfplay={sp_games}"

    def test_main_schedules_propagate_to_eval_when_locked(self, schedule_config):
        """When lock_eval_cpuct_to_selfplay, eval_mcts.cpuct equals selfplay cpuct."""
        from src.training.main import AlphaZeroTrainer

        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "cfg.yaml"
            cfg_path.write_text(yaml.safe_dump(schedule_config, sort_keys=False), encoding="utf-8")
            trainer = AlphaZeroTrainer(str(cfg_path), cli_run_dir=str(Path(tmp) / "runs" / "schedule_test"))
            trainer._apply_iteration_schedules(32)
            sp_cpuct = trainer.config["selfplay"]["mcts"]["cpuct"]
            eval_cpuct = trainer.config["evaluation"]["eval_mcts"]["cpuct"]
            assert sp_cpuct == pytest.approx(eval_cpuct)
