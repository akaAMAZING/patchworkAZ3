"""Tests for robustness and provenance improvements."""

import json
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from src.training.run_layout import (
    SELFPLAY_COMPLETE_ATTR,
    SELFPLAY_NUM_GAMES_ATTR,
    SELFPLAY_SCHEMA_VERSION,
    SELFPLAY_SCHEMA_VERSION_ATTR,
    _get_expected_games_for_iteration,
    _staging_has_complete_selfplay,
)
from src.training.replay_buffer import ReplayBuffer


def test_staging_has_complete_selfplay_with_marker():
    """Staging with explicit completion marker and enough games is complete."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "selfplay.h5"
        n_games = 100
        n_pos = 2500
        with h5py.File(path, "w") as f:
            f.create_dataset("states", shape=(n_pos, 56, 9, 9), dtype=np.float32)
            f.create_dataset("action_masks", shape=(n_pos, 2026), dtype=np.float32)
            f.create_dataset("policies", shape=(n_pos, 2026), dtype=np.float32)
            f.create_dataset("values", shape=(n_pos,), dtype=np.float32)
            f.attrs["num_games"] = n_games
            f.attrs["num_positions"] = n_pos
            f.attrs[SELFPLAY_COMPLETE_ATTR] = True
            f.attrs[SELFPLAY_NUM_GAMES_ATTR] = n_games
            f.attrs[SELFPLAY_SCHEMA_VERSION_ATTR] = SELFPLAY_SCHEMA_VERSION

        config = {"selfplay": {"games_per_iteration": 100, "bootstrap": {"games": 200}}}
        assert _staging_has_complete_selfplay(Path(tmp), 1, config) is True


def test_staging_has_complete_selfplay_without_marker_legacy():
    """Legacy files without marker but with num_games >= expected are accepted."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "selfplay.h5"
        n_games = 200
        n_pos = 5000
        with h5py.File(path, "w") as f:
            f.create_dataset("states", shape=(n_pos, 56, 9, 9), dtype=np.float32)
            f.create_dataset("action_masks", shape=(n_pos, 2026), dtype=np.float32)
            f.create_dataset("policies", shape=(n_pos, 2026), dtype=np.float32)
            f.create_dataset("values", shape=(n_pos,), dtype=np.float32)
            f.attrs["num_games"] = n_games
            f.attrs["num_positions"] = n_pos

        config = {"selfplay": {"games_per_iteration": 400, "bootstrap": {"games": 200}}}
        assert _staging_has_complete_selfplay(Path(tmp), 0, config) is True


def test_staging_has_complete_selfplay_incomplete_marker_rejected():
    """If marker says complete but num_games < expected, reject."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "selfplay.h5"
        n_games = 50
        n_pos = 1000
        with h5py.File(path, "w") as f:
            f.create_dataset("states", shape=(n_pos, 56, 9, 9), dtype=np.float32)
            f.create_dataset("action_masks", shape=(n_pos, 2026), dtype=np.float32)
            f.create_dataset("policies", shape=(n_pos, 2026), dtype=np.float32)
            f.create_dataset("values", shape=(n_pos,), dtype=np.float32)
            f.attrs[SELFPLAY_COMPLETE_ATTR] = True
            f.attrs[SELFPLAY_NUM_GAMES_ATTR] = n_games

        config = {"selfplay": {"games_per_iteration": 400, "bootstrap": {"games": 200}}}
        assert _staging_has_complete_selfplay(Path(tmp), 1, config) is False


def test_staging_has_complete_selfplay_marker_false_rejected():
    """If marker says complete=false, reject."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "selfplay.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("states", shape=(100, 56, 9, 9), dtype=np.float32)
            f.create_dataset("action_masks", shape=(100, 2026), dtype=np.float32)
            f.create_dataset("policies", shape=(100, 2026), dtype=np.float32)
            f.create_dataset("values", shape=(100,), dtype=np.float32)
            f.attrs[SELFPLAY_COMPLETE_ATTR] = False
            f.attrs[SELFPLAY_NUM_GAMES_ATTR] = 100

        config = {"selfplay": {"games_per_iteration": 400, "bootstrap": {"games": 200}}}
        assert _staging_has_complete_selfplay(Path(tmp), 1, config) is False


def test_replay_restore_enforces_window():
    """restore_state enforces window_size by evicting oldest."""
    config = {
        "paths": {"data_dir": "data"},
        "replay_buffer": {"max_size": 100000, "min_size": 1000, "window_iterations": 3},
    }
    with tempfile.TemporaryDirectory() as tmp:
        state_path = Path(tmp) / "replay_state.json"
        rb = ReplayBuffer(config, state_path=state_path)
        rb.window_size = 3

        # Create state with 5 entries
        entries = [
            {"iteration": i, "path": str(Path(tmp) / f"iter_{i:03d}.h5"), "positions": 1000}
            for i in range(5)
        ]
        for e in entries:
            Path(e["path"]).touch()
        with open(state_path, "w") as f:
            json.dump(entries, f)

        ok = rb.restore_state()
        assert ok
        assert rb.num_iterations == 3
        assert rb.window_size == 3


def test_get_expected_games_iteration_zero():
    """Iteration 0 uses bootstrap games."""
    config = {"selfplay": {"bootstrap": {"games": 200}, "games_per_iteration": 400}}
    assert _get_expected_games_for_iteration(config, 0) == 200


def test_get_expected_games_schedule():
    """Games schedule overrides base."""
    config = {
        "selfplay": {"bootstrap": {"games": 200}, "games_per_iteration": 400},
        "iteration": {"games_schedule": [{"iteration": 100, "games": 300}]},
    }
    assert _get_expected_games_for_iteration(config, 50) == 400
    assert _get_expected_games_for_iteration(config, 100) == 300
    assert _get_expected_games_for_iteration(config, 200) == 300
