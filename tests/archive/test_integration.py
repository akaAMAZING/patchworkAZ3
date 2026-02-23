"""
Integration Tests for Patchwork AlphaZero Training Pipeline

These tests verify that the reconstructed codebase works end-to-end after the SSD recovery.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml


def test_import_critical_modules():
    """Test that all critical modules can be imported."""
    from src.training.selfplay_optimized_integration import create_selfplay_generator
    from src.training.trainer import train_iteration
    from src.training.evaluation import Evaluator
    from src.training.replay_buffer import ReplayBuffer
    from src.network.model import create_network
    from src.network.encoder import StateEncoder, ActionEncoder
    from src.mcts.alphazero_mcts_optimized import create_optimized_mcts
    from src.game.patchwork_engine import new_game, apply_action, legal_actions_list

    assert create_selfplay_generator is not None
    assert train_iteration is not None
    assert Evaluator is not None
    assert ReplayBuffer is not None
    assert create_network is not None


def test_selfplay_worker_creation():
    """Test that OptimizedSelfPlayWorker can be instantiated and play a game."""
    from src.training.selfplay_optimized import OptimizedSelfPlayWorker

    config = {
        "selfplay": {
            "augment_flips": False,
            "bootstrap": {"mcts_simulations": 16},
            "mcts": {
                "temperature": 1.0,
                "temperature_threshold": 15,
                "simulations": 32,
                "parallel_leaves": 8,
                "cpuct": 1.5,
                "root_dirichlet_alpha": 0.3,
                "root_noise_weight": 0.25,
                "virtual_loss": 3.0,
                "fpu_reduction": 0.0,
            },
        },
        "hardware": {"device": "cpu"},
    }

    worker = OptimizedSelfPlayWorker(
        network_path=None,  # Bootstrap mode (pure MCTS)
        config=config,
        device="cpu",
    )

    # Should complete one game
    result = worker.play_game(game_idx=0, iteration=0, seed=42)

    assert "states" in result
    assert "policies" in result
    assert "values" in result
    assert "action_masks" in result
    assert len(result["states"]) > 0
    # states is a list of (32, 9, 9) arrays per position (gold_v2_32ch)
    assert result["states"][0].shape == (32, 9, 9)
    assert result["policies"][0].shape == (2026,)  # Action space size


def test_selfplay_worker_policy_target_uses_scheduled_temperature():
    """Stored policy target uses worker's scheduled temperature (0.5), not fixed 1.0."""
    from src.training.selfplay_optimized import OptimizedSelfPlayWorker

    captured_temps = []

    # Use a config with temperature=0.5
    config = {
        "selfplay": {
            "augmentation": "none",
            "policy_target_mode": "visits_temperature_shaped",
            "bootstrap": {"mcts_simulations": 16},
            "mcts": {
                "temperature": 0.5,
                "temperature_threshold": 15,
                "simulations": 32,
                "parallel_leaves": 8,
                "cpuct": 1.5,
                "root_dirichlet_alpha": 0.3,
                "root_noise_weight": 0.25,
                "virtual_loss": 3.0,
                "fpu_reduction": 0.0,
            },
        },
        "hardware": {"device": "cpu"},
    }

    worker = OptimizedSelfPlayWorker(
        network_path=None,
        config=config,
        device="cpu",
    )
    assert worker.temperature == 0.5

    # Patch the worker's action_encoder.create_target_policy
    enc = worker.action_encoder
    orig_create = enc.create_target_policy

    def recording_create(vc, temperature=1.0, mode="visits", **kwargs):
        captured_temps.append(float(temperature))
        return orig_create(vc, temperature=temperature, mode=mode, **kwargs)

    enc.create_target_policy = recording_create

    result = worker.play_game(game_idx=0, iteration=0, seed=42)
    assert len(result["policies"]) > 0
    # Worker should have called create_target_policy with 0.5 (scheduled temp)
    assert len(captured_temps) >= 1, "create_target_policy should have been called"
    for t in captured_temps:
        assert abs(t - 0.5) < 1e-5, f"Expected temp 0.5, got {t}"


def test_game_engine_basics():
    """Test that the game engine works correctly."""
    from src.game.patchwork_engine import (
        new_game,
        apply_action,
        legal_actions_list,
        terminal,
        compute_score,
        get_winner,
    )

    # Create new game
    state = new_game(edition="revised")
    assert state is not None
    assert not terminal(state)

    # Get legal actions
    actions = legal_actions_list(state)
    assert len(actions) > 0

    # Apply action
    next_state = apply_action(state, actions[0])
    assert next_state is not None

    # Play until terminal
    max_moves = 200
    for _ in range(max_moves):
        if terminal(state):
            break
        actions = legal_actions_list(state)
        state = apply_action(state, actions[0])

    assert terminal(state)

    # Compute scores
    score0 = compute_score(state, 0)
    score1 = compute_score(state, 1)
    assert isinstance(score0, (int, float))
    assert isinstance(score1, (int, float))

    winner = get_winner(state)
    assert winner in [0, 1]


def test_network_creation():
    """Test that the neural network can be created."""
    from src.network.model import create_network

    config = {
        "network": {
            "input_channels": 56,
            "num_res_blocks": 2,
            "channels": 32,
            "policy_channels": 16,
            "policy_hidden": 64,  # factorized policy head hidden dim
            "value_channels": 16,
            "value_hidden": 32,
            "max_actions": 2026,
            "dropout": 0.0,
            "weight_decay": 0.0001,
            "use_batch_norm": True,
            "se_ratio": 0.0,
        }
    }

    network = create_network(config)
    assert network is not None

    # Test forward pass
    import torch

    batch_size = 4
    states = torch.randn(batch_size, 56, 9, 9)
    action_masks = torch.ones(batch_size, 2026)

    policy_logits, values, scores = network(states, action_masks)
    assert policy_logits.shape == (batch_size, 2026)
    assert values.shape == (batch_size, 1)  # value head returns (B, 1)
    assert scores.shape == (batch_size, 1)  # score head returns (B, 1), unbounded


def test_state_encoder():
    """Test that the state encoder works correctly (gold_v2_32ch)."""
    from src.network.encoder import encode_state_multimodal
    from src.game.patchwork_engine import new_game, current_player
    from src.network.gold_v2_constants import C_SPATIAL_ENC

    state = new_game(edition="revised")
    to_move = current_player(state)
    x_spatial, _, _, _, _ = encode_state_multimodal(state, to_move)
    assert x_spatial.shape == (C_SPATIAL_ENC, 9, 9)
    assert x_spatial.dtype == np.float32 or str(x_spatial.dtype) == "float32"


def test_action_encoder():
    """Test that the action encoder works correctly."""
    from src.network.encoder import ActionEncoder
    from src.game.patchwork_engine import new_game, legal_actions_list

    encoder = ActionEncoder()
    state = new_game(edition="revised")
    actions = legal_actions_list(state)

    # Test encoding and decoding (encode_action/decode_action take only action/index)
    for action in actions[:5]:  # Test first 5 actions
        encoded = encoder.encode_action(action)
        assert 0 <= encoded < 2026

        decoded = encoder.decode_action(encoded)
        assert decoded is not None


def test_mcts_creation():
    """Test that MCTS can be created and run a search."""
    from src.mcts.alphazero_mcts_optimized import create_optimized_mcts
    from src.network.model import create_network
    from src.network.encoder import StateEncoder, ActionEncoder
    from src.game.patchwork_engine import new_game

    config = {
        "network": {
            "input_channels": 56,
            "num_res_blocks": 2,
            "channels": 32,
            "policy_channels": 16,
            "policy_hidden": 64,
            "value_channels": 16,
            "value_hidden": 32,
            "max_actions": 2026,
            "dropout": 0.0,
            "use_batch_norm": True,
            "se_ratio": 0.0,
        },
        "selfplay": {
            "mcts": {
                "simulations": 16,
                "parallel_leaves": 4,
                "cpuct": 1.5,
                "temperature": 1.0,
                "root_dirichlet_alpha": 0.3,
                "root_noise_weight": 0.25,
                "virtual_loss": 3.0,
                "fpu_reduction": 0.0,
            },
        },
    }

    # MCTS requires a network; use minimal 2-block network for fast test
    network = create_network(config)
    state_encoder = StateEncoder()
    action_encoder = ActionEncoder()
    mcts = create_optimized_mcts(
        network=network,
        config=config,
        device=torch.device("cpu"),
        state_encoder=state_encoder,
        action_encoder=action_encoder,
    )
    state = new_game(edition="revised")
    to_move = 0  # First move is player 0

    visit_counts, search_time, root_q = mcts.search(state=state, to_move=to_move, move_number=0, add_noise=True)
    assert visit_counts is not None
    assert isinstance(visit_counts, dict)
    assert len(visit_counts) > 0
    assert search_time >= 0


def test_selfplay_generator_factory():
    """Test that the selfplay generator factory function works."""
    from src.training.selfplay_optimized_integration import create_selfplay_generator

    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "paths": {
                "selfplay_dir": f"{tmpdir}/selfplay",
            },
            "selfplay": {
                "num_workers": 1,
                "games_per_iteration": 2,
                "bootstrap": {"games": 2},
            },
            "iteration": {
                "games_schedule": [],
            },
        }

        generator = create_selfplay_generator(config)
        assert generator is not None
        assert hasattr(generator, "generate")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
