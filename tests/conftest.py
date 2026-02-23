"""Pytest configuration for Patchwork AlphaZero tests."""

import warnings
from pathlib import Path

# Repo root: conftest lives in tests/ so parent.parent = project root
REPO_ROOT = Path(__file__).resolve().parent.parent


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")

    # Suppress noisy PyTorch warnings that spam CI logs
    warnings.filterwarnings("ignore", message=".*flash attention.*", module="torch")
    warnings.filterwarnings("ignore", message=".*nested tensors is in prototype stage.*", module="torch")
