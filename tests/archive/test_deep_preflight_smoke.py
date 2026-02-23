"""
Deep Preflight Smoke Tests

Validates the Patchwork AlphaZero pipeline end-to-end.
Steps A–D run always; Step E is optional (gated by RUN_DEEP_PREFLIGHT_E2E=1).
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

from tests.conftest import REPO_ROOT

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 56ch gold_v2_multimodal (matches production). E2E smoke validates full pipeline.
CONFIG_PATH = REPO_ROOT / "configs" / "config_e2e_smoke.yaml"


@pytest.fixture
def config():
    import yaml

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory(prefix="deep_preflight_test_") as td:
        yield Path(td)


@pytest.fixture
def device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_step_a_encoder_correctness(config):
    """Step A: Encoder correctness — state shape, channels 14/15, pos/button reconstruction, flip invariance."""
    from tools.deep_preflight import step_a_encoder_correctness

    step_a_encoder_correctness(config)


def test_step_b_network_forward(config, device, tmp_dir):
    """Step B: Network forward + mask sanity (fp32 + AMP)."""
    from tools.deep_preflight import step_b_network_forward

    step_b_network_forward(config, device, tmp_dir)


def test_step_c_training_step(config, device, tmp_dir):
    """Step C: One training step sanity — get_loss, backward, grad clip, optimizer step."""
    from tools.deep_preflight import step_c_training_step

    step_c_training_step(config, device, tmp_dir)


def test_step_d_checkpoint(config, device, tmp_dir):
    """Step D: Checkpoint save/load compatibility."""
    from tools.deep_preflight import step_d_checkpoint

    step_d_checkpoint(config, device, tmp_dir)


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("RUN_DEEP_PREFLIGHT_E2E"),
    reason="Step E skipped unless RUN_DEEP_PREFLIGHT_E2E=1",
)
def test_step_e_selfplay_hdf5_dataset(config, device, tmp_dir):
    """Step E: Minimal self-play -> HDF5 -> Dataset load (slow, optional)."""
    from tools.deep_preflight import step_e_selfplay_hdf5_dataset

    step_e_selfplay_hdf5_dataset(config, device, tmp_dir)


def test_run_all_steps_a_through_d(config, device, tmp_dir):
    """Run Steps A–D via run_all_steps (integration)."""
    from tools.deep_preflight import run_all_steps

    checklist, failed_step = run_all_steps(
        str(CONFIG_PATH),
        "auto",
        str(tmp_dir),
        skip_e=True,
        run_f=False,
    )
    assert failed_step is None, f"run_all_steps failed at step {failed_step}: {checklist}"
    assert checklist.get("A: Encoder correctness", (None,))[0] == "PASS"
    assert checklist.get("B: Network forward + AMP", (None,))[0] == "PASS"
    assert checklist.get("C: One training step", (None,))[0] == "PASS"
    assert checklist.get("D: Checkpoint save/load", (None,))[0] == "PASS"
