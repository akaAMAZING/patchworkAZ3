"""Test that synthetic H5 matches gold_v2 shapes and can run training steps."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from src.training.trainer import PatchworkDataset, Trainer, BatchIndexSampler, batch_to_dict
from src.network.model import create_network
from torch.utils.data import DataLoader


def test_synthetic_h5_gold_v2_training_steps():
    """Synthetic H5 with gold_v2_multimodal can load and run 5 training steps."""
    from tools.profile_training import create_minimal_h5

    cfg = yaml.safe_load(open("configs/config_best.yaml"))
    path = create_minimal_h5(Path("artifacts/test_synth_smoke.h5"), 256, cfg)
    ds = PatchworkDataset(path, cfg)
    assert len(ds) == 256

    batch = ds[[0, 1, 2, 3, 4]]
    bd = batch_to_dict(batch)
    assert bd["states"].shape == (5, 32, 9, 9)
    assert bd["shop_ids"].shape == (5, 33)
    assert bd["shop_feats"].shape == (5, 33, 10)
    assert bd["x_global"].shape == (5, 61)
    assert bd["x_track"].shape == (5, 8, 54)

    loader = DataLoader(
        ds,
        batch_size=None,
        sampler=BatchIndexSampler(list(range(64)), 8, shuffle=True, seed=42),
    )
    net = create_network(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(net, cfg, device, Path("artifacts/test_logs"), total_train_steps=10)
    metrics = trainer.train_epoch(loader, val_loader=None)
    assert metrics
    assert "total_loss" in metrics or "policy_loss" in metrics
