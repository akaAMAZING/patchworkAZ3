"""
PatchworkAgent: AlphaZero NN + MCTS for GUI/API solving.

Loaded by patchwork_api.py when started with --model <path>.
"""

from pathlib import Path
from typing import Optional

import torch
import yaml

from src.network.model import create_network, load_model_checkpoint
from src.network.encoder import StateEncoder, ActionEncoder
from src.mcts.alphazero_mcts_optimized import create_optimized_mcts


def _torch_load(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


class PatchworkAgent:
    """Wrapper for AlphaZero network + MCTS, used by patchwork_api for /solve_nn."""

    def __init__(self, model_path: str, config_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.config_path = config_path
        dev = torch.device(device if torch.cuda.is_available() else "cpu")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.network = create_network(config)
        ckpt = _torch_load(model_path, dev)
        load_model_checkpoint(self.network, ckpt.get("model_state_dict", ckpt))
        self.network.to(dev).eval()

        state_enc = StateEncoder()
        action_enc = ActionEncoder()
        self.mcts = create_optimized_mcts(
            self.network,
            config,
            dev,
            state_enc,
            action_enc,
            enable_tree_reuse=True,  # GUI only: training never passes this
        )
