"""Phase B: Structured conv policy head — mapping and masking correctness."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
import yaml

from tests.conftest import REPO_ROOT
from src.network.model import (
    PatchworkNetwork,
    create_network,
    load_model_checkpoint,
    PASS_INDEX,
    PATCH_START,
    BUY_START,
    NUM_SLOTS,
    NUM_ORIENTS,
    BOARD_SIZE,
    StructuredConvPolicyHead,
)


def _minimal_config(use_structured: bool) -> dict:
    """Minimal config (56ch gold_v2)."""
    cfg_path = REPO_ROOT / "configs" / "config_e2e_smoke.yaml"
    if not cfg_path.exists():
        cfg_path = REPO_ROOT / "configs" / "config_best.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["network"] = {**cfg["network"], "num_res_blocks": 2, "value_channels": 32, "value_hidden": 64}
    cfg["network"]["use_factorized_policy_head"] = use_structured
    return cfg


def _dummy_gold_v2_batch(batch_size=1, device="cpu"):
    """Dummy multimodal inputs for gold_v2."""
    from src.network.gold_v2_constants import C_SPATIAL, F_GLOBAL, C_TRACK, TRACK_LEN, NMAX, F_SHOP
    states = torch.randn(batch_size, C_SPATIAL, 9, 9) * 0.1
    x_global = torch.randn(batch_size, F_GLOBAL) * 0.1
    x_track = torch.randn(batch_size, C_TRACK, TRACK_LEN) * 0.1
    shop_ids = torch.full((batch_size, NMAX), -1, dtype=torch.int64)
    shop_ids[:, :3] = 0
    shop_feats = torch.randn(batch_size, NMAX, F_SHOP) * 0.1
    return states, x_global, x_track, shop_ids, shop_feats


def test_structured_head_mapping_correctness():
    """Verify patch_map and buy_map map to correct logit indices."""
    config = _minimal_config(use_structured=True)
    network = create_network(config)
    assert isinstance(network.policy_head, StructuredConvPolicyHead)

    state, x_global, x_track, shop_ids, shop_feats = _dummy_gold_v2_batch(1)
    trunk = network._trunk_forward(state, x_global=x_global, x_track=x_track, shop_ids=shop_ids, shop_feats=shop_feats)
    logits, patch_map, buy_map = network.policy_head(trunk, return_maps=True)

    assert logits.shape == (1, 2026)
    assert patch_map.shape == (1, 1, 9, 9)
    assert buy_map.shape == (1, 24, 9, 9)

    # patch_map[0,0,r,c] == logits[0, 1+(r*9+c)]
    for r in range(9):
        for c in range(9):
            expected_idx = PATCH_START + r * BOARD_SIZE + c
            assert torch.allclose(
                patch_map[0, 0, r, c], logits[0, expected_idx]
            ), f"patch_map (r={r},c={c}) != logits[{expected_idx}]"

    # buy_map[0, s*8+o, r, c] == logits[0, 82+(s*8+o)*81+(r*9+c)]
    for slot in range(NUM_SLOTS):
        for orient in range(NUM_ORIENTS):
            ch = slot * NUM_ORIENTS + orient
            for r in range(9):
                for c in range(9):
                    pos = r * BOARD_SIZE + c
                    expected_idx = BUY_START + ch * 81 + pos
                    assert torch.allclose(
                        buy_map[0, ch, r, c], logits[0, expected_idx]
                    ), f"buy_map (s={slot},o={orient},r={r},c={c}) != logits[{expected_idx}]"


def test_structured_head_masking_correctness():
    """Illegal indices get -inf and softmax prob 0."""
    config = _minimal_config(use_structured=True)
    network = create_network(config)

    action_mask = torch.zeros(1, 2026)
    action_mask[:, 0] = 1.0

    state, x_global, x_track, shop_ids, shop_feats = _dummy_gold_v2_batch(1)
    logits, _, _ = network(state, action_mask, x_global=x_global, x_track=x_track, shop_ids=shop_ids, shop_feats=shop_feats)

    # Masked indices must be -inf
    assert torch.isinf(logits[:, 1:]).all()
    assert (logits[:, 1:] == float("-inf")).all()
    assert not torch.isinf(logits[:, 0]).any()

    # Softmax: illegal indices get prob 0
    probs = F.softmax(logits.detach(), dim=-1)
    assert probs[0, 0].item() == pytest.approx(1.0, rel=1e-5)
    assert (probs[0, 1:] == 0.0).all()

    # Legal: pass + one patch position
    action_mask = torch.zeros(1, 2026)
    action_mask[:, 0] = 1.0
    action_mask[:, 42] = 1.0  # position 41 = row 4, col 5
    logits2, _, _ = network(state, action_mask, x_global=x_global, x_track=x_track, shop_ids=shop_ids, shop_feats=shop_feats)
    probs2 = F.softmax(logits2.detach(), dim=-1)
    # Illegal indices -inf
    illegal = (action_mask[0] == 0).nonzero(as_tuple=True)[0]
    for idx in illegal:
        assert logits2[0, idx] == float("-inf")
    assert probs2[0, 0] > 0
    assert probs2[0, 42] > 0
    assert (probs2[0, illegal] == 0.0).all()


def test_legacy_head_unchanged_when_flag_false():
    """use_factorized_policy_head=false keeps legacy head."""
    config = _minimal_config(use_structured=False)
    network = create_network(config)
    from src.network.model import PolicyHead

    assert isinstance(network.policy_head, PolicyHead)
    assert not hasattr(network.policy_head, "buy_conv")

    state, x_global, x_track, shop_ids, shop_feats = _dummy_gold_v2_batch(2)
    action_mask = torch.ones(2, 2026)
    logits, value, score = network(state, action_mask, x_global=x_global, x_track=x_track, shop_ids=shop_ids, shop_feats=shop_feats)
    assert logits.shape == (2, 2026)


def test_checkpoint_compatibility_legacy_to_structured():
    """Loading legacy checkpoint into structured model skips policy head."""
    config_legacy = _minimal_config(use_structured=False)
    config_structured = _minimal_config(use_structured=True)

    net_legacy = create_network(config_legacy)
    state_dict = net_legacy.state_dict()

    net_structured = create_network(config_structured)
    load_model_checkpoint(net_structured, state_dict)
    state, x_global, x_track, shop_ids, shop_feats = _dummy_gold_v2_batch(1)
    logits, _, _ = net_structured(state, torch.ones(1, 2026), x_global=x_global, x_track=x_track, shop_ids=shop_ids, shop_feats=shop_feats)
    assert logits.shape == (1, 2026)
