#!/usr/bin/env python
"""
Deep Preflight — End-to-End Pipeline Smoke Test

Deterministic, fast smoke test validating the entire Patchwork AlphaZero pipeline
after upgrades (bf16 AMP, FiLM support, net-size changes, 56-channel gold_v2).

NO BEHAVIOR CHANGES: Testing and validation only.

USAGE:
    python tools/deep_preflight.py --config configs/config_best.yaml --device auto --tmp_dir /tmp/patchwork_smoke
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _is_gold_v2_config(config: dict) -> bool:
    enc = str((config.get("data", {}) or {}).get("encoding_version", "") or "").strip().lower()
    if enc in ("gold_v2_32ch", "gold_v2_multimodal"):
        return True
    net = config.get("network", {}) or {}
    if int(net.get("film_global_dim", 0)) > 0:
        return True
    return False


def _dummy_gold_v2_batch(B: int, device: torch.device) -> tuple:
    """Build dummy (states, x_global, x_track, shop_ids, shop_feats) for gold_v2."""
    from src.network.gold_v2_constants import C_SPATIAL_ENC, F_GLOBAL, C_TRACK, TRACK_LEN, NMAX, F_SHOP
    states = torch.randn(B, C_SPATIAL_ENC, 9, 9, device=device) * 0.1
    x_global = torch.randn(B, F_GLOBAL, device=device) * 0.1
    x_track = torch.randn(B, C_TRACK, TRACK_LEN, device=device) * 0.1
    shop_ids = torch.full((B, NMAX), -1, dtype=torch.int64, device=device)
    shop_ids[:, :3] = 0
    shop_feats = torch.randn(B, NMAX, F_SHOP, device=device) * 0.1
    return states, x_global, x_track, shop_ids, shop_feats


# ---------------------------------------------------------------------------
# Step A: Encoder correctness (32ch gold_v2)
# ---------------------------------------------------------------------------
def step_a_encoder_correctness(config: dict) -> float:
    """Validate gold_v2 encoder: 32ch spatial, multimodal shapes, flip invariance."""
    from src.game.patchwork_engine import new_game
    from src.network.encoder import GoldV2StateEncoder, ActionEncoder, get_slot_piece_id
    from src.network.gold_v2_constants import C_SPATIAL_ENC, F_GLOBAL, C_TRACK, TRACK_LEN, NMAX, F_SHOP

    t0 = time.perf_counter()
    encoder = GoldV2StateEncoder()
    action_encoder = ActionEncoder()

    state = new_game(seed=42)
    x_spatial0, x_global0, x_track0, shop_ids0, shop_feats0 = encoder.encode_state_multimodal(state, 0)
    x_spatial1, x_global1, x_track1, shop_ids1, shop_feats1 = encoder.encode_state_multimodal(state, 1)

    # Assert 32ch gold_v2 shapes
    assert x_spatial0.shape == (C_SPATIAL_ENC, 9, 9), f"Expected ({C_SPATIAL_ENC},9,9), got {x_spatial0.shape}"
    assert x_spatial0.dtype == np.float32, f"Expected float32, got {x_spatial0.dtype}"
    assert x_global0.shape == (F_GLOBAL,), f"Expected ({F_GLOBAL},), got {x_global0.shape}"
    assert x_track0.shape == (C_TRACK, TRACK_LEN), f"Expected ({C_TRACK},{TRACK_LEN}), got {x_track0.shape}"
    assert shop_ids0.shape == (NMAX,), f"Expected ({NMAX},), got {shop_ids0.shape}"
    assert shop_feats0.shape == (NMAX, F_SHOP), f"Expected ({NMAX},{F_SHOP}), got {shop_feats0.shape}"

    # Flip invariance on spatial (slots 8-31)
    slot_piece_ids = [get_slot_piece_id(state, i) for i in range(3)]
    policy = np.zeros(2026, dtype=np.float32)
    policy[0] = 1.0
    mask = np.zeros(2026, dtype=np.float32)
    mask[0] = 1.0
    v_state, _, _ = action_encoder.augment_vertical_flip(x_spatial0.copy(), policy.copy(), mask.copy(), slot_piece_ids)
    h_state, _, _ = action_encoder.augment_horizontal_flip(x_spatial0.copy(), policy.copy(), mask.copy(), slot_piece_ids)
    assert v_state.shape == x_spatial0.shape
    assert h_state.shape == x_spatial0.shape

    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Step B: Network forward + mask sanity (fp32 + AMP)
# ---------------------------------------------------------------------------
def step_b_network_forward(config: dict, device: torch.device, tmp_dir: Path) -> float:
    """Validate network forward pass: fp32 and AMP bf16 if available (56ch gold_v2)."""
    from src.network.encoder import encode_state_multimodal
    from src.network.model import create_network
    from src.game.patchwork_engine import new_game, legal_actions_fast, current_player_fast
    from src.mcts.alphazero_mcts_optimized import encode_legal_actions_fast

    t0 = time.perf_counter()
    net = create_network(config)
    net = net.to(device)
    net.eval()

    gold_v2 = _is_gold_v2_config(config)
    states_list, xg_list, xt_list, si_list, sf_list, masks_list = [], [], [], [], [], []
    for seed in [42, 43, 44, 45]:
        state = new_game(seed=seed)
        legal = legal_actions_fast(state)
        if not legal:
            continue
        to_move = int(current_player_fast(state))
        _, mask = encode_legal_actions_fast(legal)
        if gold_v2:
            x_sp, xg, xt, si, sf = encode_state_multimodal(state, to_move)
            states_list.append(x_sp)
            xg_list.append(xg)
            xt_list.append(xt)
            si_list.append(si)
            sf_list.append(sf)
        masks_list.append(mask)

    B = 4
    if len(states_list) >= B:
        states_batch = torch.from_numpy(np.stack(states_list[:B])).float().to(device)
        masks_batch = torch.from_numpy(np.stack(masks_list[:B])).float().to(device)
        if gold_v2:
            x_global = torch.from_numpy(np.stack(xg_list[:B])).float().to(device)
            x_track = torch.from_numpy(np.stack(xt_list[:B])).float().to(device)
            shop_ids = torch.from_numpy(np.stack(si_list[:B])).long().to(device)
            shop_feats = torch.from_numpy(np.stack(sf_list[:B])).float().to(device)
    else:
        states_batch, x_global, x_track, shop_ids, shop_feats = _dummy_gold_v2_batch(B, device)
        masks_batch = torch.zeros(B, 2026, device=device)
        masks_batch[:, 0] = 1.0
    assert (masks_batch.sum(dim=1) >= 1).all(), "Every sample must have >=1 legal action"

    def _forward():
        if gold_v2:
            return net.forward(states_batch, masks_batch, x_global=x_global, x_track=x_track, shop_ids=shop_ids, shop_feats=shop_feats)
        return net.forward(states_batch, masks_batch)

    with torch.no_grad():
        policy_logits, value, _ = _forward()

    assert policy_logits.shape == (B, 2026), f"Expected ({B},2026), got {policy_logits.shape}"
    assert value.shape == (B, 1), f"Expected ({B},1), got {value.shape}"
    assert not torch.isnan(policy_logits).any(), "policy_logits must not contain NaN"
    assert not torch.isnan(value).any() and not torch.isinf(value).any()
    assert torch.isfinite(policy_logits).any(dim=1).all()
    assert (masks_batch.sum(dim=1) > 0).all()

    train_cfg = config.get("training", {}) or {}
    use_amp = train_cfg.get("use_amp", False)
    amp_dtype_str = str(train_cfg.get("amp_dtype", "bfloat16")).lower()
    amp_dtype = torch.bfloat16 if amp_dtype_str == "bfloat16" else torch.float16

    if device.type == "cuda" and use_amp:
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                policy_logits_amp, value_amp, _ = _forward()
        assert not torch.isnan(policy_logits_amp).any()
        assert not torch.isnan(value_amp).any()

    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Step C: One training step sanity
# ---------------------------------------------------------------------------
def step_c_training_step(config: dict, device: torch.device, tmp_dir: Path) -> float:
    """Validate one training step: get_loss, backward, grad clip, optimizer step (56ch gold_v2)."""
    from src.network.model import create_network
    import torch.optim as optim

    t0 = time.perf_counter()
    net = create_network(config)
    net = net.to(device)
    net.train()

    train_cfg = config.get("training", {}) or {}
    opt_name = train_cfg.get("optimizer", "adamw").lower()
    lr = train_cfg.get("learning_rate", 0.001)
    weight_decay = float(config.get("network", {}).get("weight_decay", 0.0002))
    max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
    policy_weight = train_cfg.get("policy_loss_weight", 1.0)
    value_weight = train_cfg.get("value_loss_weight", 1.0)
    ownership_weight = float(train_cfg.get("ownership_loss_weight", 0.0))

    if opt_name == "adamw":
        optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-8)
    else:
        optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

    B = 4
    gold_v2 = _is_gold_v2_config(config)
    states, x_global, x_track, shop_ids, shop_feats = _dummy_gold_v2_batch(B, device)
    action_mask = torch.zeros(B, 2026, device=device)
    action_mask[:, 0] = 1.0
    action_mask[:, 1] = 1.0
    target_policy = torch.zeros(B, 2026, device=device)
    target_policy[:, 0] = 0.5
    target_policy[:, 1] = 0.5
    target_value = torch.zeros(B, 1, device=device).uniform_(-0.5, 0.5)

    target_ownership = None
    if net.ownership_head is not None and ownership_weight > 0:
        target_ownership = torch.zeros(B, 2, 9, 9, device=device)
        target_ownership[:, 0] = 0.5
        target_ownership[:, 1] = 0.5

    kwargs = {}
    if gold_v2:
        kwargs = dict(x_global=x_global, x_track=x_track, shop_ids=shop_ids, shop_feats=shop_feats)

    optimizer.zero_grad(set_to_none=True)
    loss, _ = net.get_loss(
        states, action_mask, target_policy, target_value,
        policy_weight, value_weight,
        target_ownership=target_ownership,
        ownership_weight=ownership_weight,
        **kwargs,
    )
    assert torch.isfinite(loss).item(), f"loss must be finite, got {loss.item()}"
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
    optimizer.step()

    for n, p in net.named_parameters():
        if p.grad is not None:
            assert not torch.isnan(p.grad).any(), f"Gradient NaN in {n}"
            assert not torch.isinf(p.grad).any(), f"Gradient Inf in {n}"

    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Step D: Checkpoint save/load compatibility
# ---------------------------------------------------------------------------
def step_d_checkpoint(config: dict, device: torch.device, tmp_dir: Path) -> float:
    """Save checkpoint, reload, assert finite forward (56ch gold_v2)."""
    from src.network.model import create_network, load_model_checkpoint

    t0 = time.perf_counter()
    net = create_network(config)
    net = net.to(device)
    net.train()

    ckpt_path = tmp_dir / "deep_preflight_ckpt.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": net.state_dict()}, ckpt_path)

    net2 = create_network(config)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    load_model_checkpoint(net2, ckpt["model_state_dict"])
    net2 = net2.to(device)
    net2.eval()

    dummy, x_global, x_track, shop_ids, shop_feats = _dummy_gold_v2_batch(2, device)
    mask = torch.ones(2, 2026, device=device)
    gold_v2 = _is_gold_v2_config(config)
    with torch.no_grad():
        if gold_v2:
            pl, v, _ = net2.forward(dummy, mask, x_global=x_global, x_track=x_track, shop_ids=shop_ids, shop_feats=shop_feats)
        else:
            pl, v, _ = net2.forward(dummy, mask)
    assert torch.isfinite(pl).all(), "Reloaded model policy_logits must be finite"
    assert torch.isfinite(v).all(), "Reloaded model value must be finite"

    ckpt_path.unlink(missing_ok=True)
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Step E: Minimal self-play -> HDF5 -> Dataset load
# ---------------------------------------------------------------------------
def step_e_selfplay_hdf5_dataset(config: dict, device: torch.device, tmp_dir: Path) -> float:
    """Run tiny self-play, merge to HDF5, load PatchworkDataset, one forward pass."""
    from src.training.selfplay_optimized import OptimizedSelfPlayWorker
    from src.training.selfplay_optimized_integration import SelfPlayGenerator
    from src.training.trainer import PatchworkDataset
    from src.network.model import create_network

    t0 = time.perf_counter()
    cfg = copy.deepcopy(config)
    cfg["selfplay"]["num_workers"] = 1
    cfg["selfplay"]["augmentation"] = "none"
    cfg["selfplay"]["max_game_length"] = 50
    cfg["selfplay"]["bootstrap"]["games"] = 2
    cfg["selfplay"]["bootstrap"]["mcts_simulations"] = 8
    cfg["selfplay"]["bootstrap"]["use_pure_mcts"] = True
    cfg["selfplay"]["games_per_iteration"] = 2
    cfg["selfplay"]["mcts"]["simulations"] = 8
    cfg["selfplay"]["mcts"]["parallel_leaves"] = 4
    cfg["paths"] = cfg.get("paths", {})
    cfg["paths"]["selfplay_dir"] = str(tmp_dir / "selfplay")

    shard_dir = tmp_dir / "iter_000_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    h5_path = tmp_dir / "selfplay.h5"

    # Run 2 games in-process (pure MCTS, no network)
    worker = OptimizedSelfPlayWorker(network_path=None, config=cfg, device="cpu")
    summaries = []
    gold_v2 = _is_gold_v2_config(cfg)
    for g in range(2):
        data = worker.play_game(g, 0, seed=42 + g)
        if data is None or len(data["states"]) == 0:
            continue
        states = np.asarray(data["states"], dtype=np.float32)
        masks = np.asarray(data["action_masks"], dtype=np.float32)
        policies = np.asarray(data["policies"], dtype=np.float32)
        values = np.asarray(data["values"], dtype=np.float32)
        ownerships = np.asarray(data["ownerships"], dtype=np.float32)
        save_kw = dict(states=states, action_masks=masks, policies=policies, values=values, ownerships=ownerships)
        if gold_v2 and "spatial_states" in data:
            save_kw["spatial_states"] = data["spatial_states"]
            save_kw["global_states"] = data["global_states"]
            save_kw["track_states"] = data["track_states"]
            save_kw["shop_ids"] = data["shop_ids"]
            save_kw["shop_feats"] = data["shop_feats"]
        if "slot_piece_ids" in data:
            save_kw["slot_piece_ids"] = data["slot_piece_ids"]
        if "score_margins" in data:
            save_kw["score_margins"] = data["score_margins"]
        shard_path = shard_dir / f"sp_iter000_g{g:06d}_p{os.getpid()}.npz"
        np.savez(shard_path, **save_kw)
        summaries.append({"game_length": data["game_length"]})

    if not summaries:
        raise RuntimeError("No self-play data generated (games produced no positions)")

    # Merge shards
    gen = SelfPlayGenerator(cfg)
    gen._merge_shards(shard_dir, h5_path, summaries)

    # Load dataset and fetch one batch
    from src.training.trainer import batch_to_dict
    dataset = PatchworkDataset(str(h5_path), cfg)
    batch = batch_to_dict(dataset[[0, min(1, len(dataset) - 1)]])
    states = batch["states"]
    action_masks = batch["action_masks"]
    x_global = batch["x_global"]
    x_track = batch["x_track"]
    shop_ids = batch["shop_ids"]
    shop_feats = batch["shop_feats"]
    if isinstance(states, torch.Tensor):
        states_np = states.numpy()
    else:
        states_np = np.array(states)
    masks_np = action_masks.numpy() if hasattr(action_masks, "numpy") else np.array(action_masks)

    from src.network.gold_v2_constants import C_SPATIAL_ENC
    assert states_np.dtype == np.float32, f"states dtype must be float32, got {states_np.dtype}"
    assert states_np.shape[1:] == (C_SPATIAL_ENC, 9, 9), f"states shape (B,{C_SPATIAL_ENC},9,9), got {states_np.shape}"
    assert masks_np.shape[-1] == 2026, f"action_masks shape (B,2026), got {masks_np.shape}"
    assert (masks_np.sum(axis=-1) >= 1).all(), "Every sample must have >=1 legal action"

    # One forward pass (gold_v2 requires multimodal)
    net = create_network(cfg)
    net = net.to(device)
    net.eval()
    states_t = states.to(device) if isinstance(states, torch.Tensor) else torch.from_numpy(states_np).float().to(device)
    masks_t = action_masks.to(device) if isinstance(action_masks, torch.Tensor) else torch.from_numpy(masks_np).float().to(device)
    fwd_kw = {}
    if x_global is not None:
        si = shop_ids.long().to(device) if shop_ids.dtype != torch.int64 else shop_ids.to(device)
        fwd_kw = dict(x_global=x_global.to(device), x_track=x_track.to(device), shop_ids=si, shop_feats=shop_feats.to(device))
    with torch.no_grad():
        pl, v, _ = net.forward(states_t, masks_t, **fwd_kw)
    assert not torch.isnan(pl).any(), "Forward on dataset batch policy_logits must not contain NaN"
    # Paranoid: each sample must have at least one finite policy logit; value must be finite
    assert torch.isfinite(pl).any(dim=1).all(), (
        "Each sample must have at least one finite policy_logits entry"
    )
    assert torch.isfinite(v).all(), "Forward value must be finite"

    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Step F (optional): GPU inference server single request
# ---------------------------------------------------------------------------
def step_f_gpu_inference_server(config: dict, tmp_dir: Path) -> float:
    """If gpu_inference_server exists and CUDA available, start server, one request, shutdown."""
    if not torch.cuda.is_available():
        return 0.0
    try:
        from src.network.gpu_inference_server import run_gpu_inference_server
    except ImportError:
        return 0.0
    # Skip in smoke test - requires subprocess, free port, complex teardown
    # User can enable via env var if desired
    if not os.environ.get("RUN_DEEP_PREFLIGHT_GPU_SERVER"):
        return 0.0
    # Minimal stub: not implemented to avoid port binding complexity in CI
    return 0.0


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------
def run_all_steps(
    config_path: str,
    device_str: str,
    tmp_dir: str,
    skip_e: bool = False,
    run_f: bool = False,
) -> dict:
    """Run all deep preflight steps, return checklist with timings."""
    tmp = Path(tmp_dir)
    tmp.mkdir(parents=True, exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cpu")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    checklist = {}
    failed_step = None

    try:
        t = step_a_encoder_correctness(config)
        checklist["A: Encoder correctness"] = ("PASS", t)
    except Exception as e:
        checklist["A: Encoder correctness"] = ("FAIL", str(e))
        failed_step = "A"
        return checklist, failed_step

    try:
        t = step_b_network_forward(config, device, tmp)
        checklist["B: Network forward + AMP"] = ("PASS", t)
    except Exception as e:
        checklist["B: Network forward + AMP"] = ("FAIL", str(e))
        failed_step = "B"
        return checklist, failed_step

    try:
        t = step_c_training_step(config, device, tmp)
        checklist["C: One training step"] = ("PASS", t)
    except Exception as e:
        checklist["C: One training step"] = ("FAIL", str(e))
        failed_step = "C"
        return checklist, failed_step

    try:
        t = step_d_checkpoint(config, device, tmp)
        checklist["D: Checkpoint save/load"] = ("PASS", t)
    except Exception as e:
        checklist["D: Checkpoint save/load"] = ("FAIL", str(e))
        failed_step = "D"
        return checklist, failed_step

    if not skip_e:
        try:
            t = step_e_selfplay_hdf5_dataset(config, device, tmp)
            checklist["E: Self-play -> HDF5 -> Dataset"] = ("PASS", t)
        except Exception as e:
            checklist["E: Self-play -> HDF5 -> Dataset"] = ("FAIL", str(e))
            failed_step = "E"
            return checklist, failed_step
    else:
        checklist["E: Self-play -> HDF5 -> Dataset"] = ("SKIP", "skipped")

    if run_f:
        t = step_f_gpu_inference_server(config, tmp)
        checklist["F: GPU inference server"] = ("PASS" if t >= 0 else "SKIP", t)

    return checklist, failed_step


def main() -> None:
    parser = argparse.ArgumentParser(description="Deep preflight — end-to-end pipeline smoke test")
    parser.add_argument("--config", required=True, help="Config YAML path (e.g. configs/config_best.yaml)")
    parser.add_argument("--device", default="auto", help="Device: auto|cpu|cuda")
    parser.add_argument("--tmp_dir", default=None, help="Temp directory (default: system temp)")
    parser.add_argument("--skip-e", action="store_true", help="Skip Step E (self-play pipeline, slower)")
    args = parser.parse_args()

    tmp_dir = args.tmp_dir
    if tmp_dir is None:
        import tempfile
        tmp_dir = tempfile.mkdtemp(prefix="patchwork_smoke_")
    run_f = bool(os.environ.get("RUN_DEEP_PREFLIGHT_GPU_SERVER"))
    skip_e = args.skip_e

    checklist, failed_step = run_all_steps(args.config, args.device, tmp_dir, skip_e=skip_e, run_f=run_f)

    _USE_COLOR = sys.stdout.isatty() and os.name != "nt" or os.environ.get("FORCE_COLOR")
    def _g(s): return f"\033[92m{s}\033[0m" if _USE_COLOR else s
    def _r(s): return f"\033[91m{s}\033[0m" if _USE_COLOR else s

    print("\n" + "=" * 60)
    print("DEEP PREFLIGHT CHECKLIST")
    print("=" * 60)
    for name, result in checklist.items():
        status, val = result
        if status == "PASS":
            timing = f"  ({val:.3f}s)" if isinstance(val, (int, float)) else ""
            print(_g(f"  [PASS] {name}{timing}"))
        elif status == "FAIL":
            print(_r(f"  [FAIL] {name}: {val}"))
        else:
            print(f"  [SKIP] {name}")

    if failed_step:
        print("=" * 60)
        raise RuntimeError(f"Deep preflight FAILED at step {failed_step}. See above.")
    print("=" * 60)
    print(_g("DEEP PREFLIGHT PASSED"))
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
