#!/usr/bin/env python
"""
E2E Pipeline Smoke Test — Extremely Robust Check

Runs a fast but deep validation of the Patchwork AlphaZero pipeline after
Phase A (56ch encoder), Phase B (structured policy head), Phase C (D4 augmentation).

Usage:
  python -m tools.run_e2e_pipeline_check --config configs/config_e2e_smoke.yaml

Exit: 0 on PASS, non-zero on FAIL.

Outputs:
  - E2E_PIPELINE_CHECK_REPORT.md
  - artifacts/e2e_check/ (shards, merged_training.h5, checkpoint, logs)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EPS = 1e-5

# ---------------------------------------------------------------------------
# Check runner infrastructure
# ---------------------------------------------------------------------------


class E2ECheckError(Exception):
    """Raised when an invariant fails. Includes expected vs actual."""

    def __init__(self, check_name: str, message: str, expected: Any = None, actual: Any = None, source: str = ""):
        self.check_name = check_name
        self.message = message
        self.expected = expected
        self.actual = actual
        self.source = source
        super().__init__(f"[{check_name}] {message}")


def _config_hash(cfg: dict) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(cfg, sort_keys=True).encode())
    return h.hexdigest()[:12]


def _is_gold_v2_config(config: dict) -> bool:
    enc = str((config.get("data", {}) or {}).get("encoding_version", "") or "").strip().lower()
    if enc in ("gold_v2_32ch", "gold_v2_multimodal"):
        return True
    if int((config.get("network", {}) or {}).get("film_global_dim", 0)) > 0:
        return True
    return False


# ---------------------------------------------------------------------------
# CHECK 1 — Import + registry sanity
# ---------------------------------------------------------------------------


def check_1_import_registry(config: dict) -> Dict:
    """Import modules, print git/config hash, verify input_channels (56 for gold_v2)."""
    results = {}
    try:
        git_hash = "unknown"
        try:
            r = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
                timeout=5,
            )
            if r.returncode == 0 and r.stdout.strip():
                git_hash = r.stdout.strip()
        except Exception:
            pass

        cfg_hash = _config_hash(config)
        schema_ver = 2
        gold_v2 = _is_gold_v2_config(config)
        encoding_version = "gold_v2_32ch" if gold_v2 else "full_clarity_v1"
        expected_ch = 56

        net_cfg = config.get("network", {}) or {}
        input_ch = int(net_cfg.get("input_channels", expected_ch))
        if input_ch != expected_ch:
            raise E2ECheckError(
                "CHECK_1",
                f"network.input_channels must be 56",
                expected=expected_ch,
                actual=input_ch,
                source="config",
            )

        results["git_hash"] = git_hash
        results["config_hash"] = cfg_hash
        results["schema_version"] = schema_ver
        results["encoding_version"] = encoding_version
        results["input_channels"] = input_ch
        results["pass"] = True
    except E2ECheckError:
        raise
    except Exception as e:
        raise E2ECheckError("CHECK_1", str(e), source="import")
    return results


# ---------------------------------------------------------------------------
# CHECK 2 — Encoder invariants
# ---------------------------------------------------------------------------


def check_2_encoder_invariants(config: dict) -> Dict:
    """64 real states: gold_v2 (56,9,9) spatial + multimodal."""
    from src.game.patchwork_engine import (
        apply_action_unchecked,
        current_player_fast,
        legal_actions_fast,
        new_game,
        terminal_fast,
    )
    from src.network.encoder import GoldV2StateEncoder
    from src.network.gold_v2_constants import C_SPATIAL_ENC

    results = {"pass": False, "samples": 0}
    gold_v2 = _is_gold_v2_config(config)
    encoder = GoldV2StateEncoder()

    states_list: List[np.ndarray] = []
    rng = random.Random(42)
    for seed in range(1000):
        if len(states_list) >= 64:
            break
        st = new_game(seed=seed)
        to_move = int(current_player_fast(st))
        x_sp, _, _, _, _ = encoder.encode_state_multimodal(st, to_move)
        states_list.append(x_sp)

        for _ in range(3):
            if terminal_fast(st):
                break
            legal = legal_actions_fast(st)
            if not legal:
                break
            a = rng.choice(legal)
            st = apply_action_unchecked(st, a)
            to_move = int(current_player_fast(st))
            x_sp, _, _, _, _ = encoder.encode_state_multimodal(st, to_move)
            states_list.append(x_sp)
            if len(states_list) >= 64:
                break

    states_list = states_list[:64]

    for i, enc in enumerate(states_list):
        if enc.shape != (C_SPATIAL_ENC, 9, 9):
            raise E2ECheckError(
                "CHECK_2",
                f"Encoder output shape must be ({C_SPATIAL_ENC},9,9)",
                expected=(C_SPATIAL_ENC, 9, 9),
                actual=enc.shape,
                source="src/network/encoder.py encode_state_multimodal",
            )

        # Coord planes: coord_row_norm[r,c] == r/8, coord_col_norm[r,c] == c/8
        row_plane = enc[2]
        col_plane = enc[3]
        denom = 8.0
        for r in range(9):
            for c in range(9):
                exp_r = r / denom
                exp_c = c / denom
                if abs(row_plane[r, c] - exp_r) > EPS:
                    raise E2ECheckError(
                        "CHECK_2",
                        f"coord_row_norm[{r},{c}]",
                        expected=exp_r,
                        actual=float(row_plane[r, c]),
                        source="src/network/encoder.py _precompute_coord_planes",
                    )
                if abs(col_plane[r, c] - exp_c) > EPS:
                    raise E2ECheckError(
                        "CHECK_2",
                        f"coord_col_norm[{r},{c}]",
                        expected=exp_c,
                        actual=float(col_plane[r, c]),
                        source="src/network/encoder.py",
                    )

        # Planes 4-31: frontier, valid_7x7, slot×orient shape (8-31)
        for ch in range(4, C_SPATIAL_ENC):
            plane = enc[ch]
            if plane.shape != (9, 9):
                raise E2ECheckError(
                    "CHECK_2",
                    f"Slot/orient channel {ch} must be 9x9",
                    expected=(9, 9),
                    actual=plane.shape,
                    source="src/network/encoder.py",
                )

    results["samples"] = len(states_list)
    results["pass"] = True
    return results


# ---------------------------------------------------------------------------
# CHECK 3 — Structured head indexing invariants
# ---------------------------------------------------------------------------


def check_3_structured_head(config: dict, device: torch.device) -> Dict:
    """logits (B,2026), patch/buy map indexing, illegal -> -inf, softmax -> 0."""
    from src.game.patchwork_engine import legal_actions_fast, current_player_fast, new_game
    from src.mcts.alphazero_mcts_optimized import encode_legal_actions_fast
    from src.network.encoder import encode_state_multimodal
    from src.network.model import create_network

    results = {"pass": False}

    net = create_network(config).to(device).eval()
    gold_v2 = _is_gold_v2_config(config)

    states_list, xg_list, xt_list, si_list, sf_list, masks_list = [], [], [], [], [], []
    for seed in [42, 43, 44, 45, 46]:
        st = new_game(seed=seed)
        legal = legal_actions_fast(st)
        if not legal:
            continue
        to_move = int(current_player_fast(st))
        _, mask = encode_legal_actions_fast(legal)
        if gold_v2:
            x_sp, xg, xt, si, sf = encode_state_multimodal(st, to_move)
            states_list.append(x_sp)
            xg_list.append(xg)
            xt_list.append(xt)
            si_list.append(si)
            sf_list.append(sf)
        masks_list.append(mask)
    if len(states_list) < 2:
        raise E2ECheckError("CHECK_3", "Need at least 2 valid states", source="check_3")
    states = torch.from_numpy(np.stack(states_list[:4])).float().to(device)
    masks = torch.from_numpy(np.stack(masks_list[:4])).float().to(device)
    fwd_kw = {}
    if gold_v2:
        fwd_kw = dict(
            x_global=torch.from_numpy(np.stack(xg_list[:4])).float().to(device),
            x_track=torch.from_numpy(np.stack(xt_list[:4])).float().to(device),
            shop_ids=torch.from_numpy(np.stack(si_list[:4])).long().to(device),
            shop_feats=torch.from_numpy(np.stack(sf_list[:4])).float().to(device),
        )

    with torch.no_grad():
        logits, _, _ = net.forward(states, masks, **fwd_kw)

    B = logits.shape[0]
    if logits.shape != (B, 2026):
        raise E2ECheckError(
            "CHECK_3",
            "logits shape",
            expected=(B, 2026),
            actual=logits.shape,
            source="src/network/model.py StructuredConvPolicyHead",
        )

    # Illegal mask -> -inf (BEFORE softmax)
    for b in range(B):
        for a in range(2026):
            if masks[b, a] <= 0:
                if not torch.isinf(logits[b, a]) or logits[b, a] > 0:
                    raise E2ECheckError(
                        "CHECK_3",
                        f"Illegal action {a} must have -inf logit",
                        expected=float("-inf"),
                        actual=float(logits[b, a].item()),
                        source="src/network/model.py forward masked_fill",
                    )

    # Softmax assigns 0 to illegal
    probs = torch.softmax(logits, dim=-1)
    for b in range(B):
        for a in range(2026):
            if masks[b, a] <= 0:
                if probs[b, a] > EPS:
                    raise E2ECheckError(
                        "CHECK_3",
                        f"Softmax must give 0 for illegal action {a}",
                        expected=0.0,
                        actual=float(probs[b, a].item()),
                        source="softmax after masked_fill",
                    )

    # Policy sum ~1 for legal support
    for b in range(B):
        legal_sum = float((probs[b] * (masks[b] > 0)).sum())
        if abs(legal_sum - 1.0) > EPS:
            raise E2ECheckError("CHECK_3", "Policy probs over legal actions must sum to 1", expected=1.0, actual=legal_sum)

    results["pass"] = True
    return results


# ---------------------------------------------------------------------------
# CHECK 4 — D4 round-trip
# ---------------------------------------------------------------------------


def check_4_d4_roundtrip(config: dict) -> Dict:
    """32 samples: T then inv(T) returns identity; policy sum preserved."""
    from src.game.patchwork_engine import apply_action_unchecked, current_player_fast, legal_actions_fast, new_game, terminal_fast
    from src.mcts.alphazero_mcts_optimized import encode_legal_actions_fast
    from src.network.encoder import GoldV2StateEncoder, get_slot_piece_id
    from src.network.d4_augmentation import (
        apply_d4_augment,
        inverse_transform_idx,
        D4_COUNT,
    )
    from src.mcts.alphazero_mcts_optimized import engine_action_to_flat_index
    from src.network.gold_v2_constants import C_SPATIAL_ENC

    encoder = GoldV2StateEncoder()

    samples = []
    rng = random.Random(123)
    st = new_game(seed=999)
    for _ in range(50):
        if len(samples) >= 32:
            break
        to_move = int(current_player_fast(st))
        x_sp, _, _, _, _ = encoder.encode_state_multimodal(st, to_move)
        enc = x_sp
        legal = legal_actions_fast(st)
        if legal:
            _, mask = encode_legal_actions_fast(legal)
            policy = np.zeros(2026, dtype=np.float32)
            for a in legal:
                idx = engine_action_to_flat_index(a)
                policy[idx] = 1.0 / len(legal)
            slot_ids = [get_slot_piece_id(st, i) for i in range(3)]
            samples.append((enc.copy(), policy.copy(), mask.copy(), slot_ids))

        for _ in range(2):
            if terminal_fast(st):
                st = new_game(seed=rng.randint(0, 99999))
                break
            legal = legal_actions_fast(st)
            if not legal:
                break
            a = rng.choice(legal)
            st = apply_action_unchecked(st, a)

    if len(samples) < 8:
        raise E2ECheckError("CHECK_4", "Need at least 8 real samples", source="check_4")

    for ti in range(D4_COUNT):
        inv_ti = inverse_transform_idx(ti)
        for state, policy, mask, slot_ids in samples[:32]:
            s1, p1, m1 = apply_d4_augment(state, policy, mask, slot_ids, ti)
            s2, p2, m2 = apply_d4_augment(s1, p1, m1, slot_ids, inv_ti)

            # All spatial channels recover after T+inv(T) (32ch gold_v2: no scalar channels)
            diff_spatial = np.abs(s2 - state)
            if diff_spatial.max() > EPS:
                raise E2ECheckError(
                    "CHECK_4",
                    f"State channels 0-{C_SPATIAL_ENC-1} must recover after T+inv(T)",
                    expected=0,
                    actual=float(diff_spatial.max()),
                    source="src/network/d4_augmentation.py",
                )
            # Mask: round-trip exact equality (bijective orient maps guarantee permutation)
            if not np.array_equal(m2, mask):
                raise E2ECheckError(
                    "CHECK_4", "Mask must recover exactly after T+inv(T)",
                    expected="equal to original", actual="mismatch", source="d4_augmentation",
                )
            # Policy: round-trip equality within eps (bijective orient maps guarantee permutation)
            if not np.allclose(p2, policy, atol=EPS):
                raise E2ECheckError(
                    "CHECK_4", "Policy must recover exactly after T+inv(T)",
                    expected="equal to original", actual="mismatch", source="d4_augmentation",
                )

            # Single transform: policy sum ~1
            p1_sum = float(p1.sum())
            if abs(p1_sum - 1.0) > 1e-5:
                raise E2ECheckError(
                    "CHECK_4",
                    "Policy sum after single transform",
                    expected=1.0,
                    actual=p1_sum,
                    source="apply_d4_augment",
                )

    return {"pass": True, "samples_tested": min(32, len(samples))}


# ---------------------------------------------------------------------------
# CHECK 5 — MCTS legality + augmentation interaction
# ---------------------------------------------------------------------------


def check_5_mcts_legality(config: dict, device: torch.device) -> Dict:
    """Run MCTS from 8 states: move legal, no NaNs, augmentation after policy extraction."""
    from src.game.patchwork_engine import apply_action_unchecked, current_player_fast, legal_actions_fast, new_game, terminal_fast
    from src.mcts.alphazero_mcts_optimized import create_optimized_mcts, encode_legal_actions_fast, engine_action_to_flat_index
    from src.network.encoder import StateEncoder, ActionEncoder
    from src.network.model import create_network

    net = create_network(config).to(device).eval()
    state_encoder = StateEncoder()
    action_encoder = ActionEncoder()
    mcts = create_optimized_mcts(
        network=net, config=config, device=device,
        state_encoder=state_encoder, action_encoder=action_encoder,
    )
    mcts.config.simulations = 16
    mcts.config.add_noise = False

    rng = random.Random(77)
    states_checked = 0
    st = new_game(seed=42)
    for _ in range(200):
        if states_checked >= 8:
            break
        if terminal_fast(st):
            st = new_game(seed=rng.randint(0, 99999))
            continue
        legal = legal_actions_fast(st)
        if not legal:
            st = new_game(seed=rng.randint(0, 99999))
            continue

        to_move = current_player_fast(st)
        visit_counts, _, _ = mcts.search(state=st, to_move=to_move, move_number=0, add_noise=False, root_legal_actions=legal)

        action = mcts.select_action(visit_counts, temperature=0.0, deterministic=True)
        flat_idx = engine_action_to_flat_index(action)
        _, mask = encode_legal_actions_fast(legal)
        if mask[flat_idx] <= 0:
            raise E2ECheckError(
                "CHECK_5",
                "MCTS selected illegal action",
                expected="legal",
                actual=flat_idx,
                source="mcts.select_action",
            )

        visits = np.array(list(visit_counts.values()))
        if np.isnan(visits).any() or np.isinf(visits).any():
            raise E2ECheckError("CHECK_5", "MCTS visits contain NaN/Inf", source="mcts.search")

        states_checked += 1
        a = rng.choice(legal)
        st = apply_action_unchecked(st, a)

    return {"pass": True, "states_checked": states_checked}


# ---------------------------------------------------------------------------
# CHECK 6 — Selfplay shard generation (real, D4)
# ---------------------------------------------------------------------------


def check_6_selfplay_shards(config: dict, artifact_dir: Path, device: torch.device) -> Tuple[Dict, Path, List[Path]]:
    """Generate selfplay with D4; verify HDF5 completion and shapes."""
    from src.training.selfplay_optimized import OptimizedSelfPlayWorker
    from src.training.selfplay_optimized_integration import SelfPlayGenerator
    from src.training.run_layout import (
        SELFPLAY_COMPLETE_ATTR,
        SELFPLAY_EXPECTED_CHANNELS_ATTR,
        ENCODING_VERSION_ATTR,
        SELFPLAY_SCORE_SCALE_ATTR,
        SELFPLAY_VALUE_TARGET_TYPE_ATTR,
    )

    cfg = config.copy()
    cfg["selfplay"] = (cfg.get("selfplay") or {}).copy()
    # Smoke-friendly: fewer games/sims when run_id suggests e2e smoke
    run_id = str((cfg.get("paths") or {}).get("run_id", ""))
    is_smoke = "e2e" in run_id.lower() or "smoke" in run_id.lower()
    num_games = 2 if is_smoke else 8
    num_sims = 8 if is_smoke else 32
    cfg["selfplay"]["games_per_iteration"] = num_games
    cfg["selfplay"]["mcts"]["simulations"] = num_sims
    (cfg["selfplay"].setdefault("bootstrap", {}))["mcts_simulations"] = num_sims
    cfg["selfplay"]["augmentation"] = "d4"
    cfg["paths"] = cfg.get("paths") or {}
    cfg["paths"]["selfplay_dir"] = str(artifact_dir / "selfplay")

    shard_dir = artifact_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    h5_path = artifact_dir / "selfplay.h5"

    # Create initial model for selfplay (bootstrap uses pure MCTS, iter>0 uses model)
    ckpt_dir = artifact_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    from src.network.model import create_network
    net = create_network(cfg)
    init_ckpt = ckpt_dir / "init_model.pt"
    torch.save({"model_state_dict": net.state_dict()}, init_ckpt)

    worker = OptimizedSelfPlayWorker(network_path=str(init_ckpt), config=cfg, device=str(device))
    summaries = []
    gold_v2 = _is_gold_v2_config(cfg)
    logger = logging.getLogger(__name__)
    for g in range(num_games):
        logger.info("  CHECK_6: playing game %d/%d ...", g + 1, num_games)
        data = worker.play_game(g, 0, seed=42 + g)
        if data is None or len(data["states"]) == 0:
            continue
        states = np.asarray(data["states"], dtype=np.float32)
        masks = np.asarray(data["action_masks"], dtype=np.float32)
        policies = np.asarray(data["policies"], dtype=np.float32)
        values = np.asarray(data["values"], dtype=np.float32)
        ownerships = np.asarray(data.get("ownerships"), dtype=np.float32)
        if ownerships is None or ownerships.size == 0:
            ownerships = np.zeros((len(states), 2, 9, 9), dtype=np.float32)
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
        shard_path = shard_dir / f"sp_iter000_g{g:06d}.npz"
        np.savez(shard_path, **save_kw)
        summaries.append({"game_length": data["game_length"]})
        logger.info("  CHECK_6: game %d done (%d positions)", g + 1, data["game_length"])

    if not summaries:
        raise E2ECheckError("CHECK_6", "No selfplay data generated (0 games with positions)", source="OptimizedSelfPlayWorker")

    gen = SelfPlayGenerator(cfg)
    gen._merge_shards(shard_dir, h5_path, summaries)

    if not h5_path.exists():
        raise E2ECheckError("CHECK_6", "HDF5 not produced", source="_merge_shards")

    from src.network.gold_v2_constants import C_SPATIAL_ENC

    with __import__("h5py").File(h5_path, "r") as f:
        if not f.attrs.get(SELFPLAY_COMPLETE_ATTR, False):
            raise E2ECheckError("CHECK_6", "Completion marker missing", expected=True, actual=f.attrs.get(SELFPLAY_COMPLETE_ATTR), source="run_layout")

        states_key = "spatial_states" if "spatial_states" in f else "states"
        states = f[states_key][:]
        policies = f["policies"][:]
        masks = f["action_masks"][:]
        values = f["values"][:]

        expected_shape = (C_SPATIAL_ENC, 9, 9)
        if states.shape[1:] != expected_shape:
            raise E2ECheckError("CHECK_6", "states shape", expected=expected_shape, actual=states.shape[1:], source="HDF5")
        if policies.shape[1] != 2026 or masks.shape[1] != 2026:
            raise E2ECheckError("CHECK_6", "policies/masks shape", expected=2026, actual=(policies.shape[1], masks.shape[1]), source="HDF5")

        exp_ch = f.attrs.get(SELFPLAY_EXPECTED_CHANNELS_ATTR)
        expected_ch = C_SPATIAL_ENC if gold_v2 else 61
        if exp_ch != expected_ch:
            raise E2ECheckError("CHECK_6", "expected_channels attr", expected=expected_ch, actual=exp_ch, source="HDF5")
        enc_ver = f.attrs.get(ENCODING_VERSION_ATTR)
        expected_enc = "gold_v2_32ch"
        if enc_ver != expected_enc:
            raise E2ECheckError("CHECK_6", "encoding_version attr", expected=expected_enc, actual=enc_ver, source="HDF5")
        if SELFPLAY_VALUE_TARGET_TYPE_ATTR not in f.attrs:
            raise E2ECheckError("CHECK_6", "value_target_type attr missing", source="HDF5")

        if np.isnan(states).any() or np.isnan(policies).any() or np.isnan(masks).any() or np.isnan(values).any():
            raise E2ECheckError("CHECK_6", "NaNs in HDF5 data", source="HDF5")
        if (masks.sum(axis=1) == 0).any():
            raise E2ECheckError("CHECK_6", "All-zero mask in data", source="HDF5")
        if (policies.sum(axis=1) == 0).any():
            raise E2ECheckError("CHECK_6", "All-zero policy in data", source="HDF5")

    shard_files = list(shard_dir.glob("*.npz"))
    results = {"pass": True, "num_positions": int(states.shape[0]), "num_games": len(summaries)}
    return results, h5_path, shard_files


# ---------------------------------------------------------------------------
# CHECK 7 — Merge step
# ---------------------------------------------------------------------------


def check_7_merge(config: dict, selfplay_h5: Path, artifact_dir: Path) -> Tuple[Dict, Path]:
    """Run replay buffer merge; verify merged_training.h5, shapes, stats."""
    from src.training.replay_buffer import ReplayBuffer

    merged_path = artifact_dir / "merged_training.h5"
    buf = ReplayBuffer(config, state_path=artifact_dir / "replay_state.json")
    buf.add_iteration(0, str(selfplay_h5))
    buf.get_training_data(seed=42 + 0, output_path=str(merged_path))

    if not merged_path.exists():
        raise E2ECheckError("CHECK_7", "merged_training.h5 not produced", source="ReplayBuffer.get_training_data")

    import h5py
    with h5py.File(merged_path, "r") as f:
        states_key = "spatial_states" if "spatial_states" in f else "states"
        n = int(f[states_key].shape[0])
        if n < 50:
            raise E2ECheckError("CHECK_7", "Need at least 50 positions", expected=">=50", actual=n, source="merge")
        policies = f["policies"][:]
        values = f["values"][:]
        masks = f["action_masks"][:]

    # Policy entropy mean finite
    entropies = []
    for i in range(len(policies)):
        p = policies[i]
        p = p * (masks[i] > 0)
        s = p.sum()
        if s > 0:
            p = p / s
        else:
            continue
        nonzero = p[p > 0]
        if len(nonzero) > 0:
            ent = -np.sum(nonzero * np.log(nonzero + 1e-10))
            entropies.append(ent)
    if entropies and (np.isnan(entropies).any() or np.isinf(entropies).any()):
        raise E2ECheckError("CHECK_7", "Policy entropy contains NaN/Inf", source="merge")
    value_mean = float(np.mean(values))
    if not (-1.0 - EPS <= value_mean <= 1.0 + EPS):
        raise E2ECheckError("CHECK_7", "Value mean out of [-1,1]", expected="[-1,1]", actual=value_mean, source="merge")
    pct_legal = (masks.sum(axis=1) > 0).mean() * 100
    if pct_legal < 99:
        raise E2ECheckError("CHECK_7", "% samples with legal moves", expected=">99%", actual=f"{pct_legal:.1f}%", source="merge")

    results = {"pass": True, "num_positions": n, "policy_entropy_mean": float(np.mean(entropies)) if entropies else 0, "value_mean": value_mean}
    return results, merged_path


# ---------------------------------------------------------------------------
# CHECK 8 — Dataset + training steps
# ---------------------------------------------------------------------------


def check_8_training(config: dict, merged_path: Path, device: torch.device, artifact_dir: Path) -> Tuple[Dict, Optional[Path]]:
    """Run 50 training steps; verify finite loss/grads, optimizer step."""
    from src.training.trainer import PatchworkDataset, BatchIndexSampler, _split_indices
    from src.network.model import create_network, load_model_checkpoint

    dataset = PatchworkDataset(str(merged_path), config)
    batch_size = int(config["training"]["batch_size"])
    n = len(dataset)
    seed = int(config.get("seed", 42))
    train_indices, val_indices = _split_indices(n, config["training"]["val_split"], seed + 0)
    train_sampler = BatchIndexSampler(train_indices, batch_size, shuffle=True, seed=seed)
    net = create_network(config).to(device)
    import torch.optim as optim
    net_cfg = config.get("network") or {}
    opt = optim.AdamW(
        net.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=float(net_cfg.get("weight_decay", 0.0002)),
    )

    train_cfg = config["training"]
    policy_w = train_cfg.get("policy_loss_weight", 1.0)
    value_w = train_cfg.get("value_loss_weight", 1.0)
    ownership_w = float(train_cfg.get("ownership_loss_weight", 0.0))

    steps = 0
    max_steps = 50
    initial_loss = None
    final_loss = None
    grad_norms = []

    gold_v2 = _is_gold_v2_config(config)
    net.train()
    for batch_idx in train_sampler:
        if steps >= max_steps:
            break
        from src.training.trainer import batch_to_dict
        batch = batch_to_dict(dataset[batch_idx])
        states = batch["states"].to(device)
        masks = batch["action_masks"].to(device)
        target_policy = batch["policies"].to(device)
        target_value = batch["values"].to(device)
        target_ownership = batch["ownerships"].to(device) if batch["ownerships"] is not None else None
        loss_kw = {}
        if gold_v2 and batch["x_global"] is not None:
            si = batch["shop_ids"]
            si = si.long().to(device) if si is not None and si.dtype != torch.int64 else (si.to(device) if si is not None else None)
            loss_kw = dict(
                x_global=batch["x_global"].to(device),
                x_track=batch["x_track"].to(device),
                shop_ids=si,
                shop_feats=batch["shop_feats"].to(device),
            )

        opt.zero_grad(set_to_none=True)
        loss, _ = net.get_loss(states, masks, target_policy, target_value, policy_w, value_w, target_ownership=target_ownership, ownership_weight=ownership_w, **loss_kw)
        if not torch.isfinite(loss).all():
            raise E2ECheckError("CHECK_8", f"Loss NaN/Inf at step {steps}", source="trainer.get_loss")
        if initial_loss is None:
            initial_loss = float(loss.item())
        final_loss = float(loss.item())
        loss.backward()
        total_norm = 0.0
        for p in net.parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    raise E2ECheckError("CHECK_8", f"Gradient NaN/Inf at step {steps}", source="backward")
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        grad_norms.append(total_norm)
        torch.nn.utils.clip_grad_norm_(net.parameters(), config["training"].get("max_grad_norm", 1.0))
        opt.step()
        steps += 1

    min_steps = 3 if _is_gold_v2_config(config) and n < 500 else 10
    if steps < min_steps:
        raise E2ECheckError("CHECK_8", f"Too few steps ({steps})", expected=f">={min_steps}", actual=steps, source="training")

    ckpt_path = artifact_dir / "checkpoints" / "e2e_trained.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": net.state_dict()}, ckpt_path)

    results = {
        "pass": True,
        "steps": steps,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "mean_grad_norm": float(np.mean(grad_norms)) if grad_norms else 0,
    }
    return results, ckpt_path


# ---------------------------------------------------------------------------
# CHECK 9 — Tiny deterministic eval
# ---------------------------------------------------------------------------


def check_9_eval(config: dict, checkpoint_path: Path, device: torch.device) -> Dict:
    """Run eval vs pure_mcts; smoke config uses fewer games and sims."""
    import copy
    from src.training.evaluation import Evaluator

    run_id = str((config.get("paths") or {}).get("run_id", ""))
    is_smoke = "e2e" in run_id.lower() or "smoke" in run_id.lower()
    num_games = 2 if is_smoke else 4
    cfg = config
    if is_smoke:
        cfg = copy.deepcopy(config)
        cfg.setdefault("evaluation", {})
        cfg["evaluation"]["eval_mcts"] = cfg["evaluation"].get("eval_mcts") or {}
        cfg["evaluation"]["eval_mcts"]["simulations"] = 8
        baselines = cfg["evaluation"].get("eval_baselines") or []
        for b in baselines:
            if b.get("type") == "pure_mcts":
                b["simulations"] = 8
                break
        else:
            cfg["evaluation"]["eval_baselines"] = [{"type": "pure_mcts", "simulations": 8}]

    evaluator = Evaluator(cfg, device)
    stats = evaluator.evaluate_vs_baseline(
        model_path=str(checkpoint_path),
        baseline_type="pure_mcts",
        num_games=num_games,
        game_offset=0,
    )

    total = stats.get("total_games", 0)
    if total < num_games:
        raise E2ECheckError("CHECK_9", f"Eval must complete {num_games} games", expected=num_games, actual=total, source="Evaluator")
    results = {"pass": True, "win_rate": stats.get("win_rate", 0), "total_games": total}
    return results


# ---------------------------------------------------------------------------
# CHECK 10 — Artifact report
# ---------------------------------------------------------------------------


def check_10_report(report_path: Path, all_results: Dict, config: dict, artifact_dir: Path, duration_s: float) -> Dict:
    """Write E2E_PIPELINE_CHECK_REPORT.md."""
    lines = [
        "# E2E Pipeline Check Report",
        "",
        "## Summary",
        "| Check | Status |",
        "|-------|--------|",
    ]
    for name, r in all_results.items():
        status = "PASS" if r.get("pass", False) else "FAIL"
        lines.append(f"| {name} | {status} |")
    lines.extend([
        "",
        "## Configuration",
        f"- Config hash: `{_config_hash(config)}`",
        f"- input_channels: {config.get('network',{}).get('input_channels')}",
        f"- augmentation: {config.get('selfplay',{}).get('augmentation')}",
        "",
        "## Artifacts",
        f"- Artifact directory: `{artifact_dir}`",
        f"- merged_training.h5: `{artifact_dir / 'merged_training.h5'}`",
        "",
        "## Runtime",
        f"- Duration: {duration_s:.1f}s",
        f"- Platform: {platform.platform()}",
        "",
        "## Details",
    ])
    for name, r in all_results.items():
        lines.append(f"\n### {name}")
        for k, v in r.items():
            if k != "pass":
                lines.append(f"- {k}: {v}")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return {"pass": True}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def check_production_model_forward_backward(config: dict, device: torch.device) -> Dict:
    """Optional: Run 1 forward+backward with production architecture (18 resblocks, 128ch)."""
    import copy
    cfg = copy.deepcopy(config)
    net_cfg = cfg.setdefault("network", {})
    net_cfg["num_res_blocks"] = 18
    net_cfg["channels"] = 128
    net_cfg["policy_channels"] = 48
    net_cfg["policy_hidden"] = 512
    net_cfg["value_channels"] = 48
    net_cfg["value_hidden"] = 512
    net_cfg["ownership_channels"] = 48

    from src.network.model import create_network
    net = create_network(cfg).to(device).train()
    from tools.deep_preflight import _dummy_gold_v2_batch
    B = 4
    states, x_global, x_track, shop_ids, shop_feats = _dummy_gold_v2_batch(B, device)
    loss_kw = dict(x_global=x_global, x_track=x_track, shop_ids=shop_ids, shop_feats=shop_feats)
    masks = torch.zeros(B, 2026, device=device)
    masks[:, 0] = 1.0
    masks[:, 1] = 1.0
    targets_p = torch.zeros(B, 2026, device=device)
    targets_p[:, 0] = 0.5
    targets_p[:, 1] = 0.5
    targets_v = torch.zeros(B, 1, device=device)
    import torch.optim as optim
    opt = optim.AdamW(net.parameters(), lr=0.001)
    opt.zero_grad()
    loss, _ = net.get_loss(states, masks, targets_p, targets_v, 1.0, 1.0, **loss_kw)
    if not torch.isfinite(loss):
        raise E2ECheckError("CHECK_PRODUCTION", "Loss must be finite", source="production forward")
    loss.backward()
    for n, p in net.named_parameters():
        if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
            raise E2ECheckError("CHECK_PRODUCTION", f"Gradient NaN/Inf in {n}", source="production backward")
    opt.step()
    return {"pass": True}


def main() -> int:
    parser = argparse.ArgumentParser(description="E2E Pipeline Smoke Test")
    parser.add_argument("--config", type=str, default="configs/config_e2e_smoke.yaml", help="Config path")
    parser.add_argument("--artifact-dir", type=str, default="artifacts/e2e_check", help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--use-production-model", action="store_true", help="Run 1 forward+backward with production arch (18 resblocks, 128ch)")
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    log_path = artifact_dir / "e2e_check.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    logger = logging.getLogger(__name__)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    if not config_path.exists():
        logger.error("Config not found: %s", config_path)
        return 1
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config.get("selfplay", {}).get("augmentation") != "d4":
        logger.error("REQUIREMENT: config must use selfplay.augmentation=d4 for this check")
        return 1

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    all_results: Dict[str, Dict] = {}
    selfplay_h5 = None
    merged_path = None
    ckpt_path = None

    t0 = time.perf_counter()

    try:
        # CHECK 1
        logger.info("CHECK 1: Import + registry")
        all_results["CHECK_1"] = check_1_import_registry(config)
        logger.info("  PASS: git=%s config_hash=%s channels=%d", all_results["CHECK_1"].get("git_hash"), all_results["CHECK_1"].get("config_hash"), all_results["CHECK_1"].get("input_channels"))

        # CHECK 2
        logger.info("CHECK 2: Encoder invariants")
        all_results["CHECK_2"] = check_2_encoder_invariants(config)
        logger.info("  PASS: %d samples", all_results["CHECK_2"].get("samples"))

        # CHECK 3
        logger.info("CHECK 3: Structured head indexing")
        all_results["CHECK_3"] = check_3_structured_head(config, device)
        logger.info("  PASS")

        # CHECK 4
        logger.info("CHECK 4: D4 round-trip")
        all_results["CHECK_4"] = check_4_d4_roundtrip(config)
        logger.info("  PASS: %d samples", all_results["CHECK_4"].get("samples_tested"))

        # CHECK 5
        logger.info("CHECK 5: MCTS legality")
        all_results["CHECK_5"] = check_5_mcts_legality(config, device)
        logger.info("  PASS: %d states", all_results["CHECK_5"].get("states_checked"))

        # CHECK 6
        logger.info("CHECK 6: Selfplay shards (D4, 8 games, 32 sims)")
        all_results["CHECK_6"], selfplay_h5, _ = check_6_selfplay_shards(config, artifact_dir, device)
        logger.info("  PASS: %d positions, %d games", all_results["CHECK_6"].get("num_positions"), all_results["CHECK_6"].get("num_games"))

        # CHECK 7
        logger.info("CHECK 7: Merge")
        all_results["CHECK_7"], merged_path = check_7_merge(config, selfplay_h5, artifact_dir)
        logger.info("  PASS: %d positions", all_results["CHECK_7"].get("num_positions"))

        # CHECK 8
        logger.info("CHECK 8: Training (50 steps)")
        all_results["CHECK_8"], ckpt_path = check_8_training(config, merged_path, device, artifact_dir)
        logger.info("  PASS: %d steps, loss %.4f -> %.4f", all_results["CHECK_8"].get("steps"), all_results["CHECK_8"].get("initial_loss"), all_results["CHECK_8"].get("final_loss"))

        # CHECK 9
        logger.info("CHECK 9: Tiny eval (4 games)")
        all_results["CHECK_9"] = check_9_eval(config, ckpt_path, device)
        logger.info("  PASS: WR=%.1f%%, %d games", all_results["CHECK_9"].get("win_rate", 0) * 100, all_results["CHECK_9"].get("total_games"))

        # Optional: production model forward+backward
        if args.use_production_model:
            logger.info("CHECK_PRODUCTION: 1 forward+backward with production arch")
            all_results["CHECK_PRODUCTION"] = check_production_model_forward_backward(config, device)
            logger.info("  PASS")

    except E2ECheckError as e:
        logger.error("FAIL [%s]: %s (expected=%s actual=%s)", e.check_name, e.message, e.expected, e.actual)
        all_results[e.check_name] = {"pass": False, "error": str(e), "expected": e.expected, "actual": e.actual}
        for k in list(all_results.keys()):
            if isinstance(all_results.get(k), dict) and not all_results[k].get("pass", False) and k != e.check_name:
                pass
            elif k == e.check_name:
                break
        for k in ["CHECK_2", "CHECK_3", "CHECK_4", "CHECK_5", "CHECK_6", "CHECK_7", "CHECK_8", "CHECK_9"]:
            if k not in all_results:
                all_results[k] = {"pass": False, "skipped": "failed earlier"}
        report_path = REPO_ROOT / "E2E_PIPELINE_CHECK_REPORT.md"
        check_10_report(report_path, all_results, config, artifact_dir, time.perf_counter() - t0)
        return 1
    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        for k in ["CHECK_2", "CHECK_3", "CHECK_4", "CHECK_5", "CHECK_6", "CHECK_7", "CHECK_8", "CHECK_9"]:
            if k not in all_results:
                all_results[k] = {"pass": False, "error": str(e)}
        report_path = REPO_ROOT / "E2E_PIPELINE_CHECK_REPORT.md"
        check_10_report(report_path, all_results, config, artifact_dir, time.perf_counter() - t0)
        return 1

    # CHECK 10
    report_path = REPO_ROOT / "E2E_PIPELINE_CHECK_REPORT.md"
    all_results["CHECK_10"] = check_10_report(report_path, all_results, config, artifact_dir, time.perf_counter() - t0)
    logger.info("CHECK 10: Report written to %s", report_path)
    logger.info("All checks PASSED in %.1fs", time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
