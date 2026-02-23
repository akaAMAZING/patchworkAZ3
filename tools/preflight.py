#!/usr/bin/env python
"""
Preflight checks for overnight runs.

Validates hardware, config sanity, core correctness, and runs a tiny
smoke-run of the full pipeline (selfplay → train → eval).

USAGE:
    python tools/preflight.py --config configs/config_best.yaml
    python tools/preflight.py --config configs/overnight_strong.yaml --skip-smoke
"""

from __future__ import annotations

import argparse
import warnings
import copy
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Suppress PyTorch nested-tensor prototype warning (TransformerEncoder with src_key_padding_mask)
warnings.filterwarnings(
    "ignore",
    message="The PyTorch API of nested tensors is in prototype stage",
    category=UserWarning,
)

from src.network.model import create_network
from tools.sanity_check import (
    check_action_roundtrip,
    check_avs_a_fairness,
    check_legality_masks,
    check_state_encoder_shape,
    check_terminal_consistency,
)

# ---------------------------------------------------------------------------
# Colour helpers (degrade gracefully on dumb terminals)
# ---------------------------------------------------------------------------
_USE_COLOR = sys.stdout.isatty() and os.name != "nt" or os.environ.get("FORCE_COLOR")

def _green(s: str) -> str:
    return f"\033[92m{s}\033[0m" if _USE_COLOR else s

def _yellow(s: str) -> str:
    return f"\033[93m{s}\033[0m" if _USE_COLOR else s

def _red(s: str) -> str:
    return f"\033[91m{s}\033[0m" if _USE_COLOR else s


# ---------------------------------------------------------------------------
# Step 1 — Hardware & Environment
# ---------------------------------------------------------------------------
REQUIRED_KEYS = [
    "network", "training", "selfplay", "replay_buffer",
    "evaluation", "paths", "iteration",
]

def check_hardware(cfg: dict) -> list[str]:
    """Check hardware requirements, return list of warnings (empty = good)."""
    warnings: list[str] = []

    # Python version
    py_ver = platform.python_version()
    print(f"  Python:        {py_ver}")
    if sys.version_info < (3, 10):
        warnings.append(f"Python {py_ver} < 3.10; recommend 3.10+")

    # PyTorch + CUDA
    print(f"  PyTorch:       {torch.__version__}")
    device = cfg.get("hardware", {}).get("device", "cpu")
    if device == "cuda":
        if not torch.cuda.is_available():
            warnings.append("Config requests CUDA but torch.cuda.is_available() == False")
        else:
            gpu_name = torch.cuda.get_device_name(0)
            vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
            print(f"  GPU:           {gpu_name}")
            print(f"  VRAM:          {vram_mb:.0f} MB")
            if vram_mb < 4000:
                warnings.append(f"Low VRAM ({vram_mb:.0f} MB) — may OOM with large batches")
    else:
        print(f"  Device:        {device} (no GPU)")

    # RAM
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        print(f"  RAM:           {ram_gb:.1f} GB")
        if ram_gb < 12:
            warnings.append(f"Low RAM ({ram_gb:.1f} GB) — recommend 16+ GB")
    except ImportError:
        print("  RAM:           (psutil not installed, skipped)")

    # Disk space
    logs_dir = cfg.get("paths", {}).get("logs_dir", "logs")
    try:
        usage = shutil.disk_usage(Path(logs_dir).resolve().anchor)
        free_gb = usage.free / (1024 ** 3)
        print(f"  Disk free:     {free_gb:.1f} GB (on drive containing {logs_dir})")
        if free_gb < 20:
            warnings.append(f"Low disk space ({free_gb:.1f} GB free) — recommend 20+ GB")
    except Exception:
        print("  Disk free:     (could not determine)")

    # CPU cores
    cpu_count = os.cpu_count() or 1
    print(f"  CPU cores:     {cpu_count}")
    num_workers = cfg.get("selfplay", {}).get("num_workers", 1)
    if num_workers > cpu_count:
        warnings.append(
            f"selfplay.num_workers={num_workers} > CPU cores={cpu_count}, "
            f"will cause contention"
        )

    return warnings


# ---------------------------------------------------------------------------
# Step 2 — Config validation
# ---------------------------------------------------------------------------
def validate_config(cfg: dict) -> list[str]:
    """Check config for common mistakes, return list of errors."""
    errors: list[str] = []

    # Required top-level keys
    for key in REQUIRED_KEYS:
        if key not in cfg:
            errors.append(f"Missing required config section: '{key}'")

    # Network shape consistency — branch on encoding_version
    data_cfg = cfg.get("data", {}) or {}
    enc_ver = str(data_cfg.get("encoding_version", "")).strip().lower()
    is_gold_v2 = enc_ver in ("gold_v2_32ch", "gold_v2_multimodal")

    net = cfg.get("network", {})
    if net.get("max_actions", 2026) != 2026:
        errors.append(f"network.max_actions must be 2026, got {net.get('max_actions')}")

    if is_gold_v2:
        # Gold v2: expect 56 spatial channels (trunk) and multimodal conditioning
        if net.get("input_channels", 56) != 56:
            errors.append(
                f"gold_v2_32ch requires network.input_channels=56, got {net.get('input_channels')}"
            )
        if net.get("use_film"):
            if not net.get("film_global_dim") or not net.get("film_track_dim") or not net.get("film_shop_dim"):
                errors.append(
                    "gold_v2_32ch with use_film requires film_global_dim, film_track_dim, film_shop_dim"
                )
            if net.get("film_input_plane_indices"):
                errors.append(
                    "gold_v2_32ch must not use film_input_plane_indices; use multimodal conditioning"
                )
    else:
        # Non-gold_v2: expect 56 channels
        if net.get("input_channels", 56) != 56:
            errors.append(
                f"network.input_channels must be 56 (gold_v2); got {net.get('input_channels')}. "
                "full_clarity_v1 spatial encoding is deprecated; use gold_v2."
            )

    # Training sanity
    train = cfg.get("training", {})
    lr = train.get("learning_rate", 0)
    if lr <= 0 or lr > 0.1:
        errors.append(f"Suspicious learning_rate: {lr}")
    bs = train.get("batch_size", 0)
    if bs < 32 or bs > 8192:
        errors.append(f"Suspicious batch_size: {bs}")
    epochs = train.get("epochs_per_iteration", 0)
    if epochs < 1 or epochs > 20:
        errors.append(f"Suspicious epochs_per_iteration: {epochs}")

    # Self-play
    sp = cfg.get("selfplay", {})
    mcts_cfg = sp.get("mcts", {})
    sims = mcts_cfg.get("simulations", 0)
    if sims < 1:
        errors.append(f"selfplay.mcts.simulations must be >= 1, got {sims}")
    games = sp.get("games_per_iteration", 0)
    if games < 1:
        errors.append(f"selfplay.games_per_iteration must be >= 1, got {games}")

    # Replay buffer
    rb = cfg.get("replay_buffer", {})
    max_size = rb.get("max_size", 0)
    if max_size < 1000:
        errors.append(f"replay_buffer.max_size={max_size} is very small")

    # Evaluation (0.0 = no gating / always promote)
    ev = cfg.get("evaluation", {})
    threshold = ev.get("win_rate_threshold", 0.55)
    if not (0.0 <= threshold <= 0.7):
        errors.append(f"Suspicious win_rate_threshold: {threshold} (expected 0.0-0.7)")

    return errors


# ---------------------------------------------------------------------------
# Step 3 — VRAM estimation
# ---------------------------------------------------------------------------
def estimate_vram(cfg: dict) -> None:
    """Try to create the network and estimate peak VRAM usage."""
    device_name = cfg.get("hardware", {}).get("device", "cpu")
    if device_name != "cuda" or not torch.cuda.is_available():
        print("  VRAM test:     skipped (not using CUDA)")
        return

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)

    data_cfg = cfg.get("data", {}) or {}
    enc_ver = str(data_cfg.get("encoding_version", "")).strip().lower()
    is_gold_v2 = enc_ver in ("gold_v2_32ch", "gold_v2_multimodal")

    net = cfg.get("network", {}) or {}
    input_ch = int(net.get("input_channels", 56))

    network = create_network(cfg)
    network = network.to(device)

    bs = cfg.get("training", {}).get("batch_size", 256)
    dummy_state = torch.randn(bs, input_ch, 9, 9, device=device)
    dummy_mask = torch.ones(bs, 2026, device=device)

    # Gold v2: pass multimodal inputs for FiLM conditioning
    dummy_x_global = None
    dummy_x_track = None
    dummy_shop_ids = None
    dummy_shop_feats = None
    if is_gold_v2 and net.get("use_film") and net.get("film_global_dim"):
        from src.network.gold_v2_constants import F_GLOBAL, C_TRACK, TRACK_LEN, NMAX, F_SHOP
        dummy_x_global = torch.randn(bs, F_GLOBAL, device=device)
        dummy_x_track = torch.randn(bs, C_TRACK * TRACK_LEN, device=device)
        dummy_shop_ids = torch.randint(-1, 30, (bs, NMAX), device=device, dtype=torch.long)
        dummy_shop_feats = torch.randn(bs, NMAX, F_SHOP, device=device)
        dummy_x_track = dummy_x_track.view(bs, C_TRACK, TRACK_LEN)

    train_cfg = cfg.get("training", {}) or {}
    use_amp = train_cfg.get("use_amp", False)
    amp_dtype_str = str(train_cfg.get("amp_dtype", "float16")).lower()
    amp_dtype = torch.bfloat16 if amp_dtype_str == "bfloat16" else torch.float16
    with torch.no_grad():
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                _ = network(
                    dummy_state, dummy_mask,
                    x_global=dummy_x_global,
                    x_track=dummy_x_track,
                    shop_ids=dummy_shop_ids,
                    shop_feats=dummy_shop_feats,
                )
        else:
            _ = network(
                dummy_state, dummy_mask,
                x_global=dummy_x_global,
                x_track=dummy_x_track,
                shop_ids=dummy_shop_ids,
                shop_feats=dummy_shop_feats,
            )

    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)

    # Rough estimate: training uses ~2.5x inference peak (gradients + optimizer)
    est_train_mb = peak_mb * 2.5
    print(f"  Inference peak: {peak_mb:.0f} MB (batch_size={bs}, channels={input_ch})")
    print(f"  Est. training:  {est_train_mb:.0f} MB (rough 2.5x multiplier)")
    print(f"  GPU total:      {total_mb:.0f} MB")

    if est_train_mb > total_mb * 0.9:
        print(_yellow(f"  WARNING: estimated training VRAM ({est_train_mb:.0f} MB) "
                       f"is close to GPU limit ({total_mb:.0f} MB) — may OOM"))

    # Cleanup
    del network, dummy_state, dummy_mask
    if dummy_x_global is not None:
        del dummy_x_global, dummy_x_track, dummy_shop_ids, dummy_shop_feats
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Step 4 — Core correctness sanity checks
# ---------------------------------------------------------------------------
def run_quick_sanity(seed: int, cfg: dict) -> None:
    data_cfg = cfg.get("data", {}) or {}
    enc_ver = str(data_cfg.get("encoding_version", "")).strip().lower()
    is_gold_v2 = enc_ver in ("gold_v2_32ch", "gold_v2_multimodal")

    if is_gold_v2:
        from src.game.patchwork_engine import new_game, current_player_fast
        from src.network.encoder import encode_state_multimodal
        from src.network.gold_v2_constants import C_SPATIAL_ENC, F_GLOBAL, C_TRACK, TRACK_LEN, NMAX, F_SHOP
        state = new_game(seed=seed)
        to_move = int(current_player_fast(state))
        xs, xg, xt, sid, sfeat = encode_state_multimodal(state, to_move)
        assert xs.shape == (C_SPATIAL_ENC, 9, 9), f"x_spatial shape {xs.shape}"
        assert xg.shape == (F_GLOBAL,), f"x_global shape {xg.shape}"
        assert xt.shape == (C_TRACK, TRACK_LEN), f"x_track shape {xt.shape}"
        assert sid.shape == (NMAX,), f"shop_ids shape {sid.shape}"
        assert sfeat.shape == (NMAX, F_SHOP), f"shop_feats shape {sfeat.shape}"
    else:
        check_state_encoder_shape(seed)
    check_action_roundtrip(steps=12, seed=seed)
    check_legality_masks(steps=12, seed=seed)
    check_terminal_consistency(num_games=6, seed=seed)
    check_avs_a_fairness(num_pairs=4, seed=seed, sims=8)


def run_invariance_tests(skip_slow: bool = False, cfg: dict | None = None) -> bool:
    """Run D4 and shop-order invariance tests. Required before long training."""
    cmd = [
        sys.executable, "-m", "pytest", "-q",
        "tests/test_d4_augmentation.py",
        "tests/test_d4_action_equivariance.py",
        "tests/test_shop_order_markov_alignment.py",
    ]
    data_cfg = (cfg or {}).get("data", {}) or {}
    if str(data_cfg.get("encoding_version", "")).strip().lower() in ("gold_v2_32ch", "gold_v2_multimodal"):
        cmd.insert(-1, "tests/test_gold_v2_encoding.py")
    if skip_slow:
        cmd.extend(["-m", "not slow"])
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=120)
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Step 5 — Tiny pipeline smoke run
# ---------------------------------------------------------------------------
def _tiny_config(base_cfg: dict, temp_root: Path) -> dict:
    """Scale down config for a 2-iteration smoke test."""
    cfg = copy.deepcopy(base_cfg)
    cfg["iteration"]["auto_resume"] = False
    cfg["iteration"]["max_iterations"] = 2
    cfg["iteration"]["games_schedule"] = [{"iteration": 0, "games": 1}]
    cfg["iteration"]["mcts_schedule"] = [{"iteration": 0, "simulations": 1}]
    cfg["iteration"]["window_iterations_schedule"] = [{"iteration": 0, "window_iterations": 8}]
    # Smoke test uses iter 0 schedule values (lightweight integration check)
    cfg["iteration"]["q_value_weight_schedule"] = [{"iteration": 0, "q_value_weight": 0.13}]
    cfg["iteration"]["cpuct_schedule"] = [{"iteration": 0, "cpuct": 1.47}]

    cfg["hardware"]["device"] = "cpu"
    cfg["selfplay"]["num_workers"] = 1
    cfg["selfplay"]["augmentation"] = "none"
    cfg["selfplay"]["max_game_length"] = 40
    cfg["selfplay"]["bootstrap"]["games"] = 1
    cfg["selfplay"]["bootstrap"]["mcts_simulations"] = 1
    cfg["selfplay"]["games_per_iteration"] = 1
    cfg["selfplay"]["mcts"]["simulations"] = 1
    cfg["selfplay"]["mcts"]["parallel_leaves"] = 1

    cfg["training"]["epochs_per_iteration"] = 1
    cfg["training"]["batch_size"] = 64
    cfg["training"]["num_data_workers"] = 0
    cfg["training"]["val_frequency"] = 999999
    cfg["training"]["checkpoint_frequency"] = 999999

    cfg["evaluation"]["paired_eval"] = True
    cfg["evaluation"]["games_vs_best"] = 2
    cfg["evaluation"]["games_vs_pure_mcts"] = 2
    cfg["evaluation"]["skip_pure_mcts_after_iter"] = 0
    cfg["evaluation"]["eval_mcts"]["simulations"] = 1
    if cfg["evaluation"].get("eval_baselines"):
        cfg["evaluation"]["eval_baselines"][0]["simulations"] = 1
    cfg["evaluation"]["micro_gate"] = {
        "enabled": True,
        "start_games": 2,
        "step_games": 2,
        "max_games": 4,
        "anti_regression_threshold": 0.45,
    }

    cfg["paths"] = {
        "checkpoints_dir": str(temp_root / "checkpoints"),
        "logs_dir": str(temp_root / "logs"),
        "run_dir": str(temp_root / "runs" / "preflight_smoke"),
    }
    # Smoke: replay buffer min_size=1 so merged data is used when iter 1 has 0 new positions
    cfg.setdefault("replay_buffer", {})["min_size"] = 1
    if cfg.get("league", {}).get("enabled"):
        cfg["league"]["gate_eval_games"] = 2
        cfg["league"]["suite_eval_games"] = 2
    return cfg


def _assert_no_nans(checkpoint_path: Path) -> None:
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", {})
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            if torch.isnan(v).any().item():
                raise RuntimeError(f"NaN detected in checkpoint tensor: {k}")
            if torch.isinf(v).any().item():
                raise RuntimeError(f"Inf detected in checkpoint tensor: {k}")


def run_smoke_test(base_cfg: dict) -> dict:
    from src.training.main import AlphaZeroTrainer

    with tempfile.TemporaryDirectory(prefix="pw_preflight_") as td:
        root = Path(td)
        tiny_cfg = _tiny_config(base_cfg, root)
        tiny_cfg_path = root / "preflight_tiny.yaml"
        tiny_cfg_path.write_text(yaml.safe_dump(tiny_cfg, sort_keys=False), encoding="utf-8")

        t0 = time.time()
        trainer = AlphaZeroTrainer(
            str(tiny_cfg_path),
            cli_run_dir=str(root / "runs" / "preflight_smoke"),
        )
        trainer.train(start_iteration=0, resume_checkpoint=None)
        smoke_time = time.time() - t0

        latest_ckpt = Path(tiny_cfg["paths"]["checkpoints_dir"]) / "latest_model.pt"
        if not latest_ckpt.exists():
            raise RuntimeError("Smoke run did not produce latest_model.pt")
        _assert_no_nans(latest_ckpt)

        best_ckpt = Path(tiny_cfg["paths"]["checkpoints_dir"]) / "best_model.pt"
        if best_ckpt.exists():
            _assert_no_nans(best_ckpt)

        return {
            "smoke_time_s": round(smoke_time, 1),
            "latest_checkpoint": str(latest_ckpt),
            "best_checkpoint_exists": best_ckpt.exists(),
            "status": "ok",
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Overnight preflight checks")
    parser.add_argument("--config", required=True, help="Base config path")
    parser.add_argument("--skip-smoke", action="store_true",
                        help="Skip the slow 2-iteration smoke test")
    args = parser.parse_args()

    base_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = int(base_cfg.get("seed", 42))
    all_ok = True

    # ---- Hardware ----
    print(f"\n{'='*50}")
    print("[1/6] Hardware & Environment")
    print(f"{'='*50}")
    hw_warnings = check_hardware(base_cfg)
    for w in hw_warnings:
        print(_yellow(f"  WARNING: {w}"))
        all_ok = False
    if not hw_warnings:
        print(_green("  All hardware checks passed."))

    # ---- Config ----
    print(f"\n{'='*50}")
    print("[2/6] Config Validation")
    print(f"{'='*50}")
    cfg_errors = validate_config(base_cfg)
    for e in cfg_errors:
        print(_red(f"  ERROR: {e}"))
        all_ok = False
    if not cfg_errors:
        print(_green("  Config is valid."))

    # ---- VRAM ----
    print(f"\n{'='*50}")
    print("[3/6] VRAM Estimation")
    print(f"{'='*50}")
    try:
        estimate_vram(base_cfg)
    except Exception as e:
        print(_yellow(f"  VRAM estimation failed: {e}"))

    # ---- Correctness ----
    print(f"\n{'='*50}")
    print("[4/6] Core Correctness (sanity checks)")
    print(f"{'='*50}")
    try:
        run_quick_sanity(seed, base_cfg)
        print(_green("  All sanity checks passed."))
    except (AssertionError, AssertionError) as e:
        print(_red(f"  FAILED: {e}"))
        all_ok = False

    # ---- Invariance ----
    print(f"\n{'='*50}")
    inv_skip_slow = args.skip_smoke
    print("[5/6] Invariance Tests (D4 + shop order)" + (" (fast only)" if inv_skip_slow else ""))
    print(f"{'='*50}")
    try:
        if run_invariance_tests(skip_slow=inv_skip_slow, cfg=base_cfg):
            print(_green("  All invariance tests passed."))
        else:
            print(_red("  Invariance tests FAILED — fix before training."))
            all_ok = False
    except subprocess.TimeoutExpired:
        print(_red("  Invariance tests timed out."))
        all_ok = False

    # ---- Smoke ----
    print(f"\n{'='*50}")
    print("[6/6] Pipeline Smoke Test (2 iterations on CPU)")
    print(f"{'='*50}")
    if args.skip_smoke:
        print("  Skipped (--skip-smoke)")
    else:
        try:
            result = run_smoke_test(base_cfg)
            print(_green(f"  Smoke test passed in {result['smoke_time_s']}s"))
            print(f"  {json.dumps(result, indent=2)}")
        except Exception as e:
            print(_red(f"  SMOKE TEST FAILED: {e}"))
            all_ok = False

    # ---- Summary ----
    print(f"\n{'='*50}")
    if all_ok:
        print(_green("PREFLIGHT PASSED — safe to start overnight run."))
    else:
        print(_yellow("PREFLIGHT COMPLETED WITH WARNINGS — review above."))
    print(f"{'='*50}\n")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
