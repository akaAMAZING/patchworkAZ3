#!/usr/bin/env python
"""Unified dev tools: print-config, one-game, bench.

Usage:
    python tools/dev_tools.py print-config --config configs/config_best.yaml
    python tools/dev_tools.py one-game --config configs/config_best.yaml [--gpu-server]
    python tools/dev_tools.py bench --config configs/config_best.yaml [--quick] [--output file.json]
"""

from __future__ import annotations

import argparse
import copy
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def cmd_print_config(args) -> None:
    """Print resolved runtime config values."""
    import yaml
    import torch

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    data = cfg.get("data", {}) or {}
    enc = str(data.get("encoding_version", "") or "").strip()
    network = cfg.get("network", {}) or {}
    use_film = bool(network.get("use_film", False))
    sp = cfg.get("selfplay", {}) or {}
    api_url = sp.get("api_url")
    api_enabled = api_url is not None and str(api_url).strip() not in ("", "null", "none")
    num_workers = int(sp.get("num_workers", 1))
    cuda = torch.cuda.is_available()
    hw_device = str((cfg.get("hardware", {}) or {}).get("device", "cpu"))
    use_cuda = hw_device == "cuda" and cuda
    eval_client_would_be_used = num_workers > 1 and use_cuda

    print("Resolved config values:")
    print(f"  encoding_version:      {enc!r}")
    print(f"  use_film:             {use_film}")
    print(f"  api_url:              {api_url!r}")
    print(f"  api_url enabled:      {api_enabled}")
    print(f"  num_workers:          {num_workers}")
    print(f"  CUDA available:       {cuda}")
    print(f"  hardware.device:      {hw_device}")
    print(f"  using GPU:           {use_cuda}")
    print(f"  selfplay inference:  {'GPU server (eval_client)' if eval_client_would_be_used else 'local (in-process)'}")
    if enc.lower() in ("gold_v2_32ch", "gold_v2_multimodal") and eval_client_would_be_used:
        print("  OK: gold_v2 + GPU server (eval_client) supported")
    elif enc.lower() in ("gold_v2_32ch", "gold_v2_multimodal"):
        print("  OK: gold_v2 (local inference)")
    else:
        print("  OK")


def cmd_one_game(args) -> None:
    """Run a single selfplay game (bootstrap or with model + GPU server)."""
    import torch
    import yaml

    from src.network.model import create_network
    from src.training.selfplay_optimized_integration import SelfPlayGenerator

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    cfg = copy.deepcopy(cfg)
    cfg["selfplay"]["api_url"] = None

    if args.gpu_server:
        if not torch.cuda.is_available():
            raise RuntimeError("--gpu-server requires CUDA")
        cfg["selfplay"]["num_workers"] = 14
        cfg["hardware"]["device"] = "cuda"
        use_checkpoint = True
    else:
        cfg["selfplay"]["num_workers"] = 1
        use_checkpoint = False

    cfg.setdefault("selfplay", {})["games_per_iteration"] = 1
    cfg.setdefault("selfplay", {}).setdefault("bootstrap", {})["games"] = 1
    cfg.setdefault("paths", {})

    with tempfile.TemporaryDirectory(prefix="pw_one_game_") as td:
        td_path = Path(td)
        cfg["paths"]["checkpoints_dir"] = str(td_path / "checkpoints")
        cfg["paths"]["logs_dir"] = str(td_path / "logs")
        cfg["paths"]["selfplay_dir"] = str(td_path / "data" / "selfplay")
        run_dir = td_path / "runs" / "one_game"
        run_dir.mkdir(parents=True)
        out_dir = run_dir / "iter_000"
        out_dir.mkdir(parents=True)

        network_path = None
        if use_checkpoint:
            ckpt_dir = Path(cfg["paths"]["checkpoints_dir"])
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            net = create_network(cfg)
            ckpt_path = ckpt_dir / "test_model.pt"
            torch.save({"model_state_dict": net.state_dict()}, ckpt_path)
            network_path = str(ckpt_path)

        gen = SelfPlayGenerator(cfg)
        t0 = time.time()
        data_path, stats = gen.generate(
            iteration=1 if use_checkpoint else 0,
            network_path=network_path,
            output_dir=out_dir,
        )
        elapsed = time.time() - t0

    mode = "GPU server (num_workers=14)" if args.gpu_server else "local (num_workers=1)"
    print(f"Completed 1 game in {elapsed:.1f}s ({mode})")
    print(f"  Positions: {stats.get('num_positions', 0)}")
    print("OK: selfplay pipeline works with config")


def cmd_bench(args) -> None:
    """Run benchmark (delegates to benchmark.py)."""
    cmd = [sys.executable, str(REPO_ROOT / "tools" / "benchmark.py"), "--config", args.config]
    if args.quick:
        cmd.append("--quick")
    if args.output:
        cmd.extend(["--output", args.output])
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Patchwork dev tools")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # print-config
    p = subparsers.add_parser("print-config", help="Print resolved config values")
    p.add_argument("--config", required=True, help="Config YAML path")
    p.set_defaults(func=cmd_print_config)

    # one-game
    p = subparsers.add_parser("one-game", help="Run 1 selfplay game")
    p.add_argument("--config", required=True, help="Config YAML path")
    p.add_argument("--gpu-server", action="store_true", help="Use GPU server (14 workers)")
    p.set_defaults(func=cmd_one_game)

    # bench
    p = subparsers.add_parser("bench", help="Run benchmark")
    p.add_argument("--config", default="configs/config_best.yaml", help="Config YAML path")
    p.add_argument("--quick", action="store_true", help="Quick benchmark subset")
    p.add_argument("--output", default="", help="Optional JSON output file")
    p.set_defaults(func=cmd_bench)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
