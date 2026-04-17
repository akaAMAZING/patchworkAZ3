#!/usr/bin/env python3
"""
Windows CUDA probe — minimal repro for GPU server startup crash (nvcuda64.dll).

Imports the same GPU stack as the real GPU inference server and optionally runs
the first CUDA query and a light model load. Use to isolate whether the crash
is in torch import, first CUDA init, or model load.

Usage:
  python scripts/windows_cuda_probe.py
  python scripts/windows_cuda_probe.py --cuda-query
  python scripts/windows_cuda_probe.py --cuda-query --model-load CONFIG CHECKPOINT
  python scripts/windows_cuda_probe.py --log probe.log

Exit: 0 on success, nonzero on failure (with message to stderr).
"""

from __future__ import annotations

import argparse
import sys
import os


def _log(msg: str, log_path: str | None) -> None:
    line = f"[probe] {msg}\n"
    sys.stdout.write(line)
    sys.stdout.flush()
    if log_path:
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Windows CUDA probe for GPU server crash diagnosis")
    parser.add_argument("--cuda-query", action="store_true", help="Run first CUDA query (is_available, device)")
    parser.add_argument("--model-load", nargs=2, metavar=("CONFIG", "CHECKPOINT"),
                        help="Load config YAML and checkpoint (light model load like GPU server)")
    parser.add_argument("--log", type=str, default=None, help="Append progress to this log file")
    args = parser.parse_args()

    log_path = args.log
    _log("1_probe_start", log_path)

    try:
        _log("2_import_torch_begin", log_path)
        import torch  # noqa: E402
        _log("3_import_torch_complete", log_path)
    except Exception as e:
        _log(f"FAIL_import_torch: {e}", log_path)
        print(f"FAIL: import torch: {e}", file=sys.stderr)
        return 1

    if args.cuda_query:
        try:
            _log("4_first_cuda_query_begin", log_path)
            avail = torch.cuda.is_available()
            _log(f"5_cuda_is_available={avail}", log_path)
            if avail:
                dev = torch.device("cuda")
                _log("6_torch_device_cuda_ok", log_path)
                # One more touch: device count
                n = torch.cuda.device_count()
                _log(f"7_cuda_device_count={n}", log_path)
            _log("8_first_cuda_query_complete", log_path)
        except Exception as e:
            _log(f"FAIL_cuda_query: {e}", log_path)
            print(f"FAIL: CUDA query: {e}", file=sys.stderr)
            return 2

    if args.model_load:
        config_path, checkpoint_path = args.model_load
        if not os.path.isfile(config_path):
            _log(f"FAIL_config_missing: {config_path}", log_path)
            print(f"FAIL: config not found: {config_path}", file=sys.stderr)
            return 3
        if not os.path.isfile(checkpoint_path):
            _log(f"FAIL_checkpoint_missing: {checkpoint_path}", log_path)
            print(f"FAIL: checkpoint not found: {checkpoint_path}", file=sys.stderr)
            return 4
        try:
            _log("9_model_load_begin", log_path)
            import yaml  # noqa: E402
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            # Same imports as gpu_inference_server
            from src.network.model import create_network, load_model_checkpoint, get_state_dict_for_inference  # noqa: E402
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = create_network(config).to(device)
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            state_dict = get_state_dict_for_inference(ckpt, config, for_selfplay=True)
            load_model_checkpoint(model, state_dict)
            model.eval()
            _log("10_model_load_complete", log_path)
        except Exception as e:
            _log(f"FAIL_model_load: {e}", log_path)
            print(f"FAIL: model load: {e}", file=sys.stderr)
            return 5

    _log("11_probe_ok", log_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
