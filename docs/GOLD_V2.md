# Gold v2 Multimodal — Run Guide

Single reference for running preflight + training with `gold_v2_multimodal` on config_best.yaml.

## Quick Start

```bash
# 1. Preflight (hardware, config, invariance tests, 2-iter smoke)
python tools/preflight.py --config configs/config_best.yaml

# 2. Training (full run)
python -m src.training.main --config configs/config_best.yaml

# Short run (1 iteration)
python -m src.training.main --config configs/config_best.yaml --iterations 1 --run-dir runs/test_1iter
```

## Dev Tools

```bash
# Print resolved config values
python tools/dev_tools.py print-config --config configs/config_best.yaml

# Run 1 selfplay game (local or GPU server)
python tools/dev_tools.py one-game --config configs/config_best.yaml
python tools/dev_tools.py one-game --config configs/config_best.yaml --gpu-server

# Benchmark
python tools/dev_tools.py bench --config configs/config_best.yaml [--quick] [--output results.json]
```

## Invariants (Protected by Tests)

These must hold; preflight and CI run them:

| Test | Invariant |
|------|-----------|
| `test_shop_order_markov_alignment` | Shop order alignment |
| `test_d4_augmentation` | D4 spatial + mask remap |
| `test_d4_action_equivariance` | D4 action equivariance (buy + patch) |
| `test_gold_v2_encoding` | legalTL == buy-mask (canonical + after D4), forward shapes |

## Architecture

- **Encoding**: `gold_v2_multimodal` — 56 spatial channels, FiLM conditioning (global + track + shop)
- **Action space**: MAX_ACTIONS=2026, BUY_START=82 (unchanged)
- **D4**: Applied to spatial + policy/mask remap; legalTL via scatter-map
- **Selfplay**: Queue-based GPU server (eval_client) when num_workers > 1; 14 workers on config_best
