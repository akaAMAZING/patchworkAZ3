"""
Build training_metrics_per_iteration.csv from all
runs/patchwork_production/committed/iter_*/iteration_*.json.

Includes EVERY metric found in iteration_XXX.json:
- train_metrics (policy_loss, value_loss, score_loss, ownership_loss,
  total_loss, policy_accuracy, policy_top5_accuracy, value_mse,
  grad_norm, policy_entropy, kl_divergence, ownership_accuracy,
  step_skip_rate, etc.)
- selfplay_stats (all keys)
- eval_results (all scalar fields from vs_previous_best and vs_pure_mcts)
- applied_settings (all nested keys for selfplay/training/replay/adaptive_games)

By default includes all committed iters. Use --max-iter N to only include
iterations 0..N (exclude synthetic or non-real iterations above N).
Use --min-iter M and optionally --output PATH to write a range (e.g. 70-86) to a separate file.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def flatten_dict(d: dict, prefix: str = "") -> dict:
    """Flatten nested dict; list values become string repr (truncated)."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten_dict(v, prefix=key + "_"))
        elif isinstance(v, list):
            out[key] = str(v)[:300]
        elif v is None or isinstance(v, (bool, int, float, str)):
            out[key] = v
        else:
            out[key] = str(v)[:300]
    return out


def main():
    parser = argparse.ArgumentParser(description="Build training_metrics_per_iteration.csv from committed iteration JSONs.")
    parser.add_argument(
        "--min-iter",
        type=int,
        default=None,
        metavar="M",
        help="Only include iterations M..max (skip iters < M). Use with --output for range export (e.g. 70-86).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=None,
        metavar="N",
        help="Only include iterations 0..N (exclude iters > N, e.g. synthetic). Default: include all.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write CSV to this path instead of training_metrics_per_iteration.csv. Can be relative to this script's dir.",
    )
    args = parser.parse_args()

    run_root = Path(__file__).resolve().parents[2] / "runs" / "patchwork_production" / "committed"
    if not run_root.exists():
        run_root = Path(__file__).resolve().parents[1] / ".." / "runs" / "patchwork_production" / "committed"
        run_root = run_root.resolve()

    rows: list[dict] = []
    all_keys_ordered: list[str] = []

    # Discover all iteration_XXX.json files; optionally filter by min/max_iter.
    json_paths = sorted(run_root.glob("iter_*/iteration_*.json"))
    for p in json_paths:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        i = data.get("iteration", -1)
        if args.min_iter is not None and i < args.min_iter:
            continue
        if args.max_iter is not None and i > args.max_iter:
            continue

        row = {}

        # Top-level
        row["iteration"] = data["iteration"]
        row["accepted"] = data.get("accepted", "")
        row["iteration_time_s"] = data.get("iteration_time_s")
        row["replay_buffer_positions"] = data.get("replay_buffer_positions")
        row["replay_buffer_iterations"] = data.get("replay_buffer_iterations")

        # train_metrics — every key (KL, grad_norm, policy_entropy, etc.)
        for k, v in (data.get("train_metrics") or {}).items():
            row[f"train_{k}"] = v

        # selfplay_stats — every key
        for k, v in (data.get("selfplay_stats") or {}).items():
            row[f"selfplay_{k}"] = v

        # eval_results — every scalar/key from vs_previous_best and vs_pure_mcts (omit raw 'results' list)
        for baseline in ("vs_previous_best", "vs_pure_mcts"):
            base = (data.get("eval_results") or {}).get(baseline) or {}
            for k, v in base.items():
                if k == "results":
                    continue
                row[f"eval_{baseline}_{k}"] = v

        # applied_settings — flatten every nested key
        app = data.get("applied_settings") or {}
        for top_key, top_val in app.items():
            if not isinstance(top_val, dict):
                row[f"applied_{top_key}"] = top_val
                continue
            for k, v in top_val.items():
                if isinstance(v, list):
                    row[f"applied_{top_key}_{k}"] = str(v)[:200]
                elif isinstance(v, dict):
                    for k2, v2 in v.items():
                        row[f"applied_{top_key}_{k}_{k2}"] = v2 if not isinstance(v2, (list, dict)) else str(v2)[:200]
                else:
                    row[f"applied_{top_key}_{k}"] = v

        for k in row:
            if k not in all_keys_ordered:
                all_keys_ordered.append(k)
        rows.append(row)

    # Sort so iteration is first, then stable column order
    priority = ["iteration", "accepted", "iteration_time_s", "replay_buffer_positions", "replay_buffer_iterations"]
    rest = [k for k in all_keys_ordered if k not in priority]
    # Group by prefix for readability
    train_keys = sorted([k for k in rest if k.startswith("train_")])
    selfplay_keys = sorted([k for k in rest if k.startswith("selfplay_")])
    eval_keys = sorted([k for k in rest if k.startswith("eval_")])
    applied_keys = sorted([k for k in rest if k.startswith("applied_")])
    other = [k for k in rest if k not in train_keys + selfplay_keys + eval_keys + applied_keys]
    fieldnames = priority + other + train_keys + selfplay_keys + eval_keys + applied_keys

    out_path = args.output
    if out_path is None:
        out_path = Path(__file__).parent / "training_metrics_per_iteration.csv"
    else:
        out_path = Path(out_path)
        if not out_path.is_absolute():
            out_path = Path(__file__).parent / out_path
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    iters = [r["iteration"] for r in rows]
    range_note = []
    if args.min_iter is not None:
        range_note.append(f"min_iter={args.min_iter}")
    if args.max_iter is not None:
        range_note.append(f"max_iter={args.max_iter}")
    range_note = " " + " ".join(range_note) if range_note else ""
    if rows:
        print(
            f"Wrote {out_path} with {len(rows)} rows "
            f"(iterations {min(iters)}–{max(iters)}){range_note}, {len(fieldnames)} columns."
        )
    else:
        print(f"No rows written (no iterations match).{range_note}")
    return out_path


if __name__ == "__main__":
    main()
