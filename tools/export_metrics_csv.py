#!/usr/bin/env python
"""
Export full training metrics history to detailed CSV.

Reads metadata.jsonl (and optionally TensorBoard events) and outputs a flat CSV
with all metrics from iter000 to iterN - everything TensorBoard and metadata log.

Usage:
  python tools/export_metrics_csv.py --config configs/config_best.yaml
  python tools/export_metrics_csv.py --metadata logs/metadata.jsonl -o logs/metrics_history.csv
  python tools/export_metrics_csv.py --config configs/config_best.yaml --iters 0-33 --output logs/metrics_iter000_033.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _flatten_dict(d: dict, prefix: str = "") -> dict:
    """Flatten nested dict into prefix_key format. Skip dict/list values."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}_{k}" if prefix else k
        if v is None:
            out[key] = ""
        elif isinstance(v, dict) and not (prefix and k in ("sprt",)):
            # Recursively flatten, except sprt which we handle specially
            out.update(_flatten_dict(v, key))
        elif isinstance(v, (list, tuple)) and len(str(v)) > 100:
            out[key] = ""  # Skip huge lists
        else:
            out[key] = v
    return out


def _parse_iter_range(s: str) -> tuple[int, int]:
    """Parse '0-33' or '0' -> (0, 33) or (0, 0)."""
    if "-" in s:
        a, b = s.split("-", 1)
        return int(a.strip()), int(b.strip())
    n = int(s.strip())
    return n, n


def load_metadata(path: Path, iter_start: int, iter_end: int) -> list[dict]:
    """Load and filter metadata.jsonl by iteration range."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            it = entry.get("iteration", -1)
            if iter_start <= it <= iter_end:
                rows.append(entry)
    return rows


def build_flat_row(entry: dict) -> dict:
    """Convert one metadata entry to flat dict for CSV."""
    row = {
        "iteration": entry.get("iteration", ""),
        "timestamp_utc": entry.get("timestamp_utc", ""),
        "config_hash": entry.get("config_hash", ""),
        "accepted": entry.get("accepted", ""),
        "global_step": entry.get("global_step", ""),
        "iter_time_s": entry.get("iter_time_s", ""),
        "replay_positions": entry.get("replay_positions", ""),
        "consecutive_rejections": entry.get("consecutive_rejections", ""),
        "best_model": entry.get("best_model", ""),
        "eval_vs_best_wr": entry.get("eval_vs_best_wr", ""),
        "eval_vs_best_margin": entry.get("eval_vs_best_margin", ""),
        "eval_vs_mcts_wr": entry.get("eval_vs_mcts_wr", ""),
        "elo": entry.get("elo", ""),
    }
    # Flatten train.*
    train = entry.get("train") or {}
    for k, v in train.items():
        row[f"train_{k}"] = v
    # Flatten selfplay.*
    sp = entry.get("selfplay") or {}
    for k, v in sp.items():
        row[f"selfplay_{k}"] = v
    # Flatten sprt if present
    sprt = entry.get("sprt")
    if isinstance(sprt, dict):
        for k, v in sprt.items():
            row[f"sprt_{k}"] = v
    return row


def get_all_columns(rows: list[dict]) -> list[str]:
    """Collect all unique column names preserving order (iteration first)."""
    seen = set()
    cols = []
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                cols.append(k)
    # Prefer canonical order
    canonical = [
        "iteration", "timestamp_utc", "config_hash", "accepted", "global_step",
        "iter_time_s", "replay_positions", "consecutive_rejections", "best_model",
        "eval_vs_best_wr", "eval_vs_best_margin", "eval_vs_mcts_wr", "elo",
    ]
    train_keys = ["train_policy_loss", "train_value_loss", "train_ownership_loss",
                  "train_total_loss", "train_policy_accuracy", "train_policy_top5_accuracy",
                  "train_value_mse", "train_grad_norm", "train_policy_entropy",
                  "train_kl_divergence", "train_ownership_accuracy", "train_step_skip_rate"]
    sp_keys = ["selfplay_num_games", "selfplay_num_positions", "selfplay_avg_game_length",
               "selfplay_p0_wins", "selfplay_p1_wins", "selfplay_generation_time",
               "selfplay_games_per_minute", "selfplay_avg_policy_entropy",
               "selfplay_avg_top1_prob", "selfplay_avg_num_legal", "selfplay_avg_redundancy",
               "selfplay_unique_positions", "selfplay_avg_root_q"]
    sprt_keys = ["sprt_accept", "sprt_reject", "sprt_llr", "sprt_games"]
    ordered = [c for c in canonical + train_keys + sp_keys + sprt_keys if c in seen]
    remainder = [c for c in cols if c not in ordered]
    return ordered + remainder


def write_csv(rows: list[dict], output_path: Path) -> None:
    """Write flat rows to CSV with consistent columns."""
    if not rows:
        raise ValueError("No rows to write")
    cols = get_all_columns(rows)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"Wrote {len(rows)} rows to {output_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export full training metrics (iter000 to iterN) to detailed CSV"
    )
    ap.add_argument("--config", type=str, default=None, help="Config path (used to find logs dir)")
    ap.add_argument("--metadata", type=str, default=None, help="Path to metadata.jsonl (overrides config)")
    ap.add_argument("--output", "-o", type=str, default=None, help="Output CSV path")
    ap.add_argument("--iters", type=str, default="0-33", help="Iteration range, e.g. 0-33 or 0")
    args = ap.parse_args()

    # Resolve metadata path
    if args.metadata:
        meta_path = Path(args.metadata)
    elif args.config:
        import yaml
        cfg = Path(args.config)
        if not cfg.exists():
            raise SystemExit(f"Config not found: {cfg}")
        with open(cfg, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logs_dir = Path(config.get("paths", {}).get("logs_dir", "logs"))
        meta_path = logs_dir / "metadata.jsonl"
    else:
        meta_path = Path("logs/metadata.jsonl")

    if not meta_path.exists():
        raise SystemExit(f"Metadata not found: {meta_path}")

    # Resolve output path
    if args.output:
        out_path = Path(args.output)
    else:
        iter_start, iter_end = _parse_iter_range(args.iters)
        out_path = meta_path.parent / f"metrics_iter{iter_start:03d}_{iter_end:03d}.csv"

    iter_start, iter_end = _parse_iter_range(args.iters)
    rows = load_metadata(meta_path, iter_start, iter_end)
    if not rows:
        raise SystemExit(f"No metadata entries in range iter {iter_start}-{iter_end}")

    flat_rows = [build_flat_row(e) for e in rows]
    write_csv(flat_rows, out_path)


if __name__ == "__main__":
    main()
