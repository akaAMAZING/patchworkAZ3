#!/usr/bin/env python3
"""
Export all run metrics from metadata.jsonl and TensorBoard events
to comprehensive CSVs for downstream AI consumption.
Output: patchworkaz_root/run_metrics_export/
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "run_metrics_export"
METADATA_PATH = ROOT / "logs" / "metadata.jsonl"
TB_DIR = ROOT / "logs" / "tensorboard"
COMMITTED_DIR = ROOT / "runs" / "patchwork_production" / "committed"


def flatten_dict(d: dict, prefix: str = "") -> dict:
    """Flatten nested dict for CSV columns."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict) and not (v and isinstance(next(iter(v.values())), dict)):
            out.update(flatten_dict(v, f"{key}_"))
        else:
            out[key] = v
    return out


def load_metadata() -> list[dict]:
    """Load all iteration records from metadata.jsonl."""
    rows = []
    with open(METADATA_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def extract_tb_scalars(tb_dir: Path) -> dict[str, list[tuple[int | float, float]]]:
    """
    Extract scalar values from TensorBoard events.
    Returns: {tag: [(step_or_iter, value), ...]}
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return {}

    if not tb_dir.exists():
        return {}

    result: dict[str, list[tuple[int | float, float]]] = {}
    event_files = sorted(tb_dir.glob("events.out.tfevents.*"))

    for ef in event_files:
        try:
            ea = EventAccumulator(str(ef))
            ea.Reload()

            for tag in ea.Tags().get("scalars", []):
                if tag not in result:
                    result[tag] = []
                for e in ea.Scalars(tag):
                    result[tag].append((e.step, e.value))
        except Exception:
            continue

    # Sort each tag's values by step
    for tag in result:
        result[tag] = sorted(result[tag], key=lambda x: x[0])

    return result


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Metadata CSV (per-iteration) ---
    meta_rows = load_metadata()
    if not meta_rows:
        print("No metadata found.", file=sys.stderr)
        sys.exit(1)

    # Build flat columns from first row
    flat_sample = {}
    for r in meta_rows:
        for k, v in r.items():
            if k in ("train", "selfplay") and isinstance(v, dict):
                for sk, sv in v.items():
                    flat_sample[f"{k}_{sk}"] = None
            elif k not in ("train", "selfplay"):
                flat_sample[k] = None

    meta_columns = ["iteration", "timestamp_utc", "config_hash", "config_path", "best_model_hash",
                   "accepted", "global_step", "iter_time_s", "replay_positions", "consecutive_rejections",
                   "best_model",
                   "eval_vs_best_wr", "eval_vs_best_margin", "eval_vs_mcts_wr"]
    train_cols = [c for c in sorted(flat_sample) if c.startswith("train_")]
    selfplay_cols = [c for c in sorted(flat_sample) if c.startswith("selfplay_")]
    meta_columns = meta_columns + train_cols + selfplay_cols

    meta_path = OUT_DIR / "iter_metrics.csv"
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=meta_columns, extrasaction="ignore")
        w.writeheader()
        for r in meta_rows:
            row = {"iteration": r["iteration"], "timestamp_utc": r["timestamp_utc"],
                   "config_hash": r.get("config_hash"), "config_path": r.get("config_path"),
                   "best_model_hash": r.get("best_model_hash"), "accepted": r.get("accepted"),
                   "global_step": r.get("global_step"), "iter_time_s": r.get("iter_time_s"),
                   "replay_positions": r.get("replay_positions"),
                   "consecutive_rejections": r.get("consecutive_rejections"),
                   "best_model": r.get("best_model"),
                   "eval_vs_best_wr": r.get("eval_vs_best_wr"),
                   "eval_vs_best_margin": r.get("eval_vs_best_margin"),
                   "eval_vs_mcts_wr": r.get("eval_vs_mcts_wr")}
            if "train" in r:
                for k, v in r["train"].items():
                    row[f"train_{k}"] = v
            if "selfplay" in r:
                for k, v in r["selfplay"].items():
                    row[f"selfplay_{k}"] = v
            w.writerow(row)

    print(f"Wrote {meta_path} ({len(meta_rows)} rows)")

    # --- 1b. Filtered iter25-33 CSV ---
    filtered = [r for r in meta_rows if 25 <= r["iteration"] <= 33]
    if filtered:
        filtered_path = OUT_DIR / "iter_metrics_25_33.csv"
        with open(filtered_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=meta_columns, extrasaction="ignore")
            w.writeheader()
            for r in filtered:
                row = {"iteration": r["iteration"], "timestamp_utc": r["timestamp_utc"],
                       "config_hash": r.get("config_hash"), "config_path": r.get("config_path"),
                       "best_model_hash": r.get("best_model_hash"), "accepted": r.get("accepted"),
                       "global_step": r.get("global_step"), "iter_time_s": r.get("iter_time_s"),
                       "replay_positions": r.get("replay_positions"),
                       "consecutive_rejections": r.get("consecutive_rejections"),
                       "best_model": r.get("best_model"),
                       "eval_vs_best_wr": r.get("eval_vs_best_wr"),
                       "eval_vs_best_margin": r.get("eval_vs_best_margin"),
                       "eval_vs_mcts_wr": r.get("eval_vs_mcts_wr")}
                if "train" in r:
                    for k, v in r["train"].items():
                        row[f"train_{k}"] = v
                if "selfplay" in r:
                    for k, v in r["selfplay"].items():
                        row[f"selfplay_{k}"] = v
                w.writerow(row)
        print(f"Wrote {filtered_path} ({len(filtered)} rows, iter25-33)")

    # --- 2. TensorBoard scalars CSV (per-step / per-iteration) ---
    tb_data = extract_tb_scalars(TB_DIR)
    if tb_data:
        # Collect all unique (tag, step) -> value
        all_steps: set[int | float] = set()
        for tag, points in tb_data.items():
            for step, _ in points:
                all_steps.add(step)

        tb_path = OUT_DIR / "tensorboard_scalars.csv"
        tags_sorted = sorted(tb_data.keys())
        with open(tb_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["step_or_iteration", "tag", "value"])
            for tag in tags_sorted:
                for step, val in tb_data[tag]:
                    w.writerow([step, tag, val])
        print(f"Wrote {tb_path} (TensorBoard scalars)")
    else:
        print("TensorBoard extraction skipped (no tensorboard or no events).")

    # --- 3. Training epochs from training.log (committed iters) ---
    epoch_rows = []
    for d in sorted(COMMITTED_DIR.iterdir()):
        if not d.is_dir() or not d.name.startswith("iter_"):
            continue
        m = re.match(r"iter_(\d+)", d.name)
        if not m:
            continue
        iter_num = int(m.group(1))
        log_path = d / "training.log"
        if not log_path.exists():
            continue
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                # Epoch 1 done in 271.8s | loss=3.0831  pol_loss=2.7053  val_loss=0.3222  own_loss=0.3045  own_acc=85.2% | pol_acc=53.7%  top5=76.8% | val_mse=0.3222  grad=0.425 | LR=1.60e-03
                mo = re.search(
                    r"Epoch (\d+) done in ([\d.]+)s \| loss=([\d.]+)\s+pol_loss=([\d.]+)\s+val_loss=([\d.]+)\s+own_loss=([\d.]+)\s+own_acc=([\d.]+)%\s*\|\s*pol_acc=([\d.]+)%\s+top5=([\d.]+)%\s*\|\s*val_mse=([\d.]+)\s+grad=([\d.]+)",
                    line,
                )
                if mo:
                    epoch_rows.append({
                        "iteration": iter_num,
                        "epoch": int(mo.group(1)),
                        "epoch_time_s": float(mo.group(2)),
                        "loss": float(mo.group(3)),
                        "pol_loss": float(mo.group(4)),
                        "val_loss": float(mo.group(5)),
                        "own_loss": float(mo.group(6)),
                        "own_acc_pct": float(mo.group(7)),
                        "pol_acc_pct": float(mo.group(8)),
                        "top5_pct": float(mo.group(9)),
                        "val_mse": float(mo.group(10)),
                        "grad": float(mo.group(11)),
                    })

    if epoch_rows:
        cols = ["iteration", "epoch", "epoch_time_s", "loss", "pol_loss", "val_loss", "own_loss", "own_acc_pct", "pol_acc_pct", "top5_pct", "val_mse", "grad"]
        epoch_path = OUT_DIR / "training_epochs.csv"
        with open(epoch_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in epoch_rows:
                w.writerow({c: r.get(c, "") for c in cols})
        print(f"Wrote {epoch_path} ({len(epoch_rows)} rows)")
        # Filtered iter25-33 epochs
        epoch_25_33 = [r for r in epoch_rows if 25 <= r["iteration"] <= 33]
        if epoch_25_33:
            ep_path = OUT_DIR / "training_epochs_25_33.csv"
            with open(ep_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
                w.writeheader()
                for r in epoch_25_33:
                    w.writerow({c: r.get(c, "") for c in cols})
            print(f"Wrote {ep_path} ({len(epoch_25_33)} rows, iter25-33)")

    # --- 4. Run/ELO state (context) ---
    for name, path in [
        ("run_state", ROOT / "runs" / "patchwork_production" / "run_state.json"),
        ("elo_state", ROOT / "runs" / "patchwork_production" / "elo_state.json"),
        ("environment", ROOT / "logs" / "environment.json"),
    ]:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            out_path = OUT_DIR / f"{name}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"Wrote {out_path}")

    # --- 5. README for AI consumers ---
    readme = OUT_DIR / "README.md"
    max_iter = max(r["iteration"] for r in meta_rows) if meta_rows else 0
    readme.write_text(f"""# Patchwork AZ Run Metrics Export

All metrics from runs iter000 through iter{max_iter}. Source: logs/metadata.jsonl, logs/tensorboard/, runs/patchwork_production/committed/.

## Files

- **iter_metrics.csv**: One row per iteration (0–{max_iter}). Columns:
  - `iteration`, `timestamp_utc`, `config_hash`, `config_path`, `best_model_hash`
  - `accepted`, `global_step`, `iter_time_s`, `replay_positions`, `consecutive_rejections`
  - `best_model`, `eval_vs_best_wr`, `eval_vs_best_margin`, `eval_vs_mcts_wr`
  - `train_*`: policy_loss, value_loss, score_loss, ownership_loss, total_loss, policy_accuracy, policy_top5_accuracy, value_mse, grad_norm, policy_entropy, kl_divergence, ownership_accuracy, step_skip_rate
  - `selfplay_*`: num_games, num_positions, avg_game_length, p0_wins, p1_wins, generation_time, games_per_minute, avg_policy_entropy, avg_top1_prob, avg_num_legal, avg_redundancy, unique_positions, avg_root_q

- **tensorboard_scalars.csv**: TensorBoard scalar events. Columns: `step_or_iteration`, `tag`, `value`.
  - Tags include: train/* (per-step), val/*, iter/*, selfplay/*, buffer/*, eval/*

- **training_epochs.csv**: Per-epoch training metrics from training.log. Columns: iteration, epoch, epoch_time_s, loss, pol_loss, val_loss, own_loss, own_acc_pct, pol_acc_pct, top5_pct, val_mse, grad.

- **iter_metrics_25_33.csv**: Filtered to iterations 25–33 only (same columns as iter_metrics.csv).
- **training_epochs_25_33.csv**: Filtered to iterations 25–33 only (same columns as training_epochs.csv).

- **run_state.json**, **elo_state.json**, **environment.json**: Run context (hardware, config, ELO state).

## Source

- Metadata: `logs/metadata.jsonl`
- TensorBoard: `logs/tensorboard/`
- Training logs: `runs/patchwork_production/committed/iter_*/training.log`
""", encoding="utf-8")
    print(f"Wrote {readme}")


if __name__ == "__main__":
    main()
