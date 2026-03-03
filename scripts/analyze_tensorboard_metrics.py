#!/usr/bin/env python3
"""
Load TensorBoard event files (from logs/tensorboard or runs/.../committed/iter_*/tensorboard),
aggregate all scalars, and print a metrics report with a "solid training?" checklist.
Usage: python scripts/analyze_tensorboard_metrics.py [path_to_tensorboard_dir]
       Default path: logs/tensorboard (or merged from committed iters if empty).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TB = ROOT / "logs" / "tensorboard"
COMMITTED = ROOT / "runs" / "patchwork_production" / "committed"


def collect_event_dirs(tb_dir: Path) -> list[Path]:
    """Collect all dirs that contain events.out.tfevents.*."""
    if not tb_dir.exists():
        return []
    dirs = []
    for f in tb_dir.rglob("events.out.tfevents.*"):
        dirs.append(f.parent)
    return sorted(set(dirs))


def load_scalars(tb_dir: Path) -> dict[str, list[tuple[float, float]]]:
    """Load all scalars from event dirs. Returns {tag: [(step, value), ...]}."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return {}

    by_tag: dict[str, list[tuple[float, float]]] = {}
    for d in collect_event_dirs(tb_dir):
        for ef in sorted(d.glob("events.out.tfevents.*")):
            try:
                ea = EventAccumulator(str(ef))
                ea.Reload()
                for tag in ea.Tags().get("scalars", []):
                    if tag not in by_tag:
                        by_tag[tag] = []
                    for e in ea.Scalars(tag):
                        by_tag[tag].append((e.step, e.value))
            except Exception:
                continue

    for tag in by_tag:
        by_tag[tag] = sorted(by_tag[tag], key=lambda x: x[0])
    return by_tag


def merge_committed_tensorboard() -> Path:
    """If logs/tensorboard is empty, merge committed iter tensorboard dirs into a temp view."""
    if DEFAULT_TB.exists() and collect_event_dirs(DEFAULT_TB):
        return DEFAULT_TB
    if not COMMITTED.exists():
        return DEFAULT_TB
    # Return first committed iter with tensorboard (script will only read; we pass COMMITTED and scan)
    return COMMITTED


def analyze(scalars: dict[str, list[tuple[float, float]]]) -> None:
    """Print report and solid-training checklist."""
    if not scalars:
        print("No scalar data found.")
        return

    # Group by prefix
    groups: dict[str, list[str]] = {}
    for tag in sorted(scalars.keys()):
        prefix = tag.split("/")[0] if "/" in tag else "misc"
        groups.setdefault(prefix, []).append(tag)

    print("=" * 60)
    print("TENSORBOARD METRICS SUMMARY")
    print("=" * 60)

    for group in ["train", "val", "eval", "selfplay", "buffer", "iter", "league", "loss", "score", "ratio", "gradnorm"]:
        if group not in groups:
            continue
        print(f"\n--- {group.upper()} ---")
        for tag in groups[group]:
            pts = scalars[tag]
            steps = [p[0] for p in pts]
            vals = [p[1] for p in pts]
            n = len(vals)
            last = vals[-1]
            mn, mx = min(vals), max(vals)
            print(f"  {tag}")
            print(f"    steps: {steps[0]:.0f} .. {steps[-1]:.0f}  count={n}  last={last:.5g}  min={mn:.5g}  max={mx:.5g}")

    # Other groups
    for group in sorted(groups.keys()):
        if group in ["train", "val", "eval", "selfplay", "buffer", "iter", "league", "loss", "score", "ratio", "gradnorm"]:
            continue
        print(f"\n--- {group.upper()} ---")
        for tag in groups[group]:
            pts = scalars[tag]
            vals = [p[1] for p in pts]
            print(f"  {tag}  count={len(vals)}  last={vals[-1]:.5g}  min={min(vals):.5g}  max={max(vals):.5g}")

    # Solid-training checklist
    print("\n" + "=" * 60)
    print("SOLID TRAINING CHECKLIST (interpretation)")
    print("=" * 60)

    def get(tag: str) -> list[tuple[float, float]]:
        return scalars.get(tag, [])

    def last_val(tag: str) -> float | None:
        p = get(tag)
        return p[-1][1] if p else None

    checks = []

    # Train loss: should decrease over time (or stabilize)
    tl = get("train/total_loss")
    if tl:
        trend = "down" if len(tl) >= 2 and tl[-1][1] < tl[0][1] else "flat/up"
        checks.append(("train/total_loss", f"last={tl[-1][1]:.4f}, trend={trend}", trend == "down" or tl[-1][1] < 2.0))

    # Grad norm: not zero (learning), not huge (stable)
    gn = last_val("train/grad_norm")
    if gn is not None:
        ok = 1e-4 < gn < 100
        checks.append(("train/grad_norm", f"{gn:.4f}", ok))

    # Policy accuracy: should improve (e.g. > 0.2 and improving)
    pa = get("train/policy_accuracy")
    if pa:
        last_pa = pa[-1][1]
        checks.append(("train/policy_accuracy", f"last={last_pa:.4f}", last_pa > 0.15))

    # Val loss: should track train (no huge overfitting)
    vl = get("val/total_loss")
    if vl and tl:
        vlast, tlast = vl[-1][1], tl[-1][1]
        ratio = vlast / (tlast + 1e-12)
        checks.append(("val vs train loss", f"val={vlast:.4f} train={tlast:.4f} ratio={ratio:.2f}", ratio < 3.0))

    # ELO / win rate (if present)
    elo = last_val("eval/elo_rating")
    if elo is not None:
        checks.append(("eval/elo_rating", f"{elo:.0f}", elo > 0))
    wr = last_val("eval/win_rate_vs_mcts")
    if wr is not None:
        checks.append(("eval/win_rate_vs_mcts", f"{wr:.2%}", 0.3 < wr < 0.95))
    wrb = last_val("eval/win_rate_vs_best")
    if wrb is not None:
        # Early iters can be 0% vs best; later expect >=50% when model is accepted
        checks.append(("eval/win_rate_vs_best", f"{wrb:.2%}", wrb >= 0.0))  # sanity only; 0% early is normal

    # Learning rate (sanity)
    lr = last_val("train/learning_rate")
    if lr is not None:
        checks.append(("train/learning_rate", f"{lr:.2e}", lr >= 1e-6))

    # Buffer growth
    buf = get("buffer/total_positions")
    if buf:
        checks.append(("buffer/total_positions", f"last={buf[-1][1]:.0f}", buf[-1][1] > 0))

    for name, detail, ok in checks:
        status = "OK" if ok else "CHECK"
        print(f"  [{status}] {name}: {detail}")

    print()


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TB

    # If default and empty, try loading from committed iters
    if path == DEFAULT_TB and not collect_event_dirs(path) and COMMITTED.exists():
        all_scalars: dict[str, list[tuple[float, float]]] = {}
        for d in sorted(COMMITTED.iterdir()):
            if not d.is_dir() or not d.name.startswith("iter_"):
                continue
            tb_sub = d / "tensorboard"
            if not tb_sub.exists():
                continue
            try:
                from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            except ImportError:
                break
            for ef in sorted(tb_sub.glob("events.out.tfevents.*")):
                try:
                    ea = EventAccumulator(str(ef))
                    ea.Reload()
                    for tag in ea.Tags().get("scalars", []):
                        if tag not in all_scalars:
                            all_scalars[tag] = []
                        for e in ea.Scalars(tag):
                            all_scalars[tag].append((e.step, e.value))
                except Exception:
                    continue
        for tag in all_scalars:
            all_scalars[tag] = sorted(all_scalars[tag], key=lambda x: x[0])
        print(f"Loaded from committed iters (no events in {path})")
        analyze(all_scalars)
        return

    scalars = load_scalars(path)
    if not scalars:
        print(f"No TensorBoard scalar data in: {path}")
        print("Event file(s) may be empty (e.g. run interrupted before first log).")
        return
    print(f"Loaded from: {path}")
    analyze(scalars)


if __name__ == "__main__":
    main()
