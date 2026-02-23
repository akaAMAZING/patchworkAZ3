#!/usr/bin/env python
"""
Deep Training Analysis — Engineer-grade assessment of AlphaZero training health.

Produces a structured report that ML engineers would use to verify training is on point
or identify red flags. Performs:
  - Trend analysis (policy accuracy, value MSE, entropy over iterations)
  - Red-flag detection (collapse, spikes, anomalies)
  - Calibration quality (value head)
  - Strength evidence (ladder monotonicity, Elo progression)
  - First-player bias check
  - Validation sanity integration

Usage:
  python tools/deep_training_analysis.py --config configs/config_best.yaml
  python tools/deep_training_analysis.py --config configs/config_best.yaml --run-tools -o report.md
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def _safe_float(x: Any, default: float = 0.0) -> float:
    if x is None:
        return default
    try:
        return float(x)
    except (ValueError, TypeError):
        return default


def _trend_slope(y: list[float], x: list[float] | None = None) -> float:
    """Simple linear regression slope. Returns 0 if insufficient data."""
    n = len(y)
    if n < 3:
        return 0.0
    if x is None:
        x = list(range(n))
    x_mean = sum(x) / n
    y_mean = sum(y) / n
    num = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
    den = sum((x[i] - x_mean) ** 2 for i in range(n))
    if den == 0:
        return 0.0
    return num / den


def analyze_trends(metadata: list[dict]) -> dict:
    """Extract trend statistics from metadata."""
    if not metadata:
        return {}

    iters = [r.get("iteration", i) for i, r in enumerate(metadata)]
    train = [r.get("train") or {} for r in metadata]
    sp = [r.get("selfplay") or {} for r in metadata]

    pol_acc = [_safe_float(t.get("policy_accuracy")) for t in train]
    pol_top5 = [_safe_float(t.get("policy_top5_accuracy")) for t in train]
    val_mse = [_safe_float(t.get("value_mse")) for t in train]
    total_loss = [_safe_float(t.get("total_loss")) for t in train]
    pol_ent = [_safe_float(t.get("policy_entropy")) for t in train]
    grad_norm = [_safe_float(t.get("grad_norm")) for t in train]
    sp_ent = [_safe_float(s.get("avg_policy_entropy")) for s in sp]
    sp_top1 = [_safe_float(s.get("avg_top1_prob")) for s in sp]
    sp_legal = [_safe_float(s.get("avg_num_legal")) for s in sp]

    # Slopes (per-iter rate of change)
    half = len(metadata) // 2
    early_pol = pol_acc[:half] if half >= 5 else pol_acc
    late_pol = pol_acc[-half:] if half >= 5 else pol_acc[-10:]
    early_mse = val_mse[:half] if half >= 5 else val_mse
    late_mse = val_mse[-half:] if half >= 5 else val_mse[-10:]

    return {
        "n_iters": len(metadata),
        "pol_acc_slope_all": _trend_slope(pol_acc),
        "pol_acc_first": pol_acc[0] if pol_acc else 0,
        "pol_acc_last": pol_acc[-1] if pol_acc else 0,
        "pol_acc_recent_mean": sum(late_pol) / len(late_pol) if late_pol else 0,
        "val_mse_slope_all": _trend_slope(val_mse),
        "val_mse_first": val_mse[0] if val_mse else 0,
        "val_mse_last": val_mse[-1] if val_mse else 0,
        "pol_ent_min": min(pol_ent) if pol_ent else 0,
        "pol_ent_last": pol_ent[-1] if pol_ent else 0,
        "sp_ent_min": min(sp_ent) if sp_ent else 0,
        "sp_ent_last": sp_ent[-1] if sp_ent else 0,
        "grad_norm_median": sorted(grad_norm)[len(grad_norm) // 2] if grad_norm else 0,
        "grad_norm_max": max(grad_norm) if grad_norm else 0,
    }


def detect_red_flags(metadata: list[dict], distributions: list[dict], value_stats: list[dict]) -> list[dict]:
    """Identify potential red flags. Each entry: {severity, check, message, detail}."""
    flags: list[dict] = []
    if not metadata:
        return [{"severity": "error", "check": "data", "message": "No metadata", "detail": ""}]

    train = [r.get("train") or {} for r in metadata]
    sp = [r.get("selfplay") or {} for r in metadata]

    # 1. Policy entropy collapse (overconfidence)
    pol_ents = [_safe_float(t.get("policy_entropy")) for t in train]
    if pol_ents and min(pol_ents) < 2.0:
        flags.append({
            "severity": "critical",
            "check": "policy_entropy_collapse",
            "message": "Policy entropy collapsed below 2.0 nats",
            "detail": f"min={min(pol_ents):.2f} (iter {pol_ents.index(min(pol_ents))})",
        })
    elif pol_ents and pol_ents[-1] < 2.2:
        flags.append({
            "severity": "warn",
            "check": "policy_entropy_low",
            "message": "Policy entropy trending low",
            "detail": f"latest={pol_ents[-1]:.2f} nats",
        })

    # 2. Self-play entropy collapse
    sp_ents = [_safe_float(s.get("avg_policy_entropy")) for s in sp]
    if sp_ents and min(sp_ents) < 2.0:
        flags.append({
            "severity": "critical",
            "check": "selfplay_entropy_collapse",
            "message": "Self-play entropy collapsed",
            "detail": f"min={min(sp_ents):.2f}",
        })

    # 3. Value MSE suspiciously low (< 0.01) — might indicate trivial targets
    val_mses = [_safe_float(t.get("value_mse")) for t in train]
    if val_mses and val_mses[-1] < 0.01:
        flags.append({
            "severity": "warn",
            "check": "value_mse_very_low",
            "message": "Value MSE very low; verify targets are not trivial",
            "detail": f"latest={val_mses[-1]:.4f}",
        })

    # 4. Value target variance collapse
    if value_stats:
        vars_ = [_safe_float(r.get("value_var")) for r in value_stats]
        if vars_ and min(vars_) < 0.1:
            flags.append({
                "severity": "critical",
                "check": "value_var_collapse",
                "message": "Value target variance collapsed",
                "detail": f"min var={min(vars_):.4f}",
            })

    # 5. Grad norm spikes (skip first 5 iters — early training noisier)
    grad_norms = [_safe_float(t.get("grad_norm")) for t in train]
    if len(grad_norms) > 10:
        tail = grad_norms[5:]
        median = sorted(tail)[len(tail) // 2]
        for i in range(5, len(grad_norms)):
            g = grad_norms[i]
            if median > 0.01 and g > 3 * median:
                flags.append({
                    "severity": "warn",
                    "check": "grad_spike",
                    "message": f"Grad norm spike at iter {metadata[i].get('iteration', i)}",
                    "detail": f"grad_norm={g:.3f} (median={median:.3f})",
                })
                break

    # 6. Consecutive rejections
    rejects = [r.get("consecutive_rejections", 0) for r in metadata]
    if rejects and max(rejects) > 3:
        flags.append({
            "severity": "warn",
            "check": "consecutive_rejections",
            "message": f"Multiple consecutive eval rejections",
            "detail": f"max={max(rejects)}",
        })

    # 7. Root diversity collapse (perplexity)
    if distributions:
        perps = [_safe_float(r.get("policy_perplexity_mean")) for r in distributions]
        if perps and min(perps) < 5:
            flags.append({
                "severity": "critical",
                "check": "perplexity_collapse",
                "message": "MCTS root perplexity collapsed",
                "detail": f"min={min(perps):.1f}",
            })
        legal = [_safe_float(r.get("legal_count_mean")) for r in distributions]
        if legal and min(legal) < 30:
            flags.append({
                "severity": "warn",
                "check": "legal_count_collapse",
                "message": "Legal move count at root very low",
                "detail": f"min={min(legal):.1f}",
            })

    # 8. Policy accuracy vs baseline (majority action)
    # Typical majority baseline ~5–15% for 2026 actions; policy_acc should be >> baseline
    pol_accs = [_safe_float(t.get("policy_accuracy")) for t in train]
    if pol_accs and pol_accs[-1] < 0.25:
        flags.append({
            "severity": "critical",
            "check": "policy_acc_below_baseline",
            "message": "Policy accuracy below reasonable baseline",
            "detail": f"latest={pol_accs[-1]:.1%}",
        })

    return flags


def analyze_calibration(calibration: list[dict], last_iter: int) -> dict:
    """Compute calibration quality from decile buckets. Ideal: slope≈1, intercept≈0, low MAE."""
    # Group by iter, take last available
    by_iter: dict[int, list[dict]] = {}
    for r in calibration:
        it = _safe_float(r.get("iter", r.get("iteration", -1)), -1)
        if it >= 0:
            by_iter.setdefault(int(it), []).append(r)

    if not by_iter:
        return {"available": False}

    # Use latest iter with full deciles
    iters = sorted(by_iter.keys(), reverse=True)
    for it in iters:
        rows = by_iter[it]
        if len(rows) >= 10:
            pred_centers = []
            tgt_avgs = []
            for d in range(1, 11):
                dec_rows = [x for x in rows if _safe_float(x.get("decile")) == d]
                if dec_rows:
                    pred_lo = _safe_float(dec_rows[0].get("pred_lo"))
                    pred_hi = _safe_float(dec_rows[0].get("pred_hi"))
                    avg_tgt = _safe_float(dec_rows[0].get("avg_target"))
                    pred_centers.append((pred_lo + pred_hi) / 2)
                    tgt_avgs.append(avg_tgt)

            if len(pred_centers) >= 5:
                # Linear fit: target = slope * pred + intercept
                n = len(pred_centers)
                p_mean = sum(pred_centers) / n
                t_mean = sum(tgt_avgs) / n
                num = sum((pred_centers[i] - p_mean) * (tgt_avgs[i] - t_mean) for i in range(n))
                den = sum((pred_centers[i] - p_mean) ** 2 for i in range(n))
                slope = num / den if den > 1e-10 else 0
                intercept = t_mean - slope * p_mean
                mae = sum(abs(tgt_avgs[i] - (slope * pred_centers[i] + intercept)) for i in range(n)) / n
                return {
                    "available": True,
                    "iter": it,
                    "slope": slope,
                    "intercept": intercept,
                    "mae_decile": mae,
                    "n_deciles": n,
                    "healthy": 0.8 <= slope <= 1.2 and abs(intercept) < 0.1,
                }
    return {"available": False}


def analyze_first_player_bias(meta_analysis: dict | None) -> dict:
    """First-player advantage from meta_analysis or fallback."""
    if not meta_analysis or "first_player" not in meta_analysis:
        return {"available": False}

    fp = meta_analysis["first_player"]
    if not fp:
        return {"available": False}

    wr_list = [_safe_float(r.get("p0_win_rate"), 0.5) for r in fp]
    avg_wr = sum(wr_list) / len(wr_list) if wr_list else 0.5
    max_dev = max(abs(w - 0.5) for w in wr_list) if wr_list else 0

    return {
        "available": True,
        "avg_p0_wr": avg_wr,
        "max_dev_from_50": max_dev,
        "n_iters": len(fp),
        "concerning": avg_wr > 0.55 or avg_wr < 0.45 or max_dev > 0.1,
    }


def analyze_strength(ladder_history: list[dict], ladder_jsons: list[dict], elo_ratings: dict) -> dict:
    """Strength improvement evidence."""
    by_iter: dict[int, list[dict]] = {}
    for r in ladder_history:
        it = int(_safe_float(r.get("iteration"), -1))
        if it >= 0:
            by_iter.setdefault(it, []).append(r)

    elo_diffs_vs_1 = []
    elo_diffs_prev = []
    for it in sorted(by_iter.keys()):
        for m in by_iter[it]:
            role = m.get("role", "")
            elo = _safe_float(m.get("elo_diff"))
            if "anchor_permanent" in role or "iter_1" in str(m.get("opponent_iter", "")):
                elo_diffs_vs_1.append((it, elo))
            if "anchor_previous" in role:
                elo_diffs_prev.append((it, elo))

    monotonic_vs_1 = True
    if len(elo_diffs_vs_1) >= 2:
        for i in range(1, len(elo_diffs_vs_1)):
            if elo_diffs_vs_1[i][1] < elo_diffs_vs_1[i - 1][1] - 5:  # Allow small noise
                monotonic_vs_1 = False
                break

    sorted_elo = sorted(
        [(k, v.get("rating", 0)) for k, v in elo_ratings.items() if isinstance(v, dict)],
        key=lambda x: x[1],
        reverse=True,
    )[:5]

    return {
        "ladder_milestones": list(by_iter.keys()),
        "elo_diffs_vs_iter1": elo_diffs_vs_1,
        "monotonic_vs_anchor": monotonic_vs_1,
        "top_iters_by_elo": sorted_elo,
    }


def _table_from_dicts(rows: list[dict], cols: list[str] | None = None) -> str:
    """Format list of dicts as markdown table."""
    if not rows:
        return "(no data)"
    if cols is None:
        cols = list(rows[0].keys())
    lines = ["| " + " | ".join(str(c) for c in cols) + " |"]
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                cells.append(f"{v:.4f}" if abs(v) < 1000 else f"{v:.2f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _flatten_metadata(entry: dict) -> dict:
    """Flatten one metadata entry for table."""
    out = dict(entry)
    out.pop("train", None)
    out.pop("selfplay", None)
    train = entry.get("train") or {}
    sp = entry.get("selfplay") or {}
    for k, v in train.items():
        out[f"train_{k}"] = v
    for k, v in sp.items():
        out[f"sp_{k}"] = v
    return out


def run_validation_sanity(run_dir: Path, config_path: Path, max_samples: int = 500) -> dict:
    """Run validation sanity checks, capture output."""
    committed = run_dir / "committed"
    data_path = None
    if committed.exists():
        for d in sorted(committed.iterdir(), key=lambda x: x.name, reverse=True):
            if d.is_dir() and d.name.startswith("iter_"):
                cand = d / "selfplay.h5"
                if cand.exists():
                    data_path = cand
                    break
    if not data_path or not data_path.exists():
        return {"run": False, "reason": "no selfplay.h5"}

    try:
        out = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "tools" / "validation_sanity_checks.py"),
                "--config", str(config_path),
                "--data", str(data_path),
                "--max-val-samples", str(max_samples),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=90,
        )
        return {
            "run": True,
            "returncode": out.returncode,
            "stdout": out.stdout,
            "stderr": out.stderr,
        }
    except Exception as e:
        return {"run": False, "reason": str(e)}


def write_report(
    run_state: dict,
    metadata: list[dict],
    ladder_history: list[dict],
    ladder_jsons: list[dict],
    elo_ratings: dict,
    distributions: list[dict],
    value_stats: list[dict],
    calibration: list[dict],
    meta_analysis: dict | None,
    validation_result: dict | None,
    commit_manifests: list[dict],
    replay_state: list | dict | None,
    out_path: Path | None,
) -> str:
    """Produce full engineering report."""
    lines: list[str] = []

    last_iter = run_state.get("last_committed_iteration", -1)
    trends = analyze_trends(metadata)
    flags = detect_red_flags(metadata, distributions, value_stats)
    cal = analyze_calibration(calibration, last_iter)
    fp_bias = analyze_first_player_bias(meta_analysis)
    strength = analyze_strength(ladder_history, ladder_jsons, elo_ratings)

    # --- Executive Summary ---
    lines.append("# Deep Training Analysis Report")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    n_crit = sum(1 for f in flags if f.get("severity") == "critical")
    n_warn = sum(1 for f in flags if f.get("severity") == "warn")

    if n_crit > 0:
        lines.append(f"**Status: CRITICAL** — {n_crit} critical red flag(s) detected. Review before continuing.")
    elif n_warn > 0:
        lines.append(f"**Status: ATTENTION** — {n_warn} warning(s). Training appears functional; monitor closely.")
    else:
        lines.append("**Status: HEALTHY** — No red flags detected. Training trajectory looks normal.")

    lines.append("")
    lines.append(f"- Iterations: 0–{last_iter} ({trends.get('n_iters', 0)} committed)")
    lines.append(f"- Policy accuracy: {trends.get('pol_acc_first', 0):.1%} -> {trends.get('pol_acc_last', 0):.1%}")
    lines.append(f"- Value MSE: {trends.get('val_mse_first', 0):.4f} -> {trends.get('val_mse_last', 0):.4f}")
    lines.append(f"- Self-play entropy: {trends.get('sp_ent_min', 0):.2f}–{trends.get('sp_ent_last', 0):.2f} nats")
    if strength.get("monotonic_vs_anchor"):
        lines.append("- Ladder: Monotonic improvement vs iter_1 anchor")
    else:
        lines.append("- Ladder: Some non-monotonicity vs anchor (review)")
    lines.append("")

    # --- Red Flags ---
    lines.append("## Red Flags")
    lines.append("")
    if not flags:
        lines.append("None detected.")
    else:
        for f in flags:
            lines.append(f"- **[{f.get('severity', '?').upper()}]** {f.get('message', '')}")
            if f.get("detail"):
                lines.append(f"  - {f['detail']}")
    lines.append("")

    # --- Training Health ---
    lines.append("## Training Health")
    lines.append("")
    lines.append("| Check | Status | Detail |")
    lines.append("|-------|--------|--------|")

    def row(check: str, ok: bool, detail: str):
        status = "OK" if ok else "FAIL"
        lines.append(f"| {check} | {status} | {detail} |")

    row("Policy entropy (no collapse)", trends.get("pol_ent_min", 0) >= 2.0, f"min={trends.get('pol_ent_min', 0):.2f}")
    row("Self-play entropy", trends.get("sp_ent_min", 0) >= 2.0, f"min={trends.get('sp_ent_min', 0):.2f}")
    row("Value MSE declining", trends.get("val_mse_slope_all", 0) <= 0.01, f"slope={trends.get('val_mse_slope_all', 0):.6f}")
    row("Policy accuracy improving", trends.get("pol_acc_slope_all", 0) >= -0.001, f"slope={trends.get('pol_acc_slope_all', 0):.4f}")
    row("Value target variance", not value_stats or min(_safe_float(r.get("value_var")) for r in value_stats) >= 0.1, f"var∈[0.1, 0.35] expected")
    cal_detail = f"slope={cal.get('slope', 0):.2f} intercept={cal.get('intercept', 0):.3f}" if cal.get("available") else "no data"
    row("Calibration", cal.get("healthy", False) if cal.get("available") else True, cal_detail)
    row("First-player bias", not fp_bias.get("concerning", False) if fp_bias.get("available") else "N/A", f"avg P0 WR={fp_bias.get('avg_p0_wr', 0.5):.1%}" if fp_bias.get("available") else "")
    row("Grad norm stable", trends.get("grad_norm_max", 0) < 5 or trends.get("grad_norm_median", 0) == 0, f"max={trends.get('grad_norm_max', 0):.2f} median={trends.get('grad_norm_median', 0):.2f}")

    lines.append("")

    # --- Strength Evidence ---
    lines.append("## Strength Evidence")
    lines.append("")
    if strength.get("elo_diffs_vs_iter1"):
        lines.append("### Elo vs iter_1 (anchor)")
        for it, elo in strength["elo_diffs_vs_iter1"]:
            lines.append(f"- Iter {it}: +{elo:.0f} Elo")
        lines.append("")
    lines.append(f"Monotonic vs anchor: {'Yes' if strength.get('monotonic_vs_anchor') else 'No'}")
    lines.append("")
    if strength.get("top_iters_by_elo"):
        lines.append("Top iters by Elo:")
        for name, rat in strength["top_iters_by_elo"][:5]:
            lines.append(f"- {name}: {rat:.0f}")
    lines.append("")

    # --- Calibration Detail ---
    if cal.get("available"):
        lines.append("## Value Calibration")
        lines.append("")
        lines.append(f"- Iter: {cal.get('iter')} | Slope: {cal.get('slope', 0):.3f} | Intercept: {cal.get('intercept', 0):.4f} | MAE(decile): {cal.get('mae_decile', 0):.4f}")
        lines.append(f"- Healthy (slope 0.8–1.2, |intercept|<0.1): {cal.get('healthy', False)}")
        lines.append("")

    # --- Root Diversity ---
    if distributions:
        lines.append("## Root Diversity (MCTS)")
        lines.append("")
        last = distributions[-1]
        lines.append(f"- Legal count (mean): {last.get('legal_count_mean', '')}")
        lines.append(f"- Perplexity: {last.get('policy_perplexity_mean', '')}")
        lines.append(f"- Visit entropy: {last.get('visit_entropy_mean', '')}")
        lines.append("")

    # --- Validation Sanity ---
    if validation_result and validation_result.get("run"):
        lines.append("## Validation Sanity Checks")
        lines.append("")
        lines.append("```")
        lines.append(validation_result.get("stdout", "")[:5000])
        lines.append("```")
        lines.append("")

    # ========== FULL DATA DUMPS (preempt downstream requests) ==========

    lines.append("---")
    lines.append("")
    lines.append("# FULL DATA DUMPS")
    lines.append("")
    lines.append("## Run State (raw)")
    lines.append("```json")
    lines.append(json.dumps(run_state, indent=2))
    lines.append("```")
    lines.append("")

    if replay_state is not None:
        lines.append("## Replay State (raw)")
        lines.append("```json")
        lines.append(json.dumps(replay_state, indent=2)[:2000])
        lines.append("```")
        lines.append("")

    lines.append("## Elo Ratings (full)")
    lines.append("```json")
    lines.append(json.dumps(elo_ratings, indent=2))
    lines.append("```")
    lines.append("")

    # --- FULL LADDER (iter 25, iter 50) ---
    lines.append("## Frozen Ladder — Full JSON (per milestone)")
    for lj in ladder_jsons:
        it = lj.get("iteration", "?")
        lines.append(f"### iter_{it:03d}.json")
        lines.append("```json")
        lines.append(json.dumps(lj, indent=2))
        lines.append("```")
        lines.append("")
    if not ladder_jsons:
        lines.append("(no ladder JSONs)")
        lines.append("")

    # --- Ladder history CSV ---
    lines.append("## Ladder history.csv (full)")
    lines.append("")
    lines.append(_table_from_dicts(ladder_history))
    lines.append("")

    # --- Ladder match summary (iter50 vs iter25 etc) ---
    lines.append("## Ladder Match Summary (iter 50 vs iter 25, iter 25 vs iter 1)")
    for r in ladder_history:
        it = r.get("iteration", "")
        opp = r.get("opponent_iter", "")
        wins = r.get("wins", "")
        n = r.get("n_games", "")
        wr = r.get("win_rate", "")
        wr_p0 = r.get("wr_as_p0", "")
        wr_p1 = r.get("wr_as_p1", "")
        margin = r.get("avg_score_margin", "")
        elo = r.get("elo_diff", "")
        role = r.get("role", "")
        lines.append(f"- **iter_{it} vs iter_{opp}** ({role}): WR={wr} ({wins}/{n}) | margin={margin} | Elo=+{elo} | as P0={wr_p0} as P1={wr_p1}")
    lines.append("")

    # --- FULL METADATA TABLE ---
    lines.append("## Full Metadata (all iterations, TensorBoard-equivalent)")
    flat = [_flatten_metadata(m) for m in metadata]
    if flat:
        cols = ["iteration", "train_policy_accuracy", "train_policy_top5_accuracy", "train_value_mse",
                "train_total_loss", "train_policy_entropy", "train_grad_norm", "sp_avg_policy_entropy",
                "sp_avg_top1_prob", "sp_avg_num_legal", "sp_avg_game_length", "sp_p0_wins", "sp_p1_wins",
                "replay_positions", "iter_time_s"]
        avail = [c for c in cols if c in flat[0]]
        lines.append("")
        lines.append(_table_from_dicts(flat, avail))
    lines.append("")

    # --- Distributions ---
    lines.append("## Root Diversity — distributions.csv (full)")
    lines.append("")
    lines.append(_table_from_dicts(distributions))
    lines.append("")

    # --- Value stats ---
    lines.append("## Value Target Stats — value_stats.csv (full)")
    lines.append("")
    lines.append(_table_from_dicts(value_stats))
    lines.append("")

    # --- Calibration decile table (iter 24) ---
    by_iter: dict[int, list[dict]] = {}
    for r in calibration:
        it = int(_safe_float(r.get("iter", r.get("iteration", -1)), -1))
        if it >= 0:
            by_iter.setdefault(it, []).append(r)
    cal_iters = sorted(by_iter.keys(), reverse=True)
    for ci in cal_iters[:3]:
        rows = [x for x in by_iter[ci] if _safe_float(x.get("decile")) > 0]
        if len(rows) >= 10:
            lines.append(f"## Value Calibration Deciles — iter {ci}")
            lines.append("")
            lines.append(_table_from_dicts(rows, ["decile", "pred_lo", "pred_hi", "avg_target", "n"]))
            lines.append("")
            break

    # --- First player ---
    if meta_analysis and "first_player" in meta_analysis:
        lines.append("## First-Player Advantage (meta_analysis first_player)")
        lines.append("")
        lines.append(_table_from_dicts(meta_analysis["first_player"],
            ["iteration", "p0_wins", "p1_wins", "total_games", "p0_win_rate", "elo_advantage"]))
        lines.append("")

    # --- Action type ---
    if meta_analysis and "action_type" in meta_analysis:
        lines.append("## Action Type Distribution (meta_analysis)")
        lines.append("```json")
        lines.append(json.dumps(meta_analysis["action_type"], indent=2))
        lines.append("```")
        lines.append("")

    # --- Piece ranking ---
    if meta_analysis and "piece_ranking" in meta_analysis:
        lines.append("## Piece Ranking (policy mass when buyable)")
        pr = meta_analysis["piece_ranking"]
        lines.append("")
        lines.append(_table_from_dicts(pr[:35],
            ["piece_id", "mean_policy_mass", "mean_value_when_buyable", "mean_value_when_top_choice",
             "n_positions", "n_top_choice", "cost_buttons", "cost_time"]))
        lines.append("")

    # --- Value by phase ---
    if meta_analysis and "value_by_phase" in meta_analysis:
        lines.append("## Value by Game Phase")
        lines.append("")
        lines.append(_table_from_dicts(meta_analysis["value_by_phase"]))
        lines.append("")

    # --- Policy sharpness by phase ---
    if meta_analysis and "policy_sharpness_by_phase" in meta_analysis:
        lines.append("## Policy Sharpness by Game Phase (entropy)")
        lines.append("")
        lines.append(_table_from_dicts(meta_analysis["policy_sharpness_by_phase"]))
        lines.append("")

    # --- Calibration summary stats (from meta pred/target if present) ---
    if meta_analysis and "calibration_pred" in meta_analysis and "calibration_target" in meta_analysis:
        pred = meta_analysis["calibration_pred"]
        tgt = meta_analysis["calibration_target"]
        if isinstance(pred, list) and isinstance(tgt, list) and len(pred) == len(tgt) and len(pred) > 0:
            import statistics
            n = min(len(pred), 5000)
            p = [float(x) for x in pred[:n]]
            t = [float(x) for x in tgt[:n]]
            lines.append("## Calibration Pred vs Target (summary stats)")
            lines.append(f"- n_samples: {len(pred)}")
            sp_std = statistics.stdev(p) if len(p) > 1 else 0.0
            st_std = statistics.stdev(t) if len(t) > 1 else 0.0
            lines.append(f"- pred: mean={statistics.mean(p):.4f} std={sp_std:.4f}")
            lines.append(f"- target: mean={statistics.mean(t):.4f} std={st_std:.4f}")
            lines.append("")

    # --- Applied settings (latest commit) ---
    if commit_manifests:
        lines.append("## Applied Settings (latest commit_manifest)")
        lines.append("```json")
        lines.append(json.dumps(commit_manifests[-1].get("applied_settings", {}), indent=2))
        lines.append("```")
        lines.append("")

    # --- Recommendations ---
    lines.append("## Recommendations")
    lines.append("")
    if n_crit > 0:
        lines.append("- Address critical red flags before resuming or scaling up.")
    if not cal.get("healthy", True) and cal.get("available"):
        lines.append("- Investigate value calibration; consider score_scale or value target regime.")
    if fp_bias.get("concerning"):
        lines.append("- First-player advantage >5%; may need more games or evaluation to balance.")
    if trends.get("pol_acc_slope_all", 0) < 0.001 and trends.get("n_iters", 0) > 30:
        lines.append("- Policy accuracy plateau; typical after ~50+ iters; ensure value/ladder still improving.")
    if not strength.get("monotonic_vs_anchor") and len(strength.get("elo_diffs_vs_iter1", [])) >= 2:
        lines.append("- Ladder non-monotonicity; possible cycling—review opponent diversity and window.")
    if not any(s.startswith("- ") for s in lines[-10:]):
        lines.append("- Continue training; monitor ladder at next milestone (iter 75).")
    lines.append("")

    report = "\n".join(lines)
    if out_path:
        out_path.write_text(report, encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Deep training analysis for ML engineers")
    ap.add_argument("--config", default="configs/config_best.yaml")
    ap.add_argument("--run-dir", default="runs/patchwork_production")
    ap.add_argument("--output", "-o", help="Output report path")
    ap.add_argument("--run-validation", action="store_true", help="Run validation_sanity_checks")
    ap.add_argument("--run-tools", action="store_true",
        help="Refresh analyze_iteration_metrics + export_metrics_csv (slower)")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path

    import yaml
    with open(cfg_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logs_dir = Path(config.get("paths", {}).get("logs_dir", "logs"))
    run_dir = Path(args.run_dir)
    if not logs_dir.is_absolute():
        logs_dir = REPO_ROOT / logs_dir
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir

    # Optionally refresh analysis outputs
    if args.run_tools:
        last = _load_json(run_dir / "run_state.json") or {}
        last_iter = last.get("last_committed_iteration", 59)
        try:
            subprocess.run(
                [sys.executable, str(REPO_ROOT / "tools" / "export_metrics_csv.py"),
                 "--config", str(cfg_path), "--iters", f"0-{last_iter}",
                 "-o", str(logs_dir / f"metrics_iter000_{last_iter:03d}.csv")],
                cwd=REPO_ROOT, capture_output=True, timeout=30,
            )
            subprocess.run(
                [sys.executable, str(REPO_ROOT / "tools" / "analyze_iteration_metrics.py"),
                 "--config", str(cfg_path), "--run-dir", str(run_dir),
                 "--iters", f"0-{min(last_iter, 59)}", "--out-dir", str(logs_dir / "analysis"),
                 "--value-stats", "--calibration", "--no-display"],
                cwd=REPO_ROOT, capture_output=True, timeout=600,
            )
        except Exception as e:
            print(f"Warning: run-tools failed: {e}", file=sys.stderr)

    # Load data
    meta_path = logs_dir / "metadata.jsonl"
    metadata = _load_jsonl(meta_path)
    run_state = _load_json(run_dir / "run_state.json") or {}
    ladder_history = _load_csv(run_dir / "eval" / "ladder" / "history.csv")
    ladder_jsons = []
    for p in sorted((run_dir / "eval" / "ladder").glob("iter_*.json")):
        j = _load_json(p)
        if j:
            ladder_jsons.append(j)
    elo_ratings = _load_json(REPO_ROOT / "elo_ratings.json") or {}
    distributions = _load_csv(logs_dir / "analysis" / "distributions.csv")
    value_stats = _load_csv(logs_dir / "analysis" / "value_stats.csv")
    calibration = _load_csv(logs_dir / "analysis" / "calibration.csv")
    meta_analysis = _load_json(logs_dir / "meta_analysis" / "meta_analysis_report.json")
    replay_state = _load_json(run_dir / "replay_state.json")

    # Load commit manifests (latest 3)
    commit_manifests = []
    committed = run_dir / "committed"
    if committed.exists():
        for d in sorted(committed.iterdir(), key=lambda x: x.name, reverse=True)[:5]:
            if d.is_dir() and d.name.startswith("iter_"):
                m = _load_json(d / "commit_manifest.json")
                if m:
                    commit_manifests.append(m)
    commit_manifests.sort(key=lambda x: x.get("iteration", -1))

    validation_result = None
    if args.run_validation:
        validation_result = run_validation_sanity(run_dir, cfg_path)

    out_path = Path(args.output) if args.output else None
    if out_path and not out_path.is_absolute():
        out_path = REPO_ROOT / out_path

    report = write_report(
        run_state=run_state,
        metadata=metadata,
        ladder_history=ladder_history,
        ladder_jsons=ladder_jsons,
        elo_ratings=elo_ratings,
        distributions=distributions,
        value_stats=value_stats,
        calibration=calibration,
        meta_analysis=meta_analysis,
        validation_result=validation_result,
        commit_manifests=commit_manifests,
        replay_state=replay_state,
        out_path=out_path,
    )

    if out_path:
        print(f"Report written to {out_path}")
    else:
        print(report)


if __name__ == "__main__":
    main()
