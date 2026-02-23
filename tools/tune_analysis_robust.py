#!/usr/bin/env python3
"""
Robust parameter recommendation from Optuna tuning artifacts.
Read-only analysis: loads Optuna DB + trial_results.json, no training.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import optuna
from optuna.trial import TrialState

# t_{0.975, df=2} for 95% CI with 3 checkpoints
T_975_DF2 = 4.303


def main():
    tune_dir = Path("tuning_2d")
    storage = f"sqlite:///{tune_dir}/optuna_study.db"
    study_name = "patchwork_2d"

    study = optuna.load_study(study_name=study_name, storage=storage)
    complete = [t for t in study.trials if t.state == TrialState.COMPLETE and t.value is not None]

    # 1) List Optuna complete trials
    print("=" * 80)
    print("1) OPTUNA COMPLETE TRIALS")
    print("=" * 80)
    print(f"{'trial':>6}  {'objective':>10}  {'cpuct':>8}  {'q_value_weight':>16}")
    print("-" * 50)
    for t in complete:
        print(f"{t.number:>6}  {t.value:>+10.2f}  {t.params['cpuct']:>8.3f}  {t.params['q_value_weight']:>16.4f}")

    # 2) Load trial_results.json and compute stats per trial
    rows = []
    for t in complete:
        results_path = tune_dir / f"trial_{t.number:04d}" / "trial_results.json"
        if not results_path.exists():
            print(f"  WARNING: missing {results_path}")
            continue

        with open(results_path) as f:
            data = json.load(f)

        params = data["params"]
        avg_margin = data["avg_score_margin"]
        avg_wr = data["avg_win_rate"]
        checkpoint_evals = data.get("checkpoint_evals", [])

        # Margin stats from checkpoints
        ckpt_margins = [c["margin"] for c in checkpoint_evals]
        n_ckpt = len(ckpt_margins)
        margin_mean = sum(ckpt_margins) / n_ckpt if n_ckpt else 0
        if n_ckpt >= 2:
            variance = sum((m - margin_mean) ** 2 for m in ckpt_margins) / (n_ckpt - 1)
            margin_sd = math.sqrt(variance)
        else:
            margin_sd = 0.0
        margin_se = margin_sd / math.sqrt(n_ckpt) if n_ckpt else 0.0
        LCB95 = margin_mean - T_975_DF2 * margin_se
        UCB95 = margin_mean + T_975_DF2 * margin_se

        final_policy_acc = data.get("final_policy_accuracy")
        final_loss = data.get("final_total_loss")
        metrics = data.get("metrics_history", [])

        # Last 3 iters average
        last3 = metrics[-3:] if len(metrics) >= 3 else metrics
        avg_entropy = sum(m.get("policy_entropy", 0) for m in last3) / len(last3) if last3 else None
        avg_top1 = sum(m.get("top1_prob", 0) for m in last3) / len(last3) if last3 else None
        avg_redundancy = sum(m.get("redundancy", 0) for m in last3) / len(last3) if last3 else None
        avg_root_q = sum(m.get("avg_root_q", 0) for m in last3) / len(last3) if last3 else None

        row = {
            "trial": t.number,
            "cpuct": params["cpuct"],
            "q_value_weight": params["q_value_weight"],
            "mean_margin": margin_mean,
            "sd_ckpt": margin_sd,
            "se_ckpt": margin_se,
            "LCB95": LCB95,
            "UCB95": UCB95,
            "avg_win_rate": avg_wr,
            "final_policy_acc": final_policy_acc,
            "final_loss": final_loss,
            "avg_entropy": avg_entropy,
            "avg_top1": avg_top1,
            "avg_redundancy": avg_redundancy,
            "avg_root_q": avg_root_q,
        }
        rows.append(row)

    if not rows:
        print("No trial results loaded. Exiting.")
        return

    # 3) Rankings
    by_mean = sorted(rows, key=lambda r: r["mean_margin"], reverse=True)
    by_lcb = sorted(rows, key=lambda r: r["LCB95"], reverse=True)
    by_wr = sorted(rows, key=lambda r: r["avg_win_rate"], reverse=True)
    top5_mean = by_mean[:5]
    by_stability = sorted(top5_mean, key=lambda r: r["sd_ckpt"])

    print("\n" + "=" * 80)
    print("3) RANKINGS")
    print("=" * 80)
    print("A) Best mean margin:     ", [r["trial"] for r in by_mean[:5]])
    print("B) Best LCB95:          ", [r["trial"] for r in by_lcb[:5]])
    print("C) Best avg_win_rate:    ", [r["trial"] for r in by_wr[:5]])
    print("D) Best stability (min sd among top 5 by mean):", [r["trial"] for r in by_stability[:5]])

    # 4) Basin / robust center
    K = min(5, len(by_mean))
    top_k = by_mean[:K]
    max_margin = top_k[0]["mean_margin"]
    tau = 0.5
    weights = [math.exp((r["mean_margin"] - max_margin) / tau) for r in top_k]
    w_sum = sum(weights)
    weights = [w / w_sum for w in weights]

    cpuct_weighted = sum(r["cpuct"] * w for r, w in zip(top_k, weights))
    q_weight_weighted = sum(r["q_value_weight"] * w for r, w in zip(top_k, weights))

    cpucts = sorted(r["cpuct"] for r in top_k)
    q_weights = sorted(r["q_value_weight"] for r in top_k)
    cpuct_median = cpucts[K // 2] if K % 2 else (cpucts[K // 2 - 1] + cpucts[K // 2]) / 2
    q_median = q_weights[K // 2] if K % 2 else (q_weights[K // 2 - 1] + q_weights[K // 2]) / 2

    print("\n" + "=" * 80)
    print("4) BASIN / ROBUST CENTER (top K=%d by mean margin)" % K)
    print("=" * 80)
    print(f"  Weighted avg:  cpuct={cpuct_weighted:.4f}  q_value_weight={q_weight_weighted:.4f}")
    print(f"  Median:       cpuct={cpuct_median:.4f}  q_value_weight={q_median:.4f}")

    # 5) Training dynamics for top 5 by mean
    print("\n" + "=" * 80)
    print("5) TRAINING DYNAMICS (top 5 by mean margin)")
    print("=" * 80)
    print(f"{'trial':>6}  {'final_acc':>10}  {'final_loss':>10}  {'entropy':>8}  {'top1':>8}  {'redund':>8}  {'root_q':>8}")
    print("-" * 70)
    for r in top5_mean:
        print(
            f"{r['trial']:>6}  {r['final_policy_acc'] or 0:>10.4f}  {r['final_loss'] or 0:>10.4f}  "
            f"{r['avg_entropy'] or 0:>8.3f}  {r['avg_top1'] or 0:>8.3f}  "
            f"{r['avg_redundancy'] or 0:>8.4f}  {r['avg_root_q'] or 0:>8.4f}"
        )

    # Markdown table
    print("\n" + "=" * 80)
    print("FULL TABLE (markdown)")
    print("=" * 80)
    header = "| trial | cpuct | q_value_weight | mean_margin | sd_ckpt | se_ckpt | LCB95 | UCB95 | avg_win_rate | final_policy_acc | final_loss |"
    sep = "|-------|-------|----------------|-------------|---------|---------|-------|-------|--------------|------------------|------------|"
    print(header)
    print(sep)
    for r in rows:
        print(
            f"| {r['trial']:>5} | {r['cpuct']:.3f} | {r['q_value_weight']:.4f} | "
            f"{r['mean_margin']:+.2f} | {r['sd_ckpt']:.2f} | {r['se_ckpt']:.2f} | "
            f"{r['LCB95']:+.2f} | {r['UCB95']:+.2f} | {r['avg_win_rate']:.3f} | "
            f"{r['final_policy_acc'] or 0:.4f} | {r['final_loss'] or 0:.4f} |"
        )

    # 6) Final recommendations
    best_mean = by_mean[0]
    best_lcb = by_lcb[0]
    print("\n" + "=" * 80)
    print("6) FINAL RECOMMENDATIONS")
    print("=" * 80)
    print("\n  BEST MEAN:   trial #%d  cpuct=%.3f  q_value_weight=%.4f  margin_mean=%.2f" % (
        best_mean["trial"], best_mean["cpuct"], best_mean["q_value_weight"], best_mean["mean_margin"]))
    print("  BEST LCB:   trial #%d  cpuct=%.3f  q_value_weight=%.4f  LCB95=%.2f" % (
        best_lcb["trial"], best_lcb["cpuct"], best_lcb["q_value_weight"], best_lcb["LCB95"]))
    print("  BEST BASIN: weighted cpuct=%.4f  q_value_weight=%.4f" % (cpuct_weighted, q_weight_weighted))
    print("              median  cpuct=%.4f  q_value_weight=%.4f" % (cpuct_median, q_median))

    print("\n  --- PRODUCTION RECOMMENDATION ---")
    if best_lcb["trial"] == best_mean["trial"]:
        print("  Use trial #%d (best mean and best LCB)." % best_mean["trial"])
    else:
        print("  Prefer trial #%d (best LCB95) for long production training." % best_lcb["trial"])
    print(
        "  LCB95 optimizes for worst-case performance across checkpoints; a trial that ranks highest "
        "by LCB95 is less likely to have benefited from a lucky single-checkpoint spike. If trial #%d "
        "has comparable training dynamics (policy accuracy, loss) to #%d, use #%d. Otherwise, the "
        "basin median (cpuct=%.3f, q_value_weight=%.3f) is a conservative hedge between top performers."
        % (best_lcb["trial"], best_mean["trial"], best_lcb["trial"], cpuct_median, q_median)
    )

    # Save CSV
    csv_path = tune_dir / "analysis_summary.csv"
    with open(csv_path, "w") as f:
        f.write("trial,cpuct,q_value_weight,mean_margin,sd_ckpt,se_ckpt,LCB95,UCB95,avg_win_rate,final_policy_acc,final_loss\n")
        for r in rows:
            acc = f"{r['final_policy_acc']:.4f}" if r["final_policy_acc"] is not None else ""
            loss = f"{r['final_loss']:.4f}" if r["final_loss"] is not None else ""
            f.write(
                f"{r['trial']},{r['cpuct']:.6f},{r['q_value_weight']:.6f},"
                f"{r['mean_margin']:.4f},{r['sd_ckpt']:.4f},{r['se_ckpt']:.4f},"
                f"{r['LCB95']:.4f},{r['UCB95']:.4f},{r['avg_win_rate']:.4f},"
                f"{acc},{loss}\n"
            )
    print(f"\n  Saved: {csv_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
