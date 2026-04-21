"""
quick_diag/report.py
====================
Reads committed iteration JSONs from quick_diag/runs/qd_flat and qd_hier,
prints side-by-side metric tables, and gives a verdict on the KL question.

USAGE (from project root or quick_diag/):
    python quick_diag/report.py
    python quick_diag/report.py --runs-dir quick_diag/runs
    python quick_diag/report.py --compare-prod  # also prints production iter 0-6 alongside
"""

import argparse
import json
from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS  = PROJECT_ROOT / "quick_diag" / "runs"
PROD_RUNS     = PROJECT_ROOT / "runs" / "patchwork_fresh_v2" / "committed"

# ── Metric definitions ───────────────────────────────────────────────────────
# (display_name, json_path_in_iter_json)
TRAIN_METRICS = [
    ("total_loss",       "train_metrics.total_loss"),
    ("policy_loss",      "train_metrics.policy_loss"),
    ("value_loss",       "train_metrics.value_loss"),
    ("kl_divergence",    "train_metrics.kl_divergence"),
    ("policy_entropy",   "train_metrics.policy_entropy"),
    ("target_entropy",   "train_metrics.target_entropy"),
    ("pol_acc%",         "train_metrics.policy_accuracy"),
    ("pol_top5%",        "train_metrics.policy_top5_accuracy"),
    ("grad_norm",        "train_metrics.grad_norm"),
]

SELFPLAY_METRICS = [
    ("sp_entropy",       "selfplay_stats.avg_policy_entropy"),
    ("sp_top1_prob",     "selfplay_stats.avg_top1_prob"),
    ("avg_legal",        "selfplay_stats.avg_num_legal"),
    ("avg_game_len",     "selfplay_stats.avg_game_length"),
]

ALL_METRICS = TRAIN_METRICS + SELFPLAY_METRICS

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_nested(d: dict, path: str, default=None):
    keys = path.split(".")
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_run(run_dir: Path) -> dict[int, dict]:
    """Returns {iteration: metric_dict} from committed iter JSON files."""
    data = {}
    if not run_dir.exists():
        return data
    for iter_dir in sorted(run_dir.iterdir()):
        if not iter_dir.is_dir():
            continue
        jsons = list(iter_dir.glob("iteration_*.json"))
        if not jsons:
            continue
        with open(jsons[0]) as f:
            raw = json.load(f)
        it = raw.get("iteration", -1)
        data[it] = raw
    return data


def fmt(v, name: str) -> str:
    if v is None:
        return "  ---  "
    if name in ("pol_acc%", "pol_top5%"):
        return f"{v*100:6.1f}%"
    if name == "avg_legal":
        return f"{v:7.1f}"
    if name == "avg_game_len":
        return f"{v:6.1f}"
    return f"{v:7.4f}"


def delta_marker(flat_v, hier_v, name: str) -> str:
    """Show Δ and flag large KL gap."""
    if flat_v is None or hier_v is None:
        return ""
    d = hier_v - flat_v
    mark = ""
    if name == "kl_divergence":
        if abs(d) > 0.15:
            mark = "  *** STRUCTURAL ***"
        elif abs(d) > 0.05:
            mark = "  * notable"
    sign = "+" if d >= 0 else ""
    return f"  Δ={sign}{d:+.4f}{mark}"


# ── Table printer ─────────────────────────────────────────────────────────────

def print_table(runs: dict[str, dict[int, dict]], all_iters: list[int]) -> None:
    run_ids = list(runs.keys())
    col_w = 10

    # Header
    header = f"{'iter':>4}"
    for rid in run_ids:
        for mname, _ in ALL_METRICS:
            header += f"  {(rid+':'+mname)[:col_w]:>{col_w}}"
    if len(run_ids) == 2:
        header += f"  {'Δ KL (hier-flat)':>20}"
    print(header)
    print("-" * len(header))

    for it in all_iters:
        row = f"{it:>4}"
        vals = {}
        for rid in run_ids:
            iter_data = runs[rid].get(it, {})
            for mname, jpath in ALL_METRICS:
                v = get_nested(iter_data, jpath)
                vals[(rid, mname)] = v
                row += f"  {fmt(v, mname):>{col_w}}"
        if len(run_ids) == 2:
            flat_kl = vals.get((run_ids[0], "kl_divergence"))
            hier_kl = vals.get((run_ids[1], "kl_divergence"))
            row += delta_marker(flat_kl, hier_kl, "kl_divergence")
        print(row)


def kl_verdict(flat_data: dict, hier_data: dict) -> None:
    print("\n" + "="*60)
    print("  KL DIVERGENCE VERDICT")
    print("="*60)

    iters = sorted(set(flat_data) | set(hier_data))
    if not iters:
        print("  No data found.")
        return

    last_iter = max((i for i in iters if flat_data.get(i) and hier_data.get(i)), default=None)
    if last_iter is None:
        print("  Need both flat and hier data for verdict.")
        return

    flat_kl = get_nested(flat_data[last_iter], "train_metrics.kl_divergence")
    hier_kl = get_nested(hier_data[last_iter], "train_metrics.kl_divergence")
    flat_pe  = get_nested(flat_data[last_iter], "train_metrics.policy_entropy")
    hier_pe  = get_nested(hier_data[last_iter], "train_metrics.policy_entropy")
    flat_te  = get_nested(flat_data[last_iter], "train_metrics.target_entropy")
    hier_te  = get_nested(hier_data[last_iter], "train_metrics.target_entropy")

    if flat_kl is None or hier_kl is None:
        print("  KL not available in one or both runs.")
        return

    diff = hier_kl - flat_kl
    print(f"  At iter {last_iter}:")
    print(f"    flat  KL = {flat_kl:.4f}   (π_entropy={flat_pe:.3f}, target_entropy={flat_te:.3f})")
    print(f"    hier  KL = {hier_kl:.4f}   (π_entropy={hier_pe:.3f}, target_entropy={hier_te:.3f})")
    print(f"    Δ KL      = {diff:+.4f}")
    print()

    if diff > 0.15:
        print("  VERDICT: HIER STRUCTURAL")
        print("  Hierarchical factorization is the primary driver of elevated KL.")
        print("  Recommendation: switch production run to flat policy head.")
    elif diff > 0.05:
        print("  VERDICT: HIER CONTRIBUTES (partial)")
        print("  Hierarchical factorization adds ~{:.0f}% of KL gap.".format(100 * diff / hier_kl))
        print("  Other causes (replay averaging, low sims) are also present.")
        print("  Consider testing flat head in next production run.")
    elif diff < -0.05:
        print("  VERDICT: FLAT HEAD WORSE")
        print("  Flat head produces higher KL — hierarchical factorization is NOT the driver.")
        print("  Look elsewhere: replay buffer staleness, progressive widening, sim budget.")
    else:
        print("  VERDICT: NO SIGNIFICANT DIFFERENCE")
        print("  Both heads produce similar KL at this scale.")
        print("  KL issue is structural (low sims / replay buffer) not policy head choice.")


def print_prod_comparison(prod_data: dict) -> None:
    if not prod_data:
        print("\n  [production data not found — skipping comparison]")
        return
    print("\n" + "="*60)
    print("  PRODUCTION RUN REFERENCE (patchwork_fresh_v2, sims=224)")
    print("="*60)
    iters = sorted(prod_data.keys())
    print(f"\n  {'iter':>4}  {'total_loss':>10}  {'kl':>8}  {'pol_ent':>8}  {'tgt_ent':>8}  {'pol_acc%':>8}  {'grad':>7}")
    print("  " + "-"*65)
    for it in iters:
        d = prod_data[it]
        tl  = get_nested(d, "train_metrics.total_loss")
        kl  = get_nested(d, "train_metrics.kl_divergence")
        pe  = get_nested(d, "train_metrics.policy_entropy")
        te  = get_nested(d, "train_metrics.target_entropy")
        pa  = get_nested(d, "train_metrics.policy_accuracy")
        gn  = get_nested(d, "train_metrics.grad_norm")
        def _f(v): return f"{v:.4f}" if v is not None else "  ---"
        def _p(v): return f"{v*100:.1f}%" if v is not None else " ---"
        print(f"  {it:>4}  {_f(tl):>10}  {_f(kl):>8}  {_f(pe):>8}  {_f(te):>8}  {_p(pa):>8}  {_f(gn):>7}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="KL diagnostic report")
    parser.add_argument(
        "--runs-dir", type=Path, default=DEFAULT_RUNS,
        help="Root directory containing qd_flat/ and qd_hier/ (default: quick_diag/runs)"
    )
    parser.add_argument(
        "--compare-prod", action="store_true",
        help="Also print production run (patchwork_fresh_v2) metrics for reference"
    )
    args = parser.parse_args()

    runs_dir = args.runs_dir.resolve() if not args.runs_dir.is_absolute() else args.runs_dir

    # Auto-discover all qd_* run directories
    all_run_dirs = {
        "flat":    runs_dir / "qd_flat"     / "committed",
        "hier":    runs_dir / "qd_hier"     / "committed",
        "e3":      runs_dir / "qd_hier_e3"  / "committed",
    }

    available = {}
    for label, path in all_run_dirs.items():
        data = load_run(path)
        if data:
            available[label] = data

    if not available:
        print("No diagnostic run data found.")
        print(f"Expected committed iterations under: {runs_dir}")
        print("Run:  python quick_diag/run_diag.py --mode both")
        print("      python quick_diag/run_diag.py --mode e3")
        return

    all_iters = sorted(set(it for rd in available.values() for it in rd))

    print("\n" + "="*60)
    print("  QUICK_DIAG — POLICY HEAD KL COMPARISON")
    print("="*60)
    print(f"  Runs found: {list(available.keys())}")
    print(f"  Iterations: {all_iters}\n")

    # Per-metric section tables (cleaner than one huge table for many metrics)
    for section_name, metrics in [("TRAINING METRICS", TRAIN_METRICS), ("SELF-PLAY STATS", SELFPLAY_METRICS)]:
        print(f"\n  {section_name}")
        print("  " + "-"*55)
        col_w = 9
        hdr = f"  {'iter':>4}"
        for rid in available:
            for mname, _ in metrics:
                hdr += f"  {(rid+':'+mname)[:col_w]:>{col_w}}"
        if "flat" in available and "hier" in available and "kl_divergence" in [m for m, _ in metrics]:
            hdr += f"  {'Δ_kl(H-F)':>12}"
        if "e3" in available and "hier" in available and "kl_divergence" in [m for m, _ in metrics]:
            hdr += f"  {'Δ_kl(E3-H)':>12}"
        print(hdr)
        print("  " + "-"*len(hdr.strip()))

        for it in all_iters:
            row = f"  {it:>4}"
            row_vals = {}
            for rid in available:
                iter_data = available[rid].get(it, {})
                for mname, jpath in metrics:
                    v = get_nested(iter_data, jpath)
                    row_vals[(rid, mname)] = v
                    row += f"  {fmt(v, mname):>{col_w}}"
            if "flat" in available and "hier" in available and "kl_divergence" in [m for m, _ in metrics]:
                row += delta_marker(
                    row_vals.get(("flat", "kl_divergence")),
                    row_vals.get(("hier", "kl_divergence")),
                    "kl_divergence",
                )
            if "e3" in available and "hier" in available and "kl_divergence" in [m for m, _ in metrics]:
                row += delta_marker(
                    row_vals.get(("hier", "kl_divergence")),
                    row_vals.get(("e3",   "kl_divergence")),
                    "kl_divergence",
                )
            print(row)

    flat_data    = available.get("flat", {})
    hier_data    = available.get("hier", {})
    e3_data      = available.get("e3",   {})

    if flat_data and hier_data:
        kl_verdict(flat_data, hier_data)

    if e3_data and hier_data:
        print("\n" + "="*60)
        print("  E3 VERDICT  (optimizer resume  vs  cold-start each iter)")
        print("="*60)
        iters = sorted(set(e3_data) | set(hier_data))
        last  = max((i for i in iters if e3_data.get(i) and hier_data.get(i)), default=None)
        if last is not None:
            h_kl  = get_nested(hier_data[last], "train_metrics.kl_divergence")
            e3_kl = get_nested(e3_data[last],   "train_metrics.kl_divergence")
            h_pe  = get_nested(hier_data[last], "train_metrics.policy_entropy")
            e3_pe = get_nested(e3_data[last],   "train_metrics.policy_entropy")
            h_te  = get_nested(hier_data[last], "train_metrics.target_entropy")
            e3_te = get_nested(e3_data[last],   "train_metrics.target_entropy")
            h_pa  = get_nested(hier_data[last], "train_metrics.policy_accuracy")
            e3_pa = get_nested(e3_data[last],   "train_metrics.policy_accuracy")
            print(f"\n  At iter {last}:")
            print(f"    hier (cold)    KL={h_kl:.4f}  PE={h_pe:.3f}  TE={h_te:.3f}  pol_acc={h_pa*100:.1f}%")
            print(f"    hier (resume)  KL={e3_kl:.4f}  PE={e3_pe:.3f}  TE={e3_te:.3f}  pol_acc={e3_pa*100:.1f}%")
            diff = e3_kl - h_kl
            print(f"    Δ KL (resume−cold) = {diff:+.4f}")
            print()
            if diff < -0.15:
                print("  VERDICT: OPTIMIZER RESUME IS THE PRIMARY KL DRIVER")
                print("  Enable resume in production config immediately.")
            elif diff < -0.05:
                print("  VERDICT: OPTIMIZER RESUME CONTRIBUTES (~{:.0f}% reduction)".format(100 * abs(diff) / h_kl))
                print("  Worth enabling in production; other factors also present.")
            else:
                print("  VERDICT: OPTIMIZER RESUME NOT THE PRIMARY DRIVER")
                print("  KL is structural (sim budget / replay noise). Resume alone won't fix it.")

    if args.compare_prod:
        prod_data = load_run(PROD_RUNS)
        print_prod_comparison(prod_data)

    print()


if __name__ == "__main__":
    main()
