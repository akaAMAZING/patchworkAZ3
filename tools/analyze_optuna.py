"""Analyze Optuna study — parameter importance, statistics, and visualizations."""
import argparse
from pathlib import Path

import optuna
from optuna.trial import TrialState


def main():
    parser = argparse.ArgumentParser(description="Analyze Optuna patchwork_2d study")
    parser.add_argument("--storage", default="sqlite:///tuning_2d/optuna_study.db",
                        help="Storage URL (default: sqlite:///tuning_2d/optuna_study.db)")
    parser.add_argument("--study-name", default="patchwork_2d", help="Study name")
    parser.add_argument("--plot", action="store_true",
                        help="Generate HTML visualizations (requires plotly, scikit-learn)")
    parser.add_argument("--out-dir", default="tuning_2d/optuna_plots",
                        help="Directory for plot outputs (default: tuning_2d/optuna_plots)")
    args = parser.parse_args()

    study = optuna.load_study(
        study_name=args.study_name,
        storage=args.storage,
    )

    trials = [t for t in study.trials if t.state == TrialState.COMPLETE and t.value is not None]
    trials_sorted = sorted(trials, key=lambda t: t.value, reverse=True)

    print("=" * 60)
    print("  OPTUNA STUDY ANALYSIS")
    print("=" * 60)
    print(f"\n  Total trials:   {len(study.trials)}")
    print(f"  Completed:      {len(trials)}")
    print(f"  Failed/other:   {len(study.trials) - len(trials)}")

    if not trials_sorted:
        print("\n  No completed trials with values. Nothing to analyze.")
        return

    # Best and top 5 (use study.best_trial — Optuna's answer for "what is best?")
    print("\n  BEST (Optuna):")
    bt = study.best_trial
    print(f"    Trial #{bt.number}  margin={bt.value:+.1f}  {bt.params}")

    print("\n  TOP 5:")
    for t in trials_sorted[:5]:
        print(f"    #{t.number:>2}  margin={t.value:>+6.1f}  cpuct={t.params['cpuct']:.3f}  q_value_weight={t.params['q_value_weight']:.3f}")

    # Optuna's parameter importance (fANOVA) — which params matter most
    if len(trials) >= 5:
        try:
            importances = optuna.importance.get_param_importances(study)
            print("\n  PARAMETER IMPORTANCE (fANOVA, normalized to sum=1):")
            for param, imp in importances.items():
                bar = "#" * int(imp * 40) + " " * (40 - int(imp * 40))
                print(f"    {param:>18s}: {imp:.3f}  [{bar}]")
        except Exception as e:
            print(f"\n  Parameter importance failed: {e}")
            print("    (needs scikit-learn: pip install scikit-learn)")
    else:
        print(f"\n  Parameter importance needs ≥5 completed trials (have {len(trials)})")

    # Basic stats
    values = [t.value for t in trials]
    print(f"\n  OBJECTIVE STATS:")
    print(f"    mean:   {sum(values)/len(values):+.1f}")
    print(f"    min:    {min(values):+.1f}")
    print(f"    max:    {max(values):+.1f}")
    print(f"    spread: {max(values) - min(values):.1f}")

    # Visualizations
    if args.plot and len(trials) >= 3:
        try:
            import optuna.visualization as vis
            out = Path(args.out_dir)
            out.mkdir(parents=True, exist_ok=True)

            # Parameter importance
            fig = vis.plot_param_importances(study)
            fig.write_html(str(out / "param_importances.html"))
            print(f"\n  Saved: {out / 'param_importances.html'}")

            # Optimization history
            fig = vis.plot_optimization_history(study)
            fig.write_html(str(out / "optimization_history.html"))
            print(f"  Saved: {out / 'optimization_history.html'}")

            # Slice — effect of each param
            fig = vis.plot_slice(study, params=list(study.best_trial.params.keys()))
            fig.write_html(str(out / "slice.html"))
            print(f"  Saved: {out / 'slice.html'}")

            # Contour — cpuct vs q_value_weight
            fig = vis.plot_contour(study, params=["cpuct", "q_value_weight"])
            fig.write_html(str(out / "contour.html"))
            print(f"  Saved: {out / 'contour.html'}")

        except ImportError as e:
            print(f"\n  Plot failed (missing deps): {e}")
            print("    pip install plotly scikit-learn")
        except Exception as e:
            print(f"\n  Plot failed: {e}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
