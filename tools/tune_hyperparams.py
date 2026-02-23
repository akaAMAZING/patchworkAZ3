#!/usr/bin/env python
"""
Optuna hyperparameter tuning for Patchwork AlphaZero — 2D FOCUSED PASS.

Tunes the two most impactful parameters identified from the 5D exploration study:
  1. cpuct            — MCTS exploration constant (importance: 0.48)
  2. q_value_weight   — Mix of MCTS root Q-value vs game outcome (importance: 0.44)

Fixed from 5D best trial (#4, margin=+6.525pts):
  - learning_rate:    0.002  (production value from Adam scaling rule)
  - score_scale:      19.3   (importance: 0.03, any reasonable value works)
  - dirichlet_alpha:  0.08   (importance: 0.00, any reasonable value works)

Key improvements over 5D pass:
  - NO PRUNING — every trial runs to completion (eliminates false negatives)
  - 3-CHECKPOINT EVAL — evaluates iter 7, 9, 11 vs iter1 (smooths trajectory noise)
  - NO IN-LOOP EVAL — saves compute per iteration (was only monitoring noise)

Usage:
    python tools/tune_hyperparams.py --config configs/overnight_strong.yaml \\
        --n-trials 15 --max-iters 12 --eval-games 360 \\
        --tune-dir tuning_2d --study-name patchwork_2d

Resume (automatic — just re-run the same command):
    # Ctrl+C during trial 8? Just re-run. Completed trials are in SQLite DB.
    # --n-trials is a TOTAL target, not additive.

Dashboard:
    pip install optuna-dashboard
    optuna-dashboard sqlite:///tuning_2d/optuna_study.db
"""

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("tune")

# --- Fixed parameters from 5D study best trial (#4) ---
FIXED_LEARNING_RATE = 0.002       # Production LR (Adam scaling rule)
FIXED_SCORE_SCALE = 19.3          # 5D importance: 0.03
FIXED_DIRICHLET_ALPHA = 0.08      # 5D importance: 0.00


def create_trial_config(base_config: dict, trial, trial_dir: Path, max_iters: int) -> Path:
    """Create a modified config for this trial with sampled hyperparameters.

    2D search: only cpuct and q_value_weight are sampled.
    All other parameters are fixed to known-good values.
    In-loop evaluation is disabled (no games_vs_best).
    """
    config = _deep_copy_config(base_config)

    # --- Sample the 2 important hyperparameters ---

    # 1. CPUCT — MCTS exploration constant (5D importance: 0.48)
    #    Range narrowed from 5D (1.0-3.0) based on best results clustering at 1.5-1.7
    cpuct = trial.suggest_float("cpuct", 1.0, 3.0)
    config["selfplay"]["mcts"]["cpuct"] = cpuct

    # 2. Q-value weight — mix MCTS Q-values into training targets (5D importance: 0.44)
    #    Low values (<0.05) consistently scored worst; moderate (0.2-0.35) scored best
    q_value_weight = trial.suggest_float("q_value_weight", 0.05, 0.5)
    config["selfplay"]["q_value_weight"] = q_value_weight

    # --- Fix the 3 unimportant parameters ---
    config["training"]["learning_rate"] = FIXED_LEARNING_RATE
    config["selfplay"]["score_scale"] = FIXED_SCORE_SCALE
    config["selfplay"]["mcts"]["root_dirichlet_alpha"] = FIXED_DIRICHLET_ALPHA

    # --- Isolated per-trial paths ---
    trial_dir.mkdir(parents=True, exist_ok=True)
    config["paths"]["checkpoints_dir"] = str(trial_dir / "checkpoints")
    config["paths"]["logs_dir"] = str(trial_dir / "logs")
    config["paths"]["run_root"] = str(trial_dir)
    config["paths"]["run_id"] = "data"
    config["logging"]["log_file"] = str(trial_dir / "logs" / "training.log")
    config["logging"]["tensorboard"]["log_dir"] = str(trial_dir / "logs" / "tensorboard")

    # --- Fixed overrides for tuning ---
    config["iteration"]["max_iterations"] = max_iters
    config["iteration"]["auto_resume"] = False
    config["selfplay"]["games_per_iteration"] = 200
    # NO in-loop evaluation — saves ~12 min/trial, was only monitoring noise
    config["evaluation"]["games_vs_best"] = 0
    config["league"]["enabled"] = False
    # Disable all schedules — flat values for clean parameter isolation
    config["iteration"]["games_schedule"] = []
    config["iteration"]["mcts_schedule"] = []
    config["iteration"]["window_iterations_schedule"] = []
    config["iteration"]["temperature_schedule"] = []
    config["iteration"]["dirichlet_alpha_schedule"] = []
    config["iteration"]["noise_weight_schedule"] = []
    config["iteration"]["q_value_weight_schedule"] = []
    config["iteration"]["cpuct_schedule"] = []
    # Clear LR schedule too (use fixed LR throughout)
    config["iteration"]["lr_schedule"] = []

    config_path = trial_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    return config_path


def _deep_copy_config(config: dict) -> dict:
    """Deep copy a config dict (avoids YAML anchors / shared references)."""
    return yaml.safe_load(yaml.safe_dump(config))


def _cleanup_gpu():
    """Release GPU memory between trials to prevent OOM accumulation."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass
    gc.collect()


def _release_locks():
    """Release any held run locks from the previous trial."""
    try:
        from src.training import run_layout
        run_layout._release_run_lock()
    except Exception:
        pass


def run_trial_training(config_path: Path, trial) -> dict:
    """Run training for max_iters iterations. No pruning — every trial completes.

    Still reports policy_accuracy to Optuna for dashboard monitoring.
    """
    from src.training.main import AlphaZeroTrainer

    metrics_history = []

    def iteration_callback(iteration, eval_results, train_metrics, selfplay_stats):
        """Called after each iteration. Reports metrics for dashboard (no pruning)."""
        policy_acc = train_metrics.get("policy_accuracy", 0)

        metrics_history.append({
            "iteration": iteration,
            "policy_accuracy": policy_acc,
            "total_loss": train_metrics.get("total_loss", 999),
            "value_mse": train_metrics.get("value_mse", 999),
            "policy_entropy": selfplay_stats.get("avg_policy_entropy", 0),
            "top1_prob": selfplay_stats.get("avg_top1_prob", 1),
            "redundancy": selfplay_stats.get("avg_redundancy", 0),
            "avg_root_q": selfplay_stats.get("avg_root_q", 0),
        })

        # Report for dashboard visualization (NopPruner ignores this)
        trial.report(policy_acc, iteration)

        return True  # always continue — no pruning

    trainer = AlphaZeroTrainer(
        str(config_path),
        iteration_callback=iteration_callback,
    )

    try:
        trainer.train()
    except Exception as e:
        logger.error(f"[Trial {trial.number}] Training failed: {e}", exc_info=True)
        return {"error": str(e)}
    finally:
        _release_locks()
        _cleanup_gpu()

    return {
        "metrics_history": metrics_history,
        "best_model_path": trainer.best_model_path,
        "best_model_iteration": trainer.best_model_iteration,
        "final_policy_accuracy": metrics_history[-1]["policy_accuracy"] if metrics_history else 0,
        "final_total_loss": metrics_history[-1]["total_loss"] if metrics_history else 999,
    }


def evaluate_final_strength(config_path: Path, eval_games: int, max_iters: int) -> dict:
    """Evaluate 3 checkpoints against iter1 to measure strength with trajectory smoothing.

    Instead of evaluating a single checkpoint (which could catch a "bad iteration"),
    evaluates 3 checkpoints spread across the second half of training and averages:
      - iter (max-5): mid-late training
      - iter (max-3): late training
      - iter (max-1): final model

    Each gets eval_games/3 games. The average margin is the Optuna objective.
    This smooths over single-iteration variance while keeping total games constant.

    Returns dict with 'margin' (averaged), 'win_rate' (averaged), and per-checkpoint details.
    """
    import torch
    from src.training.evaluation import Evaluator

    FAIL_RESULT = {"margin": -50.0, "win_rate": 0.0, "checkpoints": []}

    with open(config_path) as f:
        config = yaml.safe_load(f)

    run_root = Path(config["paths"]["run_root"]) / config["paths"]["run_id"]

    # Find iter1 baseline
    iter1_path = run_root / "committed" / "iter_001" / "iteration_001.pt"
    if not iter1_path.exists():
        iter1_path = run_root / "committed" / "iter_000" / "iteration_000.pt"
        if not iter1_path.exists():
            logger.warning("[Eval] No iter1 checkpoint at %s — cannot evaluate", run_root)
            return FAIL_RESULT

    # Determine which 3 checkpoints to evaluate
    # max_iters=12 → iterations 0-11 → evaluate iter 7, 9, 11
    last_iter = max_iters - 1
    eval_iters = [last_iter - 4, last_iter - 2, last_iter]

    # Validate: all must be > 1 (can't be the same as baseline)
    eval_iters = [i for i in eval_iters if i > 1]
    if not eval_iters:
        logger.warning("[Eval] No valid checkpoints to evaluate (max_iters too small)")
        return FAIL_RESULT

    # Find checkpoint paths
    checkpoint_paths = []
    for it in eval_iters:
        cp = run_root / "committed" / f"iter_{it:03d}" / f"iteration_{it:03d}.pt"
        if cp.exists():
            checkpoint_paths.append((it, cp))
        else:
            logger.warning(f"[Eval] Checkpoint iter_{it:03d} not found, skipping")

    if not checkpoint_paths:
        logger.warning("[Eval] No checkpoints found")
        return FAIL_RESULT

    # Distribute games evenly across checkpoints
    games_per_checkpoint = max(20, eval_games // len(checkpoint_paths))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = Evaluator(config, device)

    margins = []
    win_rates = []
    checkpoint_details = []

    try:
        for it, cp_path in checkpoint_paths:
            logger.info(f"[Eval] iter{it:03d} vs iter001 ({games_per_checkpoint} games)")
            try:
                results = evaluator.evaluate_vs_baseline(
                    str(cp_path),
                    baseline_type="previous_best",
                    baseline_path=str(iter1_path),
                    num_games=games_per_checkpoint,
                )
                wr = results.get("win_rate", 0.5)
                margin = results.get("avg_model_score_margin", 0.0)
                logger.info(f"[Eval] iter{it:03d}: WR={wr:.1%}, margin={margin:+.1f}pts")
                margins.append(margin)
                win_rates.append(wr)
                checkpoint_details.append({"iter": it, "margin": margin, "win_rate": wr})
            except Exception as e:
                logger.error(f"[Eval] iter{it:03d} failed: {e}")
                # Skip this checkpoint, don't fail the whole eval
                continue

        if not margins:
            return FAIL_RESULT

        avg_margin = sum(margins) / len(margins)
        avg_wr = sum(win_rates) / len(win_rates)

        logger.info(f"[Eval] AVERAGED: margin={avg_margin:+.1f}pts, WR={avg_wr:.1%} "
                     f"(from {len(margins)} checkpoints)")

        return {"margin": avg_margin, "win_rate": avg_wr, "checkpoints": checkpoint_details}

    finally:
        del evaluator
        _cleanup_gpu()


def objective(trial, base_config: dict, tune_dir: Path, max_iters: int, eval_games: int) -> float:
    """Optuna objective: averaged score margin across 3 checkpoints vs iter1.

    2D search: only cpuct and q_value_weight vary.
    No pruning: every trial runs to completion.
    3-checkpoint eval: smooths over single-iteration variance.
    """
    trial_dir = tune_dir / f"trial_{trial.number:04d}"

    config_path = create_trial_config(base_config, trial, trial_dir, max_iters)

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  TRIAL {trial.number}")
    logger.info(f"  cpuct={trial.params['cpuct']:.3f}  "
                f"q_weight={trial.params['q_value_weight']:.3f}")
    logger.info(f"  Fixed: lr={FIXED_LEARNING_RATE}  scale={FIXED_SCORE_SCALE}  "
                f"dir_alpha={FIXED_DIRICHLET_ALPHA}")
    logger.info(f"  Directory: {trial_dir}")
    logger.info("=" * 60)

    trial_start = time.time()

    trial_result = run_trial_training(config_path, trial)

    if "error" in trial_result:
        logger.warning(f"[Trial {trial.number}] FAILED: {trial_result['error']}")
        return -50.0

    # 3-checkpoint evaluation vs iter1
    eval_result = evaluate_final_strength(config_path, eval_games, max_iters)
    margin = eval_result["margin"]
    win_rate = eval_result["win_rate"]

    trial_time = (time.time() - trial_start) / 60

    # Save detailed results
    results_path = trial_dir / "trial_results.json"
    save_data = {
        "trial_number": trial.number,
        "params": dict(trial.params),
        "fixed_params": {
            "learning_rate": FIXED_LEARNING_RATE,
            "score_scale": FIXED_SCORE_SCALE,
            "dirichlet_alpha": FIXED_DIRICHLET_ALPHA,
        },
        "avg_score_margin": margin,
        "avg_win_rate": win_rate,
        "checkpoint_evals": eval_result.get("checkpoints", []),
        "final_policy_accuracy": trial_result.get("final_policy_accuracy", 0),
        "final_total_loss": trial_result.get("final_total_loss", 999),
        "trial_time_minutes": round(trial_time, 1),
        "metrics_history": trial_result.get("metrics_history", []),
    }
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    logger.info(f"[Trial {trial.number}] DONE — avg_margin={margin:+.1f}pts, "
                f"avg_WR={win_rate:.1%} ({trial_time:.0f} min)")
    return margin


def print_study_results(study):
    """Print summary of the 2D tuning study."""
    print("\n" + "=" * 70)
    print("  HYPERPARAMETER TUNING RESULTS (2D: cpuct + q_value_weight)")
    print("=" * 70)

    all_trials = study.trials
    completed = [t for t in all_trials if t.value is not None and t.state.name == "COMPLETE"]
    failed = [t for t in all_trials if t.state.name == "FAIL"]

    print(f"\n  Total trials:     {len(all_trials)}")
    print(f"  Completed:        {len(completed)}")
    if failed:
        print(f"  Failed:           {len(failed)}")

    print(f"\n  Fixed parameters:")
    print(f"    learning_rate:    {FIXED_LEARNING_RATE}")
    print(f"    score_scale:      {FIXED_SCORE_SCALE}")
    print(f"    dirichlet_alpha:  {FIXED_DIRICHLET_ALPHA}")

    if study.best_trial:
        bt = study.best_trial
        print(f"\n  BEST TRIAL: #{bt.number}")
        print(f"  Avg score margin vs iter1: {bt.value:+.1f} pts")
        print(f"  Parameters:")
        for k, v in bt.params.items():
            print(f"    {k}: {v:.6f}")

    # Show all completed trials sorted by margin
    completed.sort(key=lambda t: t.value or -999, reverse=True)
    if completed:
        print(f"\n  ALL TRIALS (sorted by margin):")
        print(f"  {'#':>4}  {'Margin':>8}  {'CPUCT':>7}  {'QWeight':>8}")
        print(f"  {'-'*4}  {'-'*8}  {'-'*7}  {'-'*8}")
        for t in completed:
            p = t.params
            print(f"  {t.number:>4}  {t.value:>+7.1f}  {p.get('cpuct', 0):>7.3f}  "
                  f"{p.get('q_value_weight', 0):>8.3f}")

    print("\n" + "=" * 70)

    # YAML snippet for production config
    if study.best_trial:
        p = study.best_trial.params
        print("\n  APPLY BEST CONFIG — paste into config_best.yaml:\n")
        print(f"  selfplay:")
        print(f"    q_value_weight: {p['q_value_weight']:.3f}")
        print(f"    mcts:")
        print(f"      cpuct: {p['cpuct']:.3f}")
        print()

    # Parameter importance
    if len(completed) >= 5:
        try:
            import optuna
            importances = optuna.importance.get_param_importances(study)
            print("\n  PARAMETER IMPORTANCE:")
            for param, importance in importances.items():
                bar = "#" * int(importance * 40)
                print(f"    {param:>20s}: {importance:.3f}  {bar}")
            print()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="Optuna 2D hyperparameter tuning (cpuct + q_value_weight)")
    parser.add_argument("--config", type=str, default="configs/overnight_strong.yaml",
                        help="Base config file to tune from")
    parser.add_argument("--n-trials", type=int, default=15,
                        help="Total number of trials (default: 15)")
    parser.add_argument("--max-iters", type=int, default=12,
                        help="Training iterations per trial (default: 12, minimum: 6)")
    parser.add_argument("--eval-games", type=int, default=360,
                        help="Total eval games split across 3 checkpoints (default: 360, 120 each)")
    parser.add_argument("--tune-dir", type=str, default="tuning_2d",
                        help="Directory for all tuning data (default: tuning_2d/)")
    parser.add_argument("--study-name", type=str, default="patchwork_2d",
                        help="Optuna study name")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    try:
        import optuna
    except ImportError:
        print("ERROR: optuna not installed. Run: pip install optuna")
        sys.exit(1)

    # Need at least 6 iterations for 3-checkpoint eval (checkpoints at iter 1, 3, 5 minimum)
    if args.max_iters < 6:
        print(f"ERROR: --max-iters must be >= 6 (got {args.max_iters}). "
              "Need enough iterations for 3-checkpoint evaluation.")
        sys.exit(1)

    # Load base config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}")
        sys.exit(1)
    with open(config_path) as f:
        base_config = yaml.safe_load(f)

    tune_dir = Path(args.tune_dir)
    tune_dir.mkdir(parents=True, exist_ok=True)

    # Optuna study — NO PRUNING (NopPruner)
    db_path = tune_dir / "optuna_study.db"
    storage = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",  # maximize averaged score margin vs iter1
        pruner=optuna.pruners.NopPruner(),  # No pruning — every trial completes
        load_if_exists=True,
    )

    # Time estimate: ~10 min/iter, no pruning, all trials complete
    min_per_trial = args.max_iters * 10 + args.eval_games * 0.05
    est_hours = args.n_trials * min_per_trial / 60

    logger.info("")
    logger.info("=" * 60)
    logger.info("  PATCHWORK ALPHAZERO — 2D HYPERPARAMETER TUNING")
    logger.info("=" * 60)
    logger.info(f"  Search:               cpuct (1.0-3.0) + q_value_weight (0.05-0.5)")
    logger.info(f"  Fixed:                lr={FIXED_LEARNING_RATE}  scale={FIXED_SCORE_SCALE}  "
                f"alpha={FIXED_DIRICHLET_ALPHA}")
    logger.info(f"  Trials:               {args.n_trials}")
    logger.info(f"  Iters per trial:      {args.max_iters}")
    logger.info(f"  Eval:                 3 checkpoints x {args.eval_games // 3} games vs iter1")
    logger.info(f"  Pruning:              DISABLED (every trial completes)")
    logger.info(f"  Estimated time:       {est_hours:.0f} hours")
    logger.info(f"  Tune directory:       {tune_dir.resolve()}")
    logger.info(f"  Dashboard:            optuna-dashboard sqlite:///{db_path}")
    logger.info("=" * 60)

    # Resume logic
    existing = len([t for t in study.trials if t.state.name in ("COMPLETE", "PRUNED")])
    remaining = max(0, args.n_trials - existing)

    if existing > 0:
        logger.info(f"  Resuming: {existing} trials done, {remaining} remaining")

    if remaining == 0:
        logger.info(f"  Study already has {existing} trials (target: {args.n_trials}). Nothing to do.")
        print_study_results(study)
        return

    logger.info("")

    study.optimize(
        lambda trial: objective(trial, base_config, tune_dir, args.max_iters, args.eval_games),
        n_trials=remaining,
        show_progress_bar=True,
    )

    print_study_results(study)


if __name__ == "__main__":
    main()
