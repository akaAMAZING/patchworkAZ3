#!/usr/bin/env python3
"""
Build a single comprehensive CSV of ALL metrics for every iteration (0 through 192).

Sources:
- logs/metadata.jsonl: one row per iteration (train.*, selfplay.*, and top-level fields).
- runs/patchwork_production/committed/iter_*/iteration_*.json: when present, merges
  train_metrics, selfplay_stats, eval_results, applied_settings (flattened) so no metric is omitted.

Output:
- docs/full_metrics_0_192.csv: one row per iteration, columns = union of all keys (flattened).
- docs/METRICS_GLOSSARY.md: each column name, source, and how it is calculated (code refs).

Usage:
  python tools/build_full_metrics_csv.py
  python tools/build_full_metrics_csv.py --metadata logs/metadata.jsonl --out docs/full_metrics_0_192.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def flatten_dict(d: dict, prefix: str = "") -> dict:
    """Flatten nested dict into prefix_key. Leaves are scalar or list (converted to string if long)."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}_{k}" if prefix else k
        if v is None:
            out[key] = ""
        elif isinstance(v, dict) and not (k in ("sprt",) and prefix):
            out.update(flatten_dict(v, key))
        elif isinstance(v, (list, tuple)):
            if len(v) > 20 or (v and isinstance(v[0], dict)):
                out[key] = ""  # Skip huge or nested lists in CSV
            else:
                out[key] = ";".join(str(x) for x in v)
        else:
            out[key] = v
    return out


def load_metadata(path: Path, iter_start: int, iter_end: int) -> list[dict]:
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


def load_committed_iteration_json(run_root: Path, iteration: int) -> dict | None:
    p = run_root / "committed" / f"iter_{iteration}" / f"iteration_{iteration}.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def row_from_metadata(entry: dict) -> dict:
    """Build flat row from metadata.jsonl entry (train + selfplay nested)."""
    row = {
        "iteration": entry.get("iteration", ""),
        "timestamp_utc": entry.get("timestamp_utc", ""),
        "config_hash": entry.get("config_hash", ""),
        "config_path": entry.get("config_path", ""),
        "best_model_hash": entry.get("best_model_hash", ""),
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
    train = entry.get("train") or {}
    for k, v in train.items():
        row[f"train_{k}"] = v
    selfplay = entry.get("selfplay") or {}
    for k, v in selfplay.items():
        row[f"selfplay_{k}"] = v
    if "sprt" in entry and isinstance(entry["sprt"], dict):
        for k, v in entry["sprt"].items():
            row[f"sprt_{k}"] = v
    return row


def merge_committed_into_row(row: dict, committed: dict) -> None:
    """Merge committed iteration_*.json into row (in-place). Prefer committed for overlapping keys."""
    # train_metrics
    for k, v in (committed.get("train_metrics") or {}).items():
        row[f"train_{k}"] = v
    # selfplay_stats (keys may already be "selfplay_*" in iteration_*.json)
    for k, v in (committed.get("selfplay_stats") or {}).items():
        col = k if k.startswith("selfplay_") else f"selfplay_{k}"
        row[col] = v
    # top-level
    for k in ("accepted", "iteration_time_s", "replay_buffer_positions", "replay_buffer_iterations"):
        if k in committed:
            if k == "replay_buffer_positions":
                row["replay_positions"] = committed[k]
            elif k == "replay_buffer_iterations":
                row["replay_buffer_iterations"] = committed[k]
            else:
                row[k] = committed[k]
    # eval_results.vs_previous_best (flatten)
    er = committed.get("eval_results") or {}
    vb = er.get("vs_previous_best") or {}
    for k, v in vb.items():
        if k != "results":
            row[f"eval_vs_previous_best_{k}"] = v
    # applied_settings (flatten)
    ap = committed.get("applied_settings") or {}
    for section, sub in ap.items():
        if isinstance(sub, dict):
            for k, v in sub.items():
                if isinstance(v, (list, dict)) and k != "results":
                    continue
                row[f"applied_{section}_{k}"] = v
        else:
            row[f"applied_{section}"] = sub


def get_all_columns(rows: list[dict]) -> list[str]:
    seen = set()
    cols = []
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                cols.append(k)
    # Canonical order: iteration first, then logical groups
    priority = (
        "iteration", "timestamp_utc", "config_hash", "config_path", "best_model_hash",
        "accepted", "global_step", "iter_time_s", "iteration_time_s",
        "replay_positions", "replay_buffer_positions", "replay_buffer_iterations",
        "consecutive_rejections", "best_model",
        "eval_vs_best_wr", "eval_vs_best_margin", "eval_vs_mcts_wr", "elo",
    )
    ordered = [c for c in priority if c in seen]
    rest = [c for c in cols if c not in ordered]
    return ordered + sorted(rest)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build full metrics CSV (iter 0–192) and glossary")
    ap.add_argument("--metadata", type=str, default=None, help="Path to metadata.jsonl")
    ap.add_argument("--out", type=str, default=None, help="Output CSV path")
    ap.add_argument("--run-root", type=str, default=None, help="Run root (committed dir)")
    ap.add_argument("--iter-end", type=int, default=192, help="Last iteration (inclusive)")
    args = ap.parse_args()

    metadata_path = Path(args.metadata) if args.metadata else REPO_ROOT / "logs" / "metadata.jsonl"
    out_path = Path(args.out) if args.out else REPO_ROOT / "docs" / "full_metrics_0_192.csv"
    run_root = Path(args.run_root) if args.run_root else REPO_ROOT / "runs" / "patchwork_production"
    iter_end = args.iter_end

    if not metadata_path.exists():
        print(f"Metadata not found: {metadata_path}", file=sys.stderr)
        sys.exit(1)

    meta_rows = load_metadata(metadata_path, 0, iter_end)
    if not meta_rows:
        print(f"No metadata entries in range 0–{iter_end}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for entry in meta_rows:
        it = entry.get("iteration", 0)
        row = row_from_metadata(entry)
        committed = load_committed_iteration_json(run_root, it)
        if committed:
            merge_committed_into_row(row, committed)
        rows.append(row)

    cols = get_all_columns(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, "") for c in cols})

    print(f"Wrote {len(rows)} rows, {len(cols)} columns to {out_path}")

    # Write glossary
    glossary_path = REPO_ROOT / "docs" / "METRICS_GLOSSARY.md"
    write_glossary(cols, glossary_path)
    print(f"Wrote glossary to {glossary_path}")


def write_glossary(columns: list[str], path: Path) -> None:
    """Write METRICS_GLOSSARY.md with source/code ref for each column."""
    # Predefined glossary entries (column_name -> (description, source_file, code_ref))
    GLOSSARY = {
        "iteration": ("Zero-based iteration index (0..192).", "logs/metadata.jsonl", "entry['iteration']"),
        "timestamp_utc": ("UTC timestamp when iteration was committed.", "src/training/main.py", "_append_metadata"),
        "config_hash": ("Hash of config used for this run.", "src/training/main.py", "_config_hash"),
        "config_path": ("Path to YAML config file.", "configs/config_best.yaml", "paths"),
        "best_model_hash": ("File hash of best model checkpoint.", "src/training/main.py", "_file_hash(best_model_path)"),
        "accepted": ("True if model was accepted (promoted). With eval disabled, always True.", "src/training/main.py", "_original_gate / _should_accept_model"),
        "global_step": ("Cumulative training step count across all iterations.", "src/training/main.py", "train_iteration() return"),
        "iter_time_s": ("Wall-clock time in seconds for this iteration (selfplay + train + commit).", "src/training/main.py", "time.time() - iter_start"),
        "replay_positions": ("Total positions in replay buffer after this iteration.", "src/training/main.py", "self.replay_buffer.total_positions"),
        "replay_buffer_positions": ("Same as replay_positions (from committed iteration_*.json).", "runs/.../committed/iter_*/iteration_*.json", "summary['replay_buffer_positions']"),
        "replay_buffer_iterations": ("Number of iterations currently in replay window.", "src/training/replay_buffer.py", "len(self._entries)"),
        "consecutive_rejections": ("Count of consecutive gate rejections (0 when no gating).", "src/training/main.py", "self.consecutive_rejections"),
        "best_model": ("Path to best model checkpoint.", "src/training/main.py", "self.best_model_path"),
        "eval_vs_best_wr": ("Win rate vs previous best (when eval games run).", "src/training/main.py", "eval_results['vs_previous_best']['win_rate']"),
        "eval_vs_best_margin": ("Avg score margin vs previous best.", "src/training/evaluation.py", "avg_model_score_margin"),
        "eval_vs_mcts_wr": ("Win rate vs pure MCTS (when eval games run).", "src/training/evaluation.py", "evaluate_vs_baseline(..., 'pure_mcts')"),
        "elo": ("ELO rating (when evaluation.elo.enabled).", "tools/elo_system.py", "get_rating(player_id)"),
        "train_policy_loss": ("Cross-entropy loss between target policy and network policy.", "src/network/model.py", "get_loss: -(target_policy_norm * log_probs).sum(dim=-1).mean()"),
        "train_value_loss": ("MSE between value head and target value.", "src/network/model.py", "get_loss: F.mse_loss(value, target_value)"),
        "train_score_loss": ("Cross-entropy of 201-bin score head vs soft target from score_margins.", "src/network/model.py", "get_loss: -(tgt * logp).sum(dim=-1).mean()"),
        "train_ownership_loss": ("BCE with logits for ownership head (2×9×9).", "src/network/model.py", "get_loss: F.binary_cross_entropy_with_logits(ownership_logits, target_ownership)"),
        "train_total_loss": ("policy_weight*policy_loss + value_weight*value_loss + score_weight*score_loss + ownership_weight*ownership_loss.", "src/network/model.py", "get_loss"),
        "train_policy_accuracy": ("Fraction of samples where argmax(policy) == argmax(target).", "src/network/model.py", "get_loss: (pred_actions == target_actions).float().mean()"),
        "train_policy_top5_accuracy": ("Fraction where target action in top-5 predicted.", "src/network/model.py", "get_loss: (top5_pred == target_actions.unsqueeze(1)).any(dim=1).float().mean()"),
        "train_value_mse": ("Same as train_value_loss (MSE).", "src/network/model.py", "value_loss.item()"),
        "train_grad_norm": ("Global gradient norm (before clip).", "src/training/trainer.py", "torch.nn.utils.clip_grad_norm_; grad_norm.item()"),
        "train_policy_entropy": ("Mean entropy of network policy over batch.", "src/network/model.py", "-(policy_probs * log_probs).sum(dim=-1).mean()"),
        "train_target_entropy": ("Mean entropy of MCTS target policy.", "src/network/model.py", "-(target_policy_norm * target_log).sum(dim=-1).mean()"),
        "train_policy_cross_entropy": ("Same as policy_loss (CE).", "src/network/model.py", "policy_loss"),
        "train_ce_minus_policy_entropy": ("CE - H(pi); diagnostic.", "src/network/model.py", "policy_cross_entropy - policy_entropy"),
        "train_kl_divergence": ("KL(MCTS_target || network_policy).", "src/network/model.py", "(target_policy_norm * (target_log - log_probs)).sum(dim=-1).mean()"),
        "train_approx_identity_check": ("|H(pi)+KL(pi||p)-CE| mean; should be ~0.", "src/network/model.py", "approx_identity_error"),
        "train_ownership_accuracy": ("Binary accuracy of ownership head (threshold 0.5).", "src/network/model.py", "(ownership_pred == target_ownership).float().mean()"),
        "train_ownership_filled_fraction_mean": ("Mean of target ownership (filled cells).", "src/network/model.py", "target_ownership.mean()"),
        "train_ownership_accuracy_all_filled_baseline": ("Baseline acc if predicting all filled.", "src/network/model.py", "ownership_filled_fraction_mean"),
        "train_ownership_empty_recall": ("Recall for empty cells (target=0).", "src/network/model.py", "get_loss ownership_empty_recall"),
        "train_ownership_empty_precision": ("Precision for empty cells.", "src/network/model.py", "get_loss ownership_empty_precision"),
        "train_ownership_balanced_accuracy": ("0.5*(empty_recall + filled_recall).", "src/network/model.py", "get_loss ownership_balanced_accuracy"),
        "train_ownership_mae_empty_count": ("MAE of predicted vs true empty count per sample.", "src/network/model.py", "get_loss ownership_mae_empty_count"),
        "train_step_skip_rate": ("Fraction of steps skipped (e.g. NaN).", "src/training/trainer.py", "steps_skipped / num_batches"),
        "selfplay_num_games": ("Number of self-play games this iteration.", "src/training/selfplay_optimized_integration.py", "_compute_stats: len(summaries)"),
        "selfplay_num_positions": ("Total positions collected.", "src/training/selfplay_optimized_integration.py", "sum(s.get('num_positions',0) for s in summaries)"),
        "selfplay_avg_game_length": ("Mean game length in moves.", "src/training/selfplay_optimized_integration.py", "np.mean(game_lengths)"),
        "selfplay_p0_wins": ("Games won by player 0.", "src/training/selfplay_optimized_integration.py", "winners.count(0)"),
        "selfplay_p1_wins": ("Games won by player 1.", "src/training/selfplay_optimized_integration.py", "winners.count(1)"),
        "selfplay_generation_time": ("Wall time for self-play (seconds).", "src/training/selfplay_optimized_integration.py", "generation_time"),
        "selfplay_games_per_minute": ("num_games / (generation_time/60).", "src/training/selfplay_optimized_integration.py", "_compute_stats"),
        "selfplay_avg_policy_entropy": ("Mean policy entropy over root nodes.", "src/training/selfplay_optimized_integration.py", "np.mean(entropies)"),
        "selfplay_avg_top1_prob": ("Mean top-1 action probability.", "src/training/selfplay_optimized_integration.py", "np.mean(top1_probs)"),
        "selfplay_avg_num_legal": ("Mean number of legal actions at root.", "src/training/selfplay_optimized_integration.py", "np.mean(num_legals)"),
        "selfplay_avg_redundancy": ("Position redundancy (0=unique).", "src/training/selfplay_optimized_integration.py", "np.mean(redundancies)"),
        "selfplay_unique_positions": ("Unique position count.", "src/training/selfplay_optimized_integration.py", "sum(s.get('unique_positions',0) for s in summaries)"),
        "selfplay_avg_root_q": ("Mean root Q value.", "src/training/selfplay_optimized_integration.py", "np.mean(root_qs)"),
        "selfplay_avg_final_empty_squares_mean": ("Mean empty squares at game end (packing quality).", "src/utils/packing_metrics.py", "aggregate_packing_over_games"),
        "selfplay_avg_final_empty_components_mean": ("Mean connected empty components.", "src/utils/packing_metrics.py", "aggregate_packing_over_games"),
        "selfplay_avg_final_isolated_1x1_holes_mean": ("Mean isolated 1x1 holes.", "src/utils/packing_metrics.py", "aggregate_packing_over_games"),
        "selfplay_p50_final_empty_squares_mean": ("Median empty squares.", "src/utils/packing_metrics.py", "aggregate_packing_over_games"),
        "selfplay_p90_final_empty_squares_mean": ("90th percentile empty squares.", "src/utils/packing_metrics.py", "aggregate_packing_over_games"),
        "selfplay_avg_root_legal_count": ("Mean root legal action count.", "src/utils/packing_metrics.py", "aggregate_root_over_moves"),
        "selfplay_avg_root_expanded_count": ("Mean root expanded nodes.", "src/utils/packing_metrics.py", "aggregate_root_over_moves"),
        "selfplay_avg_root_expanded_ratio": ("expanded/legal ratio.", "src/utils/packing_metrics.py", "aggregate_root_over_moves"),
        "selfplay_frac_games_vs_packer": ("Fraction of games vs packer (opponent_mix).", "src/training/selfplay_optimized_integration.py", "len(vs_packer_games)/len(summaries)"),
        "selfplay_nn_vs_packer_winrate": ("Win rate in vs-packer games.", "src/training/selfplay_optimized_integration.py", "_compute_stats"),
        "applied_selfplay_games": ("Games actually run (after adaptive_games).", "src/training/main.py", "applied_settings.selfplay.games"),
        "applied_selfplay_temperature": ("Temperature used.", "src/training/main.py", "schedule lookup"),
        "applied_selfplay_simulations": ("MCTS simulations used.", "src/training/main.py", "mcts_schedule"),
        "applied_selfplay_cpuct": ("PUCT constant used.", "src/training/main.py", "cpuct_schedule"),
        "applied_training_lr": ("Learning rate used.", "src/training/main.py", "lr_schedule"),
        "applied_replay_window_iterations": ("Replay window size.", "src/training/main.py", "window_iterations_schedule"),
    }
    lines = [
        "# Full Metrics Glossary",
        "",
        "Every column in `full_metrics_0_192.csv` with source and how it is calculated.",
        "TensorBoard tags map to these: `train/*` → train_*, `selfplay/*` → selfplay_*, `iter/*` → train_*, `buffer/*` → replay_*.",
        "",
        "| Column | Description | Source | Code / Formula |",
        "|--------|-------------|--------|----------------|",
    ]
    for col in columns:
        desc, src, code = GLOSSARY.get(col, ("(See code or TensorBoard tag.)", "—", "—"))
        lines.append(f"| `{col}` | {desc} | {src} | {code} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
