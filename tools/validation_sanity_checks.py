#!/usr/bin/env python
"""
Validation set sanity checks for Patchwork AlphaZero.

Prints ownership class balance, baseline accuracies, and more informative metrics
(per-class precision/recall, macro-F1, ECE) to interpret model performance.

Worth running once per training run to establish baselines and sanity.

Usage:
  # From latest committed iteration (auto-discovers run from config):
  python tools/validation_sanity_checks.py --config configs/config_best.yaml

  # From specific HDF5 file:
  python tools/validation_sanity_checks.py --config configs/config_best.yaml --data path/to/selfplay.h5

  # With model (adds per-class metrics, ECE):
  python tools/validation_sanity_checks.py --config configs/config_best.yaml --checkpoint path/to/iteration_025.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _split_indices(n: int, val_split: float, seed: int):
    """Same logic as trainer._split_indices."""
    import torch
    val_size = int(n * val_split)
    train_size = n - val_size
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    return perm[:train_size], perm[train_size:]


def compute_ownership_metrics(
    ownerships: np.ndarray,
    preds: np.ndarray | None = None,
    probs: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
) -> dict:
    """
    ownerships: (N, 2, 9, 9) float32, values in {0, 1} or -1 (invalid)
    preds: (N, 2, 9, 9) float32, 0/1 predictions (if from model)
    probs: (N, 2, 9, 9) float32, probabilities (if from model, for ECE)
    valid_mask: (N,) bool, True = sample has valid ownership targets
    """
    out = {}
    flat = ownerships.reshape(ownerships.shape[0], -1)
    if valid_mask is None:
        valid_mask = (flat.min(axis=1) >= 0)
    valid = ownerships[valid_mask]
    if valid.size == 0:
        return out

    # Per-channel class balance (ch0 = P0 board, ch1 = P1 board)
    for ch, name in enumerate(["P0_board", "P1_board"]):
        v = valid[:, ch, :, :].flatten()
        n_total = len(v)
        n_empty = int((v == 0).sum())
        n_filled = int((v == 1).sum())
        out[f"ch{ch}_{name}_empty_frac"] = n_empty / n_total if n_total else 0
        out[f"ch{ch}_{name}_filled_frac"] = n_filled / n_total if n_total else 0

    # Combined (both channels) for "per-cell" view
    all_cells = valid.reshape(-1)
    n_empty = int((all_cells == 0).sum())
    n_filled = int((all_cells == 1).sum())
    n_total = n_empty + n_filled
    out["combined_empty_frac"] = n_empty / n_total if n_total else 0
    out["combined_filled_frac"] = n_filled / n_total if n_total else 0

    # Baseline: always predict empty
    baseline_empty_acc = n_empty / n_total if n_total else 0
    out["baseline_always_empty_acc"] = baseline_empty_acc

    if preds is not None and valid_mask is not None:
        pred_valid = preds[valid_mask]
        target_flat = valid.reshape(-1)
        pred_flat = pred_valid.reshape(-1)
        # Overall accuracy
        out["accuracy"] = float(np.mean(target_flat == pred_flat))

        # Per-class precision/recall/F1 (binary: 0=empty, 1=filled)
        for cls, label in [(0, "empty"), (1, "filled")]:
            tp = ((pred_flat == cls) & (target_flat == cls)).sum()
            fp = ((pred_flat == cls) & (target_flat != cls)).sum()
            fn = ((pred_flat != cls) & (target_flat == cls)).sum()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            out[f"{label}_precision"] = float(prec)
            out[f"{label}_recall"] = float(rec)
            out[f"{label}_f1"] = float(f1)

        # Macro-F1
        out["macro_f1"] = (out["empty_f1"] + out["filled_f1"]) / 2

        # ECE (Expected Calibration Error) if probs provided
        if probs is not None:
            prob_valid = probs[valid_mask].reshape(-1)
            out["ece"] = _compute_ece(prob_valid, pred_flat, target_flat.astype(np.float32))

    return out


def _compute_ece(probs: np.ndarray, preds: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> float:
    """
    ECE for binary classification: bin by predicted probability, compare
    mean predicted prob vs mean actual accuracy in each bin.
    """
    # Use predicted class's probability: for class 1, that's probs; for class 0, it's 1-probs
    # We predict 1 when probs > 0.5, so confidence = max(probs, 1-probs)
    confidence = np.where(preds == 1, probs, 1 - probs)
    confidence = np.clip(confidence, 1e-6, 1 - 1e-6)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = 0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidence >= lo) & (confidence < hi)
        if i == n_bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        n = mask.sum()
        if n == 0:
            continue
        acc = (preds[mask] == targets[mask]).mean()
        conf = confidence[mask].mean()
        ece += n * abs(acc - conf)
        total += n
    return float(ece / total) if total > 0 else 0.0


def compute_policy_metrics(policies: np.ndarray, valid_mask: np.ndarray | None = None) -> dict:
    """
    policies: (N, 2026) float32, target policy (MCTS)
    """
    out = {}
    if valid_mask is None:
        valid_mask = np.ones(policies.shape[0], dtype=bool)
    pol = policies[valid_mask]
    targets = pol.argmax(axis=1)

    # Action distribution
    unique, counts = np.unique(targets, return_counts=True)
    sorted_idx = np.argsort(-counts)
    top10_actions = [(int(unique[sorted_idx[i]]), int(counts[sorted_idx[i]])) for i in range(min(10, len(unique)))]
    out["top10_actions"] = top10_actions
    majority_action = int(unique[sorted_idx[0]])
    majority_count = int(counts[sorted_idx[0]])
    out["majority_action"] = majority_action
    out["majority_frac"] = majority_count / len(targets) if len(targets) else 0
    out["baseline_always_majority_acc"] = out["majority_frac"]
    out["unique_actions"] = len(unique)
    return out


def run_checks(
    data_path: str | Path,
    config: dict,
    checkpoint_path: str | Path | None = None,
    max_val_samples: int | None = None,
) -> None:
    import h5py
    from src.network.model import create_network, load_model_checkpoint

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    val_split = float(config.get("training", {}).get("val_split", 0.08))
    seed = int(config.get("seed", 42))

    print("=" * 70)
    print("VALIDATION SET SANITY CHECKS")
    print("=" * 70)
    print(f"Data: {data_path}")
    print(f"Val split: {val_split:.1%}  Seed: {seed}")
    print()

    with h5py.File(data_path, "r") as f:
        n = int(f["states"].shape[0])
        has_ownership = "ownerships" in f

    _, val_indices = _split_indices(n, val_split, seed)
    if max_val_samples:
        val_indices = val_indices[:max_val_samples]
    n_val = len(val_indices)
    print(f"Validation samples: {n_val}")

    # HDF5 fancy indexing requires sorted indices
    val_sorted = sorted(val_indices)

    # Load validation slice
    with h5py.File(data_path, "r") as f:
        policies = np.array(f["policies"][val_sorted], dtype=np.float32)
        if has_ownership:
            ownerships = np.array(f["ownerships"][val_sorted], dtype=np.float32)
            valid_own = (ownerships.reshape(n_val, -1).min(axis=1) >= 0)
        else:
            ownerships = None
            valid_own = None

    # --- Policy metrics ---
    print("\n--- POLICY (action prediction) ---")
    pol_metrics = compute_policy_metrics(policies)
    print(f"  Unique target actions in val set: {pol_metrics['unique_actions']}")
    print(f"  Top-10 most common actions: (action_id, count)")
    for aid, cnt in pol_metrics["top10_actions"]:
        pct = 100 * cnt / n_val
        print(f"    {aid:5d}: {cnt:6d} ({pct:5.1f}%)")
    print(f"  Baseline 'always predict majority' accuracy: {pol_metrics['baseline_always_majority_acc']:.1%}")
    print("  -> If your policy_accuracy is only slightly above this, gains are modest.")

    # --- Ownership metrics ---
    if has_ownership and valid_own is not None and valid_own.any():
        n_valid = int(valid_own.sum())
        print(f"\n--- OWNERSHIP (per-cell) - {n_valid} samples with valid targets ---")
        own_metrics = compute_ownership_metrics(ownerships, valid_mask=valid_own)

        print("  Class balance (per-cell, both channels combined):")
        print(f"    Empty:  {own_metrics.get('combined_empty_frac', 0):.1%}")
        print(f"    Filled: {own_metrics.get('combined_filled_frac', 0):.1%}")
        print("  Per-channel balance:")
        for ch, label in [(0, "P0 board"), (1, "P1 board")]:
            key_base = f"ch{ch}_P{'0' if ch == 0 else '1'}_board"
            ef = own_metrics.get(f"{key_base}_empty_frac", 0)
            ff = own_metrics.get(f"{key_base}_filled_frac", 0)
            print(f"    {label}: empty={ef:.1%}  filled={ff:.1%}")

        print(f"  Baseline 'always predict empty' accuracy: {own_metrics.get('baseline_always_empty_acc', 0):.1%}")
        print("  -> If empty is ~80%+, then 84% accuracy is not surprising.")

        # Model predictions if checkpoint provided
        if checkpoint_path and Path(checkpoint_path).exists():
            import torch

            net = create_network(config)
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            load_model_checkpoint(net, ckpt["model_state_dict"])
            net.to(device)
            net.eval()

            # Need states for forward pass — load a batch at a time
            batch_size = 256
            all_preds = []
            all_probs = []
            valid_idx = np.where(valid_own)[0]

            with torch.no_grad():
                for i in range(0, len(valid_idx), batch_size):
                    idx_batch = [val_sorted[j] for j in valid_idx[i : i + batch_size]]
                    with h5py.File(data_path, "r") as h5:
                        states = np.array(h5["states"][idx_batch], dtype=np.float32)
                    x = torch.from_numpy(states).to(device)
                    trunk = net._trunk_forward(x)
                    if net.ownership_head is None:
                        break
                    logits = net.ownership_head(trunk)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    preds = (logits > 0).float().cpu().numpy()
                    all_preds.append(preds)
                    all_probs.append(probs)

            if all_preds:
                preds = np.concatenate(all_preds, axis=0)
                probs = np.concatenate(all_probs, axis=0)
                own_metrics = compute_ownership_metrics(ownerships[valid_own], preds=preds, probs=probs)

                print("\n  With model predictions:")
                print(f"    Accuracy: {own_metrics.get('accuracy', 0):.1%}")
                print(f"    Empty  - precision: {own_metrics.get('empty_precision', 0):.3f}  recall: {own_metrics.get('empty_recall', 0):.3f}  F1: {own_metrics.get('empty_f1', 0):.3f}")
                print(f"    Filled - precision: {own_metrics.get('filled_precision', 0):.3f}  recall: {own_metrics.get('filled_recall', 0):.3f}  F1: {own_metrics.get('filled_f1', 0):.3f}")
                print(f"    Macro-F1: {own_metrics.get('macro_f1', 0):.3f}")
                if "ece" in own_metrics:
                    print(f"    ECE (calibration): {own_metrics['ece']:.4f}")
    else:
        if not has_ownership:
            print("\n--- OWNERSHIP: skipped (no ownerships in data) ---")
        else:
            print("\n--- OWNERSHIP: skipped (no valid ownership samples in val set) ---")

    print()
    print("=" * 70)


def main() -> None:
    import yaml
    from src.training.run_layout import committed_dir, get_run_root, max_committed_iteration

    ap = argparse.ArgumentParser(
        description="Print validation set sanity checks (class balance, baselines, metrics)"
    )
    ap.add_argument("--config", type=str, default="configs/config_best.yaml")
    ap.add_argument("--data", type=str, default=None, help="HDF5 path (overrides auto-discovery)")
    ap.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint for ownership metrics")
    ap.add_argument("--max-val-samples", type=int, default=None, help="Cap val samples for quick run")
    args = ap.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_path = args.data
    if not data_path:
        run_root = get_run_root(config)
        if not run_root.exists():
            raise SystemExit(
                f"Run root {run_root} does not exist. Run training first or pass --data <path_to_selfplay.h5>"
            )
        last_iter = max_committed_iteration(run_root)
        if last_iter < 0:
            raise SystemExit(
                f"No committed iterations in {run_root}. Run training first or pass --data <path>"
            )
        data_path = committed_dir(run_root, last_iter) / "selfplay.h5"
        if not Path(data_path).exists():
            raise SystemExit(
                f"Self-play data not found at {data_path}. Pass --data <path> explicitly."
            )
        print(f"Using latest committed iter {last_iter}: {data_path}")

    run_checks(
        data_path=data_path,
        config=config,
        checkpoint_path=args.checkpoint,
        max_val_samples=args.max_val_samples,
    )


if __name__ == "__main__":
    main()
