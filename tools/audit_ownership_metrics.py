"""
E) Detect the "metric lie" (class imbalance).

On a training batch we now compute (in model.get_loss and return in metrics):
  - ownership_filled_fraction_mean: actual mean filled fraction in targets
  - ownership_accuracy_all_filled_baseline: accuracy if we predicted "all filled" (= filled fraction)
  - ownership_empty_recall: among true empty cells, fraction correctly predicted empty
  - ownership_empty_precision: among predicted empty, fraction truly empty
  - ownership_balanced_accuracy: 0.5 * (empty_recall + filled_recall)
  - ownership_mae_empty_count: mean absolute error of predicted empty count per sample

If train_ownership_accuracy plateaus at ~0.86 but filled fraction is ~0.90, then
predicting "all filled" would give 90% accuracy — so the 86% is actually *worse*
than the baseline for the majority class. Empty-cell recall and balanced_accuracy
reveal whether the model is learning empty cells at all.

Code additions: see src/network/model.py get_loss() — new metrics in the returned
dict and in the ownership block. They are logged to TensorBoard when in
main.py _iter_whitelist (iter/ownership_*).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main():
    print("E) Ownership metric lie (class imbalance)")
    print()
    print("New metrics added in model.get_loss (see src/network/model.py):")
    print("  - ownership_filled_fraction_mean: mean(target) = filled fraction in batch")
    print("  - ownership_accuracy_all_filled_baseline: accuracy if predict all 1 (= filled fraction)")
    print("  - ownership_empty_recall: TP_empty / (TP_empty + FN_empty)")
    print("  - ownership_empty_precision: TP_empty / (TP_empty + FP_empty)")
    print("  - ownership_balanced_accuracy: 0.5 * (empty_recall + filled_recall)")
    print("  - ownership_mae_empty_count: mean over samples of |pred_empty_count - target_empty_count|")
    print()
    print("Interpretation: If train_ownership_accuracy ~ 0.86 and filled_fraction_mean ~ 0.90,")
    print("then 'all filled' baseline would get 90%%; true accuracy 86%% is worse. Check")
    print("ownership_empty_recall and ownership_balanced_accuracy to see if empty cells are learned.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
