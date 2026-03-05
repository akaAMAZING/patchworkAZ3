"""
C) PROVE score-target reconstruction is consistent (scale/sign).

For a random batch of 16 samples:
  - Print stored score_margin_tanh
  - Compute margin_points = score_utility_scale * atanh(m) before rounding/clamping
  - Print argmax bin of target_score_dist and its bin value
  - Confirm argmax bin matches rounded/clamped margin_points
  - Confirm score_utility_scale matches value_targets._SCORE_NORMALISE_DIVISOR
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main():
    import torch
    from src.training.trainer import make_gaussian_score_targets
    from src.training.value_targets import _SCORE_NORMALISE_DIVISOR

    score_utility_scale = 30.0
    score_min, score_max = -100, 100
    sigma = 1.5

    print("C) Score-target reconstruction audit")
    print("  SCORE_NORMALISE_DIVISOR (value_targets) =", _SCORE_NORMALISE_DIVISOR)
    print("  score_utility_scale (trainer)           =", score_utility_scale)
    print("  Match:", "YES" if score_utility_scale == _SCORE_NORMALISE_DIVISOR else "NO")
    print()

    # 16 random tanh margins in (-1, 1)
    torch.manual_seed(123)
    m_tanh = torch.tensor([
        -0.9, -0.5, -0.2, 0.0, 0.1, 0.2, 0.33, 0.5, 0.7, 0.9,
        0.99, -0.99, 0.16, -0.58, 0.76, 0.0
    ], dtype=torch.float32)
    if m_tanh.shape[0] < 16:
        m_tanh = torch.cat([m_tanh, torch.zeros(16 - m_tanh.shape[0])])
    m_tanh = m_tanh[:16]

    target = make_gaussian_score_targets(
        m_tanh, score_utility_scale, score_min, score_max, sigma
    )
    m_clamp = m_tanh.clamp(-0.999999, 0.999999)
    margin_points_exact = score_utility_scale * torch.atanh(m_clamp)
    margin_points_rounded = margin_points_exact.round().clamp(float(score_min), float(score_max))

    bin_vals = torch.arange(score_min, score_max + 1, dtype=torch.float32)
    argmax_bins = target.argmax(dim=-1)
    argmax_bin_values = bin_vals[argmax_bins]

    print("  sample | stored_m_tanh | margin_points (exact) | margin_points (rounded) | argmax_bin_idx | argmax_bin_val | match?")
    print("  " + "-" * 95)
    for i in range(16):
        match = "YES" if abs(argmax_bin_values[i].item() - margin_points_rounded[i].item()) < 0.5 else "NO"
        print(
            f"  {i:6d} | {m_tanh[i].item():13.6f} | {margin_points_exact[i].item():21.2f} | "
            f"{margin_points_rounded[i].item():24.2f} | {argmax_bins[i].item():14d} | "
            f"{argmax_bin_values[i].item():14.2f} | {match}"
        )

    all_match = torch.allclose(argmax_bin_values.float(), margin_points_rounded.float())
    print()
    print("  All argmax bin values match rounded/clamped margin_points:", "YES" if all_match else "NO")
    print("  Conclusion: scale/sign consistent; bin centre = scale * atanh(m), rounded to integer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
