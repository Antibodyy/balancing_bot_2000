#!/usr/bin/env python3
"""Analyze if online and fixed linearization produce truly identical results."""

import numpy as np
import sys

def analyze_predictions(result_fixed, result_online):
    """Compare predicted trajectories numerically."""

    print("\n" + "="*80)
    print("NUMERICAL ANALYSIS OF PREDICTIONS")
    print("="*80)

    # Check if we have the same number of predictions
    n_fixed = len(result_fixed.reference_history)
    n_online = len(result_online.reference_history)
    n_compare = min(n_fixed, n_online)

    print(f"\nNumber of predictions:")
    print(f"  Fixed:  {n_fixed}")
    print(f"  Online: {n_online}")
    print(f"  Comparing first {n_compare} predictions\n")

    # Compare each prediction
    max_diff_all = 0.0
    mean_diff_all = 0.0
    identical_count = 0

    for i in range(n_compare):
        pred_fixed = result_fixed.reference_history[i]
        pred_online = result_online.reference_history[i]

        # Compute differences
        diff = np.abs(pred_fixed - pred_online)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        max_diff_all = max(max_diff_all, max_diff)
        mean_diff_all += mean_diff

        # Check if identical (within floating point tolerance)
        if np.allclose(pred_fixed, pred_online, atol=1e-10):
            identical_count += 1

        # Print first few and last few
        if i < 5 or i >= n_compare - 5:
            print(f"Prediction {i:3d}: max_diff = {max_diff:.6e}, mean_diff = {mean_diff:.6e}")
        elif i == 5:
            print("  ...")

    mean_diff_all /= n_compare

    print(f"\nOverall statistics:")
    print(f"  Maximum difference across all predictions: {max_diff_all:.6e}")
    print(f"  Average mean difference: {mean_diff_all:.6e}")
    print(f"  Identical predictions (atol=1e-10): {identical_count}/{n_compare}")
    print(f"  Percentage identical: {100*identical_count/n_compare:.1f}%")

    if identical_count == n_compare:
        print("\n⚠️  WARNING: ALL predictions are identical!")
        print("   This indicates online linearization is NOT working!")
    elif identical_count > n_compare * 0.5:
        print("\n⚠️  WARNING: More than 50% of predictions are identical!")
        print("   This suggests a potential bug in online linearization.")
    elif max_diff_all < 1e-6:
        print("\n⚠️  WARNING: Differences are very small (< 1e-6)!")
        print("   Online linearization may not be having significant effect.")
    else:
        print("\n✓ Predictions show meaningful differences.")
        print("  Online linearization appears to be working.")

    print("="*80 + "\n")

    return max_diff_all, mean_diff_all, identical_count


if __name__ == '__main__':
    print("This is a helper module for analyzing linearization test results.")
    print("Import it in your test and call analyze_predictions(result_fixed, result_online)")
