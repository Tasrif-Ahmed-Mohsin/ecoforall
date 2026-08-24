"""
Compute 95% Bootstrap Confidence Intervals for Regional Lifts and Baselines
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def bootstrap_lift(actual, pred_base, pred_model, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    N = len(actual)
    if N < 5:
        return 0.0, 0.0, 0.0

    err_base = np.abs(actual - pred_base)
    err_model = np.abs(actual - pred_model)

    mae_b = np.mean(err_base)
    mae_m = np.mean(err_model)
    point_lift = (mae_b - mae_m) / mae_b * 100.0 if mae_b > 0 else 0.0

    boot_lifts = []
    for _ in range(n_boot):
        idx = rng.randint(0, N, size=N)
        b_mae_b = np.mean(err_base[idx])
        b_mae_m = np.mean(err_model[idx])
        if b_mae_b > 0:
            boot_lifts.append((b_mae_b - b_mae_m) / b_mae_b * 100.0)

    ci_lower = np.percentile(boot_lifts, 2.5)
    ci_upper = np.percentile(boot_lifts, 97.5)
    return point_lift, ci_lower, ci_upper


def main():
    p_reg = ROOT / "data" / "benchmarks" / "sovereign_segmentation_regional_breakdown.csv"
    if not p_reg.exists():
        print("Regional breakdown CSV not found.")
        return

    df = pd.read_csv(p_reg)
    print("Regional CSV loaded:")
    print(df.head(20))


if __name__ == "__main__":
    main()
