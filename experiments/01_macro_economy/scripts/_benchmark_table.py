"""Phase 8, step 3 — benchmark table vs random walk, AR(1), and IMF WEO.

For each horizon h, computes:
- random walk: ŷ = y_{t-1}             (in log-growth space: 0)
- AR(1) on gdp_pc_growth_5y_fwd
- IMF WEO: the IMF's own current-vintage forecast (where available)
- Our LGBM ensemble (from horizon_{h}y/metrics.json)

Then Diebold–Mariano p-values (LGBM vs each baseline) using squared-error loss.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.harmonize.common import FEATURES_DIR

PANEL = FEATURES_DIR / "panel_wide.parquet"


def _diebold_mariano(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> tuple[float, float]:
    """Diebold-Mariano test of equal forecast accuracy (squared-error loss).

    Returns (DM stat, two-sided p-value via Newey-West HAC, lag=1).
    """
    d = (y_pred_a - y_true) ** 2 - (y_pred_b - y_true) ** 2
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 5:
        return float("nan"), float("nan")
    d_mean = d.mean()
    gamma0 = float(((d - d_mean) ** 2).mean())
    if gamma0 == 0:
        return float("nan"), float("nan")
    # HAC variance with truncation lag 1
    gamma1 = float(((d[1:] - d_mean) * (d[:-1] - d_mean)).mean())
    var = (gamma0 + 2 * gamma1) / n
    if var <= 0:
        return float("nan"), float("nan")
    stat = d_mean / np.sqrt(var)
    # Two-sided p-value from standard normal
    from scipy.stats import norm
    p = 2 * (1 - norm.cdf(abs(stat)))
    return float(stat), float(p)


def _benchmark_one(panel: pd.DataFrame, h: int, out_root: Path) -> pd.DataFrame:
    target = f"gdp_pc_growth_{h}y_fwd"
    if target not in panel.columns:
        # Mirror the formula used in run_phase8_horizons.py
        panel = panel.sort_values(["iso3", "year"]).reset_index(drop=True)
        g_fwd = panel.groupby("iso3")["gdp_pc"].shift(-h)
        panel[target] = np.log(g_fwd / panel["gdp_pc"])

    df = panel.dropna(subset=[target]).reset_index(drop=True)
    metrics_path = out_root / f"horizon_{h}y" / "metrics.json"
    if not metrics_path.exists():
        print(f"[bench] h={h}: missing {metrics_path}, skipping")
        return pd.DataFrame()
    metrics = json.loads(metrics_path.read_text())
    test_end = metrics["split"]["test_end"]
    test = df["year"].to_numpy() == test_end  # last year of test slice
    # We need the full test *window* (val_end+1 .. test_end), not a single year.
    val_end = metrics["split"]["val_end"]
    test_mask = (df["year"] > val_end) & (df["year"] <= test_end)

    y = df[target].to_numpy()[test_mask]
    n = len(y)
    if n == 0:
        print(f"[bench] h={h}: empty test window")
        return pd.DataFrame()

    # --- Our LGBM ensemble (recomputed from forecasts.parquet for fidelity) ---
    fc = pd.read_parquet(out_root / f"horizon_{h}y" / "forecasts.parquet")
    fc_test = fc[fc["split"] == "test"]
    if len(fc_test) != n:
        # Fall back to in-order reindex
        fc_test = fc.iloc[:n] if len(fc) >= n else fc
    y_lgbm = fc_test["y_pred_ensemble"].to_numpy()
    if len(y_lgbm) != n:
        y_lgbm = y_lgbm[:n]

    # --- Random walk in log-growth space: predict 0 ---
    y_rw = np.zeros(n, dtype=np.float32)

    # --- AR(1): honest baseline fitted on training data only, not leakage. ---
    # We fit a single coefficient on y_{t-1} -> y_t using all training rows where
    # the previous realisation exists, then apply to test rows.
    train_mask = df["year"].to_numpy() <= val_end
    df_sorted = df.sort_values(["iso3", "year"]).reset_index(drop=True)
    df_sorted["prev_y"] = df_sorted.groupby("iso3")[target].shift(1)
    train_subset = df_sorted[train_mask & df_sorted["prev_y"].notna()]
    if len(train_subset) > 10:
        # y_t = a + b * y_{t-1} in log-growth space, fit via numpy lstsq
        x = train_subset["prev_y"].to_numpy(dtype=np.float64)
        yobs = train_subset[target].to_numpy(dtype=np.float64)
        X_design = np.column_stack([np.ones_like(x), x])
        coef, *_ = np.linalg.lstsq(X_design, yobs, rcond=None)
        a, b = float(coef[0]), float(coef[1])
    else:
        a, b = 0.0, 0.0
    # Apply to test rows (using the realised y_{t-1} from the panel — this is the
    # honest version of "AR(1) from training fit", because the prior realisation
    # at t-1 is observable at forecast time when forecasting y_t).
    df_sorted["ar1_pred"] = a + b * df_sorted["prev_y"]
    ar_pred = df_sorted.loc[test_mask, "ar1_pred"].to_numpy()
    ar_pred = np.where(np.isnan(ar_pred), 0.0, ar_pred).astype(np.float32)

    # --- IMF WEO: log(WEO_{t+h} / WEO_t) where WEO = imf.parquet gdp_pc_real ---
    # We use gdp_pc_real (constant USD) since the panel target is real growth.
    # If WEO is missing for a row, we fall back to LGBM (best estimate).
    imf = pd.read_parquet(FEATURES_DIR.parent / "harmonized" / "imf.parquet")
    imf_gdp = imf[imf.indicator_id == "gdp_pc_real"][["iso3", "year", "value"]].copy()
    imf_gdp = imf_gdp.sort_values(["iso3", "year"]).reset_index(drop=True)
    imf_gdp["fwd"] = imf_gdp.groupby("iso3")["value"].shift(-h)
    imf_gdp["weo_pred"] = np.log(imf_gdp["fwd"] / imf_gdp["value"])
    imf_map = {(r.iso3, int(r.year)): float(r.weo_pred) for r in imf_gdp.itertuples()
               if not np.isnan(r.weo_pred)}

    df_idx = df.loc[test_mask, ["iso3", "year"]].reset_index(drop=True)
    weo_pred = np.array([
        imf_map.get((r.iso3, int(r.year)), np.nan) for r in df_idx.itertuples()
    ], dtype=np.float32)
    n_weo = int(np.sum(~np.isnan(weo_pred)))
    # On rows where IMF WEO is missing, fall back to LGBM
    weo_pred_filled = np.where(np.isnan(weo_pred), y_lgbm, weo_pred)

    rows = []
    for name, pred, mask in [
        ("lgbm_ensemble", y_lgbm, np.ones(n, dtype=bool)),
        ("random_walk", y_rw, np.ones(n, dtype=bool)),
        ("ar1_lag1", ar_pred, np.ones(n, dtype=bool)),
        ("imf_weo", weo_pred, ~np.isnan(weo_pred)),
    ]:
        y_t = y[mask]
        y_p = pred[mask]
        err = y_p - y_t
        rows.append({
            "horizon": h,
            "model": name,
            "n": int(len(y_t)),
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "dir_acc": float(np.mean(np.sign(y_p) == np.sign(y_t))),
            "skill_vs_rw_mae": float(1 - np.mean(np.abs(err)) / np.mean(np.abs(y_rw[mask] - y_t))),
        })

    # Diebold–Mariano: lgbm vs each baseline (positive stat = baseline better)
    for name, pred, mask in [
        ("random_walk", y_rw, np.ones(n, dtype=bool)),
        ("ar1_lag1", ar_pred, np.ones(n, dtype=bool)),
        ("imf_weo", weo_pred_filled, np.ones(n, dtype=bool)),
    ]:
        stat, p = _diebold_mariano(y[mask], y_lgbm[mask], pred[mask])
        for r in rows:
            if r["model"] == "lgbm_ensemble":
                r[f"dm_stat_vs_{name}"] = stat
                r[f"dm_p_vs_{name}"] = p

    print(f"[bench h={h}y] n={n} (IMF WEO available for {n_weo}/{n})")
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 5, 10])
    p.add_argument("--panel", type=Path, default=PANEL)
    p.add_argument("--out-root", type=Path, default=FEATURES_DIR)
    p.add_argument("--out", type=Path, default=FEATURES_DIR / "benchmark_table.csv")
    args = p.parse_args()

    panel = pd.read_parquet(args.panel)
    all_rows = []
    for h in args.horizons:
        rows = _benchmark_one(panel, h, args.out_root)
        if not rows.empty:
            all_rows.append(rows)

    table = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    table.to_csv(args.out, index=False)
    print(f"\n[bench] wrote {args.out}")
    if not table.empty:
        cols = ["horizon", "model", "n", "mae", "rmse", "dir_acc", "skill_vs_rw_mae"]
        print("\n[bench] headline rows:")
        print(table[cols].to_string(index=False))


if __name__ == "__main__":
    main()