"""
External Out-of-Sample Generalization Benchmark Suite (Robust Preprocessing)
============================================================================
Evaluates two rigorous, reviewer-demanded external validation protocols:

Protocol 1: Spatial Out-of-Distribution Transfer (20% Zero-Overlap Unseen Countries)
  - Train strictly on 80% of countries (N=136).
  - Test on 20% completely unseen countries (N=33).
  - Evaluates spatial generalization without country identity memorization.

Protocol 2: Pure Temporal Decade Freeze (2015–2025 Out-of-Time Era)
  - Train strictly on historical years (t <= 2014).
  - Freeze all models completely (zero parameter updates).
  - Evaluate on the 2015–2025 era across all countries, including COVID & inflation shocks.
"""

from __future__ import annotations
import sys
import time
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from src.gating.conformal_router import compute_conformal_uncertainty_weights
from src.econometrics.panel_granger import diebold_mariano_test


def _sanitize_matrix(X):
    arr = np.array(X, dtype=np.float64, copy=True)
    arr[~np.isfinite(arr)] = np.nan
    return arr


def train_robust_specialists(X_train, y_train, seed=42):
    X_clean = _sanitize_matrix(X_train)
    
    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_tr_imp = imp.fit_transform(X_clean)
    X_tr_sc = scaler.fit_transform(X_tr_imp)
    X_tr_sc = np.clip(X_tr_sc, -5.0, 5.0)

    y_arr = np.asarray(y_train, dtype=np.float64)

    # 1. Ridge Trend Expert
    exp1 = Ridge(alpha=100.0, random_state=seed)
    exp1.fit(X_tr_sc, y_arr)
    pred1_tr = exp1.predict(X_tr_sc)
    res1 = np.abs(y_arr - pred1_tr)

    # 2. LightGBM Deep Quad Expert
    exp2 = lgb.LGBMRegressor(
        n_estimators=180, learning_rate=0.03, max_depth=5,
        num_leaves=24, min_child_samples=25, subsample=0.8,
        colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=2.0,
        random_state=seed, verbose=-1, n_jobs=-1, objective="regression_l1"
    )
    exp2.fit(X_tr_imp, y_arr)
    pred2_tr = exp2.predict(X_tr_imp)
    res2 = np.abs(y_arr - pred2_tr)

    # 3. Robust Huber Outlier Expert
    exp3 = HuberRegressor(max_iter=500, alpha=50.0)
    exp3.fit(X_tr_sc, y_arr)
    pred3_tr = exp3.predict(X_tr_sc)
    res3 = np.abs(y_arr - pred3_tr)

    # 4. Stress Downturn Specialist
    p25 = np.percentile(y_arr, 25)
    sample_weights = np.where(y_arr < p25, 3.0, 1.0)
    exp4 = lgb.LGBMRegressor(
        n_estimators=120, learning_rate=0.04, max_depth=4,
        num_leaves=16, min_child_samples=20, subsample=0.8,
        random_state=seed, verbose=-1, n_jobs=-1, objective="regression_l1"
    )
    exp4.fit(X_tr_imp, y_arr, sample_weight=sample_weights)
    pred4_tr = exp4.predict(X_tr_imp)
    res4 = np.abs(y_arr - pred4_tr)

    residuals = {"ridge": res1, "lgbm": res2, "huber": res3, "stress": res4}
    models = (exp1, exp2, exp3, exp4)
    preprocessors = (imp, scaler)

    return models, preprocessors, residuals


def predict_robust_specialists(models, preprocessors, X_test):
    imp, scaler = preprocessors
    exp1, exp2, exp3, exp4 = models

    X_clean = _sanitize_matrix(X_test)
    X_te_imp = imp.transform(X_clean)
    X_te_sc = scaler.transform(X_te_imp)
    X_te_sc = np.clip(X_te_sc, -5.0, 5.0)

    pred1 = np.clip(exp1.predict(X_te_sc), -0.5, 0.5)
    pred2 = np.clip(exp2.predict(X_te_imp), -0.5, 0.5)
    pred3 = np.clip(exp3.predict(X_te_sc), -0.5, 0.5)
    pred4 = np.clip(exp4.predict(X_te_imp), -0.5, 0.5)

    return {"ridge": pred1, "lgbm": pred2, "huber": pred3, "stress": pred4}


def run_benchmarks():
    t0 = time.time()
    panel_path = ROOT / "data" / "processed_panels" / "quad_domain_annual_panel.parquet"
    if not panel_path.exists():
        panel_path = ROOT / "data" / "quad_domain_annual_panel.parquet"

    print("=" * 85)
    print("  EXTERNAL OUT-OF-SAMPLE GENERALIZATION BENCHMARK SUITE")
    print("  1. Spatial OOD Transfer (Zero Country Overlap, N=33 Unseen)")
    print("  2. Pure Temporal Decade Freeze (2015-2025 Out-of-Time Era)")
    print("=" * 85)

    df = pd.read_parquet(panel_path)
    exclude_cols = {"iso3", "country", "year", "region", "income_level"}
    feature_cols = [c for c in df.columns if c not in exclude_cols and not c.startswith("gdp_pc_growth_")]

    horizons = [1, 3, 5]
    all_countries = sorted(df["iso3"].unique())
    n_countries = len(all_countries)

    # ──────────────────────────────────────────────────────────────────────────
    # PROTOCOL 1: Spatial Out-of-Distribution Transfer
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "#" * 85)
    print("  PROTOCOL 1: SPATIAL OUT-OF-DISTRIBUTION TRANSFER (20% HELD-OUT UNSEEN COUNTRIES)")
    print("#" * 85)

    np.random.seed(42)
    shuffled_countries = np.random.permutation(all_countries)
    n_test_countries = int(0.20 * n_countries)
    test_countries = set(shuffled_countries[:n_test_countries])
    train_countries = set(shuffled_countries[n_test_countries:])

    spatial_records = []

    for h in horizons:
        target_col = f"gdp_pc_growth_{h}y_fwd"
        clean = df.dropna(subset=[target_col]).copy()

        tr_df = clean[clean["iso3"].isin(train_countries)]
        te_df = clean[clean["iso3"].isin(test_countries)]

        X_tr = tr_df[feature_cols].values
        y_tr = tr_df[target_col].values
        X_te = te_df[feature_cols].values
        y_te = te_df[target_col].values

        # 1. Honest AR(1) baseline on test countries
        ar_preds = []
        for c in test_countries:
            c_data = te_df[te_df["iso3"] == c].sort_values("year")
            if len(c_data) >= 2 and "gdp_pc_growth_1y_fwd" in c_data.columns:
                lag = c_data["gdp_pc_growth_1y_fwd"].shift(1).fillna(0.02)
                ar_pred = 0.5 * lag + 0.5 * 0.02
                ar_preds.extend(ar_pred.values)
            else:
                ar_preds.extend([0.02] * len(c_data))
        ar_preds = np.array(ar_preds)[:len(y_te)]
        mae_ar1 = float(np.mean(np.abs(y_te - ar_preds)))

        # 2. Train on train countries only
        models, preprocessors, residuals = train_robust_specialists(X_tr, y_tr, seed=42)
        preds = predict_robust_specialists(models, preprocessors, X_te)

        # 3. Conformal Router on Unseen Countries
        gated_pred, weights = compute_conformal_uncertainty_weights(
            preds, residuals, alpha=0.10, tau=0.05
        )

        mae_eco = float(np.mean(np.abs(y_te - preds["ridge"])))
        mae_lgbm = float(np.mean(np.abs(y_te - preds["lgbm"])))
        mae_gated = float(np.mean(np.abs(y_te - gated_pred)))

        lift_vs_ar1 = (mae_ar1 - mae_gated) / mae_ar1 * 100.0
        lift_vs_eco = (mae_eco - mae_gated) / mae_eco * 100.0
        dm_stat, dm_p = diebold_mariano_test(y_te, preds["ridge"], gated_pred, h=h)

        spatial_records.append({
            "Protocol": "1_Spatial_OOD",
            "Horizon": f"{h}Y",
            "N_Test_Obs": len(y_te),
            "Test_Domain": f"N={len(test_countries)} Unseen Countries",
            "MAE_AR1": round(mae_ar1, 5),
            "MAE_Eco_Ridge": round(mae_eco, 5),
            "MAE_LGBM_Quad": round(mae_lgbm, 5),
            "MAE_LGCF_v2_Gated": round(mae_gated, 5),
            "Lift_vs_AR1_pct": round(lift_vs_ar1, 2),
            "Lift_vs_Eco_pct": round(lift_vs_eco, 2),
            "DM_Stat": round(dm_stat, 3),
            "DM_p_value": dm_p
        })

        print(f"Horizon h={h}Y (Unseen Countries N={len(test_countries)}, Obs={len(y_te)}):")
        print(f"  AR(1) MAE: {mae_ar1:.4f} | Eco Ridge: {mae_eco:.4f} | LGCF-v2: {mae_gated:.4f}")
        print(f"  Lift vs AR(1): +{lift_vs_ar1:.2f}% | Lift vs Eco: +{lift_vs_eco:.2f}% (DM p={dm_p:.4f})")

    # ──────────────────────────────────────────────────────────────────────────
    # PROTOCOL 2: Pure Temporal Decade Freeze (2015–2025 Era)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "#" * 85)
    print("  PROTOCOL 2: PURE TEMPORAL DECADE FREEZE (TRAIN <= 2014, EVALUATE 2015-2025)")
    print("#" * 85)

    temporal_records = []

    for h in horizons:
        target_col = f"gdp_pc_growth_{h}y_fwd"
        clean = df.dropna(subset=[target_col]).copy()

        train_cutoff = 2014 - (h - 1)
        tr_df = clean[clean["year"] <= train_cutoff]
        te_df = clean[clean["year"] >= 2015]

        X_tr = tr_df[feature_cols].values
        y_tr = tr_df[target_col].values
        X_te = te_df[feature_cols].values
        y_te = te_df[target_col].values

        # 1. AR(1) baseline
        ar_preds = []
        for c in te_df["iso3"].unique():
            c_data = te_df[te_df["iso3"] == c].sort_values("year")
            if len(c_data) >= 2 and "gdp_pc_growth_1y_fwd" in c_data.columns:
                lag = c_data["gdp_pc_growth_1y_fwd"].shift(1).fillna(0.02)
                ar_pred = 0.6 * lag + 0.4 * 0.02
                ar_preds.extend(ar_pred.values)
            else:
                ar_preds.extend([0.02] * len(c_data))
        ar_preds = np.array(ar_preds)[:len(y_te)]
        mae_ar1 = float(np.mean(np.abs(y_te - ar_preds)))

        # 2. Train on pre-2015 data only
        models, preprocessors, residuals = train_robust_specialists(X_tr, y_tr, seed=42)
        preds = predict_robust_specialists(models, preprocessors, X_te)

        # 3. Conformal Router on post-2015 frozen evaluation
        gated_pred, weights = compute_conformal_uncertainty_weights(
            preds, residuals, alpha=0.10, tau=0.05
        )

        mae_eco = float(np.mean(np.abs(y_te - preds["ridge"])))
        mae_lgbm = float(np.mean(np.abs(y_te - preds["lgbm"])))
        mae_gated = float(np.mean(np.abs(y_te - gated_pred)))

        lift_vs_ar1 = (mae_ar1 - mae_gated) / mae_ar1 * 100.0
        lift_vs_eco = (mae_eco - mae_gated) / mae_eco * 100.0
        dm_stat, dm_p = diebold_mariano_test(y_te, preds["ridge"], gated_pred, h=h)

        # Shock regime analysis (2020 COVID shock & 2022 energy shock)
        shock_mask = te_df["year"].isin([2020, 2021, 2022]).values
        if shock_mask.sum() > 20:
            mae_eco_shock = float(np.mean(np.abs(y_te[shock_mask] - preds["ridge"][shock_mask])))
            mae_gated_shock = float(np.mean(np.abs(y_te[shock_mask] - gated_pred[shock_mask])))
            lift_shock = (mae_eco_shock - mae_gated_shock) / mae_eco_shock * 100.0
        else:
            lift_shock = 0.0

        temporal_records.append({
            "Protocol": "2_Temporal_Decade_Freeze",
            "Horizon": f"{h}Y",
            "N_Test_Obs": len(y_te),
            "Test_Domain": "2015-2025 Era Freeze",
            "MAE_AR1": round(mae_ar1, 5),
            "MAE_Eco_Ridge": round(mae_eco, 5),
            "MAE_LGBM_Quad": round(mae_lgbm, 5),
            "MAE_LGCF_v2_Gated": round(mae_gated, 5),
            "Lift_vs_AR1_pct": round(lift_vs_ar1, 2),
            "Lift_vs_Eco_pct": round(lift_vs_eco, 2),
            "Lift_in_Shocks_pct": round(lift_shock, 2),
            "DM_Stat": round(dm_stat, 3),
            "DM_p_value": dm_p
        })

        print(f"Horizon h={h}Y (2015-2025 Era Freeze, Obs={len(y_te)}):")
        print(f"  AR(1) MAE: {mae_ar1:.4f} | Eco Ridge: {mae_eco:.4f} | LGCF-v2: {mae_gated:.4f}")
        print(f"  Lift vs AR(1): +{lift_vs_ar1:.2f}% | Lift vs Eco: +{lift_vs_eco:.2f}% | Shock Lift: +{lift_shock:.2f}% (DM p={dm_p:.4f})")

    # Combine and save results
    all_res = pd.concat([pd.DataFrame(spatial_records), pd.DataFrame(temporal_records)], ignore_index=True)
    out_csv = ROOT / "data" / "benchmarks" / "external_generalization_benchmarks.csv"
    all_res.to_csv(out_csv, index=False)

    print("\n" + "=" * 85)
    print(f"Benchmark Suite Complete in {time.time() - t0:.2f}s.")
    print(f"Results saved to: {out_csv}")
    print("=" * 85)
    print(all_res.to_string())


if __name__ == "__main__":
    run_benchmarks()
