"""
Oracle Gap Tournament Benchmark Suite
=====================================
Evaluates 4 advanced mathematical routing architectures against the Theoretical Oracle Ceiling
under 5-fold rolling-origin walk-forward cross-validation (1960–2025).

Architectures Evaluated:
  1. Single-Domain Economy Baseline (Ridge)
  2. Baseline Softmax Meta-Router
  3. Candidate A: Sparsemax Direct MAE Router (Exact Simplex Boundary Projection)
  4. Candidate B: Cost-Sensitive Regret Router (Learning-to-Defer Surrogate)
  5. Candidate C: Koop-Korobilis Dynamic Model Selection (DMS State-Space Filter)
  6. Candidate D: Hybrid DMS-Sparsemax Router (State-Space Memory + Sparse Simplex)
  7. Theoretical Bound: Oracle Dynamic Model Selector (Hindsight Upper Bound)
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
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, HuberRegressor
import lightgbm as lgb

from src.gating.sparsemax_router import SparsemaxDirectMAERouter
from src.gating.cost_sensitive_router import CostSensitiveRegretRouter
from src.gating.dms_state_space_router import DynamicModelSelectionRouter
from src.gating.hybrid_dms_sparsemax import HybridDMSSparsemaxRouter
from src.gating.conformal_router import compute_conformal_uncertainty_weights
from src.econometrics.panel_granger import diebold_mariano_test


def _sanitize_matrix(X):
    arr = np.array(X, dtype=np.float64, copy=True)
    arr[~np.isfinite(arr)] = np.nan
    return arr


def train_orthogonal_specialists(X_train, y_train, seed=42):
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

    # 2. LightGBM Deep Quad Expert
    exp2 = lgb.LGBMRegressor(
        n_estimators=180, learning_rate=0.03, max_depth=5,
        num_leaves=24, min_child_samples=25, subsample=0.8,
        colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=2.0,
        random_state=seed, verbose=-1, n_jobs=-1, objective="regression_l1"
    )
    exp2.fit(X_tr_imp, y_arr)

    # 3. Robust Huber Outlier Expert
    exp3 = HuberRegressor(max_iter=500, alpha=50.0)
    exp3.fit(X_tr_sc, y_arr)

    # 4. Stress Downturn Specialist
    p25 = np.percentile(y_arr, 25)
    sample_weights = np.where(y_arr < p25, 3.0, 1.0)
    exp4 = lgb.LGBMRegressor(
        n_estimators=120, learning_rate=0.04, max_depth=4,
        num_leaves=16, min_child_samples=20, subsample=0.8,
        random_state=seed, verbose=-1, n_jobs=-1, objective="regression_l1"
    )
    exp4.fit(X_tr_imp, y_arr, sample_weight=sample_weights)

    models = (exp1, exp2, exp3, exp4)
    preprocessors = (imp, scaler)
    return models, preprocessors


def predict_orthogonal_specialists(models, preprocessors, X_test):
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

    return np.column_stack([pred1, pred2, pred3, pred4])


def run_tournament():
    t0 = time.time()
    panel_path = ROOT / "data" / "processed_panels" / "quad_domain_annual_panel.parquet"
    if not panel_path.exists():
        panel_path = ROOT / "data" / "quad_domain_annual_panel.parquet"

    print("=" * 95)
    print("  ORACLE GAP TOURNAMENT BENCHMARK SUITE")
    print("  Evaluating 4 Advanced Routing Architectures vs. the +18.46% Oracle Ceiling")
    print("  5-Fold Rolling-Origin Walk-Forward CV (1960–2025)")
    print("=" * 95)

    df = pd.read_parquet(panel_path)
    exclude_cols = {"iso3", "country", "year", "region", "income_level"}
    feature_cols = [c for c in df.columns if c not in exclude_cols and not c.startswith("gdp_pc_growth_")]

    horizons = [1, 3, 5]
    
    # 5 Rolling Folds
    folds = [
        {"test_start": 2019, "test_end": 2024},
        {"test_start": 2015, "test_end": 2018},
        {"test_start": 2011, "test_end": 2014},
        {"test_start": 2007, "test_end": 2010},
        {"test_start": 2003, "test_end": 2006},
    ]

    records = []

    for h in horizons:
        target_col = f"gdp_pc_growth_{h}y_fwd"
        clean = df.dropna(subset=[target_col]).sort_values(["iso3", "year"]).copy()

        print(f"\n>>> Running Evaluation for Horizon h={h} Year(s)...")

        # Accumulators across folds
        y_true_all = []
        preds_eco_all = []
        preds_softmax_all = []
        preds_sparsemax_all = []
        preds_cost_sens_all = []
        preds_dms_all = []
        preds_hybrid_all = []
        preds_oracle_all = []

        for f_idx, fold in enumerate(folds):
            t_start, t_end = fold["test_start"], fold["test_end"]
            train_end = t_start - (h - 1) - 1

            tr_df = clean[clean["year"] <= train_end]
            te_df = clean[(clean["year"] >= t_start) & (clean["year"] <= t_end)]

            if len(te_df) == 0:
                continue

            X_tr = tr_df[feature_cols].values
            y_tr = tr_df[target_col].values
            X_te = te_df[feature_cols].values
            y_te = te_df[target_col].values

            # 1. Train 4 Orthogonal Specialists
            models, preprocessors = train_orthogonal_specialists(X_tr, y_tr, seed=42 + f_idx)
            
            # Predict in-sample on train to train routers
            E_tr = predict_orthogonal_specialists(models, preprocessors, X_tr)
            # Predict out-of-fold on test
            E_te = predict_orthogonal_specialists(models, preprocessors, X_te)

            imp, scaler = preprocessors
            X_tr_sc = scaler.transform(imp.transform(_sanitize_matrix(X_tr)))
            X_te_sc = scaler.transform(imp.transform(_sanitize_matrix(X_te)))
            X_tr_sc = np.clip(X_tr_sc, -5.0, 5.0)
            X_te_sc = np.clip(X_te_sc, -5.0, 5.0)

            # --- Model 1: Baseline Single Economy (Ridge) ---
            pred_eco = E_te[:, 0]

            # --- Model 2: Baseline Softmax Router ---
            # Softmax on calibrated residuals
            res_tr = np.abs(y_tr[:, None] - E_tr)
            sigma_tr = np.mean(res_tr, axis=0) + 1e-4
            weights_soft = np.exp(- res_tr / sigma_tr)
            weights_soft /= np.sum(weights_soft, axis=1, keepdims=True)
            pred_softmax = np.sum(E_te * np.mean(weights_soft, axis=0), axis=1)

            # --- Model 3 (Candidate A): Sparsemax Direct MAE Router ---
            sparse_router = SparsemaxDirectMAERouter(n_features=X_tr_sc.shape[1], n_experts=4, lr=0.03, max_iter=250)
            sparse_router.fit(X_tr_sc, E_tr, y_tr)
            pred_sparsemax, _ = sparse_router.route(X_te_sc, E_te)

            # --- Model 4 (Candidate B): Cost-Sensitive Regret Router ---
            cs_router = CostSensitiveRegretRouter(n_features=X_tr_sc.shape[1], n_experts=4, lr=0.03, max_iter=250)
            cs_router.fit(X_tr_sc, E_tr, y_tr)
            pred_cost_sens, _ = cs_router.route(X_te_sc, E_te)

            # --- Model 5 (Candidate C): Koop-Korobilis DMS State-Space Filter ---
            dms_router = DynamicModelSelectionRouter(n_experts=4, forgetting_factor=0.92, mode="dma")
            pred_dms, _ = dms_router.route_panel(te_df, E_te, y_true=y_te)

            # --- Model 6 (Candidate D): Hybrid DMS-Sparsemax Router ---
            hybrid_router = HybridDMSSparsemaxRouter(n_features=X_tr_sc.shape[1], n_experts=4, forgetting_factor=0.92, lr=0.03, max_iter=250)
            hybrid_router.fit(X_tr_sc, E_tr, y_tr)
            pred_hybrid, _ = hybrid_router.route_panel(te_df, X_te_sc, E_te, y_true=y_te)

            # --- Model 7: Theoretical Oracle Upper Bound ---
            abs_errs = np.abs(E_te - y_te[:, None])
            oracle_idx = np.argmin(abs_errs, axis=1)
            pred_oracle = E_te[np.arange(len(y_te)), oracle_idx]

            # Accumulate
            y_true_all.extend(y_te)
            preds_eco_all.extend(pred_eco)
            preds_softmax_all.extend(pred_softmax)
            preds_sparsemax_all.extend(pred_sparsemax)
            preds_cost_sens_all.extend(pred_cost_sens)
            preds_dms_all.extend(pred_dms)
            preds_hybrid_all.extend(pred_hybrid)
            preds_oracle_all.extend(pred_oracle)

        # Convert to arrays
        y_true_arr = np.array(y_true_all)
        p_eco = np.array(preds_eco_all)
        p_soft = np.array(preds_softmax_all)
        p_sparse = np.array(preds_sparsemax_all)
        p_cs = np.array(preds_cost_sens_all)
        p_dms = np.array(preds_dms_all)
        p_hybrid = np.array(preds_hybrid_all)
        p_oracle = np.array(preds_oracle_all)

        mae_eco = float(np.mean(np.abs(y_true_arr - p_eco)))
        mae_soft = float(np.mean(np.abs(y_true_arr - p_soft)))
        mae_sparse = float(np.mean(np.abs(y_true_arr - p_sparse)))
        mae_cs = float(np.mean(np.abs(y_true_arr - p_cs)))
        mae_dms = float(np.mean(np.abs(y_true_arr - p_dms)))
        mae_hybrid = float(np.mean(np.abs(y_true_arr - p_hybrid)))
        mae_oracle = float(np.mean(np.abs(y_true_arr - p_oracle)))

        oracle_gap_total = mae_eco - mae_oracle

        models_eval = [
            ("1. Economy Baseline (Ridge)", mae_eco, p_eco),
            ("2. Baseline Softmax Router", mae_soft, p_soft),
            ("3. Sparsemax Direct MAE", mae_sparse, p_sparse),
            ("4. Cost-Sensitive L2D", mae_cs, p_cs),
            ("5. Koop-Korobilis DMS", mae_dms, p_dms),
            ("6. Hybrid DMS-Sparsemax", mae_hybrid, p_hybrid),
            ("7. Theoretical Oracle Bound", mae_oracle, p_oracle),
        ]

        for name, mae_val, pred_vec in models_eval:
            lift_vs_eco = (mae_eco - mae_val) / mae_eco * 100.0
            gap_closed_pct = (mae_eco - mae_val) / max(1e-8, oracle_gap_total) * 100.0
            
            if name.startswith("1."):
                dm_p = 1.0
                dm_stat = 0.0
            else:
                dm_stat, dm_p = diebold_mariano_test(y_true_arr, p_eco, pred_vec, h=h)

            records.append({
                "Horizon": f"{h}Y",
                "Total_Obs": len(y_true_arr),
                "Model_Router": name,
                "MAE": round(mae_val, 5),
                "Lift_vs_Eco_pct": round(lift_vs_eco, 2),
                "Oracle_Gap_Closed_pct": round(gap_closed_pct, 2),
                "DM_Stat": round(dm_stat, 3),
                "DM_p_value": dm_p
            })

            print(f"  {name:<32} | MAE: {mae_val:.5f} | Lift: {lift_vs_eco:+.2f}% | Gap Closed: {gap_closed_pct:5.1f}% (DM p={dm_p:.4f})")

    res_df = pd.DataFrame(records)
    out_csv = ROOT / "data" / "benchmarks" / "oracle_gap_tournament_results.csv"
    res_df.to_csv(out_csv, index=False)

    print("\n" + "=" * 95)
    print(f"Tournament Benchmark Complete in {time.time() - t0:.2f}s.")
    print(f"Results successfully saved to: {out_csv}")
    print("=" * 95)
    print(res_df.to_string())


if __name__ == "__main__":
    run_tournament()
