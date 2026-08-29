"""
Hyperparameter Sensitivity & Robustness Grid Audit (Reconciled with Headline Tournament)
========================================================================================
Evaluates stability of Dynamic Model Selection (DMS) and Specialist Models across:
  1. Forgetting factor lambda in {0.85, 0.88, 0.90, 0.92, 0.95, 0.98, 1.00}
     (lambda = 1.00 represents static recursive Bayesian Model Averaging without discounting)
  2. Ridge regularization alpha in {10, 25, 50, 100, 200}
     (Evaluates the Information Dilution Penalty across regularizations)

Across the exact 5-Fold Rolling-Origin Walk-Forward CV (2000-2024) and exact headline preprocessing.
Outputs verified results to data/benchmarks/real_robustness_lambda_results.csv and real_robustness_alpha_results.csv.
"""

from __future__ import annotations
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.reproducibility import seed_everything
from src.gating.dms_state_space_router import DynamicModelSelectionRouter
from src.models.macro_baselines import (
    PerCountryARForecaster,
    add_lagged_growth,
    DEFAULT_LAG_GROWTH_COL,
)

PRED_CLIP = 0.5
FEATURE_CLIP = 5.0


def run_robustness_grid():
    seed_everything(42)
    panel_path = ROOT / "data" / "processed_panels" / "real_cross_domain_annual_panel.parquet"
    if not panel_path.exists():
        raise FileNotFoundError(f"Real panel not found at {panel_path}")

    df = pd.read_parquet(panel_path)
    df = add_lagged_growth(df, level_col="gdp_pc_real")

    exclude_meta_cols = {
        "iso3", "country", "year", "region", "income_level", "region_wb",
        DEFAULT_LAG_GROWTH_COL,
    }

    all_feature_cols = [c for c in df.columns if c not in exclude_meta_cols and not c.endswith("_fwd")]
    eco_cols = [c for c in all_feature_cols if not c.startswith("vdem_") and not c.startswith("climate_")]
    pol_cols = [c for c in all_feature_cols if c.startswith("vdem_")]
    cli_cols = [c for c in all_feature_cols if c.startswith("climate_")]

    horizons = [1, 3, 5]
    lambda_grid = [0.85, 0.88, 0.90, 0.92, 0.95, 0.98, 1.00]
    alpha_grid = [10.0, 25.0, 50.0, 100.0, 200.0]

    folds = [
        {"test_start": 2019, "test_end": 2024},
        {"test_start": 2015, "test_end": 2018},
        {"test_start": 2011, "test_end": 2014},
        {"test_start": 2007, "test_end": 2010},
        {"test_start": 2000, "test_end": 2006},
    ]

    print("=" * 105)
    print("  HYPERPARAMETER SENSITIVITY & ROBUSTNESS GRID AUDIT (RECONCILED WITH TOURNAMENT)")
    print(f"  Features: Total {len(all_feature_cols)} | Economy: {len(eco_cols)} | Politics: {len(pol_cols)} | Climate: {len(cli_cols)}")
    print("  Evaluating Horizons h in {1, 3, 5} across 5-Fold Rolling Walk-Forward CV (2000-2024)")
    print("=" * 105)

    lambda_results = []
    alpha_results = []

    for h in horizons:
        target_col = f"gdp_pc_growth_{h}y_fwd"
        clean = df.dropna(subset=[target_col]).sort_values(["iso3", "year"]).copy()

        # Precompute specialist predictions across folds for the headline setup (alpha=100)
        fold_data = []
        for fold in folds:
            t_start, t_end = fold["test_start"], fold["test_end"]
            train_end = t_start - h - 1
            warm_start = t_start - h

            tr_df = clean[clean["year"] <= train_end].copy()
            ev_df = clean[(clean["year"] >= warm_start) & (clean["year"] <= t_end)].copy()
            if len(tr_df) == 0 or len(ev_df) == 0:
                continue

            ev_df["is_score"] = ev_df["year"] >= t_start
            y_tr = tr_df[target_col].values.astype(np.float64)
            y_ev = ev_df[target_col].values.astype(np.float64)

            imp_eco, scaler_eco = SimpleImputer(strategy="median"), StandardScaler()
            imp_pol, scaler_pol = SimpleImputer(strategy="median"), StandardScaler()
            imp_cli, scaler_cli = SimpleImputer(strategy="median"), StandardScaler()
            imp_all, scaler_all = SimpleImputer(strategy="median"), StandardScaler()

            def clean_mat(mat):
                m = np.array(mat, dtype=np.float64, copy=True)
                m[~np.isfinite(m)] = np.nan
                return m

            def fit_domain(cols, imp, scaler):
                A = clean_mat(tr_df[cols].values)
                B = clean_mat(ev_df[cols].values)
                A_sc = np.clip(scaler.fit_transform(imp.fit_transform(A)), -FEATURE_CLIP, FEATURE_CLIP)
                B_sc = np.clip(scaler.transform(imp.transform(B)), -FEATURE_CLIP, FEATURE_CLIP)
                return A, B, A_sc, B_sc

            X_tr_eco, X_ev_eco, X_tr_eco_sc, X_ev_eco_sc = fit_domain(eco_cols, imp_eco, scaler_eco)
            _, _, X_tr_pol_sc, X_ev_pol_sc = fit_domain(pol_cols, imp_pol, scaler_pol)
            _, _, X_tr_cli_sc, X_ev_cli_sc = fit_domain(cli_cols, imp_cli, scaler_cli)
            X_tr_all, X_ev_all, X_tr_all_sc, X_ev_all_sc = fit_domain(all_feature_cols, imp_all, scaler_all)

            # 1. AR(1)
            ar_model = PerCountryARForecaster(horizon=h, prediction_clip=PRED_CLIP)
            ar_model.fit(tr_df, target_col=target_col, lag_growth_col=DEFAULT_LAG_GROWTH_COL)
            pred_ar1 = ar_model.predict_panel(ev_df, lag_growth_col=DEFAULT_LAG_GROWTH_COL)

            # 2. Economy Ridge (alpha=100)
            m_eco = Ridge(alpha=100.0, random_state=42).fit(X_tr_eco_sc, y_tr)
            pred_eco_ridge = np.clip(m_eco.predict(X_ev_eco_sc), -PRED_CLIP, PRED_CLIP)

            # 3. Politics Ridge (alpha=100)
            m_pol = Ridge(alpha=100.0, random_state=42).fit(X_tr_pol_sc, y_tr)
            pred_pol_ridge = np.clip(m_pol.predict(X_ev_pol_sc), -PRED_CLIP, PRED_CLIP)

            # 4. Climate Ridge (alpha=100)
            m_cli = Ridge(alpha=100.0, random_state=42).fit(X_tr_cli_sc, y_tr)
            pred_cli_ridge = np.clip(m_cli.predict(X_ev_cli_sc), -PRED_CLIP, PRED_CLIP)

            # 5. Economy LightGBM
            g_eco = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.03, max_depth=5,
                                      random_state=42, verbose=-1, n_jobs=-1)
            g_eco.fit(imp_eco.transform(X_tr_eco), y_tr)
            pred_eco_lgbm = np.clip(g_eco.predict(imp_eco.transform(X_ev_eco)), -PRED_CLIP, PRED_CLIP)

            specialist_matrix = np.column_stack([pred_ar1, pred_eco_ridge, pred_pol_ridge, pred_cli_ridge, pred_eco_lgbm])

            fold_data.append({
                "ev_df": ev_df,
                "y_ev": y_ev,
                "specialist_matrix": specialist_matrix,
                "X_tr_eco_sc": X_tr_eco_sc,
                "X_ev_eco_sc": X_ev_eco_sc,
                "X_tr_all_sc": X_tr_all_sc,
                "X_ev_all_sc": X_ev_all_sc,
                "y_tr": y_tr,
                "is_score": ev_df["is_score"].values,
                "target_col": target_col,
            })

        # --- A. LAMBDA SENSITIVITY GRID ---
        print(f"\n--- Horizon h={h}: Lambda Sensitivity Grid ---")
        for lam in lambda_grid:
            all_preds = []
            all_true = []

            for fd in fold_data:
                router = DynamicModelSelectionRouter(
                    n_experts=5, forgetting_factor=lam, mode="dma"
                )
                preds, _ = router.route_panel(
                    fd["ev_df"], fd["specialist_matrix"], fd["y_ev"],
                    horizon=h, year_col="year", iso_col="iso3"
                )
                score_mask = fd["is_score"]
                all_preds.extend(preds[score_mask])
                all_true.extend(fd["ev_df"].loc[score_mask, fd["target_col"]].values)

            y_t = np.array(all_true, dtype=np.float64)
            y_p = np.array(all_preds, dtype=np.float64)
            mae = float(np.mean(np.abs(y_t - y_p)))
            rmse = float(np.sqrt(np.mean((y_t - y_p) ** 2)))

            lambda_results.append({
                "Parameter": "Forgetting Factor (lambda)",
                "Value": lam,
                "Horizon": h,
                "MAE": round(mae, 5),
                "RMSE": round(rmse, 5),
                "Description": "Static Recursive BMA" if lam == 1.00 else f"DMS Memory (decay={(1-lam)*100:.0f}%/yr)",
            })
            print(f"  [Lambda={lam:.2f}] h={h} -> MAE: {mae:.5f}, RMSE: {rmse:.5f}")

        # --- B. ALPHA REGULARIZATION GRID ---
        print(f"\n--- Horizon h={h}: Alpha Regularization Grid ---")
        for alpha in alpha_grid:
            all_preds_eco = []
            all_preds_all = []
            all_true = []

            for fd in fold_data:
                r_eco = Ridge(alpha=alpha, random_state=42).fit(fd["X_tr_eco_sc"], fd["y_tr"])
                p_eco = np.clip(r_eco.predict(fd["X_ev_eco_sc"]), -PRED_CLIP, PRED_CLIP)

                r_all = Ridge(alpha=alpha, random_state=42).fit(fd["X_tr_all_sc"], fd["y_tr"])
                p_all = np.clip(r_all.predict(fd["X_ev_all_sc"]), -PRED_CLIP, PRED_CLIP)

                score_mask = fd["is_score"]
                all_preds_eco.extend(p_eco[score_mask])
                all_preds_all.extend(p_all[score_mask])
                all_true.extend(fd["ev_df"].loc[score_mask, fd["target_col"]].values)

            y_t = np.array(all_true, dtype=np.float64)
            p_e = np.array(all_preds_eco, dtype=np.float64)
            p_a = np.array(all_preds_all, dtype=np.float64)

            mae_eco = float(np.mean(np.abs(y_t - p_e)))
            mae_all = float(np.mean(np.abs(y_t - p_a)))
            dilution_penalty = ((mae_all - mae_eco) / mae_eco) * 100.0

            alpha_results.append({
                "Parameter": "Ridge Shrinkage (alpha)",
                "Value": alpha,
                "Horizon": h,
                "MAE_Eco_Ridge": round(mae_eco, 5),
                "MAE_All_Domain_Concat": round(mae_all, 5),
                "Dilution_Penalty_pct": round(dilution_penalty, 2),
            })
            print(f"  [Alpha={alpha:5.1f}] h={h} -> Eco Ridge MAE: {mae_eco:.5f} vs Concat: {mae_all:.5f} (Dilution Penalty: {dilution_penalty:+.2f}%)")

    # Export SSoT CSV artifacts
    out_dir = ROOT / "data" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    lam_df = pd.DataFrame(lambda_results)
    alp_df = pd.DataFrame(alpha_results)

    lam_path = out_dir / "real_robustness_lambda_results.csv"
    alp_path = out_dir / "real_robustness_alpha_results.csv"

    lam_df.to_csv(lam_path, index=False)
    alp_df.to_csv(alp_path, index=False)
    print(f"\n[SSoT] Reconciled Robustness Lambda results saved to: {lam_path}")
    print(f"[SSoT] Reconciled Robustness Alpha results saved to: {alp_path}")


if __name__ == "__main__":
    run_robustness_grid()

