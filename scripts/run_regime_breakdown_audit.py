"""
Real Regime-Conditional Forecast Evaluation & Dilution Audit Script
==================================================================
Disaggregates out-of-sample forecast accuracy across:
  1. Institutional Transition Regimes (|Delta V-Dem| >= 0.05 in 3yr window) vs Stable Regimes
  2. Macro-Financial Crisis Regimes (GFC 2008-09, COVID 2020, Growth < -3%) vs Tranquil Regimes

Directly tests Proposition 1 (Information Dilution in Tranquil Regimes vs Stress Utility).
Exports verified single-source-of-truth artifact to:
  data/benchmarks/real_regime_breakdown_results.csv
"""

from __future__ import annotations
import sys
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
from src.gating.sovereign_segmentation_router import get_wb_region
from src.gating.dms_state_space_router import DynamicModelSelectionRouter
from src.models.macro_baselines import (
    PerCountryARForecaster,
    EqualWeightCombinationForecaster,
    DynamicFactorForecaster,
    add_lagged_growth,
    DEFAULT_LAG_GROWTH_COL,
)
from src.evaluation.regime_breakdown import identify_regimes, evaluate_regime_performance

PRED_CLIP = 0.5
FEATURE_CLIP = 5.0


def run_regime_audit():
    seed_everything(42)
    panel_path = ROOT / "data" / "processed_panels" / "real_cross_domain_annual_panel.parquet"
    if not panel_path.exists():
        raise FileNotFoundError(f"Real panel not found at {panel_path}")

    df = pd.read_parquet(panel_path)
    df["region_wb"] = df["iso3"].apply(get_wb_region)
    df = add_lagged_growth(df, level_col="gdp_pc_real")
    df = identify_regimes(df)

    exclude_meta_cols = {
        "iso3", "country", "year", "region", "income_level", "region_wb",
        DEFAULT_LAG_GROWTH_COL, "is_inst_transition", "is_inst_stable",
        "is_macro_crisis", "is_macro_tranquil", "vdem_delta_3y",
    }

    all_feature_cols = [c for c in df.columns if c not in exclude_meta_cols and not c.endswith("_fwd")]
    eco_cols = [c for c in all_feature_cols if not c.startswith("vdem_") and not c.startswith("climate_")]
    pol_cols = [c for c in all_feature_cols if c.startswith("vdem_")]
    cli_cols = [c for c in all_feature_cols if c.startswith("climate_")]

    horizons = [1, 3, 5]
    all_regime_records = []

    print("=" * 105)
    print("  REAL REGIME-CONDITIONAL DILUTION & FORECAST EVALUATION AUDIT")
    print("  Disaggregating walk-forward out-of-sample predictions across Institutional and Macro Regimes")
    print("=" * 105)

    for h in horizons:
        target_col = f"gdp_pc_growth_{h}y_fwd"
        clean = df.dropna(subset=[target_col]).sort_values(["iso3", "year"]).copy()

        folds = [
            {"test_start": 2019, "test_end": 2024},
            {"test_start": 2015, "test_end": 2018},
            {"test_start": 2011, "test_end": 2014},
            {"test_start": 2007, "test_end": 2010},
            {"test_start": 2000, "test_end": 2006},
        ]

        h_preds_list = []

        for fold_idx, fold in enumerate(folds):
            t_start, t_end = fold["test_start"], fold["test_end"]
            train_end = t_start - h - 1
            warm_start = t_start - h

            tr_df = clean[clean["year"] <= train_end].copy()
            ev_df = clean[(clean["year"] >= warm_start) & (clean["year"] <= t_end)].copy()
            if len(ev_df) == 0 or len(tr_df) == 0:
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

            ar_model = PerCountryARForecaster(horizon=h, prediction_clip=PRED_CLIP)
            ar_model.fit(tr_df, target_col=target_col, lag_growth_col=DEFAULT_LAG_GROWTH_COL)
            ev_df["pred_ar1"] = ar_model.predict_panel(ev_df, lag_growth_col=DEFAULT_LAG_GROWTH_COL)

            for name, Xtr, Xev in (
                ("pred_eco_ridge", X_tr_eco_sc, X_ev_eco_sc),
                ("pred_all_ridge", X_tr_all_sc, X_ev_all_sc),
                ("pred_pol_ridge", X_tr_pol_sc, X_ev_pol_sc),
                ("pred_cli_ridge", X_tr_cli_sc, X_ev_cli_sc),
            ):
                m = Ridge(alpha=100.0, random_state=42)
                m.fit(Xtr, y_tr)
                ev_df[name] = np.clip(m.predict(Xev), -PRED_CLIP, PRED_CLIP)

            for name, imp, Xtr_raw, Xev_raw in (
                ("pred_eco_lgbm", imp_eco, X_tr_eco, X_ev_eco),
                ("pred_all_lgbm", imp_all, X_tr_all, X_ev_all),
            ):
                g = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.03, max_depth=5,
                                      random_state=42, verbose=-1, n_jobs=-1)
                g.fit(imp.transform(Xtr_raw), y_tr)
                ev_df[name] = np.clip(g.predict(imp.transform(Xev_raw)), -PRED_CLIP, PRED_CLIP)

            expert_cols = ["pred_ar1", "pred_eco_ridge", "pred_pol_ridge",
                           "pred_cli_ridge", "pred_eco_lgbm"]
            specialist_matrix = np.column_stack([ev_df[c].values for c in expert_cols])
            ev_df["pred_equal_weight"] = EqualWeightCombinationForecaster.combine(
                [ev_df[c].values for c in expert_cols]
            )

            dms_router = DynamicModelSelectionRouter(n_experts=len(expert_cols), forgetting_factor=0.92, mode="dma")
            dms_preds, _ = dms_router.route_panel(
                ev_df, specialist_matrix, y_ev, horizon=h, year_col="year", iso_col="iso3"
            )
            ev_df["pred_dms_gated"] = np.clip(dms_preds, -PRED_CLIP, PRED_CLIP)

            h_preds_list.append(ev_df[ev_df["is_score"]].copy())

        h_combined = pd.concat(h_preds_list, ignore_index=True)

        model_dict = {
            "AR(1) Baseline": "pred_ar1",
            "Economy-Only Ridge": "pred_eco_ridge",
            "All-Domain Ridge (Concat)": "pred_all_ridge",
            "Economy LightGBM": "pred_eco_lgbm",
            "All-Domain LightGBM (Concat)": "pred_all_lgbm",
            "Equal-Weight Multi-Domain": "pred_equal_weight",
            "DMS State-Space Router": "pred_dms_gated",
        }

        # Regimes to evaluate:
        regimes = [
            ("Full Sample (All Scored)", pd.Series(True, index=h_combined.index)),
            ("Tranquil Macro Expansion", h_combined["is_macro_tranquil"]),
            ("Macro Crisis / Global Shock", h_combined["is_macro_crisis"]),
            ("Stable Democratic Regime", h_combined["is_inst_stable"]),
            ("Institutional Transition / Shift", h_combined["is_inst_transition"]),
        ]

        print(f"\n--- Horizon h = {h} Year(s) Regime Breakdown ---")
        for reg_name, mask in regimes:
            res = evaluate_regime_performance(
                h_combined,
                target_col=target_col,
                model_pred_cols=model_dict,
                regime_mask=mask,
                regime_name=reg_name,
                horizon=h,
            )
            all_regime_records.append(res)
            print(f"  {reg_name:<35} (N={res['N_obs']:4d}) | Eco-Ridge: {res.get('MAE_Economy-Only Ridge', 0):.5f} | All-Ridge: {res.get('MAE_All-Domain Ridge (Concat)', 0):.5f} (Penalty: {res.get('Ridge_Concat_Penalty_pct', 0):+5.2f}%) | DMS: {res.get('MAE_DMS State-Space Router', 0):.5f}")

    out_df = pd.DataFrame(all_regime_records)
    out_dir = ROOT / "data" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "real_regime_breakdown_results.csv"
    out_df.to_csv(out_path, index=False)

    print("\n" + "=" * 105)
    print(f"  Regime Breakdown audit complete. Artifact saved to:\n  {out_path}")
    print("=" * 105)


if __name__ == "__main__":
    run_regime_audit()
