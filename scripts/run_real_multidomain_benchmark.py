"""
Real Multi-Domain Macroeconomic Forecasting Tournament Suite
============================================================
Evaluates competitive forecasters across:
  - Real GMD Economic Panel + Real V-Dem v14 Politics + Real ERA5 Climate Panel
  - Horizons: h in {1, 3, 5} Years
  - 5-Fold Rolling-Origin Walk-Forward CV (1960-2024)
  - Strict Target Purging & Fold Quarantine (t_train <= t_start - h - 1)
  - Single-Domain Specialists vs. Static Concatenation (THE PARADOX TEST):
      1. Economy-Only Ridge (206 GMD features only)
      2. All-Domain Ridge (241 features: Economy + Politics + Climate)
      3. Politics-Only Ridge (25 V-Dem v14 features only)
      4. Climate-Only Ridge (10 ERA5 / CO2 features only)
      5. Economy-Only LightGBM (206 GMD features only)
      6. All-Domain LightGBM (241 features)
  - Multi-Domain Combinations & Routers:
      7. Stock-Watson Dynamic Factor Model (DFM)
      8. Equal-Weight Combination (1/M Average)
      9. Recursive State-Space DMS (Koop-Korobilis 2012, lambda=0.92), REAL-TIME feedback
     10. The same DMS with feedback disabled (diagnostic: DMA with no updates is
         algebraically the 1/M average, so this row must equal row 8)
  - Honest Autoregressive Benchmark:
      11. Fitted Per-Country AR(1) with empirical-Bayes shrinkage, on a
          contract-checked regressor (see src/models/macro_baselines.py)

Two defects that invalidated the previous run of this script are fixed here:

  1. AR REGRESSOR MISMATCH. The baseline was fit on pct_change(gdp_pc_real) and
     predicted with gdp_pc_real_logret5 -- a 5-year cumulative log return at 4.3x
     the fitted scale. Baseline MAE was inflated 20-55%, and every reported lift was
     measured against it. Both ends now use `growth_into_origin`, and
     PerCountryARForecaster raises on any mismatch.

  2. DMS TEST-TARGET LEAKAGE. The router received the full test-fold target vector
     and updated its weights immediately, so weights at origin t depended on targets
     unrealised for another h-1 years. route_panel now gates each realisation by its
     realisation date (origin + h <= t). To let the filter leave its uniform prior
     without seeing the future, each fold adds a warm-up window of origins
     [t_start - h, t_start - 1]: out-of-sample for the fold's models (training ends at
     t_start - h - 1) and with targets realising at [t_start, t_start + h - 1], so at
     origin t_start exactly one is observable. Warm-up rows are NOT scored.

Inference is reported both pooled (for continuity with the prior artifact) and
year-clustered; the pooled statistic overstates |DM| by roughly 2.6x on this panel.

Exports single-source-of-truth CSV artifacts to data/benchmarks/real_cross_domain_benchmark_results.csv.
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
from src.gating.sovereign_segmentation_router import get_wb_region
from src.gating.dms_state_space_router import DynamicModelSelectionRouter
from src.models.macro_baselines import (
    PerCountryARForecaster,
    EqualWeightCombinationForecaster,
    DynamicFactorForecaster,
    add_lagged_growth,
    DEFAULT_LAG_GROWTH_COL,
)
from src.econometrics.panel_granger import (
    diebold_mariano_test,
    clark_west_test,
    year_clustered_forecast_test,
)

PRED_CLIP = 0.5
FEATURE_CLIP = 5.0


def run_real_multidomain_tournament():
    seed_everything(42)
    start_time = time.time()

    p = ROOT / "data" / "processed_panels" / "real_cross_domain_annual_panel.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Real panel not found at {p}")

    df = pd.read_parquet(p)
    df["region_wb"] = df["iso3"].apply(get_wb_region)

    # AR regressor: growth realised INTO the origin year. Must be built on the full
    # panel before splitting -- it is a within-country lagged difference, so it cannot
    # be reconstructed from a training slice alone.
    df = add_lagged_growth(df, level_col="gdp_pc_real")

    # DEFAULT_LAG_GROWTH_COL is excluded from the feature matrix so the AR baseline
    # stays distinct and the reported feature count remains d = 241.
    exclude_meta_cols = {
        "iso3", "country", "year", "region", "income_level", "region_wb",
        DEFAULT_LAG_GROWTH_COL,
    }

    all_feature_cols = [c for c in df.columns if c not in exclude_meta_cols and not c.endswith("_fwd")]
    eco_cols = [c for c in all_feature_cols if not c.startswith("vdem_") and not c.startswith("climate_")]
    pol_cols = [c for c in all_feature_cols if c.startswith("vdem_")]
    cli_cols = [c for c in all_feature_cols if c.startswith("climate_")]

    print("=" * 105)
    print("  REAL MULTI-DOMAIN MACROECONOMIC FORECASTING TOURNAMENT & PARADOX AUDIT")
    print(f"  Features: Total {len(all_feature_cols)} | Economy: {len(eco_cols)} | Politics (V-Dem): {len(pol_cols)} | Climate (ERA5): {len(cli_cols)}")
    print("  Evaluating Horizons h in {1, 3, 5} across 5-Fold Rolling Walk-Forward CV (1960-2024)")
    print("  AR regressor: growth_into_origin (fit == predict, contract-enforced)")
    print("  DMS feedback: real-time, gated by realisation date (origin + h <= t)")
    print("=" * 105)

    horizons = [1, 3, 5]
    global_records = []

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
        pending_never_released = 0

        for fold_idx, fold in enumerate(folds):
            t_start, t_end = fold["test_start"], fold["test_end"]
            train_end = t_start - h - 1
            warm_start = t_start - h  # DMS warm-up origins; never scored

            tr_df = clean[clean["year"] <= train_end].copy()
            ev_df = clean[(clean["year"] >= warm_start) & (clean["year"] <= t_end)].copy()

            if len(ev_df) == 0 or len(tr_df) == 0:
                continue

            ev_df["is_score"] = ev_df["year"] >= t_start
            y_tr = tr_df[target_col].values.astype(np.float64)
            y_ev = ev_df[target_col].values.astype(np.float64)

            # Imputers & Scalers per domain (Zero Cross-Domain Contamination),
            # fitted exclusively on the historical training slice.
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

            # 1. Baseline: Fitted Per-Country AR(1), empirical-Bayes shrinkage.
            #    fit() pins the regressor; predict_panel() names it to assert intent
            #    and raises on any mismatch.
            ar_model = PerCountryARForecaster(horizon=h, prediction_clip=PRED_CLIP)
            ar_model.fit(tr_df, target_col=target_col, lag_growth_col=DEFAULT_LAG_GROWTH_COL)
            ev_df["pred_ar1"] = ar_model.predict_panel(ev_df, lag_growth_col=DEFAULT_LAG_GROWTH_COL)

            # 2-5. Domain-quarantined and concatenated Ridge specialists
            for name, Xtr, Xev in (
                ("pred_eco_ridge", X_tr_eco_sc, X_ev_eco_sc),
                ("pred_all_ridge", X_tr_all_sc, X_ev_all_sc),
                ("pred_pol_ridge", X_tr_pol_sc, X_ev_pol_sc),
                ("pred_cli_ridge", X_tr_cli_sc, X_ev_cli_sc),
            ):
                m = Ridge(alpha=100.0, random_state=42)
                m.fit(Xtr, y_tr)
                ev_df[name] = np.clip(m.predict(Xev), -PRED_CLIP, PRED_CLIP)

            # 6-7. LightGBM specialists (median-imputed, unscaled)
            for name, imp, Xtr_raw, Xev_raw in (
                ("pred_eco_lgbm", imp_eco, X_tr_eco, X_ev_eco),
                ("pred_all_lgbm", imp_all, X_tr_all, X_ev_all),
            ):
                g = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.03, max_depth=5,
                                      random_state=42, verbose=-1, n_jobs=-1)
                g.fit(imp.transform(Xtr_raw), y_tr)
                ev_df[name] = np.clip(g.predict(imp.transform(Xev_raw)), -PRED_CLIP, PRED_CLIP)

            # 8. Baseline: Dynamic Factor Model (Stock-Watson 2002)
            dfm = DynamicFactorForecaster(n_factors=5, alpha=20.0, random_state=42)
            dfm.fit(X_tr_eco_sc, y_tr)
            ev_df["pred_dfm"] = np.clip(dfm.predict(X_ev_eco_sc), -PRED_CLIP, PRED_CLIP)

            # 9. Baseline: Equal-Weight Multi-Domain Combination (1/M)
            expert_cols = ["pred_ar1", "pred_eco_ridge", "pred_pol_ridge",
                           "pred_cli_ridge", "pred_eco_lgbm"]
            specialist_matrix = np.column_stack([ev_df[c].values for c in expert_cols])
            ev_df["pred_equal_weight"] = EqualWeightCombinationForecaster.combine(
                [ev_df[c].values for c in expert_cols]
            )

            # 10. Recursive State-Space DMS with REAL-TIME feedback.
            #     horizon=h delays each realisation until origin + h, so no weight
            #     ever conditions on a target that has not yet been observed.
            dms_router = DynamicModelSelectionRouter(
                n_experts=len(expert_cols), forgetting_factor=0.92, mode="dma"
            )
            dms_preds, _ = dms_router.route_panel(
                ev_df, specialist_matrix, y_ev, horizon=h, year_col="year", iso_col="iso3"
            )
            ev_df["pred_dms_gated"] = np.clip(dms_preds, -PRED_CLIP, PRED_CLIP)
            pending_never_released += dms_router.n_pending()

            # 11. Diagnostic: the same router with feedback disabled. DMA whose
            #     posterior never updates stays at the 1/M prior, so this must equal
            #     row 9's equal-weight average exactly.
            nf_router = DynamicModelSelectionRouter(
                n_experts=len(expert_cols), forgetting_factor=0.92, mode="dma"
            )
            nf_preds, _ = nf_router.route_panel(
                ev_df, specialist_matrix, None, horizon=h, year_col="year", iso_col="iso3"
            )
            ev_df["pred_dms_nofeedback"] = np.clip(nf_preds, -PRED_CLIP, PRED_CLIP)

            ev_df["horizon"] = h
            ev_df["fold"] = fold_idx + 1
            h_preds_list.append(ev_df[ev_df["is_score"]].copy())

        h_combined = pd.concat(h_preds_list, ignore_index=True)

        y_true = h_combined[target_col].values
        years = h_combined["year"].values
        models = {
            "AR(1) Baseline": h_combined["pred_ar1"].values,
            "Economy-Only Ridge": h_combined["pred_eco_ridge"].values,
            "All-Domain Ridge (Concat)": h_combined["pred_all_ridge"].values,
            "Politics Ridge (V-Dem)": h_combined["pred_pol_ridge"].values,
            "Climate Ridge (ERA5)": h_combined["pred_cli_ridge"].values,
            "Economy LightGBM": h_combined["pred_eco_lgbm"].values,
            "All-Domain LightGBM (Concat)": h_combined["pred_all_lgbm"].values,
            "Stock-Watson DFM": h_combined["pred_dfm"].values,
            "Equal-Weight Multi-Domain": h_combined["pred_equal_weight"].values,
            "DMS State-Space Router": h_combined["pred_dms_gated"].values,
            "DMS (feedback disabled)": h_combined["pred_dms_nofeedback"].values,
        }

        mae_ar1 = np.mean(np.abs(y_true - models["AR(1) Baseline"]))
        mae_eco_ridge = np.mean(np.abs(y_true - models["Economy-Only Ridge"]))
        mae_ew = np.mean(np.abs(y_true - models["Equal-Weight Multi-Domain"]))

        print(f"\n--- Horizon h = {h} Year(s) (N = {len(y_true)} scored country-years) ---")
        print(f"    realisations queued but never observable within a fold: {pending_never_released}")
        print(f"    {'model':<30} {'MAE':>9} {'vs AR(1)':>10} {'DM pooled':>11} {'DM yr-clust':>12} {'p':>9}")

        for m_name, m_preds in models.items():
            mae = float(np.mean(np.abs(y_true - m_preds)))
            rmse = float(np.sqrt(np.mean((y_true - m_preds) ** 2)))
            lift_vs_ar1 = ((mae_ar1 - mae) / mae_ar1) * 100.0
            lift_vs_eco_ridge = ((mae_eco_ridge - mae) / mae_eco_ridge) * 100.0
            lift_vs_ew = ((mae_ew - mae) / mae_ew) * 100.0

            cw_stat, cw_p = clark_west_test(y_true, models["AR(1) Baseline"], m_preds, h=h)
            dm_stat, dm_p = diebold_mariano_test(y_true, models["AR(1) Baseline"], m_preds, h=h)
            dmc_stat, dmc_p, n_years = year_clustered_forecast_test(
                y_true, models["AR(1) Baseline"], m_preds, years, criterion="mae", h=h
            )

            global_records.append({
                "Horizon": h,
                "Model": m_name,
                "MAE": round(mae, 5),
                "RMSE": round(rmse, 5),
                "Lift_vs_AR1_pct": round(float(lift_vs_ar1), 2),
                "Lift_vs_EcoRidge_pct": round(float(lift_vs_eco_ridge), 2),
                "Lift_vs_EW_pct": round(float(lift_vs_ew), 2),
                # Pooled country-year inference: retained for continuity, but it
                # ignores cross-sectional dependence and overstates |stat| ~2.6x.
                "CW_stat_pooled": round(float(cw_stat), 3),
                "CW_pval_pooled": float(cw_p),
                "DM_stat_pooled": round(float(dm_stat), 3),
                "DM_pval_pooled": float(dm_p),
                # Year-clustered (Driscoll-Kraay style with Newey-West h-1 adjustment):
                "DM_stat_yearclustered": round(float(dmc_stat), 3),
                "DM_pval_yearclustered": float(dmc_p),
                "N_years_clusters": n_years,
                "N_obs": len(y_true),
            })

            sign = "+" if lift_vs_ar1 >= 0 else ""
            print(f"    {m_name:<30} {mae:9.5f} {sign}{lift_vs_ar1:8.2f}% "
                  f"{dm_stat:11.2f} {dmc_stat:12.2f} {dmc_p:9.4f}")

        # Direct pairwise tests: DMS vs Equal-Weight and DMS vs Economy LightGBM
        dms_preds = models["DMS State-Space Router"]
        eq_preds = models["Equal-Weight Multi-Domain"]
        lgbm_preds = models["Economy LightGBM"]
        eco_ridge_preds = models["Economy-Only Ridge"]
        all_ridge_preds = models["All-Domain Ridge (Concat)"]
        all_lgbm_preds = models["All-Domain LightGBM (Concat)"]

        dm_vs_eq, p_vs_eq, _ = year_clustered_forecast_test(y_true, eq_preds, dms_preds, years, criterion="mae", h=h)
        dm_vs_lgbm, p_vs_lgbm, _ = year_clustered_forecast_test(y_true, lgbm_preds, dms_preds, years, criterion="mae", h=h)
        dm_vs_ecoridge, p_vs_ecoridge, _ = year_clustered_forecast_test(y_true, eco_ridge_preds, dms_preds, years, criterion="mae", h=h)

        # Clark-West (2007) nested model tests (Small = Economy-Only, Large = All-Domain)
        cw_ridge, p_cw_ridge, _ = year_clustered_forecast_test(y_true, eco_ridge_preds, all_ridge_preds, years, nested=True, h=h)
        cw_lgbm, p_cw_lgbm, _ = year_clustered_forecast_test(y_true, lgbm_preds, all_lgbm_preds, years, nested=True, h=h)

        # Save pairwise test results to CSV
        for test_label, stat_val, p_val, model1, model2 in [
            ("DMS_vs_EqualWeight", dm_vs_eq, p_vs_eq, "Equal-Weight Multi-Domain", "DMS State-Space Router"),
            ("DMS_vs_EcoLGBM", dm_vs_lgbm, p_vs_lgbm, "Economy LightGBM", "DMS State-Space Router"),
            ("DMS_vs_EcoRidge", dm_vs_ecoridge, p_vs_ecoridge, "Economy-Only Ridge", "DMS State-Space Router"),
            ("CW_EcoRidge_vs_AllRidge", cw_ridge, p_cw_ridge, "Economy-Only Ridge", "All-Domain Ridge (Concat)"),
            ("CW_EcoLGBM_vs_AllLGBM", cw_lgbm, p_cw_lgbm, "Economy LightGBM", "All-Domain LightGBM (Concat)"),
        ]:
            global_records.append({
                "Horizon": h,
                "Model": f"[Pairwise] {test_label}",
                "MAE": 0.0,
                "RMSE": 0.0,
                "Lift_vs_AR1_pct": 0.0,
                "Lift_vs_EcoRidge_pct": 0.0,
                "Lift_vs_EW_pct": 0.0,
                "CW_stat_pooled": 0.0,
                "CW_pval_pooled": 0.0,
                "DM_stat_pooled": 0.0,
                "DM_pval_pooled": 0.0,
                "DM_stat_yearclustered": round(float(stat_val), 3),
                "DM_pval_yearclustered": float(p_val),
                "N_years_clusters": n_years,
                "N_obs": len(y_true),
            })

        print(f"\n    [Direct Tests of DMS vs Nearest Rivals (Year-Clustered with Newey-West h={h})]:") 
        print(f"      DMS vs Equal-Weight (1/M) : DM = {dm_vs_eq:6.3f} | p = {p_vs_eq:.4f}")
        print(f"      DMS vs Economy LightGBM   : DM = {dm_vs_lgbm:6.3f} | p = {p_vs_lgbm:.4f}")
        print(f"      DMS vs Economy Ridge      : DM = {dm_vs_ecoridge:6.3f} | p = {p_vs_ecoridge:.4f}")
        print(f"    [Clark-West (2007) Nested Tests: Single-Domain vs All-Domain Concat]:")
        print(f"      Eco-Ridge vs All-Domain Ridge : CW = {cw_ridge:6.3f} | p = {p_cw_ridge:.4f} ({'Concat fails to add signal' if p_cw_ridge >= 0.05 else 'Concat adds signal'})")
        print(f"      Eco-LGBM vs All-Domain LGBM   : CW = {cw_lgbm:6.3f} | p = {p_cw_lgbm:.4f}")

        gap = abs(
            np.mean(np.abs(y_true - models["DMS (feedback disabled)"]))
            - np.mean(np.abs(y_true - models["Equal-Weight Multi-Domain"]))
        )
        print(f"    [check] |MAE(DMS no-feedback) - MAE(equal-weight)| = {gap:.2e} "
               f"({'as expected: DMA without updates IS the 1/M average' if gap < 1e-12 else 'UNEXPECTED'})")

    results_df = pd.DataFrame(global_records)
    out_dir = ROOT / "data" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "real_cross_domain_benchmark_results.csv"
    results_df.to_csv(out_path, index=False)

    print("\n" + "=" * 105)
    print(f"  Multi-Domain Tournament complete in {time.time() - start_time:.2f}s. Artifact saved to:")
    print(f"  {out_path}")
    print("=" * 105)


if __name__ == "__main__":
    run_real_multidomain_tournament()
