"""
Hansen, Lunde & Nason (2011) Model Confidence Set (MCS) Tournament Audit
=======================================================================
Computes the Model Confidence Set at 90% and 75% confidence levels across
all competing macro-forecasting architectures and horizons h in {1, 3, 5}.
Captures time-series / panel correlation via moving-block bootstrap (B=1,000).

Exports verified single-source-of-truth artifact to:
  data/benchmarks/real_model_confidence_set_results.csv
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
from src.econometrics.model_confidence_set import model_confidence_set

PRED_CLIP = 0.5
FEATURE_CLIP = 5.0


def run_mcs_audit():
    seed_everything(42)
    panel_path = ROOT / "data" / "processed_panels" / "real_cross_domain_annual_panel.parquet"
    if not panel_path.exists():
        raise FileNotFoundError(f"Real panel not found at {panel_path}")

    df = pd.read_parquet(panel_path)
    df["region_wb"] = df["iso3"].apply(get_wb_region)
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
    all_mcs_records = []

    print("=" * 105)
    print("  HANSEN, LUNDE & NASON (2011) MODEL CONFIDENCE SET (MCS) TOURNAMENT AUDIT")
    print("  Evaluating M=10 candidate forecasters across h in {1, 3, 5} via Block Bootstrap (B=1,000)")
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
            ev_df["AR(1) Baseline"] = ar_model.predict_panel(ev_df, lag_growth_col=DEFAULT_LAG_GROWTH_COL)

            for name, Xtr, Xev in (
                ("Economy-Only Ridge", X_tr_eco_sc, X_ev_eco_sc),
                ("All-Domain Ridge (Concat)", X_tr_all_sc, X_ev_all_sc),
                ("Politics Ridge (V-Dem)", X_tr_pol_sc, X_ev_pol_sc),
                ("Climate Ridge (ERA5)", X_tr_cli_sc, X_ev_cli_sc),
            ):
                m = Ridge(alpha=100.0, random_state=42)
                m.fit(Xtr, y_tr)
                ev_df[name] = np.clip(m.predict(Xev), -PRED_CLIP, PRED_CLIP)

            for name, imp, Xtr_raw, Xev_raw in (
                ("Economy LightGBM", imp_eco, X_tr_eco, X_ev_eco),
                ("All-Domain LightGBM (Concat)", imp_all, X_tr_all, X_ev_all),
            ):
                g = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.03, max_depth=5,
                                      random_state=42, verbose=-1, n_jobs=-1)
                g.fit(imp.transform(Xtr_raw), y_tr)
                ev_df[name] = np.clip(g.predict(imp.transform(Xev_raw)), -PRED_CLIP, PRED_CLIP)

            dfm = DynamicFactorForecaster(n_factors=5, alpha=20.0, random_state=42)
            dfm.fit(X_tr_eco_sc, y_tr)
            ev_df["Stock-Watson DFM"] = np.clip(dfm.predict(X_ev_eco_sc), -PRED_CLIP, PRED_CLIP)

            expert_cols = ["AR(1) Baseline", "Economy-Only Ridge", "Politics Ridge (V-Dem)",
                           "Climate Ridge (ERA5)", "Economy LightGBM"]
            specialist_matrix = np.column_stack([ev_df[c].values for c in expert_cols])
            ev_df["Equal-Weight Multi-Domain"] = EqualWeightCombinationForecaster.combine(
                [ev_df[c].values for c in expert_cols]
            )

            dms_router = DynamicModelSelectionRouter(n_experts=len(expert_cols), forgetting_factor=0.92, mode="dma")
            dms_preds, _ = dms_router.route_panel(
                ev_df, specialist_matrix, y_ev, horizon=h, year_col="year", iso_col="iso3"
            )
            ev_df["DMS State-Space Router"] = np.clip(dms_preds, -PRED_CLIP, PRED_CLIP)

            h_preds_list.append(ev_df[ev_df["is_score"]].copy())

        h_combined = pd.concat(h_preds_list, ignore_index=True)
        y_true = h_combined[target_col].values

        candidate_models = [
            "AR(1) Baseline",
            "Economy-Only Ridge",
            "All-Domain Ridge (Concat)",
            "Politics Ridge (V-Dem)",
            "Climate Ridge (ERA5)",
            "Economy LightGBM",
            "All-Domain LightGBM (Concat)",
            "Stock-Watson DFM",
            "Equal-Weight Multi-Domain",
            "DMS State-Space Router",
        ]

        # Year-clustered mean losses for the block bootstrap
        h_combined["year_int"] = h_combined["year"].astype(int)
        years_unique = sorted(h_combined["year_int"].unique())
        
        # Build yearly mean MAE matrix of shape (n_years, M)
        yearly_loss_list = []
        for yr in years_unique:
            yr_mask = h_combined["year_int"] == yr
            y_sub = y_true[yr_mask]
            yr_losses = []
            for m_name in candidate_models:
                p_sub = h_combined.loc[yr_mask, m_name].values
                mae_yr = float(np.mean(np.abs(y_sub - p_sub)))
                yr_losses.append(mae_yr)
            yearly_loss_list.append(yr_losses)

        yearly_losses = np.array(yearly_loss_list, dtype=np.float64)  # (T_years, M)

        mcs_res = model_confidence_set(
            losses=yearly_losses,
            model_names=candidate_models,
            alpha=0.10,
            n_boot=1000,
            block_size=max(2, h),  # Block size >= h captures multi-step serial correlation
            seed=42,
            stat_type="t_max",
        )

        print(f"\n--- Horizon h = {h} Year(s) Model Confidence Set (MCS 90%) ---")
        for _, row in mcs_res.iterrows():
            in_str = "IN MCS (90%)" if row["In_MCS_90pct"] else "ELIMINATED"
            print(f"  Rank {row['Rank']:2d} | {row['Model']:<30} | Mean MAE: {row['Mean_Loss']:.5f} | MCS p-val: {row['MCS_P_Value']:.4f} | {in_str}")
            all_mcs_records.append({
                "Horizon": h,
                "Rank": row["Rank"],
                "Model": row["Model"],
                "Mean_Loss_MAE": row["Mean_Loss"],
                "MCS_P_Value": row["MCS_P_Value"],
                "In_MCS_90pct": row["In_MCS_90pct"],
                "In_MCS_75pct": row["In_MCS_75pct"],
            })

    out_df = pd.DataFrame(all_mcs_records)
    out_dir = ROOT / "data" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "real_model_confidence_set_results.csv"
    out_df.to_csv(out_path, index=False)

    print("\n" + "=" * 105)
    print(f"  MCS Audit complete. Single-source-of-truth artifact saved to:\n  {out_path}")
    print("=" * 105)


if __name__ == "__main__":
    run_mcs_audit()
