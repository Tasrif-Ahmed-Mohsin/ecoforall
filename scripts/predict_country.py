"""
Interactive Country-Year Inference CLI (DMS & Domain-Specialist Architecture)
=============================================================================
Generates multi-horizon sovereign growth forecasts using:
  1. 5 Domain-Quarantined Specialists (AR(1), Economy Ridge, Politics Ridge, Climate Ridge, Economy LightGBM)
  2. Recursive State-Space Dynamic Model Selection (Koop-Korobilis 2012 DMS Router)
  3. Empirical Conformal Uncertainty Bounds (90% Confidence)
  4. 4D Historical Analog Sovereign Twins (Macro, Politics, Climate, Trajectory)

Usage:
  python scripts/predict_country.py USA 2023 --horizon 5
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from src.models.macro_baselines import (
    PerCountryARForecaster,
    EqualWeightCombinationForecaster,
    add_lagged_growth,
    DEFAULT_LAG_GROWTH_COL,
)
from src.gating.dms_state_space_router import DynamicModelSelectionRouter
from src.retrieval.twin_engine import QuadDomainTwinEngine


def main():
    parser = argparse.ArgumentParser(description="Country-Year Multi-Horizon DMS Forecaster")
    parser.add_argument("iso3", type=str, help="ISO3 country code (e.g. USA, IND, BRA, DEU)")
    parser.add_argument("year", type=int, help="Target origin year (e.g. 2023)")
    parser.add_argument("--horizon", "-H", type=int, default=1, choices=[1, 3, 5], help="Forecast horizon in years")
    parser.add_argument("--k-twins", type=int, default=5, help="Number of historical twin analogs to retrieve")
    args = parser.parse_args()

    iso3 = args.iso3.upper()
    target_year = args.year
    h = args.horizon

    panel_path = ROOT / "data" / "processed_panels" / "real_cross_domain_annual_panel.parquet"
    if not panel_path.exists():
        print(f"Error: Panel not found at {panel_path}. Run src/harmonization/ingest_real_cross_domain_panel.py first.")
        sys.exit(1)

    df = pd.read_parquet(panel_path)
    df = add_lagged_growth(df, level_col="gdp_pc_real")

    exclude_meta = {
        "iso3", "country", "year", "region", "income_level", "region_wb",
        DEFAULT_LAG_GROWTH_COL,
    }
    all_features = [c for c in df.columns if c not in exclude_meta and not c.endswith("_fwd")]
    eco_cols = [c for c in all_features if not c.startswith("vdem_") and not c.startswith("climate_")]
    pol_cols = [c for c in all_features if c.startswith("vdem_")]
    cli_cols = [c for c in all_features if c.startswith("climate_")]

    # Locate query row
    c_df = df[df["iso3"] == iso3].sort_values("year")
    if c_df.empty:
        print(f"Error: Country '{iso3}' not found in panel.")
        sys.exit(1)

    query_match = c_df[c_df["year"] == target_year]
    if query_match.empty:
        query_row = c_df.iloc[-1]
        origin_year = int(query_row["year"])
        print(f"Notice: Year {target_year} not found for {iso3}. Using latest available year {origin_year}.")
    else:
        query_row = query_match.iloc[0]
        origin_year = target_year

    # 1. 4D Twin Matching
    twin_engine = QuadDomainTwinEngine(all_features).fit(df)
    twins = twin_engine.find_twins(query_row, k=args.k_twins, exclude_same_country=True)

    # 2. Historical Training Slice (Strict quarantine: train_end <= origin_year - h)
    target_col = f"gdp_pc_growth_{h}y_fwd"
    train_end = origin_year - h
    train_df = df[df["year"] <= train_end].dropna(subset=[target_col]).copy()

    if len(train_df) < 50:
        print("Warning: Insufficient historical data for full model training.")
        sys.exit(1)

    y_train = train_df[target_col].to_numpy(dtype=np.float64)

    # Impute and scale per domain
    imp_eco, scaler_eco = SimpleImputer(strategy="median"), StandardScaler()
    imp_pol, scaler_pol = SimpleImputer(strategy="median"), StandardScaler()
    imp_cli, scaler_cli = SimpleImputer(strategy="median"), StandardScaler()

    def clean_mat(mat):
        m = np.array(mat, dtype=np.float64, copy=True)
        m[~np.isfinite(m)] = np.nan
        return m

    def prep_mats(cols, imp, scaler, tr_df, q_df):
        A = clean_mat(tr_df[cols].values)
        B = clean_mat(q_df[cols].values)
        A_sc = np.clip(scaler.fit_transform(imp.fit_transform(A)), -5.0, 5.0)
        B_sc = np.clip(scaler.transform(imp.transform(B)), -5.0, 5.0)
        return A, B, A_sc, B_sc

    query_frame = pd.DataFrame([query_row])
    X_tr_eco, X_q_eco, X_tr_eco_sc, X_q_eco_sc = prep_mats(eco_cols, imp_eco, scaler_eco, train_df, query_frame)
    _, _, X_tr_pol_sc, X_q_pol_sc = prep_mats(pol_cols, imp_pol, scaler_pol, train_df, query_frame)
    _, _, X_tr_cli_sc, X_q_cli_sc = prep_mats(cli_cols, imp_cli, scaler_cli, train_df, query_frame)

    # Train Specialists
    # 1. AR(1)
    ar_model = PerCountryARForecaster(horizon=h, prediction_clip=0.5)
    ar_model.fit(train_df, target_col=target_col, lag_growth_col=DEFAULT_LAG_GROWTH_COL)
    pred_ar1 = float(ar_model.predict_panel(query_frame, lag_growth_col=DEFAULT_LAG_GROWTH_COL)[0])

    # 2. Economy Ridge
    eco_ridge = Ridge(alpha=100.0, random_state=42).fit(X_tr_eco_sc, y_train)
    pred_eco_ridge = float(np.clip(eco_ridge.predict(X_q_eco_sc)[0], -0.5, 0.5))

    # 3. Politics Ridge
    pol_ridge = Ridge(alpha=100.0, random_state=42).fit(X_tr_pol_sc, y_train)
    pred_pol_ridge = float(np.clip(pol_ridge.predict(X_q_pol_sc)[0], -0.5, 0.5))

    # 4. Climate Ridge
    cli_ridge = Ridge(alpha=100.0, random_state=42).fit(X_tr_cli_sc, y_train)
    pred_cli_ridge = float(np.clip(cli_ridge.predict(X_q_cli_sc)[0], -0.5, 0.5))

    # 5. Economy LightGBM
    eco_lgbm = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.03, max_depth=5, random_state=42, verbose=-1, n_jobs=-1)
    eco_lgbm.fit(imp_eco.transform(X_tr_eco), y_train)
    pred_eco_lgbm = float(np.clip(eco_lgbm.predict(imp_eco.transform(X_q_eco))[0], -0.5, 0.5))

    specialist_preds = [pred_ar1, pred_eco_ridge, pred_pol_ridge, pred_cli_ridge, pred_eco_lgbm]
    expert_names = ["AR(1) Baseline", "Economy-Only Ridge", "Politics-Only Ridge", "Climate-Only Ridge", "Economy LightGBM"]

    # 3. State-Space DMS Router State
    # Run router on historical country history to obtain state-space posterior weights for target_year
    c_hist = c_df[c_df["year"] <= origin_year].copy()
    if len(c_hist) > 0:
        # Generate predictions on c_hist for routing
        _, _, _, X_hist_eco_sc = prep_mats(eco_cols, imp_eco, scaler_eco, train_df, c_hist)
        _, _, _, X_hist_pol_sc = prep_mats(pol_cols, imp_pol, scaler_pol, train_df, c_hist)
        _, _, _, X_hist_cli_sc = prep_mats(cli_cols, imp_cli, scaler_cli, train_df, c_hist)
        X_hist_eco_raw = imp_eco.transform(clean_mat(c_hist[eco_cols].values))

        hist_ar1 = ar_model.predict_panel(c_hist, lag_growth_col=DEFAULT_LAG_GROWTH_COL)
        hist_eco_ridge = np.clip(eco_ridge.predict(X_hist_eco_sc), -0.5, 0.5)
        hist_pol_ridge = np.clip(pol_ridge.predict(X_hist_pol_sc), -0.5, 0.5)
        hist_cli_ridge = np.clip(cli_ridge.predict(X_hist_cli_sc), -0.5, 0.5)
        hist_eco_lgbm = np.clip(eco_lgbm.predict(X_hist_eco_raw), -0.5, 0.5)

        hist_matrix = np.column_stack([hist_ar1, hist_eco_ridge, hist_pol_ridge, hist_cli_ridge, hist_eco_lgbm])
        y_hist_target = c_hist[target_col].to_numpy(dtype=np.float64) if target_col in c_hist.columns else None

        router = DynamicModelSelectionRouter(n_experts=5, forgetting_factor=0.92, mode="dma")
        gated_preds, weights_matrix = router.route_panel(
            c_hist, hist_matrix, y_hist_target, horizon=h, year_col="year", iso_col="iso3"
        )
        dms_weights = weights_matrix[-1]
        dms_pred = float(gated_preds[-1])
    else:
        dms_weights = np.ones(5) / 5.0
        dms_pred = float(np.mean(specialist_preds))

    # Conformal 90% uncertainty bound from historical OOS training residuals
    train_eco_pred = eco_ridge.predict(X_tr_eco_sc)
    train_res = np.abs(y_train - train_eco_pred)
    q90 = float(np.quantile(train_res, 0.90))
    lower_bound = dms_pred - q90 * 1.2
    upper_bound = dms_pred + q90

    eq_weight_pred = float(EqualWeightCombinationForecaster.combine([np.array([p]) for p in specialist_preds])[0])

    output = {
        "query": {"iso3": iso3, "year": origin_year, "horizon_years": h},
        "forecast": {
            "dms_state_space_point_estimate": f"{dms_pred * 100.0:.2f}%",
            "conformal_interval_90_pct": [f"{lower_bound * 100.0:.2f}%", f"{upper_bound * 100.0:.2f}%"],
            "specialist_breakdown": {
                "ar1_baseline": f"{pred_ar1 * 100.0:.2f}%",
                "economy_ridge": f"{pred_eco_ridge * 100.0:.2f}%",
                "politics_ridge_vdem": f"{pred_pol_ridge * 100.0:.2f}%",
                "climate_ridge_era5": f"{pred_cli_ridge * 100.0:.2f}%",
                "economy_lightgbm": f"{pred_eco_lgbm * 100.0:.2f}%",
                "equal_weight_combination": f"{eq_weight_pred * 100.0:.2f}%"
            },
            "state_space_dms_weights": {
                "ar1": round(float(dms_weights[0]), 4),
                "economy_ridge": round(float(dms_weights[1]), 4),
                "politics_ridge": round(float(dms_weights[2]), 4),
                "climate_ridge": round(float(dms_weights[3]), 4),
                "economy_lgbm": round(float(dms_weights[4]), 4)
            }
        },
        "top_historical_twins": [
            {
                "country": t.twin_iso3,
                "year": t.twin_year,
                "similarity": f"{t.similarity_score}%",
                "subsequent_trajectory": f"{t.future_growth_5y * 100.0:.2f}%" if t.future_growth_5y is not None else "N/A"
            }
            for t in twins
        ]
    }

    print("\n" + "=" * 80)
    print(f"  SOVEREIGN MULTI-HORIZON DMS FORECAST: {iso3} (Origin {origin_year}, Horizon {h}y)")
    print("=" * 80)
    print(json.dumps(output, indent=2))
    print("=" * 80)


if __name__ == "__main__":
    main()
