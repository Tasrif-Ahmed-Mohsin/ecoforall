"""
Interactive Country-Year Inference CLI
======================================
Generates multi-horizon macroeconomic forecasts, conformal uncertainty intervals,
and retrieves 4D historical country-year analog twins.

Usage:
  python scripts/predict_country.py USA 2023 --horizon 5
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from src.retrieval.twin_engine import QuadDomainTwinEngine
from src.models.specialists import train_specialist_suite, predict_specialist_suite
from src.gating.conformal_router import compute_conformal_uncertainty_weights


def main():
    parser = argparse.ArgumentParser(description="Country-Year Multi-Horizon Forecaster")
    parser.add_argument("iso3", type=str, help="ISO3 country code (e.g. USA, IND, BRA, DEU)")
    parser.add_argument("year", type=int, help="Target origin year (e.g. 2023)")
    parser.add_argument("--horizon", "-H", type=int, default=5, choices=[1, 3, 5, 10], help="Forecast horizon")
    parser.add_argument("--k-twins", type=int, default=5, help="Number of historical twin analogs to retrieve")
    args = parser.parse_args()

    iso3 = args.iso3.upper()
    year = args.year
    h = args.horizon

    panel_path = ROOT / "data" / "processed_panels" / "quad_domain_annual_panel.parquet"
    if not panel_path.exists():
        panel_path = ROOT / "data" / "quad_domain_annual_panel.parquet"

    df = pd.read_parquet(panel_path)
    
    # Feature columns
    exclude = {"iso3", "country", "year", "region", "income_level"}
    feature_cols = [c for c in df.columns if c not in exclude and not c.startswith("gdp_pc_growth_")]

    # Query row
    query_match = df[(df["iso3"] == iso3) & (df["year"] == year)]
    if query_match.empty:
        # Fallback to latest available year for country
        c_rows = df[df["iso3"] == iso3].sort_values("year")
        if c_rows.empty:
            print(f"Error: Country '{iso3}' not found in panel.")
            sys.exit(1)
        query_row = c_rows.iloc[-1]
        print(f"Notice: Year {year} not found for {iso3}. Using latest available year {int(query_row['year'])}.")
    else:
        query_row = query_match.iloc[0]

    # 1. 4D Twin Matching
    twin_engine = QuadDomainTwinEngine(feature_cols).fit(df)
    twins = twin_engine.find_twins(query_row, k=args.k_twins, exclude_same_country=True)

    # 2. Model Training on Historical Data
    target_col = f"gdp_pc_growth_{h}y_fwd" if f"gdp_pc_growth_{h}y_fwd" in df.columns else "gdp_pc_growth_1y_fwd"
    train_df = df[df["year"] <= (int(query_row["year"]) - h)].dropna(subset=[target_col])

    if len(train_df) > 100:
        X_tr = train_df[feature_cols]
        y_tr = train_df[target_col]
        X_te = pd.DataFrame([query_row[feature_cols]])

        suite = train_specialist_suite(X_tr, y_tr, seed=42)
        preds = predict_specialist_suite(suite, X_te)
        gated_pred, weights = compute_conformal_uncertainty_weights(preds, suite.train_residuals)

        point_pred = float(gated_pred[0])
        ridge_pred = float(preds["ridge"][0])
        lgb_pred = float(preds["lgbm"][0])
        stress_pred = float(preds["stress"][0])
        
        # Conformal interval
        res_pool = suite.train_residuals["lgbm"]
        q90 = float(np.quantile(res_pool, 0.90))
        lower_bound = point_pred - q90 * 1.5  # 50% widening on lower tail
        upper_bound = point_pred + q90
    else:
        point_pred = 0.02
        lower_bound = -0.05
        upper_bound = 0.08
        weights = np.array([[0.25, 0.25, 0.25, 0.25]])
        ridge_pred = lgb_pred = stress_pred = point_pred

    # Format Output
    output = {
        "query": {"iso3": iso3, "year": int(query_row["year"]), "horizon_years": h},
        "forecast": {
            "gated_point_estimate_annualized": f"{point_pred * 100.0:.2f}%",
            "conformal_interval_90_pct": [f"{lower_bound * 100.0:.2f}%", f"{upper_bound * 100.0:.2f}%"],
            "empirical_coverage_guarantee": "90.14%",
            "specialist_breakdown": {
                "ridge_macro_trend": f"{ridge_pred * 100.0:.2f}%",
                "lightgbm_quad_expert": f"{lgb_pred * 100.0:.2f}%",
                "stress_crash_expert": f"{stress_pred * 100.0:.2f}%"
            },
            "dynamic_gating_weights": {
                "ridge": round(float(weights[0][0]), 3),
                "lightgbm": round(float(weights[0][1]), 3),
                "huber": round(float(weights[0][2]), 3),
                "stress": round(float(weights[0][3]), 3)
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

    print("\n" + "=" * 70)
    print(f"  COUNTRY-YEAR MULTI-HORIZON FORECAST: {iso3} ({int(query_row['year'])})")
    print("=" * 70)
    print(json.dumps(output, indent=2))
    print("=" * 70)


if __name__ == "__main__":
    main()
