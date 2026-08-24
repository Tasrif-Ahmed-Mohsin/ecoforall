"""
Sovereign Segmentation & Region-Adaptive Multi-Regime Router
===========================================================
Addresses the fundamental heterogeneity between:
  1. Low-Volatility / High-Institutional Regimes (e.g. North America, Western Europe)
     -> Where AR(1) persistence & low-variance Ridge are optimal in steady-state.
  2. Emerging / Transition / Volatile Regimes (e.g. Asia, Latin America, Sub-Saharan Africa)
     -> Where non-linear multi-domain specialists (Politics, Climate, Society) dominate.

We evaluate:
  - Global Baseline AR(1)
  - Global Economy Ridge
  - Uniform Global Router (LGCF)
  - Sovereign-Segmented Adaptive Router (LGCF-Segmented)
Across 5-Fold Rolling-Origin Walk-Forward CV (1960-2025)
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent

REGION_MAP = {
    "USA": "North America", "CAN": "North America",
    "DEU": "Europe & Central Asia", "FRA": "Europe & Central Asia", "GBR": "Europe & Central Asia",
    "ITA": "Europe & Central Asia", "ESP": "Europe & Central Asia", "NLD": "Europe & Central Asia",
    "CHE": "Europe & Central Asia", "SWE": "Europe & Central Asia", "NOR": "Europe & Central Asia",
    "POL": "Europe & Central Asia", "AUT": "Europe & Central Asia", "BEL": "Europe & Central Asia",
    "RUS": "Europe & Central Asia", "TUR": "Europe & Central Asia", "UKR": "Europe & Central Asia",
    "GRC": "Europe & Central Asia", "PRT": "Europe & Central Asia", "IRL": "Europe & Central Asia",
    "FIN": "Europe & Central Asia", "DNK": "Europe & Central Asia", "CZE": "Europe & Central Asia",
    "ROU": "Europe & Central Asia", "HUN": "Europe & Central Asia", "KAZ": "Europe & Central Asia",
    "CHN": "East Asia & Pacific", "JPN": "East Asia & Pacific", "KOR": "East Asia & Pacific",
    "AUS": "East Asia & Pacific", "IDN": "East Asia & Pacific", "MYS": "East Asia & Pacific",
    "PHL": "East Asia & Pacific", "SGP": "East Asia & Pacific", "THA": "East Asia & Pacific",
    "VNM": "East Asia & Pacific", "NZL": "East Asia & Pacific", "MMR": "East Asia & Pacific",
    "BRA": "Latin America & Caribbean", "MEX": "Latin America & Caribbean", "ARG": "Latin America & Caribbean",
    "CHL": "Latin America & Caribbean", "COL": "Latin America & Caribbean", "PER": "Latin America & Caribbean",
    "VEN": "Latin America & Caribbean", "ECU": "Latin America & Caribbean", "URY": "Latin America & Caribbean",
    "BOL": "Latin America & Caribbean", "PRY": "Latin America & Caribbean", "PAN": "Latin America & Caribbean",
    "CRI": "Latin America & Caribbean", "DOM": "Latin America & Caribbean", "GTM": "Latin America & Caribbean",
    "IND": "South Asia", "PAK": "South Asia", "BGD": "South Asia",
    "LKA": "South Asia", "NPL": "South Asia", "AFG": "South Asia",
    "NGA": "Sub-Saharan Africa", "ZAF": "Sub-Saharan Africa", "KEN": "Sub-Saharan Africa",
    "ETH": "Sub-Saharan Africa", "GHA": "Sub-Saharan Africa", "TZA": "Sub-Saharan Africa",
    "UGA": "Sub-Saharan Africa", "AGO": "Sub-Saharan Africa", "CIV": "Sub-Saharan Africa",
    "SEN": "Sub-Saharan Africa", "CMR": "Sub-Saharan Africa", "ZMB": "Sub-Saharan Africa",
    "ZWE": "Sub-Saharan Africa", "MOZ": "Sub-Saharan Africa", "RWA": "Sub-Saharan Africa",
    "SAU": "Middle East & North Africa", "EGY": "Middle East & North Africa", "IRN": "Middle East & North Africa",
    "ISR": "Middle East & North Africa", "ARE": "Middle East & North Africa", "IRQ": "Middle East & North Africa",
    "DZA": "Middle East & North Africa", "MAR": "Middle East & North Africa", "QAT": "Middle East & North Africa",
    "KWT": "Middle East & North Africa", "OMN": "Middle East & North Africa", "JOR": "Middle East & North Africa",
    "TUN": "Middle East & North Africa", "LBN": "Middle East & North Africa",
}

def get_region(iso):
    return REGION_MAP.get(iso, "Other / Emerging")

def run():
    p = ROOT / "data" / "processed_panels" / "quad_domain_annual_panel.parquet"
    df = pd.read_parquet(p)
    df["region_wb"] = df["iso3"].apply(get_region)

    exclude_cols = {"iso3", "country", "year", "region", "income_level", "region_wb"}
    feature_cols = [c for c in df.columns if c not in exclude_cols and not c.startswith("gdp_pc_growth_")]

    horizons = [1, 5]

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

        test_records = []

        for fold in folds:
            t_start, t_end = fold["test_start"], fold["test_end"]
            train_end = t_start - (h - 1) - 1

            tr_df = clean[clean["year"] <= train_end].copy()
            te_df = clean[(clean["year"] >= t_start) & (clean["year"] <= t_end)].copy()

            if len(te_df) == 0 or len(tr_df) == 0:
                continue

            X_tr = tr_df[feature_cols].values
            y_tr = tr_df[target_col].values
            X_te = te_df[feature_cols].values
            y_te = te_df[target_col].values

            # Preprocessing
            imp = SimpleImputer(strategy="median")
            scaler = StandardScaler()
            X_tr_clean = np.array(X_tr, dtype=np.float64, copy=True)
            X_tr_clean[~np.isfinite(X_tr_clean)] = np.nan
            X_te_clean = np.array(X_te, dtype=np.float64, copy=True)
            X_te_clean[~np.isfinite(X_te_clean)] = np.nan

            X_tr_imp = imp.fit_transform(X_tr_clean)
            X_tr_sc = np.clip(scaler.fit_transform(X_tr_imp), -5.0, 5.0)

            X_te_imp = imp.transform(X_te_clean)
            X_te_sc = np.clip(scaler.transform(X_te_imp), -5.0, 5.0)

            # 1. AR(1) Baseline
            ar_preds = []
            for idx, row in te_df.iterrows():
                iso = row["iso3"]
                hist = clean[(clean["iso3"] == iso) & (clean["year"] < row["year"])]
                if len(hist) >= 2 and "gdp_pc_growth_1y_fwd" in hist.columns:
                    lag = hist["gdp_pc_growth_1y_fwd"].iloc[-1]
                    if np.isfinite(lag):
                        ar_pred = 0.5 * lag + 0.5 * 0.02
                    else:
                        ar_pred = 0.02
                else:
                    ar_pred = 0.02
                ar_preds.append(ar_pred)
            te_df["pred_ar1"] = np.clip(ar_preds, -0.5, 0.5)

            # 2. Economy Ridge Specialist
            exp1 = Ridge(alpha=100.0, random_state=42)
            exp1.fit(X_tr_sc, y_tr)
            te_df["pred_eco"] = np.clip(exp1.predict(X_te_sc), -0.5, 0.5)

            # 3. LightGBM Quad Specialist
            exp2 = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.03, max_depth=5, random_state=42, verbose=-1, n_jobs=-1)
            exp2.fit(X_tr_imp, y_tr)
            te_df["pred_lgbm"] = np.clip(exp2.predict(X_te_imp), -0.5, 0.5)

            # 4. Huber Regressor
            exp3 = HuberRegressor(max_iter=300, alpha=50.0)
            exp3.fit(X_tr_sc, y_tr)
            te_df["pred_huber"] = np.clip(exp3.predict(X_te_sc), -0.5, 0.5)

            # 5. Global Unsegmented Router (Uniform blend)
            te_df["pred_global_router"] = 0.40 * te_df["pred_eco"] + 0.40 * te_df["pred_lgbm"] + 0.20 * te_df["pred_huber"]

            # ── 6. SOVEREIGN-SEGMENTED ADAPTIVE ROUTER ──
            # Calculate historical sovereign volatility for each country from training slice
            vol_map = {}
            for iso, grp in tr_df.groupby("iso3"):
                vol = grp[target_col].std()
                vol_map[iso] = vol if np.isfinite(vol) and vol > 0 else 0.03
            
            med_vol = np.median(list(vol_map.values())) if len(vol_map) > 0 else 0.03

            segmented_preds = []
            for idx, row in te_df.iterrows():
                iso = row["iso3"]
                reg = row["region_wb"]
                c_vol = vol_map.get(iso, med_vol)

                # Classification into Sovereign Typologies:
                # Type A: Ultra-Tranquil / Developed (e.g. North America, Low-Vol Western Europe)
                # -> Strong prior weight on AR(1) + Ridge
                if reg in ["North America"] or (reg in ["Europe & Central Asia"] and c_vol < med_vol):
                    pred_s = 0.65 * row["pred_ar1"] + 0.25 * row["pred_eco"] + 0.10 * row["pred_lgbm"]
                
                # Type B: Emerging / High Volatility / Rapid Industrialization (East Asia, South Asia, LatAm, Africa)
                # -> Strong activation of Non-Linear Quad-Domain Specialists
                elif reg in ["East Asia & Pacific", "South Asia", "Sub-Saharan Africa", "Latin America & Caribbean"]:
                    pred_s = 0.10 * row["pred_ar1"] + 0.35 * row["pred_eco"] + 0.40 * row["pred_lgbm"] + 0.15 * row["pred_huber"]
                
                # Type C: Resource / Transition / Mixed Volatility
                else:
                    pred_s = 0.25 * row["pred_ar1"] + 0.35 * row["pred_eco"] + 0.25 * row["pred_lgbm"] + 0.15 * row["pred_huber"]
                
                segmented_preds.append(pred_s)

            te_df["pred_segmented_router"] = np.clip(segmented_preds, -0.5, 0.5)
            te_df["actual"] = y_te

            test_records.append(te_df)

        full_test = pd.concat(test_records, ignore_index=True)

        print(f"\n==========================================================================================")
        print(f"  SOVEREIGN-SEGMENTED ROUTER TOURNAMENT RESULTS (h={h}Y, N={len(full_test)})")
        print(f"==========================================================================================")

        mae_ar1 = np.mean(np.abs(full_test["actual"] - full_test["pred_ar1"]))
        mae_eco = np.mean(np.abs(full_test["actual"] - full_test["pred_eco"]))
        mae_global = np.mean(np.abs(full_test["actual"] - full_test["pred_global_router"]))
        mae_seg = np.mean(np.abs(full_test["actual"] - full_test["pred_segmented_router"]))

        print(f"Global Benchmark:")
        print(f"  1. Honest AR(1) Baseline:          MAE = {mae_ar1:.5f} (Baseline)")
        print(f"  2. Single Economy (Ridge):         MAE = {mae_eco:.5f} (Lift vs AR1: {(mae_ar1-mae_eco)/mae_ar1*100:+.2f}%)")
        print(f"  3. Unsegmented Global Router:      MAE = {mae_global:.5f} (Lift vs AR1: {(mae_ar1-mae_global)/mae_ar1*100:+.2f}%)")
        print(f"  4. SOVEREIGN-SEGMENTED ROUTER:     MAE = {mae_seg:.5f} (Lift vs AR1: {(mae_ar1-mae_seg)/mae_ar1*100:+.2f}%, vs Eco: {(mae_eco-mae_seg)/mae_eco*100:+.2f}%)")

        print("\n--- REGIONAL COMPARISON: AR(1) vs. Global Router vs. Sovereign-Segmented Router ---")
        reg_summary = []
        for reg, grp in full_test.groupby("region_wb"):
            r_ar1 = np.mean(np.abs(grp["actual"] - grp["pred_ar1"]))
            r_eco = np.mean(np.abs(grp["actual"] - grp["pred_eco"]))
            r_global = np.mean(np.abs(grp["actual"] - grp["pred_global_router"]))
            r_seg = np.mean(np.abs(grp["actual"] - grp["pred_segmented_router"]))
            lift_vs_ar1 = (r_ar1 - r_seg) / r_ar1 * 100.0
            lift_vs_global = (r_global - r_seg) / r_global * 100.0
            reg_summary.append({
                "Region": reg,
                "Obs": len(grp),
                "MAE_AR1": round(r_ar1, 4),
                "MAE_Global": round(r_global, 4),
                "MAE_Segmented": round(r_seg, 4),
                "Segmented_vs_AR1(%)": round(lift_vs_ar1, 2),
                "Segmented_vs_Global(%)": round(lift_vs_global, 2),
            })
        print(pd.DataFrame(reg_summary).to_string(index=False))

if __name__ == "__main__":
    run()
