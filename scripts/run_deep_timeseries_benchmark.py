"""
NeurIPS Benchmark Suite: StateSpaceMoE vs. Modern Deep Time-Series Baselines
============================================================================
Evaluates StateSpaceMoE against:
  - PatchTST (Nie et al., ICLR 2023)
  - iTransformer (Liu et al., ICLR 2024)
  - DLinear (Zheng et al., AAAI 2023)
  - Single-Domain Economy Ridge
  - Autoregressive AR(1)

Across 5-Fold Rolling-Origin Walk-Forward CV (1960-2025; Horizons h in {1, 3, 5} Years).
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.state_space_moe import StateSpaceMoE
from src.models.deep_baselines import DLinearForecaster, PatchTSTForecaster, iTransformerForecaster
from src.econometrics.panel_granger import diebold_mariano_test


def compute_ar1(clean_df: pd.DataFrame, te_df: pd.DataFrame) -> list[float]:
    ar_preds = []
    for _, row in te_df.iterrows():
        iso = row["iso3"]
        hist = clean_df[(clean_df["iso3"] == iso) & (clean_df["year"] < row["year"])]
        if len(hist) >= 2 and "gdp_pc_growth_1y_fwd" in hist.columns:
            lag = hist["gdp_pc_growth_1y_fwd"].iloc[-1]
            if np.isfinite(lag):
                ar_pred = 0.5 * lag + 0.5 * 0.02
            else:
                ar_pred = 0.02
        else:
            ar_pred = 0.02
        ar_preds.append(ar_pred)
    return ar_preds


def run():
    start_time = time.time()
    p = ROOT / "data" / "processed_panels" / "quad_domain_annual_panel.parquet"
    df = pd.read_parquet(p)

    exclude_cols = {"iso3", "country", "year", "region", "income_level", "region_wb"}
    feature_cols = [c for c in df.columns if c not in exclude_cols and not c.startswith("gdp_pc_growth_")]

    horizons = [1, 3, 5]
    results = []

    print("=" * 105)
    print("  NEURIPS BENCHMARK SUITE: StateSpaceMoE vs. MODERN DEEP TIME-SERIES BASELINES")
    print("  5-Fold Rolling-Origin Walk-Forward CV (1960-2025) across 169 Sovereign Economies")
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
            te_df["pred_ar1"] = np.clip(compute_ar1(clean, te_df), -0.5, 0.5)

            # 2. Economy Ridge
            m_ridge = Ridge(alpha=100.0, random_state=42)
            m_ridge.fit(X_tr_sc, y_tr)
            te_df["pred_ridge"] = np.clip(m_ridge.predict(X_te_sc), -0.5, 0.5)

            # 3. DLinear (AAAI 2023)
            m_dlinear = DLinearForecaster(alpha=20.0, random_state=42)
            m_dlinear.fit(X_tr_sc, y_tr)
            te_df["pred_dlinear"] = m_dlinear.predict(X_te_sc)

            # 4. PatchTST (ICLR 2023)
            m_patchtst = PatchTSTForecaster(patch_len=16, stride=8, random_state=42)
            m_patchtst.fit(X_tr_sc, y_tr)
            te_df["pred_patchtst"] = m_patchtst.predict(X_te_sc)

            # 5. iTransformer (ICLR 2024)
            m_itrans = iTransformerForecaster(d_embed=32, random_state=42)
            m_itrans.fit(X_tr_sc, y_tr)
            te_df["pred_itransformer"] = m_itrans.predict(X_te_sc)

            # 6. StateSpaceMoE (Our Proposed Unified Algorithm)
            m_moe = StateSpaceMoE(horizon=h, lambda_forget=0.92, random_state=42)
            m_moe.fit(X_tr_sc, y_tr, tr_df["iso3"].tolist(), tr_df["year"].values)
            te_df["pred_statespace_moe"] = m_moe.predict(X_te_sc, te_df["iso3"].tolist(), te_df["pred_ar1"].values)

            te_df["actual"] = y_te
            test_records.append(te_df)

        full_test = pd.concat(test_records, ignore_index=True)
        N = len(full_test)
        y_true = full_test["actual"].values

        models = [
            ("1. Autoregressive AR(1)", "pred_ar1"),
            ("2. Single Economy (Ridge)", "pred_ridge"),
            ("3. DLinear (AAAI 2023)", "pred_dlinear"),
            ("4. PatchTST (ICLR 2023)", "pred_patchtst"),
            ("5. iTransformer (ICLR 2024)", "pred_itransformer"),
            ("6. StateSpaceMoE (Ours)", "pred_statespace_moe"),
        ]

        mae_eco = np.mean(np.abs(y_true - full_test["pred_ridge"].values))
        y_ours = full_test["pred_statespace_moe"].values

        print(f"\n>>> Results for Horizon h={h} Year(s) (N={N} Out-of-Fold Obs):")

        for name, col in models:
            preds = full_test[col].values
            mae = np.mean(np.abs(y_true - preds))
            lift_vs_eco = (mae_eco - mae) / mae_eco * 100.0

            dm_stat, p_val = diebold_mariano_test(y_true, preds, y_ours, h=h, criterion="mae")

            results.append({
                "Horizon": f"{h}Y",
                "Total_Obs": N,
                "Model_Architecture": name,
                "MAE": round(mae, 5),
                "Lift_vs_Eco_pct": round(lift_vs_eco, 2),
                "DM_Stat_vs_Ours": round(dm_stat, 3),
                "p_value_vs_Ours": float(p_val),
            })

            print(f"  {name:<32} | MAE: {mae:.5f} | Lift vs Eco: {lift_vs_eco:+.2f}% | DM vs Ours: {dm_stat:6.2f} (p={p_val:.4e})")

    out_df = pd.DataFrame(results)
    out_path = ROOT / "data" / "benchmarks" / "deep_timeseries_benchmark_results.csv"
    out_df.to_csv(out_path, index=False)

    print("\n" + "=" * 105)
    print("  FINAL DEEP TIME-SERIES BENCHMARK MATRIX (Saved to data/benchmarks/deep_timeseries_benchmark_results.csv)")
    print("=" * 105)
    print(out_df.to_string(index=False))

    elapsed = time.time() - start_time
    print(f"\nBenchmark completed successfully in {elapsed:.2f}s.")


if __name__ == "__main__":
    run()
