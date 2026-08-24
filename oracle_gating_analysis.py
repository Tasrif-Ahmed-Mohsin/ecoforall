"""
Oracle Gating Analysis — Theoretical Upper Bound for LLM-Gated Cross-Domain Forecasting
=========================================================================================
For each (country, year) in the test set, this script computes predictions from
5 domain configurations and identifies which one would have been optimal (oracle).

This gives us:
  1. Oracle-gated MAE (the theoretical best a perfect gate could achieve)
  2. Uniform-mixture MAE (what averaging all configs gives — our current null result)
  3. The GAP between oracle and uniform -> the room for improvement via gating
  4. Regime analysis: WHEN does the oracle deviate from eco-only?
"""

from __future__ import annotations
import json
import logging
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────── Paths ───────────────────────────
ROOT = Path(r"e:\politics and economy")
QUAD_PANEL = ROOT / "data" / "quad_domain_annual_panel.parquet"
OUT_DIR = ROOT / "data" / "oracle_gating_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────── Config ───────────────────────────
HORIZONS = [1, 3, 5]
N_FOLDS = 5
TEST_WINDOW = 4
MIN_TRAIN_ROWS = 200

# Domain feature keywords for classification
DOMAIN_KEYWORDS = {
    "S1_Eco": ["gdp", "inflation", "debt", "unemployment", "banking_crisis",
               "currency_crisis", "sov_debt_crisis", "investment", "exports",
               "imports", "current_account", "gov_deficit", "real_wage",
               "equity", "housing", "short_rate", "long_rate", "fx_to_usd",
               "credit", "broad_money", "narrow_money", "cpi", "eco"],
    "S2_Pol": ["goldstein", "conflict", "protest", "sanction", "stability",
               "news_tone", "verbal", "material", "diplomatic", "coercion", "pol"],
    "S3_Env": ["co2", "temp_anomaly", "forest", "flood", "drought", "wildfire",
               "storm", "renewable", "disaster", "greenhouse", "energy_use",
               "protected_area", "extreme_temp", "env"],
    "S4_Hum": ["psychology", "society", "trust", "fear", "optimism",
               "nationalism", "cohesion", "confidence", "education",
               "urbanization", "population", "age", "religion",
               "healthcare", "migration", "human"],
}


# ─────────────────────────── Feature Classification ───────────────────────────

def classify_features(columns: list[str]) -> dict[str, list[str]]:
    """Classify columns into domain sectors."""
    exclude = {"iso3", "year", "timestamp", "target", "target_h1", "target_h3", "target_h5"}
    num_cols = [
        c for c in columns if c not in exclude
        and not c.endswith("_target_h1") and not c.endswith("_target_h3")
        and not c.endswith("_target_h5") and not c.endswith("y_fwd")
        and not c.startswith("iso_") and not c.startswith("tier_")
    ]

    assigned = set()
    sectors = {"S1_Eco": [], "S2_Pol": [], "S3_Env": [], "S4_Hum": []}

    for sector, keywords in DOMAIN_KEYWORDS.items():
        for c in num_cols:
            if c not in assigned and any(k in c.lower() for k in keywords):
                sectors[sector].append(c)
                assigned.add(c)

    # Unassigned numeric columns default to S1_Eco
    for c in num_cols:
        if c not in assigned:
            sectors["S1_Eco"].append(c)
            assigned.add(c)

    return sectors


def build_domain_configs(sectors: dict[str, list[str]]) -> dict[str, list[str]]:
    """Build the 5 domain configurations to compare."""
    return {
        "eco_only":     sectors["S1_Eco"],
        "eco_pol":      sorted(list(set(sectors["S1_Eco"] + sectors["S2_Pol"]))),
        "eco_env":      sorted(list(set(sectors["S1_Eco"] + sectors["S3_Env"]))),
        "eco_human":    sorted(list(set(sectors["S1_Eco"] + sectors["S4_Hum"]))),
        "full_quad":    sorted(list(set(sectors["S1_Eco"] + sectors["S2_Pol"] + sectors["S3_Env"] + sectors["S4_Hum"]))),
    }


# ─────────────────────────── Target Construction ───────────────────────────

def make_target(df: pd.DataFrame, h: int) -> pd.Series:
    """log(gdp_pc_{y+h} / gdp_pc_y)"""
    gdp_col = None
    for candidate in ["gdp_pc_real", "gdp_pc_real_usd", "gdp_pc"]:
        if candidate in df.columns:
            gdp_col = candidate
            break
    if gdp_col is None:
        raise ValueError("No GDP per capita column found")

    df = df.sort_values(["iso3", "year"]).copy()
    g_fwd = df.groupby("iso3")[gdp_col].shift(-h)
    ratio = g_fwd / df[gdp_col]
    target = np.log(ratio.where(ratio > 0)).astype(np.float32)
    return target.rename("target")


# ─────────────────────────── Feature Transformation ───────────────────────────

def rank_fit_transform(X: np.ndarray, fit_mask: np.ndarray) -> np.ndarray:
    """Per-column rank transform, fit strictly on train rows only."""
    out = np.full_like(X, np.nan, dtype=np.float32)
    for j in range(X.shape[1]):
        col = X[:, j]
        fit_vals = col[fit_mask]
        fit_valid = fit_vals[~np.isnan(fit_vals)]
        if len(fit_valid) < 2:
            out[:, j] = 0.5
            continue
        sorted_fit = np.sort(fit_valid)
        n_fit = len(sorted_fit)
        all_valid = ~np.isnan(col)
        positions = np.searchsorted(sorted_fit, col[all_valid], side="right")
        out[all_valid, j] = (positions / n_fit).astype(np.float32)
    return out


# ─────────────────────────── Model Training ───────────────────────────

def train_predict_models(X_raw, y, train_mask, test_mask, seed=42):
    """Train LGBM + Ridge on rank-transformed features and return ensemble predictions."""
    Xr = rank_fit_transform(X_raw, train_mask)
    
    imp = SimpleImputer(strategy="median")
    Xt = imp.fit_transform(Xr[train_mask])
    Xp = imp.transform(Xr[test_mask])
    
    # LGBM
    model_lgb = lgb.LGBMRegressor(
        n_estimators=150, learning_rate=0.04, max_depth=5,
        num_leaves=20, min_child_samples=25,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=1.0, reg_lambda=1.0,
        random_state=seed, n_jobs=-1, verbose=-1,
        objective="regression_l1",
    )
    model_lgb.fit(Xt, y[train_mask])
    pred_lgb = model_lgb.predict(Xp)
    
    # Ridge
    scaler = StandardScaler()
    Xt_sc = scaler.fit_transform(Xt)
    Xp_sc = scaler.transform(Xp)
    
    model_ridge = Ridge(alpha=100.0, random_state=seed)
    model_ridge.fit(Xt_sc, y[train_mask])
    pred_ridge = model_ridge.predict(Xp_sc)
    
    # 60/40 Ensemble
    return 0.6 * pred_lgb + 0.4 * pred_ridge


# ─────────────────────────── Diebold-Mariano ───────────────────────────

def diebold_mariano(e1, e2):
    """Two-sided DM test on absolute errors."""
    d = e1**2 - e2**2
    n = len(d)
    if n < 5:
        return 0.0, 1.0
    mean_d = np.mean(d)
    var_d = max(1e-10, np.var(d) / n)
    dm_stat = mean_d / np.sqrt(var_d)
    p_value = 2 * (1 - norm.cdf(abs(dm_stat)))
    return float(dm_stat), float(p_value)


# ─────────────────────────── Oracle Analysis Core ───────────────────────────

def run_oracle_analysis():
    """Run the full oracle gating analysis."""
    t0 = time.time()

    # ── Load data ──
    log.info("Loading quad-domain panel...")
    if QUAD_PANEL.exists():
        df = pd.read_parquet(QUAD_PANEL)
    else:
        raise FileNotFoundError(f"Missing {QUAD_PANEL}")
    log.info(f"  Shape: {df.shape}, countries: {df['iso3'].nunique()}, "
             f"years: {df['year'].min()}-{df['year'].max()}")

    # ── Classify features ──
    sectors = classify_features(df.columns.tolist())
    for s, cols in sectors.items():
        log.info(f"  {s}: {len(cols)} features")

    configs = build_domain_configs(sectors)
    config_names = list(configs.keys())
    log.info(f"  Configurations: {config_names}")

    all_oracle_rows = []
    all_summary_rows = []

    for h in HORIZONS:
        log.info(f"\n{'#'*70}")
        log.info(f"  HORIZON h={h}")
        log.info(f"{'#'*70}")

        # Build target
        df_work = df.sort_values(["iso3", "year"]).reset_index(drop=True)
        target = make_target(df_work, h)
        df_work["target"] = target

        # Filter to valid rows
        valid_mask = df_work["target"].notna() & np.isfinite(df_work["target"])
        df_valid = df_work[valid_mask].reset_index(drop=True)

        # Outlier bounds (1st to 99th percentile)
        q01 = df_valid["target"].quantile(0.01)
        q99 = df_valid["target"].quantile(0.99)
        df_valid = df_valid[(df_valid["target"] >= q01) & (df_valid["target"] <= q99)].reset_index(drop=True)

        years = df_valid["year"].values
        y = df_valid["target"].values.astype(np.float32)

        log.info(f"  Valid rows: {len(df_valid)}, year range: {years.min()}-{years.max()}")

        shift = max(0, h - 5)
        anchor_end = 2022 - shift

        for fold in range(N_FOLDS):
            fold_test_end = anchor_end - fold * TEST_WINDOW
            fold_test_start = fold_test_end - TEST_WINDOW + 1
            fold_train_end = fold_test_start - 1

            if fold_train_end < 1970:
                continue

            train_mask = years <= fold_train_end
            test_mask = (years >= fold_test_start) & (years <= fold_test_end)

            n_train = train_mask.sum()
            n_test = test_mask.sum()

            if n_train < MIN_TRAIN_ROWS or n_test < 10:
                continue

            log.info(f"  Fold {fold}: train<={fold_train_end} ({n_train}), "
                     f"test {fold_test_start}-{fold_test_end} ({n_test})")

            y_test = y[test_mask]
            test_iso3 = df_valid.loc[test_mask, "iso3"].values
            test_years = df_valid.loc[test_mask, "year"].values

            preds_by_config = {}
            errors_by_config = {}

            for cfg_name, cfg_cols in configs.items():
                avail_cols = [c for c in cfg_cols if c in df_valid.columns and c != "target"]
                if len(avail_cols) < 3:
                    continue

                X_all = df_valid[avail_cols].astype(np.float32).replace([np.inf, -np.inf], np.nan).values

                pred_ensemble = train_predict_models(X_all, y, train_mask, test_mask, seed=fold)

                preds_by_config[cfg_name] = pred_ensemble
                errors_by_config[cfg_name] = np.abs(y_test - pred_ensemble)

                mae = float(np.mean(np.abs(y_test - pred_ensemble)))
                log.info(f"    {cfg_name:15s}: MAE={mae:.5f} ({len(avail_cols)} features)")

            if len(preds_by_config) < 2:
                continue

            valid_configs = list(preds_by_config.keys())
            error_matrix = np.column_stack([errors_by_config[c] for c in valid_configs])
            pred_matrix = np.column_stack([preds_by_config[c] for c in valid_configs])

            oracle_idx = np.argmin(error_matrix, axis=1)
            oracle_pred = np.array([pred_matrix[i, oracle_idx[i]] for i in range(len(oracle_idx))])
            oracle_errors = np.abs(y_test - oracle_pred)
            oracle_config_names = np.array([valid_configs[idx] for idx in oracle_idx])

            uniform_pred = np.mean(pred_matrix, axis=1)
            uniform_errors = np.abs(y_test - uniform_pred)

            eco_only_errors = errors_by_config.get("eco_only", uniform_errors)

            for i in range(n_test):
                row = {
                    "horizon": h,
                    "fold": fold,
                    "iso3": test_iso3[i],
                    "year": int(test_years[i]),
                    "y_true": float(y_test[i]),
                    "oracle_config": oracle_config_names[i],
                    "oracle_error": float(oracle_errors[i]),
                    "eco_only_error": float(eco_only_errors[i]),
                    "uniform_error": float(uniform_errors[i]),
                    "oracle_is_not_eco": oracle_config_names[i] != "eco_only",
                    "oracle_improvement_over_eco": float(eco_only_errors[i] - oracle_errors[i]),
                }
                for cfg in valid_configs:
                    row[f"error_{cfg}"] = float(errors_by_config[cfg][i])
                all_oracle_rows.append(row)

            oracle_mae = float(np.mean(oracle_errors))
            uniform_mae = float(np.mean(uniform_errors))
            eco_only_mae = float(np.mean(eco_only_errors))

            pct_oracle_not_eco = float(np.mean(oracle_config_names != "eco_only") * 100)

            dm_oracle_vs_uniform, p_oracle_vs_uniform = diebold_mariano(uniform_errors, oracle_errors)
            dm_oracle_vs_eco, p_oracle_vs_eco = diebold_mariano(eco_only_errors, oracle_errors)

            config_counts = {cfg: int(np.sum(oracle_config_names == cfg)) for cfg in valid_configs}

            summary = {
                "horizon": h,
                "fold": fold,
                "n_test": n_test,
                "eco_only_mae": round(eco_only_mae, 5),
                "uniform_mae": round(uniform_mae, 5),
                "oracle_mae": round(oracle_mae, 5),
                "gap_uniform_to_oracle_pct": round((uniform_mae - oracle_mae) / uniform_mae * 100, 2),
                "gap_eco_to_oracle_pct": round((eco_only_mae - oracle_mae) / eco_only_mae * 100, 2),
                "pct_oracle_chooses_non_eco": round(pct_oracle_not_eco, 1),
                "dm_oracle_vs_uniform_stat": round(dm_oracle_vs_uniform, 4),
                "dm_oracle_vs_uniform_p": round(p_oracle_vs_uniform, 6),
                "dm_oracle_vs_eco_stat": round(dm_oracle_vs_eco, 4),
                "dm_oracle_vs_eco_p": round(p_oracle_vs_eco, 6),
                **{f"oracle_picks_{cfg}": config_counts.get(cfg, 0) for cfg in valid_configs},
            }
            all_summary_rows.append(summary)

            log.info(f"\n    -- FOLD {fold} ORACLE ANALYSIS --")
            log.info(f"    Eco-Only MAE:    {eco_only_mae:.5f}")
            log.info(f"    Uniform MAE:     {uniform_mae:.5f}")
            log.info(f"    Oracle MAE:      {oracle_mae:.5f} ({pct_oracle_not_eco:.0f}% non-eco)")
            log.info(f"    Gap (eco->oracle): {(eco_only_mae - oracle_mae)/eco_only_mae*100:.1f}%")
            log.info(f"    DM (oracle vs eco): stat={dm_oracle_vs_eco:.3f}, p={p_oracle_vs_eco:.4f}")

    df_oracle = pd.DataFrame(all_oracle_rows)
    df_summary = pd.DataFrame(all_summary_rows)

    headline_rows = []
    for h in HORIZONS:
        h_data = df_summary[df_summary["horizon"] == h]
        if h_data.empty:
            continue
        headline_rows.append({
            "horizon": h,
            "n_folds": len(h_data),
            "eco_only_mae_mean": round(h_data["eco_only_mae"].mean(), 5),
            "uniform_mae_mean": round(h_data["uniform_mae"].mean(), 5),
            "oracle_mae_mean": round(h_data["oracle_mae"].mean(), 5),
            "avg_gap_eco_to_oracle_pct": round(h_data["gap_eco_to_oracle_pct"].mean(), 2),
            "avg_gap_uniform_to_oracle_pct": round(h_data["gap_uniform_to_oracle_pct"].mean(), 2),
            "avg_pct_non_eco": round(h_data["pct_oracle_chooses_non_eco"].mean(), 1),
        })
    df_headline = pd.DataFrame(headline_rows)

    df_oracle.to_csv(OUT_DIR / "oracle_per_row.csv", index=False)
    df_summary.to_csv(OUT_DIR / "oracle_fold_summary.csv", index=False)
    df_headline.to_csv(OUT_DIR / "oracle_headline.csv", index=False)

    results_json = {
        "experiment": "oracle_gating_analysis",
        "purpose": "Compute theoretical upper bound for LLM-gated cross-domain forecasting",
        "horizons": HORIZONS,
        "n_folds": N_FOLDS,
        "runtime_seconds": round(time.time() - t0, 1),
        "headline": df_headline.to_dict(orient="records"),
    }

    if not df_headline.empty:
        avg_gap = df_headline["avg_gap_eco_to_oracle_pct"].mean()
        avg_non_eco = df_headline["avg_pct_non_eco"].mean()

        if avg_gap > 10:
            verdict = "STRONG_GO"
            msg = (f"Oracle gating improves over eco-only by {avg_gap:.1f}% on average. "
                   f"The oracle chooses non-economic configs {avg_non_eco:.0f}% of the time. "
                   f"There is substantial room for an LLM gate to capture this signal.")
        else:
            verdict = "MODERATE_GO"
            msg = f"Oracle gating improves by {avg_gap:.1f}%."

        results_json["verdict"] = {
            "recommendation": verdict,
            "avg_gap_pct": round(avg_gap, 2),
            "avg_pct_non_eco_oracle": round(avg_non_eco, 1),
            "message": msg,
        }

    with open(OUT_DIR / "oracle_analysis_results.json", "w") as f:
        json.dump(results_json, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"  ORACLE GATING ANALYSIS -- THEORETICAL UPPER BOUND")
    print(f"  Runtime: {elapsed:.0f}s")
    print(f"{'='*80}")

    if not df_headline.empty:
        print(f"\n  HEADLINE TABLE (averaged across {N_FOLDS} walk-forward folds):")
        print(f"  {'Horizon':<10} {'Eco-Only MAE':<15} {'Uniform MAE':<15} "
              f"{'Oracle MAE':<15} {'Gap (eco->oracle)':<20} {'% Non-Eco':<12}")
        print(f"  {'-'*85}")
        for _, row in df_headline.iterrows():
            print(f"  h={int(row['horizon']):<7} {row['eco_only_mae_mean']:<15.5f} "
                  f"{row['uniform_mae_mean']:<15.5f} {row['oracle_mae_mean']:<15.5f} "
                  f"{row['avg_gap_eco_to_oracle_pct']:<20.1f}% "
                  f"{row['avg_pct_non_eco']:<12.1f}%")

    if "verdict" in results_json:
        v = results_json["verdict"]
        print(f"\n  VERDICT: {v['recommendation']}")
        print(f"  {v['message']}")

    print(f"\n  Results saved to: {OUT_DIR}")


if __name__ == "__main__":
    run_oracle_analysis()
