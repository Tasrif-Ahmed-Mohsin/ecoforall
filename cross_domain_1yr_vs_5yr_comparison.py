"""
Cross-Domain 1-Year vs 5-Year Horizon Comparison Engine
--------------------------------------------------------
Compares predictive performance, cross-domain feature importances,
and statistical synergy between 1-Year (h=1) and 5-Year (h=5) forecast horizons.
"""

import os
import logging
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import norm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

ANNUAL_PANEL_PATH = "data/joint_annual_eco_political_panel.parquet"
OUTPUT_COMPARISON_PATH = "data/1yr_vs_5yr_comparison_results.csv"
OUTPUT_IMPORTANCE_PATH = "data/1yr_vs_5yr_feature_importances.csv"


def diebold_mariano_test(e1, e2, h=1):
    """
    Diebold-Mariano test for predictive accuracy equality.
    """
    d = e1**2 - e2**2
    n = len(d)
    if n < 5:
        return 0.0, 1.0

    mean_d = np.mean(d)
    autocov = np.var(d)
    for lag in range(1, h):
        if len(d[lag:]) > 0:
            gamma = np.cov(d[lag:], d[:-lag])[0, 1]
            autocov += 2 * (1 - lag / h) * gamma

    var_d = max(1e-8, autocov / n)
    dm_stat = mean_d / np.sqrt(var_d)
    p_value = 2 * (1 - norm.cdf(abs(dm_stat)))
    return float(dm_stat), float(p_value)


def build_multi_horizon_targets_and_lags(df):
    logging.info("Preparing 1-Year and 5-Year targets and lagged features...")
    df = df.sort_values(["iso3", "year"]).reset_index(drop=True)

    # 5-year political forward targets
    df["material_conflict_annual_sum_5y_fwd"] = df.groupby("iso3")["material_conflict_annual_sum"].shift(-5)
    df["goldstein_annual_mean_5y_fwd"] = df.groupby("iso3")["goldstein_annual_mean"].shift(-5)

    eco_base = ["gdp_pc", "inflation_rate", "gov_debt_gdp", "gov_deficit_gdp", "unemployment_rate", "current_account_gdp", "investment_gdp", "exports_gdp"]
    pol_base = ["goldstein_annual_mean", "news_tone_annual_mean", "material_conflict_annual_sum", "protest_unrest_annual_sum", "conflict_intensity_annual_mean"]

    eco_base = [c for c in eco_base if c in df.columns]
    pol_base = [c for c in pol_base if c in df.columns]

    for col in eco_base + pol_base:
        df[f"{col}_lag1"] = df.groupby("iso3")[col].shift(1)
        df[f"{col}_lag5"] = df.groupby("iso3")[col].shift(5)

    return df


def train_eval_walk_forward(df, target_col, horizon, eco_features, pol_features, n_folds=5):
    """
    Runs 5-Fold Walk-Forward CV for target column at given horizon.
    """
    clean_df = df.dropna(subset=[target_col]).copy()
    clean_df = clean_df[np.isfinite(clean_df[target_col])].copy()

    q_low = clean_df[target_col].quantile(0.01)
    q_high = clean_df[target_col].quantile(0.99)
    clean_df = clean_df[(clean_df[target_col] >= q_low) & (clean_df[target_col] <= q_high)].sort_values("year").reset_index(drop=True)

    if len(clean_df) < 200:
        return None, None

    is_eco_target = "gdp" in target_col.lower()

    base_eco_features = [f for f in eco_features if f in clean_df.columns]
    base_pol_features = [f for f in pol_features if f in clean_df.columns]

    baseline_cols = base_eco_features if is_eco_target else base_pol_features
    all_features = sorted(list(set(base_eco_features + base_pol_features)))
    hybrid_cols = [c for c in all_features if c != target_col]

    clean_df[hybrid_cols] = clean_df[hybrid_cols].fillna(0.0)

    y = clean_df[target_col].values
    X_base = clean_df[baseline_cols].values
    X_hybr = clean_df[hybrid_cols].values

    n_total = len(clean_df)
    min_train_pct = 0.50
    step_pct = (1.0 - min_train_pct) / n_folds

    oof_y_true = []
    oof_pred_base = []
    oof_pred_hybr = []

    feature_importance_accum = np.zeros(X_hybr.shape[1])

    for fold in range(1, n_folds + 1):
        train_end_pct = min_train_pct + (fold - 1) * step_pct
        test_end_pct = min_train_pct + fold * step_pct

        train_idx = int(n_total * train_end_pct)
        test_idx = int(n_total * test_end_pct)

        X_b_tr, X_h_tr, y_tr = X_base[:train_idx], X_hybr[:train_idx], y[:train_idx]
        X_b_val, X_h_val, y_val = X_base[train_idx:test_idx], X_hybr[train_idx:test_idx], y[train_idx:test_idx]

        if len(y_val) == 0:
            continue

        l_b = lgb.LGBMRegressor(n_estimators=80, learning_rate=0.03, max_depth=5, num_leaves=15, n_jobs=-1, random_state=42, verbose=-1).fit(X_b_tr, y_tr)
        pred_b = l_b.predict(X_b_val)

        l_h = lgb.LGBMRegressor(n_estimators=80, learning_rate=0.03, max_depth=5, num_leaves=15, n_jobs=-1, random_state=42, verbose=-1).fit(X_h_tr, y_tr)
        pred_h = l_h.predict(X_h_val)

        oof_y_true.extend(y_val)
        oof_pred_base.extend(pred_b)
        oof_pred_hybr.extend(pred_h)

        feature_importance_accum += l_h.feature_importances_

    oof_y = np.array(oof_y_true)
    oof_b = np.array(oof_pred_base)
    oof_h = np.array(oof_pred_hybr)

    rmse_base = np.sqrt(mean_squared_error(oof_y, oof_b))
    rmse_hybr = np.sqrt(mean_squared_error(oof_y, oof_h))

    mae_base = mean_absolute_error(oof_y, oof_b)
    mae_hybr = mean_absolute_error(oof_y, oof_h)

    r2_base = r2_score(oof_y, oof_b)
    r2_hybr = r2_score(oof_y, oof_h)

    pct_rmse_imprv = ((rmse_base - rmse_hybr) / (rmse_base + 1e-8)) * 100.0

    err_base = abs(oof_y - oof_b)
    err_hybr = abs(oof_y - oof_h)
    dm_stat, p_val = diebold_mariano_test(err_base, err_hybr, h=horizon)

    summary = {
        "target_variable": target_col,
        "horizon_years": f"{horizon} Year",
        "target_domain": f"{'Economic Growth' if is_eco_target else 'Political Risk'} ({horizon}Y)",
        "baseline_type": "Economic Only" if is_eco_target else "Political Only",
        "hybrid_type": f"{horizon}Y Eco + Pol Synergy",
        "n_samples": len(oof_y),
        "rmse_baseline": round(float(rmse_base), 5),
        "rmse_hybrid": round(float(rmse_hybr), 5),
        "rmse_improvement_pct": round(float(pct_rmse_imprv), 2),
        "mae_baseline": round(float(mae_base), 5),
        "mae_hybrid": round(float(mae_hybr), 5),
        "r2_baseline": round(float(r2_base), 4),
        "r2_hybrid": round(float(r2_hybr), 4),
        "dm_stat": round(float(dm_stat), 4),
        "p_value": round(float(p_val), 4),
        "significant_synergy": bool(p_val < 0.05 and rmse_hybr < rmse_base)
    }

    imp_df = pd.DataFrame({
        "horizon_years": f"{horizon} Year",
        "target_variable": target_col,
        "feature_name": hybrid_cols,
        "importance": feature_importance_accum / n_folds
    }).sort_values("importance", ascending=False)

    return summary, imp_df


def main():
    df = pd.read_parquet(ANNUAL_PANEL_PATH)
    df = build_multi_horizon_targets_and_lags(df)

    eco_cols = [
        "gdp_pc_lag1", "gdp_pc_lag5", "inflation_rate", "inflation_rate_lag1", "gov_debt_gdp", "gov_debt_gdp_lag1",
        "gov_deficit_gdp", "unemployment_rate", "banking_crisis", "currency_crisis", "sov_debt_crisis",
        "current_account_gdp", "investment_gdp", "exports_gdp"
    ]

    pol_cols = [
        "goldstein_annual_mean", "goldstein_annual_mean_lag1", "goldstein_annual_mean_lag5",
        "news_tone_annual_mean", "news_tone_annual_mean_lag1", "news_tone_annual_mean_lag5",
        "material_conflict_annual_sum", "material_conflict_annual_sum_lag1", "material_conflict_annual_sum_lag5",
        "protest_unrest_annual_sum", "conflict_intensity_annual_mean", "stability_momentum_annual_mean"
    ]

    tasks = [
        ("gdp_pc_growth_1y_fwd", 1),
        ("gdp_pc_growth_5y_fwd", 5),
        ("goldstein_annual_mean_1y_fwd", 1),
        ("goldstein_annual_mean_5y_fwd", 5),
        ("material_conflict_annual_sum_1y_fwd", 1),
        ("material_conflict_annual_sum_5y_fwd", 5)
    ]

    summaries = []
    all_importances = []

    for target_col, h in tasks:
        if target_col not in df.columns:
            continue
        logging.info(f"Evaluating {h}-Year Horizon for target: {target_col}...")
        sum_res, imp_res = train_eval_walk_forward(df, target_col, h, eco_cols, pol_cols, n_folds=5)
        if sum_res is not None:
            summaries.append(sum_res)
            all_importances.append(imp_res)
            logging.info(f"-> Target: {target_col} ({h}Y) | Baseline RMSE: {sum_res['rmse_baseline']} -> Hybrid RMSE: {sum_res['rmse_hybrid']} (Imprv: {sum_res['rmse_improvement_pct']}%) | R2: {sum_res['r2_baseline']} -> {sum_res['r2_hybrid']}")

    summary_df = pd.DataFrame(summaries)
    importance_df = pd.concat(all_importances, ignore_index=True)

    summary_df.to_csv(OUTPUT_COMPARISON_PATH, index=False)
    importance_df.to_csv(OUTPUT_IMPORTANCE_PATH, index=False)

    print("\n" + "=" * 90)
    print(" 1-YEAR vs 5-YEAR HORIZON CROSS-DOMAIN COMPARISON TOURNAMENT")
    print("=" * 90)
    print(summary_df[["horizon_years", "target_variable", "rmse_baseline", "rmse_hybrid", "rmse_improvement_pct", "r2_baseline", "r2_hybrid", "p_value"]].to_string(index=False))
    print("=" * 90)


if __name__ == "__main__":
    main()
