"""
Cross-Domain Multi-Target Synergy Engine
-----------------------------------------
Evaluates cross-domain predictive synergy across diverse Economic & Political targets:

Economic Targets:
1. Inflation Rate (inflation_rate_1y_fwd)
2. Government Debt / GDP (gov_debt_gdp_1y_fwd)
3. Unemployment Rate (unemployment_rate_1y_fwd)
4. Banking Crisis Flag (banking_crisis_1y_fwd)
5. Sovereign Debt Crisis Flag (sov_debt_crisis_1y_fwd)

Political Targets:
1. Protest Unrest Count (protest_unrest_annual_sum_1y_fwd)
2. News Sentiment Tone (news_tone_annual_mean_1y_fwd)
3. Sanctions & Coercion Count (sanctions_coercion_annual_sum_1y_fwd)
4. Conflict Intensity Pct (conflict_intensity_annual_mean_1y_fwd)
"""

import os
import logging
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score
from scipy.stats import norm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

ANNUAL_PANEL_PATH = "data/joint_annual_eco_political_panel.parquet"
OUTPUT_MULTI_TARGET_SUMMARY = "data/multi_target_synergy_results.csv"
OUTPUT_MULTI_TARGET_IMPORTANCE = "data/multi_target_feature_importances.csv"


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


def build_additional_forward_targets(df):
    logging.info("Constructing forward targets for Inflation, Debt, Unemployment, Crises, Protests, Tone, and Sanctions...")
    df = df.sort_values(["iso3", "year"]).reset_index(drop=True)

    # Economic forward targets (1-year ahead)
    df["inflation_rate_1y_fwd"] = df.groupby("iso3")["inflation_rate"].shift(-1)
    df["gov_debt_gdp_1y_fwd"] = df.groupby("iso3")["gov_debt_gdp"].shift(-1)
    df["unemployment_rate_1y_fwd"] = df.groupby("iso3")["unemployment_rate"].shift(-1)
    df["banking_crisis_1y_fwd"] = df.groupby("iso3")["banking_crisis"].shift(-1)
    df["sov_debt_crisis_1y_fwd"] = df.groupby("iso3")["sov_debt_crisis"].shift(-1)

    # Political forward targets (1-year ahead)
    df["protest_unrest_annual_sum_1y_fwd"] = df.groupby("iso3")["protest_unrest_annual_sum"].shift(-1)
    df["news_tone_annual_mean_1y_fwd"] = df.groupby("iso3")["news_tone_annual_mean"].shift(-1)
    df["sanctions_coercion_annual_sum_1y_fwd"] = df.groupby("iso3")["sanctions_coercion_annual_sum"].shift(-1)
    df["conflict_intensity_annual_mean_1y_fwd"] = df.groupby("iso3")["conflict_intensity_annual_mean"].shift(-1)

    # Lags
    eco_base = ["gdp_pc", "inflation_rate", "gov_debt_gdp", "gov_deficit_gdp", "unemployment_rate", "current_account_gdp", "investment_gdp", "exports_gdp"]
    pol_base = ["goldstein_annual_mean", "news_tone_annual_mean", "material_conflict_annual_sum", "protest_unrest_annual_sum", "sanctions_coercion_annual_sum", "conflict_intensity_annual_mean"]

    eco_base = [c for c in eco_base if c in df.columns]
    pol_base = [c for c in pol_base if c in df.columns]

    for col in eco_base + pol_base:
        df[f"{col}_lag1"] = df.groupby("iso3")[col].shift(1)

    return df


def train_eval_walk_forward_multi(df, target_col, is_eco_target, eco_features, pol_features, is_binary=False, n_folds=5):
    clean_df = df.dropna(subset=[target_col]).copy()
    clean_df = clean_df[np.isfinite(clean_df[target_col])].copy()

    if not is_binary:
        q_low = clean_df[target_col].quantile(0.01)
        q_high = clean_df[target_col].quantile(0.99)
        clean_df = clean_df[(clean_df[target_col] >= q_low) & (clean_df[target_col] <= q_high)].sort_values("year").reset_index(drop=True)

    if len(clean_df) < 200:
        return None, None

    base_eco_features = [f for f in eco_features if f in clean_df.columns and f != target_col]
    base_pol_features = [f for f in pol_features if f in clean_df.columns and f != target_col]

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

        if len(y_val) == 0 or len(np.unique(y_tr)) < 2:
            continue

        if is_binary:
            l_b = lgb.LGBMClassifier(n_estimators=60, learning_rate=0.03, max_depth=4, n_jobs=-1, random_state=42, verbose=-1).fit(X_b_tr, y_tr)
            pred_b = l_b.predict_proba(X_b_val)[:, 1]

            l_h = lgb.LGBMClassifier(n_estimators=60, learning_rate=0.03, max_depth=4, n_jobs=-1, random_state=42, verbose=-1).fit(X_h_tr, y_tr)
            pred_h = l_h.predict_proba(X_h_val)[:, 1]
        else:
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

    if is_binary:
        auc_base = roc_auc_score(oof_y, oof_b) if len(np.unique(oof_y)) > 1 else 0.5
        auc_hybr = roc_auc_score(oof_y, oof_h) if len(np.unique(oof_y)) > 1 else 0.5
        pct_imprv = ((auc_hybr - auc_base) / (auc_base + 1e-8)) * 100.0
        dm_stat, p_val = diebold_mariano_test(abs(oof_y - oof_b), abs(oof_y - oof_h))

        summary = {
            "target_variable": target_col,
            "target_domain": "Economic Crises" if is_eco_target else "Political Risk",
            "metric_name": "ROC-AUC",
            "baseline_val": round(float(auc_base), 4),
            "hybrid_val": round(float(auc_hybr), 4),
            "improvement_pct": round(float(pct_imprv), 2),
            "p_value": round(float(p_val), 4),
            "significant_synergy": bool(p_val < 0.05 and auc_hybr > auc_base)
        }
    else:
        rmse_base = np.sqrt(mean_squared_error(oof_y, oof_b))
        rmse_hybr = np.sqrt(mean_squared_error(oof_y, oof_h))
        r2_base = r2_score(oof_y, oof_b)
        r2_hybr = r2_score(oof_y, oof_h)

        pct_imprv = ((rmse_base - rmse_hybr) / (rmse_base + 1e-8)) * 100.0
        dm_stat, p_val = diebold_mariano_test(abs(oof_y - oof_b), abs(oof_y - oof_h))

        summary = {
            "target_variable": target_col,
            "target_domain": "Economic Metrics" if is_eco_target else "Political Metrics",
            "metric_name": "RMSE (R2)",
            "baseline_val": round(float(rmse_base), 4),
            "hybrid_val": round(float(rmse_hybr), 4),
            "improvement_pct": round(float(pct_imprv), 2),
            "p_value": round(float(p_val), 4),
            "significant_synergy": bool(p_val < 0.05 and rmse_hybr < rmse_base)
        }

    imp_df = pd.DataFrame({
        "target_variable": target_col,
        "feature_name": hybrid_cols,
        "importance": feature_importance_accum / n_folds
    }).sort_values("importance", ascending=False)

    return summary, imp_df


def main():
    df = pd.read_parquet(ANNUAL_PANEL_PATH)
    df = build_additional_forward_targets(df)

    eco_cols = [
        "gdp_pc", "gdp_pc_lag1", "inflation_rate", "inflation_rate_lag1", "gov_debt_gdp", "gov_debt_gdp_lag1",
        "gov_deficit_gdp", "unemployment_rate", "banking_crisis", "currency_crisis", "sov_debt_crisis",
        "current_account_gdp", "investment_gdp", "exports_gdp"
    ]

    pol_cols = [
        "goldstein_annual_mean", "goldstein_annual_mean_lag1", "news_tone_annual_mean", "news_tone_annual_mean_lag1",
        "material_conflict_annual_sum", "material_conflict_annual_sum_lag1", "protest_unrest_annual_sum",
        "sanctions_coercion_annual_sum", "conflict_intensity_annual_mean", "stability_momentum_annual_mean"
    ]

    target_configs = [
        # (target_col, is_eco_target, is_binary)
        ("inflation_rate_1y_fwd", True, False),
        ("gov_debt_gdp_1y_fwd", True, False),
        ("unemployment_rate_1y_fwd", True, False),
        ("banking_crisis_1y_fwd", True, True),
        ("sov_debt_crisis_1y_fwd", True, True),
        ("protest_unrest_annual_sum_1y_fwd", False, False),
        ("news_tone_annual_mean_1y_fwd", False, False),
        ("sanctions_coercion_annual_sum_1y_fwd", False, False),
        ("conflict_intensity_annual_mean_1y_fwd", False, False)
    ]

    summaries = []
    all_importances = []

    for t_col, is_eco, is_bin in target_configs:
        if t_col not in df.columns:
            continue
        logging.info(f"Evaluating synergy for target: {t_col}...")
        sum_res, imp_res = train_eval_walk_forward_multi(df, t_col, is_eco, eco_cols, pol_cols, is_binary=is_bin, n_folds=5)
        if sum_res is not None:
            summaries.append(sum_res)
            all_importances.append(imp_res)
            logging.info(f"-> Target: {t_col} | Baseline: {sum_res['baseline_val']} -> Hybrid: {sum_res['hybrid_val']} (Imprv: {sum_res['improvement_pct']}%)")

    summary_df = pd.DataFrame(summaries)
    importance_df = pd.concat(all_importances, ignore_index=True)

    summary_df.to_csv(OUTPUT_MULTI_TARGET_SUMMARY, index=False)
    importance_df.to_csv(OUTPUT_MULTI_TARGET_IMPORTANCE, index=False)

    print("\n" + "=" * 90)
    print(" DIVERSE MULTI-TARGET CROSS-DOMAIN SYNERGY TOURNAMENT RESULTS")
    print("=" * 90)
    print(summary_df.to_string(index=False))
    print("=" * 90)


if __name__ == "__main__":
    main()
