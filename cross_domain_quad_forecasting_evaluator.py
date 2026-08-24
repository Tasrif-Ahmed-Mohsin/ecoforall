"""
Quad-Domain Cross-Forecasting Power & Synergy Evaluator (LightGBM Pipeline)
---------------------------------------------------------------------------
Evaluates cross-domain forecasting improvement across Economy, Politics, Environment, and Human/Society
using 5-Fold Walk-Forward Cross Validation, Diebold-Mariano tests, and LightGBM feature importances.
"""

import os
import logging
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import norm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

QUAD_PANEL_PATH = "data/quad_domain_annual_panel.parquet"
OUTPUT_SUMMARY_PATH = "data/quad_domain_forecasting_tournament_results.csv"
OUTPUT_IMPORTANCE_PATH = "data/quad_domain_feature_importances.csv"


def diebold_mariano_test(e1, e2, h=1):
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


def train_eval_walk_forward_quad(df, target_col, feature_sets, n_folds=5):
    clean_df = df.dropna(subset=[target_col]).copy()
    clean_df = clean_df[np.isfinite(clean_df[target_col])].copy()

    # Outlier truncation for stability
    q_low = clean_df[target_col].quantile(0.01)
    q_high = clean_df[target_col].quantile(0.99)
    clean_df = clean_df[(clean_df[target_col] >= q_low) & (clean_df[target_col] <= q_high)].sort_values("year").reset_index(drop=True)

    if len(clean_df) < 150:
        logging.warning(f"Insufficient clean samples for {target_col}: {len(clean_df)}")
        return None, None

    y = clean_df[target_col].values
    n_total = len(clean_df)
    min_train_pct = 0.50
    step_pct = (1.0 - min_train_pct) / n_folds

    oof_predictions = {cfg: [] for cfg in feature_sets.keys()}
    oof_y_true = []

    prepared_features = {}
    for cfg_name, cols in feature_sets.items():
        valid_cols = [c for c in cols if c in clean_df.columns and c != target_col]
        clean_df[valid_cols] = clean_df[valid_cols].fillna(0.0)
        prepared_features[cfg_name] = (valid_cols, clean_df[valid_cols].values)

    feature_importances_accum = {cfg: np.zeros(prepared_features[cfg][1].shape[1]) for cfg in feature_sets.keys()}

    for fold in range(1, n_folds + 1):
        train_end_pct = min_train_pct + (fold - 1) * step_pct
        test_end_pct = min_train_pct + fold * step_pct

        train_idx = int(n_total * train_end_pct)
        test_idx = int(n_total * test_end_pct)

        y_tr = y[:train_idx]
        y_val = y[train_idx:test_idx]

        if len(y_val) == 0:
            continue

        if fold == 1 or len(oof_y_true) < test_idx - int(n_total * min_train_pct):
            oof_y_true.extend(y_val)

        for cfg_name, (cols, X_mat) in prepared_features.items():
            if len(cols) == 0:
                continue
            X_tr = X_mat[:train_idx]
            X_val = X_mat[train_idx:test_idx]

            model = lgb.LGBMRegressor(
                n_estimators=100, learning_rate=0.03, max_depth=5, num_leaves=15, n_jobs=-1, random_state=42, verbose=-1
            ).fit(X_tr, y_tr)

            pred_val = model.predict(X_val)
            oof_predictions[cfg_name].extend(pred_val)
            feature_importances_accum[cfg_name] += model.feature_importances_

    oof_y = np.array(oof_y_true)

    tournament_rows = []
    base_rmse = None
    base_err = None

    for cfg_name in ["Domain_Only", "Eco+Pol", "Eco+Pol+Env", "Full_Quad_Domain"]:
        if cfg_name not in oof_predictions or len(oof_predictions[cfg_name]) != len(oof_y):
            continue

        oof_pred = np.array(oof_predictions[cfg_name])
        rmse = float(np.sqrt(mean_squared_error(oof_y, oof_pred)))
        mae = float(mean_absolute_error(oof_y, oof_pred))
        r2 = float(r2_score(oof_y, oof_pred))
        err = abs(oof_y - oof_pred)

        if cfg_name == "Domain_Only":
            base_rmse = rmse
            base_err = err
            imprv_pct = 0.0
            dm_stat, p_val = 0.0, 1.0
        else:
            imprv_pct = float(((base_rmse - rmse) / (base_rmse + 1e-8)) * 100.0)
            dm_stat, p_val = diebold_mariano_test(base_err, err)

        tournament_rows.append({
            "target_variable": target_col,
            "configuration": cfg_name,
            "oof_rmse": round(rmse, 4),
            "oof_mae": round(mae, 4),
            "oof_r2": round(r2, 4),
            "rmse_improvement_pct": round(imprv_pct, 2),
            "dm_statistic": round(dm_stat, 4),
            "p_value": round(p_val, 4),
            "is_stat_significant": p_val < 0.05
        })

    importances_rows = []
    if "Full_Quad_Domain" in prepared_features:
        cols, _ = prepared_features["Full_Quad_Domain"]
        imp_scores = feature_importances_accum["Full_Quad_Domain"] / n_folds
        top_idx = np.argsort(imp_scores)[::-1][:20]

        for rank, idx in enumerate(top_idx, 1):
            importances_rows.append({
                "target_variable": target_col,
                "feature_name": cols[idx],
                "importance_score": round(float(imp_scores[idx]), 4),
                "rank": rank
            })

    return tournament_rows, importances_rows


def main():
    if not os.path.exists(QUAD_PANEL_PATH):
        raise FileNotFoundError(f"Missing {QUAD_PANEL_PATH}")

    df = pd.read_parquet(QUAD_PANEL_PATH)
    logging.info(f"Loaded Quad-Domain Annual Panel shape: {df.shape}")

    num_cols = [c for c in df.columns if c not in ["iso3", "year", "timestamp"]]

    eco_cols = [c for c in num_cols if any(k in c for k in ["gdp", "inflation", "debt", "unemployment", "crisis", "eco"])]
    pol_cols = [c for c in num_cols if any(k in c for k in ["goldstein", "conflict", "protest", "sanction", "stability", "news", "pol"])]
    env_cols = [c for c in num_cols if any(k in c for k in ["co2", "temp", "forest", "flood", "drought", "wildfire", "storm", "renewable", "disaster", "env"])]
    human_cols = [c for c in num_cols if any(k in c for k in ["psychology", "society", "trust", "fear", "cohesion", "education", "urbanization", "population"])]

    targets = [
        "gdp_pc_growth_1y_fwd",
        "stability_momentum_annual_mean",
        "co2_emissions_per_capita",
        "psychology_trust",
        "psychology_fear"
    ]
    targets = [t for t in targets if t in df.columns]

    all_tournament_rows = []
    all_importances_rows = []

    for target in targets:
        logging.info(f"Running Synergy Tournament for Target: {target}...")

        domain_only = eco_cols if target in eco_cols else (pol_cols if target in pol_cols else (env_cols if target in env_cols else human_cols))

        feature_sets = {
            "Domain_Only": domain_only,
            "Eco+Pol": list(set(eco_cols + pol_cols)),
            "Eco+Pol+Env": list(set(eco_cols + pol_cols + env_cols)),
            "Full_Quad_Domain": list(set(eco_cols + pol_cols + env_cols + human_cols))
        }

        t_rows, i_rows = train_eval_walk_forward_quad(df, target, feature_sets)
        if t_rows:
            all_tournament_rows.extend(t_rows)
        if i_rows:
            all_importances_rows.extend(i_rows)

    df_tournament = pd.DataFrame(all_tournament_rows)
    df_importances = pd.DataFrame(all_importances_rows)

    df_tournament.to_csv(OUTPUT_SUMMARY_PATH, index=False)
    df_importances.to_csv(OUTPUT_IMPORTANCE_PATH, index=False)

    logging.info(f"\nSaved Tournament Results to {OUTPUT_SUMMARY_PATH}")
    logging.info(f"\nSaved Feature Importances to {OUTPUT_IMPORTANCE_PATH}")
    print(df_tournament.to_string(index=False))


if __name__ == "__main__":
    main()
