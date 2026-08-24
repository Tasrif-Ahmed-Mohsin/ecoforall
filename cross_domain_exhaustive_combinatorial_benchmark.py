"""
Exhaustive Multidimensional Sector Combination Benchmark Engine
----------------------------------------------------------------
Evaluates all 15 non-empty sector combinations across 4 domains:
1. Sector 1 (Economy)
2. Sector 2 (Politics)
3. Sector 3 (Environment)
4. Sector 4 (Human/Society)

Runs 5-Fold Walk-Forward Cross Validation, Diebold-Mariano hypothesis testing,
and identifies the optimal sector combination that produces maximum predictive lift.
"""

import os
import itertools
import logging
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import norm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

QUAD_PANEL_PATH = "data/quad_domain_annual_panel.parquet"
OUTPUT_FULL_BENCHMARK_PATH = "data/exhaustive_combinatorial_benchmark_results.csv"
OUTPUT_OPTIMAL_SUMMARY_PATH = "data/optimal_sector_combinations_summary.csv"


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


def get_sector_features(df):
    num_cols = [c for c in df.columns if c not in ["iso3", "year", "timestamp"] and not c.endswith("_target_h1") and not c.endswith("_target_h3") and not c.endswith("_target_h5")]

    s1_keywords = ["gdp", "inflation", "debt", "unemployment", "banking_crisis", "currency_crisis", "sov_debt_crisis", "eco"]
    s2_keywords = ["goldstein", "conflict", "protest", "sanction", "stability", "news", "pol"]
    s3_keywords = ["co2", "temp", "forest", "flood", "drought", "wildfire", "storm", "renewable", "disaster", "env", "protected_area"]
    s4_keywords = ["psychology", "society", "trust", "fear", "optimism", "nationalism", "cohesion", "confidence", "education", "urbanization", "population", "age", "religion", "healthcare", "migration"]

    s1_cols = [c for c in num_cols if any(k in c for k in s1_keywords)]
    s2_cols = [c for c in num_cols if any(k in c for k in s2_keywords) and c not in s1_cols]
    s3_cols = [c for c in num_cols if any(k in c for k in s3_keywords) and c not in s1_cols and c not in s2_cols]
    s4_cols = [c for c in num_cols if any(k in c for k in s4_keywords) and c not in s1_cols and c not in s2_cols and c not in s3_cols]

    sectors = {
        "S1_Eco": s1_cols,
        "S2_Pol": s2_cols,
        "S3_Env": s3_cols,
        "S4_Hum": s4_cols
    }
    return sectors


def generate_all_sector_combinations(sectors):
    sector_names = list(sectors.keys())
    combos = {}

    for r in range(1, len(sector_names) + 1):
        for combo in itertools.combinations(sector_names, r):
            combo_name = "+".join(combo)
            cols = []
            for s in combo:
                cols.extend(sectors[s])
            combos[combo_name] = list(set(cols))

    return combos


def run_exhaustive_combinatorial_tournament(df, target_col, combinations, n_folds=5):
    clean_df = df.dropna(subset=[target_col]).copy()
    clean_df = clean_df[np.isfinite(clean_df[target_col])].copy()

    q_low = clean_df[target_col].quantile(0.01)
    q_high = clean_df[target_col].quantile(0.99)
    clean_df = clean_df[(clean_df[target_col] >= q_low) & (clean_df[target_col] <= q_high)].sort_values("year").reset_index(drop=True)

    if len(clean_df) < 150:
        logging.warning(f"Insufficient clean samples for {target_col}: {len(clean_df)}")
        return []

    y = clean_df[target_col].values
    n_total = len(clean_df)
    min_train_pct = 0.50
    step_pct = (1.0 - min_train_pct) / n_folds

    oof_predictions = {cfg: [] for cfg in combinations.keys()}
    oof_y_true = []

    prepared_features = {}
    for cfg_name, cols in combinations.items():
        valid_cols = [c for c in cols if c in clean_df.columns and c != target_col]
        clean_df[valid_cols] = clean_df[valid_cols].fillna(0.0)
        prepared_features[cfg_name] = (valid_cols, clean_df[valid_cols].values)

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

    oof_y = np.array(oof_y_true)

    # Determine baseline (single sector baseline corresponding to target's native domain)
    if "psychology" in target_col or "society" in target_col:
        base_cfg = "S4_Hum"
    elif "gdp" in target_col or "inflation" in target_col or "unemployment" in target_col:
        base_cfg = "S1_Eco"
    elif "stability" in target_col or "goldstein" in target_col or "conflict" in target_col:
        base_cfg = "S2_Pol"
    else:
        base_cfg = "S3_Env"

    if base_cfg not in oof_predictions or len(oof_predictions[base_cfg]) != len(oof_y):
        base_cfg = list(combinations.keys())[0]

    base_oof_pred = np.array(oof_predictions[base_cfg])
    base_rmse = float(np.sqrt(mean_squared_error(oof_y, base_oof_pred)))
    base_err = abs(oof_y - base_oof_pred)

    results_rows = []

    for cfg_name in combinations.keys():
        if cfg_name not in oof_predictions or len(oof_predictions[cfg_name]) != len(oof_y):
            continue

        oof_pred = np.array(oof_predictions[cfg_name])
        rmse = float(np.sqrt(mean_squared_error(oof_y, oof_pred)))
        mae = float(mean_absolute_error(oof_y, oof_pred))
        r2 = float(r2_score(oof_y, oof_pred))
        err = abs(oof_y - oof_pred)

        num_sectors = len(cfg_name.split("+"))
        imprv_pct = float(((base_rmse - rmse) / (base_rmse + 1e-8)) * 100.0)

        if cfg_name == base_cfg:
            dm_stat, p_val = 0.0, 1.0
        else:
            dm_stat, p_val = diebold_mariano_test(base_err, err)

        results_rows.append({
            "target_variable": target_col,
            "combination_name": cfg_name,
            "num_sectors": num_sectors,
            "baseline_sector": base_cfg,
            "oof_rmse": round(rmse, 4),
            "oof_mae": round(mae, 4),
            "oof_r2": round(r2, 4),
            "rmse_improvement_pct": round(imprv_pct, 2),
            "dm_statistic": round(dm_stat, 4),
            "p_value": round(p_val, 4),
            "is_stat_significant": p_val < 0.05
        })

    return results_rows


def run_benchmark():
    if not os.path.exists(QUAD_PANEL_PATH):
        raise FileNotFoundError(f"Missing {QUAD_PANEL_PATH}")

    df = pd.read_parquet(QUAD_PANEL_PATH)
    logging.info(f"Loaded dataset shape: {df.shape}")

    sectors = get_sector_features(df)
    for s_name, cols in sectors.items():
        logging.info(f"Sector {s_name}: {len(cols)} features")

    combinations = generate_all_sector_combinations(sectors)
    logging.info(f"Generated {len(combinations)} non-empty sector combinations.")

    targets = [
        "gdp_pc_growth_1y_fwd",
        "stability_momentum_annual_mean",
        "co2_emissions_per_capita",
        "psychology_trust",
        "psychology_fear",
        "disaster_economic_damage_usd"
    ]
    targets = [t for t in targets if t in df.columns]

    all_benchmark_rows = []
    optimal_summary_rows = []

    for target in targets:
        logging.info(f"\n--- Benchmarking Target: {target} across all 15 Combinations ---")
        t_rows = run_exhaustive_combinatorial_tournament(df, target, combinations)
        all_benchmark_rows.extend(t_rows)

        if t_rows:
            df_t = pd.DataFrame(t_rows)
            best_row = df_t.sort_values("oof_rmse").iloc[0]
            optimal_summary_rows.append({
                "target_variable": target,
                "best_combination": best_row["combination_name"],
                "best_num_sectors": best_row["num_sectors"],
                "best_oof_rmse": best_row["oof_rmse"],
                "best_oof_r2": best_row["oof_r2"],
                "max_improvement_pct": best_row["rmse_improvement_pct"],
                "dm_p_value": best_row["p_value"],
                "stat_significant_lift": best_row["is_stat_significant"]
            })

    df_full = pd.DataFrame(all_benchmark_rows)
    df_optimal = pd.DataFrame(optimal_summary_rows)

    os.makedirs("data", exist_ok=True)
    df_full.to_csv(OUTPUT_FULL_BENCHMARK_PATH, index=False)
    df_optimal.to_csv(OUTPUT_OPTIMAL_SUMMARY_PATH, index=False)

    logging.info(f"Saved full benchmark results to {OUTPUT_FULL_BENCHMARK_PATH}")
    logging.info(f"Saved optimal summary to {OUTPUT_OPTIMAL_SUMMARY_PATH}")

    print("\n" + "=" * 80)
    print("  OPTIMAL SECTOR COMBINATIONS SUMMARY")
    print("=" * 80)
    print(df_optimal.to_string(index=False))


if __name__ == "__main__":
    run_benchmark()
