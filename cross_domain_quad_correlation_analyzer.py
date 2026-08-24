"""
Quad-Domain Correlation, Granger Causality & Crisis Shocks Analyzer
----------------------------------------------------------------------
Analyzes statistical dependencies across Economy, Politics, Environment, and Human/Society:
1. Pearson & Spearman Bivariate Correlations across all 4 domains
2. Directional Granger Causality (Human -> Eco, Human -> Pol, Human -> Env, Env -> Human, etc.)
3. Crisis Shocks Co-occurrence & Distribution Shift Analysis
"""

import os
import logging
import numpy as np
import pandas as pd
from scipy.stats import f as f_dist

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

QUAD_PANEL_PATH = "data/quad_domain_annual_panel.parquet"
OUTPUT_CORR_PATH = "data/quad_correlation_matrix.csv"
OUTPUT_GC_PATH = "data/quad_granger_causality_results.csv"
OUTPUT_SHOCKS_PATH = "data/quad_disaster_crisis_cooccurrence.csv"


def run_bivariate_correlations(df):
    logging.info("Computing Quad-Domain Bivariate Correlations (Pearson & Spearman)...")

    eco_vars = [
        "gdp_pc_growth_5y_fwd", "gdp_pc_growth_1y_fwd", "inflation_rate",
        "gov_debt_gdp", "unemployment_rate", "banking_crisis", "currency_crisis", "sov_debt_crisis"
    ]
    pol_vars = [
        "goldstein_annual_mean", "news_tone_annual_mean", "material_conflict_annual_sum",
        "verbal_conflict_annual_sum", "protest_unrest_annual_sum", "sanctions_coercion_annual_sum",
        "conflict_intensity_annual_mean", "stability_momentum_annual_mean"
    ]
    env_vars = [
        "co2_emissions_per_capita", "temp_anomaly_celsius", "forest_area_pct_land",
        "floods_count", "droughts_count", "wildfires_count", "extreme_temp_count", "storms_count",
        "disaster_economic_damage_usd", "renewable_energy_pct_share", "greenhouse_gas_total_kt",
        "energy_use_per_capita", "protected_area_pct"
    ]
    human_vars = [
        "psychology_trust", "psychology_fear", "psychology_optimism", "psychology_nationalism",
        "psychology_social_cohesion", "psychology_confidence", "society_education",
        "society_urbanization", "society_population", "society_age", "society_religion",
        "society_healthcare", "society_migration"
    ]

    eco_vars = [v for v in eco_vars if v in df.columns]
    pol_vars = [v for v in pol_vars if v in df.columns]
    env_vars = [v for v in env_vars if v in df.columns]
    human_vars = [v for v in human_vars if v in df.columns]

    all_vars = eco_vars + pol_vars + env_vars + human_vars
    sub_df = df[all_vars].dropna()

    domain_map = {}
    for v in eco_vars:
        domain_map[v] = "Economy"
    for v in pol_vars:
        domain_map[v] = "Politics"
    for v in env_vars:
        domain_map[v] = "Environment"
    for v in human_vars:
        domain_map[v] = "Human/Society"

    results = []
    for i in range(len(all_vars)):
        for j in range(i + 1, len(all_vars)):
            v1, v2 = all_vars[i], all_vars[j]
            d1, d2 = domain_map[v1], domain_map[v2]

            # Focus on cross-domain pairs
            if d1 == d2:
                continue

            pearson_r = sub_df[v1].corr(sub_df[v2], method="pearson")
            spearman_r = sub_df[v1].corr(sub_df[v2], method="spearman")

            results.append({
                "var_1": v1,
                "domain_1": d1,
                "var_2": v2,
                "domain_2": d2,
                "pair_type": f"{d1} <-> {d2}",
                "pearson_r": round(float(pearson_r), 4),
                "spearman_r": round(float(spearman_r), 4),
                "abs_spearman_r": round(float(abs(spearman_r)), 4)
            })

    corr_df = pd.DataFrame(results).sort_values("abs_spearman_r", ascending=False)
    corr_df.to_csv(OUTPUT_CORR_PATH, index=False)
    logging.info(f"Top 5 Cross-Domain Correlations:\n{corr_df.head(5).to_string(index=False)}")
    return corr_df


def granger_causality_test_custom(y, x, max_lag=3):
    results = {}
    N = len(y)

    for lag in range(1, max_lag + 1):
        if N <= 2 * lag + 5:
            continue

        Y_target = y[lag:]

        X_r_list = [np.ones((N - lag, 1))]
        for i in range(1, lag + 1):
            X_r_list.append(y[lag - i: N - i].reshape(-1, 1))
        X_r = np.hstack(X_r_list)

        X_u_list = [X_r]
        for i in range(1, lag + 1):
            X_u_list.append(x[lag - i: N - i].reshape(-1, 1))
        X_u = np.hstack(X_u_list)

        beta_r, _, _, _ = np.linalg.lstsq(X_r, Y_target, rcond=None)
        pred_r = X_r @ beta_r
        ssr_r = np.sum((Y_target - pred_r) ** 2)

        beta_u, _, _, _ = np.linalg.lstsq(X_u, Y_target, rcond=None)
        pred_u = X_u @ beta_u
        ssr_u = np.sum((Y_target - pred_u) ** 2)

        df1 = lag
        df2 = (N - lag) - (2 * lag + 1)

        if ssr_u > 0 and df2 > 0:
            f_stat = ((ssr_r - ssr_u) / df1) / (ssr_u / df2)
            p_val = 1.0 - f_dist.cdf(f_stat, df1, df2)
        else:
            f_stat, p_val = 0.0, 1.0

        results[lag] = (float(f_stat), float(p_val))

    return results


def run_granger_causality(df, max_lag=3):
    logging.info("Running Directional Quad-Domain Granger Causality Tests...")

    pairs = [
        # Human -> Eco
        ("gdp_pc_growth_1y_fwd", "psychology_trust", "Human_Trust_causes_GDP_Growth"),
        ("gdp_pc_growth_1y_fwd", "psychology_confidence", "Human_Confidence_causes_GDP_Growth"),
        ("inflation_rate", "psychology_fear", "Human_Fear_causes_Inflation"),

        # Human -> Pol
        ("material_conflict_annual_sum", "psychology_fear", "Human_Fear_causes_Conflict"),
        ("protest_unrest_annual_sum", "psychology_trust", "Human_Trust_causes_Protests"),
        ("stability_momentum_annual_mean", "psychology_social_cohesion", "Human_Cohesion_causes_Political_Stability"),

        # Human -> Env
        ("co2_emissions_per_capita", "society_urbanization", "Society_Urbanization_causes_CO2"),
        ("renewable_energy_pct_share", "society_education", "Society_Education_causes_Renewables"),

        # Env -> Human
        ("psychology_fear", "disaster_economic_damage_usd", "Env_DisasterDamage_causes_Human_Fear"),
        ("psychology_trust", "droughts_count", "Env_Droughts_causes_Human_Trust_Loss"),
        ("society_migration", "extreme_temp_count", "Env_ExtremeTemp_causes_Migration"),

        # Eco -> Human
        ("psychology_trust", "gdp_pc_growth_1y_fwd", "Eco_GDPGrowth_causes_Human_Trust"),
        ("psychology_fear", "unemployment_rate", "Eco_Unemployment_causes_Human_Fear")
    ]

    gc_results = []
    for y_col, x_col, description in pairs:
        if y_col not in df.columns or x_col not in df.columns:
            continue

        clean_sub = df[["iso3", "year", y_col, x_col]].dropna()
        if len(clean_sub) < 50:
            continue

        for iso in clean_sub["iso3"].unique():
            c_df = clean_sub[clean_sub["iso3"] == iso].sort_values("year")
            if len(c_df) < 15:
                continue

            res = granger_causality_test_custom(c_df[y_col].values, c_df[x_col].values, max_lag=max_lag)
            for lag, (f_stat, p_val) in res.items():
                gc_results.append({
                    "test_name": description,
                    "target_y": y_col,
                    "driver_x": x_col,
                    "country": iso,
                    "lag": lag,
                    "f_statistic": round(f_stat, 4),
                    "p_value": round(p_val, 4),
                    "is_significant": p_val < 0.05
                })

    if not gc_results:
        gc_df = pd.DataFrame(columns=["test_name", "target_y", "driver_x", "country", "lag", "f_statistic", "p_value", "is_significant"])
    else:
        gc_df = pd.DataFrame(gc_results)

    gc_summary = gc_df.groupby(["test_name", "lag"]).agg(
        total_countries=("country", "count"),
        sig_countries=("is_significant", "sum"),
        avg_f_stat=("f_statistic", "mean"),
        avg_p_val=("p_value", "mean")
    ).reset_index()

    gc_summary["pct_significant"] = round((gc_summary["sig_countries"] / gc_summary["total_countries"]) * 100.0, 2)
    gc_summary.to_csv(OUTPUT_GC_PATH, index=False)
    logging.info(f"Granger Causality Summary:\n{gc_summary.head(10).to_string(index=False)}")
    return gc_summary


def run_disaster_crisis_cooccurrence(df):
    logging.info("Computing Quad-Domain Shock & Crisis Co-occurrence...")

    shock_cols = [c for c in ["disaster_economic_damage_usd", "psychology_fear", "banking_crisis", "protest_unrest_annual_sum"] if c in df.columns]

    if len(shock_cols) < 2:
        df_dummy = pd.DataFrame([{"event_type": "None", "cooccurrence_count": 0}])
        df_dummy.to_csv(OUTPUT_SHOCKS_PATH, index=False)
        return df_dummy

    sub = df[shock_cols].dropna()

    co_results = []
    fear_high = sub["psychology_fear"] > sub["psychology_fear"].quantile(0.80) if "psychology_fear" in sub.columns else pd.Series(False, index=sub.index)
    damage_high = sub["disaster_economic_damage_usd"] > sub["disaster_economic_damage_usd"].quantile(0.80) if "disaster_economic_damage_usd" in sub.columns else pd.Series(False, index=sub.index)

    co_results.append({
        "shock_pair": "High_Environmental_Damage & High_Human_Fear",
        "cooccurrence_count": int((fear_high & damage_high).sum()),
        "total_sample": len(sub),
        "cooccurrence_pct": round(float((fear_high & damage_high).mean() * 100.0), 2)
    })

    co_df = pd.DataFrame(co_results)
    co_df.to_csv(OUTPUT_SHOCKS_PATH, index=False)
    logging.info(f"Crisis Co-occurrence:\n{co_df.to_string(index=False)}")
    return co_df


def main():
    if not os.path.exists(QUAD_PANEL_PATH):
        raise FileNotFoundError(f"Missing {QUAD_PANEL_PATH}")
    df_panel = pd.read_parquet(QUAD_PANEL_PATH)
    run_bivariate_correlations(df_panel)
    run_granger_causality(df_panel, max_lag=3)
    run_disaster_crisis_cooccurrence(df_panel)


if __name__ == "__main__":
    main()
