"""
Dumitrescu-Hurlin (2012) Panel Granger Causality Test (Pure NumPy Implementation)
================================================================================
Tests for Granger non-causality in heterogeneous panels with fixed-T asymptotics.
Reference: Dumitrescu, E. I., & Hurlin, C. (2012). Testing for Granger non-causality
in heterogeneous panels. Economic Modelling, 29(4), 1450-1460.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist, norm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(r"e:\politics and economy")
QUAD_PANEL = ROOT / "data" / "quad_domain_annual_panel.parquet"
OUT_CSV = ROOT / "data" / "dumitrescu_hurlin_panel_granger_results.csv"


@dataclass
class DHTestResult:
    y_var: str
    x_var: str
    lags: int
    n_countries: int
    avg_obs_per_country: float
    w_bar: float
    z_tilde: float
    p_value: float
    p_value_bonferroni: float
    significant_at_05: bool
    pct_countries_causal: float


def ols_ssr(y: np.ndarray, X: np.ndarray) -> tuple[float, int] | None:
    """Compute Sum of Squared Residuals (SSR) and degrees of freedom via numpy lstsq."""
    try:
        n, p = X.shape
        if n <= p:
            return None
        beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        if rank < p:
            return None
        if len(residuals) > 0:
            ssr = float(residuals[0])
        else:
            y_pred = X @ beta
            ssr = float(np.sum((y - y_pred) ** 2))
        df_resid = n - p
        return ssr, df_resid
    except Exception:
        return None


def individual_granger_wald(y_series: np.ndarray, x_series: np.ndarray, lags: int) -> float | None:
    """Compute individual Wald statistic for unit i testing x -> y with K lags."""
    T = len(y_series)
    if T <= 2 * lags + 3:
        return None

    Y_dep = y_series[lags:]
    X_mat = [np.ones(T - lags)]

    for k in range(1, lags + 1):
        X_mat.append(y_series[lags - k : T - k])

    for k in range(1, lags + 1):
        X_mat.append(x_series[lags - k : T - k])

    X_u = np.column_stack(X_mat)
    X_r = X_u[:, : (lags + 1)]

    res_u = ols_ssr(Y_dep, X_u)
    res_r = ols_ssr(Y_dep, X_r)

    if res_u is None or res_r is None:
        return None

    ssr_u, df_u = res_u
    ssr_r, df_r = res_r

    if ssr_u <= 1e-12 or df_u <= 0:
        return None

    f_stat = ((ssr_r - ssr_u) / lags) / (ssr_u / df_u)
    if f_stat < 0 or not np.isfinite(f_stat):
        return None

    wald_i = lags * f_stat
    return float(wald_i)


def run_dumitrescu_hurlin_test(df: pd.DataFrame, y_var: str, x_var: str,
                                lags: int = 2, min_obs: int = 15) -> DHTestResult | None:
    """Run Dumitrescu-Hurlin (2012) test for x_var -> y_var on panel df[iso3, year]."""
    valid = df[["iso3", "year", y_var, x_var]].dropna().sort_values(["iso3", "year"])
    countries = valid["iso3"].unique()

    wald_stats = []
    indiv_p_vals = []
    t_lengths = []

    for c in countries:
        c_data = valid[valid["iso3"] == c]
        if len(c_data) < min_obs:
            continue

        y_c = c_data[y_var].values.astype(np.float64)
        x_c = c_data[x_var].values.astype(np.float64)

        w_i = individual_granger_wald(y_c, x_c, lags)
        if w_i is not None and np.isfinite(w_i):
            wald_stats.append(w_i)
            t_lengths.append(len(c_data))
            f_val = w_i / lags
            df2 = len(c_data) - 2 * lags - 1
            p_i = 1.0 - f_dist.cdf(f_val, lags, df2) if df2 > 0 else 1.0
            indiv_p_vals.append(p_i)

    N = len(wald_stats)
    if N < 10:
        return None

    w_bar = float(np.mean(wald_stats))
    T_bar = float(np.mean(t_lengths))
    K = lags

    e_w = K * (T_bar - 2 * K - 1) / max(1.0, (T_bar - 2 * K - 3))
    var_w = (2 * K * ((T_bar - 2 * K - 1) ** 2) * (T_bar - K - 3)) / max(
        1.0, (((T_bar - 2 * K - 3) ** 2) * (T_bar - 2 * K - 5))
    )

    if var_w <= 0:
        return None

    z_tilde = float(np.sqrt(N / var_w) * (w_bar - e_w))
    p_val = float(2 * (1 - norm.cdf(abs(z_tilde))))
    pct_causal = float(np.mean([p < 0.05 for p in indiv_p_vals]) * 100)

    return DHTestResult(
        y_var=y_var,
        x_var=x_var,
        lags=lags,
        n_countries=N,
        avg_obs_per_country=round(T_bar, 1),
        w_bar=round(w_bar, 4),
        z_tilde=round(z_tilde, 4),
        p_value=round(p_val, 6),
        p_value_bonferroni=1.0,
        significant_at_05=p_val < 0.05,
        pct_countries_causal=round(pct_causal, 1),
    )


def run_full_dh_suite(df: pd.DataFrame) -> pd.DataFrame:
    """Run DH tests across all key cross-domain hypotheses with multiplicity corrections."""
    log.info("=" * 70)
    log.info("  DUMITRESCU-HURLIN (2012) PANEL GRANGER CAUSALITY SUITE")
    log.info("=" * 70)

    hypotheses = [
        ("gdp_pc_growth_1y_fwd", "psychology_trust", "Trust -> GDP Growth"),
        ("gdp_pc_growth_1y_fwd", "psychology_social_cohesion", "Social Cohesion -> GDP Growth"),
        ("gdp_pc_growth_1y_fwd", "society_education", "Education -> GDP Growth"),
        ("material_conflict_annual_sum", "psychology_fear", "Fear -> Material Conflict"),
        ("protest_unrest_annual_sum", "psychology_fear", "Fear -> Protest Unrest"),
        ("stability_momentum_annual_mean", "psychology_trust", "Trust -> Political Stability"),
        ("psychology_fear", "disaster_economic_damage_usd", "Disaster Damage -> Social Fear"),
        ("psychology_fear", "temp_anomaly_celsius", "Temp Anomaly -> Social Fear"),
        ("psychology_trust", "disaster_economic_damage_usd", "Disaster Damage -> Institutional Trust"),
        ("gdp_pc_growth_1y_fwd", "disaster_economic_damage_usd", "Disaster Damage -> GDP Growth"),
        ("gdp_pc_growth_1y_fwd", "temp_anomaly_celsius", "Temp Anomaly -> GDP Growth"),
        ("renewable_energy_pct_share", "stability_momentum_annual_mean", "Stability -> Renewable Adoption"),
        ("co2_emissions_per_capita", "sanctions_coercion_annual_sum", "Sanctions -> CO2 Emissions"),
        ("psychology_trust", "gdp_pc_growth_1y_fwd", "GDP Growth -> Trust (Feedback)"),
        ("psychology_fear", "inflation_rate", "Inflation -> Fear (Feedback)"),
    ]

    valid_hyps = [
        (y, x, desc) for (y, x, desc) in hypotheses
        if y in df.columns and x in df.columns
    ]

    log.info(f"Testing {len(valid_hyps)} directional panel hypotheses...")
    results = []

    for y, x, desc in valid_hyps:
        for lag in [1, 2]:
            res = run_dumitrescu_hurlin_test(df, y, x, lags=lag)
            if res:
                results.append({
                    "hypothesis": desc,
                    "direction": f"{x} -> {y}",
                    "y_var": y,
                    "x_var": x,
                    "lags": lag,
                    "n_countries": res.n_countries,
                    "avg_t": res.avg_obs_per_country,
                    "w_bar": res.w_bar,
                    "z_tilde": res.z_tilde,
                    "p_value": res.p_value,
                    "pct_countries_causal": res.pct_countries_causal,
                })

    df_res = pd.DataFrame(results)

    if not df_res.empty:
        m = len(df_res)
        df_res["p_bonferroni"] = (df_res["p_value"] * m).clip(upper=1.0).round(6)
        
        sorted_indices = np.argsort(df_res["p_value"].values)
        p_sorted = df_res["p_value"].values[sorted_indices]
        fdr_thresh = (np.arange(1, m + 1) / m) * 0.05
        fdr_sig = p_sorted <= fdr_thresh
        
        df_res["significant_bonferroni"] = df_res["p_bonferroni"] < 0.05
        df_res["significant_fdr_05"] = False
        df_res.iloc[sorted_indices[fdr_sig], df_res.columns.get_loc("significant_fdr_05")] = True

    df_res.to_csv(OUT_CSV, index=False)
    log.info(f"Saved DH panel Granger results to {OUT_CSV}")

    print("\n" + "=" * 90)
    print("  DUMITRESCU-HURLIN (2012) PANEL GRANGER CAUSALITY TEST RESULTS")
    print("=" * 90)
    cols_show = ["hypothesis", "lags", "n_countries", "w_bar", "z_tilde", "p_value", "p_bonferroni", "significant_fdr_05"]
    print(df_res[cols_show].to_string(index=False))

    return df_res


if __name__ == "__main__":
    if QUAD_PANEL.exists():
        df_panel = pd.read_parquet(QUAD_PANEL)
        run_full_dh_suite(df_panel)
    else:
        log.error(f"Missing {QUAD_PANEL}")
