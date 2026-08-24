"""
Econometric Testing: Pure NumPy Dumitrescu-Hurlin Panel Granger & Diebold-Mariano
==================================================================================
Reference: Dumitrescu, E. I., & Hurlin, C. (2012). Testing for Granger non-causality
in heterogeneous panels. Economic Modelling, 29(4), 1450-1460.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy.stats import norm


def diebold_mariano_test(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray,
    h: int = 1,
    criterion: str = "mae"
) -> Tuple[float, float]:
    """
    Diebold-Mariano (1995) forecast accuracy equivalence test.
    """
    y_true = np.asarray(y_true)
    y_pred1 = np.asarray(y_pred1)
    y_pred2 = np.asarray(y_pred2)

    if criterion == "mae":
        d = np.abs(y_true - y_pred1) - np.abs(y_true - y_pred2)
    else:
        d = (y_true - y_pred1) ** 2 - (y_true - y_pred2) ** 2

    n = len(d)
    if n < 5:
        return 0.0, 1.0

    mean_d = np.mean(d)
    gamma0 = np.var(d, ddof=1)

    autocov = 0.0
    for lag in range(1, h):
        if lag < n:
            cov = np.sum((d[lag:] - mean_d) * (d[:-lag] - mean_d)) / n
            autocov += 2.0 * cov

    var_d = (gamma0 + autocov) / n
    if var_d <= 0.0:
        return 0.0, 1.0

    dm_stat = mean_d / np.sqrt(var_d)
    p_value = 2.0 * (1.0 - norm.cdf(np.abs(dm_stat)))

    return float(dm_stat), float(p_value)


def _ols_ssr(y: np.ndarray, X: np.ndarray) -> Optional[Tuple[float, int]]:
    """Compute Sum of Squared Residuals (SSR) and df via numpy lstsq."""
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


def individual_granger_wald(y_series: np.ndarray, x_series: np.ndarray, lags: int) -> Optional[float]:
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

    res_u = _ols_ssr(Y_dep, X_u)
    res_r = _ols_ssr(Y_dep, X_r)

    if res_u is None or res_r is None:
        return None

    ssr_u, df_u = res_u
    ssr_r, df_r = res_r

    if ssr_u <= 1e-12 or df_u <= 0:
        return None

    f_stat = ((ssr_r - ssr_u) / lags) / (ssr_u / df_u)
    if f_stat < 0 or not np.isfinite(f_stat):
        return None

    return float(lags * f_stat)


def dumitrescu_hurlin_test(
    panel_df: pd.DataFrame,
    cause_col: str,
    effect_col: str,
    max_lag: int = 2,
    min_obs_per_country: int = 15
) -> Dict[str, Any]:
    """
    Dumitrescu-Hurlin (2012) Panel Granger Causality Test.
    """
    countries = panel_df["iso3"].unique()
    w_stats: List[float] = []

    for c in countries:
        c_data = panel_df[panel_df["iso3"] == c].sort_values("year")[[cause_col, effect_col]].dropna()
        if len(c_data) < max(min_obs_per_country, max_lag * 3):
            continue

        w_i = individual_granger_wald(c_data[effect_col].values, c_data[cause_col].values, lags=max_lag)
        if w_i is not None and np.isfinite(w_i):
            w_stats.append(w_i)

    n_countries = len(w_stats)
    if n_countries < 5:
        return {
            "w_bar": 0.0,
            "z_tilde": 0.0,
            "p_value": 1.0,
            "n_countries": n_countries,
            "significant": False
        }

    k = float(max_lag)
    w_bar = float(np.mean(w_stats))
    
    # Asymptotic Z-statistic
    z_stat = np.sqrt(n_countries / (2.0 * k)) * (w_bar - k)
    p_val = float(2.0 * (1.0 - norm.cdf(np.abs(z_stat))))

    return {
        "w_bar": round(w_bar, 4),
        "z_tilde": round(float(z_stat), 4),
        "p_value": p_val,
        "n_countries": n_countries,
        "significant": bool(p_val < 0.05)
    }
