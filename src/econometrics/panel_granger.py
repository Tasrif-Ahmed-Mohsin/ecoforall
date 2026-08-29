"""
Econometric Testing: Panel Granger Causality, Non-Stationarity Pre-Testing & Robust Forecast Inference
======================================================================================================
References:
  - Dumitrescu, E. I., & Hurlin, C. (2012). Testing for Granger non-causality in heterogeneous panels.
    Economic Modelling, 29(4), 1450-1460.
  - Granger, C. W., & Newbold, P. (1974). Spurious regressions in econometrics. Journal of Econometrics, 2(2), 111-120.
  - Pesaran, M. H. (2004, 2021). General Diagnostic Tests for Cross-Sectional Dependence in Panels.
    Empirical Economics, 60(1), 13-50.
  - Clark, T. E., & West, K. D. (2007). Approximately normal tests for equal predictive accuracy
    in nested models. Journal of Econometrics, 138(1), 291-311.
  - Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. JBES, 13(3), 253-263.
  - Romano, J. P., & Wolf, M. (2005). Stepwise multiple testing as formalised by data-snooping.
    Econometrica, 73(4), 1237-1282.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import t as t_dist
from statsmodels.tsa.stattools import adfuller


def diebold_mariano_test(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray,
    h: int = 1,
    criterion: str = "mae"
) -> Tuple[float, float]:
    """
    Diebold-Mariano (1995) forecast accuracy test with Newey-West (1987) HAC covariance.
    NOTE: Only valid for strictly non-nested models on non-pooled time series.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred1 = np.asarray(y_pred1, dtype=np.float64)
    y_pred2 = np.asarray(y_pred2, dtype=np.float64)

    valid = np.isfinite(y_true) & np.isfinite(y_pred1) & np.isfinite(y_pred2)
    y_true = y_true[valid]
    y_pred1 = y_pred1[valid]
    y_pred2 = y_pred2[valid]

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
    bandwidth = max(1, h)
    for lag in range(1, bandwidth):
        if lag < n:
            weight = 1.0 - (lag / float(bandwidth))
            cov = np.sum((d[lag:] - mean_d) * (d[:-lag] - mean_d)) / float(n)
            autocov += 2.0 * weight * cov

    var_d = max(1e-12, (gamma0 + autocov) / float(n))
    dm_stat = mean_d / np.sqrt(var_d)
    p_value = float(2.0 * (1.0 - norm.cdf(np.abs(dm_stat))))

    return float(dm_stat), float(p_value)


def year_clustered_forecast_test(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray,
    years: np.ndarray,
    criterion: str = "mae",
    nested: bool = False,
    h: int = 1,
) -> Tuple[float, float, int]:
    r"""
    Forecast-accuracy test with year-clustered standard errors (Driscoll-Kraay style).

    Pooled panel DM/CW divides by :math:`\sqrt{N_{\text{country-years}}}`, treating the
    ~170 sovereigns that share each year's global shocks (oil, 2008, COVID) as
    independent draws. This averages the loss differential within each origin year and
    tests the year-level means, so the effective sample is the number of *years*.

    On this panel the pooled statistic overstates :math:`|DM|` by roughly 2.6x
    (e.g. 30.08 pooled vs 11.49 year-clustered at :math:`h=1`).

    Set ``nested=True`` to apply the Clark-West adjustment term
    :math:`(\hat{y}_1 - \hat{y}_2)^2` before clustering, which is required when
    ``y_pred1`` is nested in ``y_pred2``.

    Returns
    -------
    (stat, p_value, n_years)
        Two-sided ``p`` from a :math:`t` distribution with ``n_years - 1`` degrees of
        freedom. Clark-West is one-sided by construction, so ``p`` is halved when
        ``nested=True`` and the statistic is positive.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y1 = np.asarray(y_pred1, dtype=np.float64)
    y2 = np.asarray(y_pred2, dtype=np.float64)
    yr = np.asarray(years)

    valid = np.isfinite(y_true) & np.isfinite(y1) & np.isfinite(y2)
    y_true, y1, y2, yr = y_true[valid], y1[valid], y2[valid], yr[valid]
    if len(y_true) < 5:
        return 0.0, 1.0, 0

    if nested:
        d = (y_true - y1) ** 2 - ((y_true - y2) ** 2 - (y1 - y2) ** 2)
    elif criterion == "mae":
        d = np.abs(y_true - y1) - np.abs(y_true - y2)
    else:
        d = (y_true - y1) ** 2 - (y_true - y2) ** 2

    per_year_df = pd.DataFrame({"d": d, "yr": yr}).groupby("yr")["d"].mean().sort_index()
    per_year = per_year_df.to_numpy()
    g = len(per_year)
    if g < 3:
        return 0.0, 1.0, g

    # Mean loss differential
    mean_d = float(np.mean(per_year))
    gamma0 = float(np.var(per_year, ddof=1))

    # Newey-West Bartlett kernel adjustment for multi-step horizon serial correlation (h - 1 lags)
    autocov = 0.0
    bandwidth = max(1, h)
    for lag in range(1, bandwidth):
        if lag < g:
            weight = 1.0 - (lag / float(bandwidth))
            cov = float(np.sum((per_year[lag:] - mean_d) * (per_year[:-lag] - mean_d)) / float(g))
            autocov += 2.0 * weight * cov

    var_d = max(1e-12, (gamma0 + autocov) / float(g))
    se = float(np.sqrt(var_d))

    stat = float(mean_d / se)
    p_two = float(2.0 * (1.0 - t_dist.cdf(abs(stat), df=g - 1)))
    if nested:
        p_val = p_two / 2.0 if stat > 0 else 1.0 - p_two / 2.0
    else:
        p_val = p_two
    return stat, float(p_val), g


def clark_west_test(
    y_true: np.ndarray,
    y_pred_small: np.ndarray,
    y_pred_large: np.ndarray,
    h: int = 1
) -> Tuple[float, float]:
    r"""
    Clark-West (2007) test for comparing nested forecasting models.
    Adjusts MSFE differential for finite-sample parameter estimation noise in the nested model:
      f_t = (y_t - \hat{y}_{1,t})^2 - [(y_t - \hat{y}_{2,t})^2 - (\hat{y}_{1,t} - \hat{y}_{2,t})^2]
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y1 = np.asarray(y_pred_small, dtype=np.float64)
    y2 = np.asarray(y_pred_large, dtype=np.float64)

    valid = np.isfinite(y_true) & np.isfinite(y1) & np.isfinite(y2)
    y_true = y_true[valid]
    y1 = y1[valid]
    y2 = y2[valid]

    n = len(y_true)
    if n < 5:
        return 0.0, 1.0

    e1_sq = (y_true - y1) ** 2
    e2_sq = (y_true - y2) ** 2
    adj = (y1 - y2) ** 2
    f = e1_sq - (e2_sq - adj)

    mean_f = np.mean(f)
    gamma0 = np.var(f, ddof=1)

    autocov = 0.0
    bandwidth = max(1, h)
    for lag in range(1, bandwidth):
        if lag < n:
            weight = 1.0 - (lag / float(bandwidth))
            cov = np.sum((f[lag:] - mean_f) * (f[:-lag] - mean_f)) / float(n)
            autocov += 2.0 * weight * cov

    var_f = max(1e-12, (gamma0 + autocov) / float(n))
    cw_stat = mean_f / np.sqrt(var_f)
    # One-sided test (larger model has strictly lower true prediction error)
    p_value = float(1.0 - norm.cdf(cw_stat))

    return float(cw_stat), float(p_value)


def pesaran_cd_test(panel_residuals: pd.DataFrame, time_col: str = "year", unit_col: str = "iso3", res_col: str = "residual") -> Tuple[float, float]:
    """
    Pesaran (2004, 2021) Cross-Sectional Dependence (CD) Test.
    Tests null hypothesis of cross-sectional independence across panel units.
    """
    piv = panel_residuals.pivot(index=time_col, columns=unit_col, values=res_col)
    units = piv.columns.tolist()
    N = len(units)
    if N < 2:
        return 0.0, 1.0

    # Compute pairwise correlation on common time observations
    sum_rho = 0.0
    pair_count = 0
    T_ij_sum = 0.0

    for i in range(N):
        for j in range(i + 1, N):
            s1 = piv[units[i]]
            s2 = piv[units[j]]
            valid = s1.notna() & s2.notna()
            T_ij = valid.sum()
            if T_ij > 3:
                r = np.corrcoef(s1[valid], s2[valid])[0, 1]
                if np.isfinite(r):
                    sum_rho += np.sqrt(T_ij) * r
                    pair_count += 1
                    T_ij_sum += T_ij

    if pair_count == 0:
        return 0.0, 1.0

    cd_stat = np.sqrt(2.0 / (N * (N - 1))) * sum_rho
    p_value = float(2.0 * (1.0 - norm.cdf(np.abs(cd_stat))))
    return float(cd_stat), float(p_value)


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


def individual_granger_wald(y_series: np.ndarray, x_series: np.ndarray, lags: int) -> Optional[Tuple[float, int]]:
    """Compute individual Wald statistic for unit i testing x -> y with K lags and effective sample T_i."""
    T = len(y_series)
    if T <= 2 * lags + 5:
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

    w_i = float(lags * f_stat)
    T_eff = T - lags
    return w_i, T_eff


def dumitrescu_hurlin_test(
    panel_df: pd.DataFrame,
    cause_col: str,
    effect_col: str,
    max_lag: int = 2,
    min_obs_per_country: int = 15,
    difference_if_nonstationary: bool = True,
    cause_order: Optional[str] = None,
    effect_order: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Dumitrescu-Hurlin (2012) Heterogeneous Panel Granger Causality Test.
    Implements the EXACT finite-T standardized statistic Z_tilde (equation 6).
    
    When cause_order / effect_order are supplied (e.g. from Pesaran CIPS panel unit root tests),
    differencing is applied globally to prevent univariate ADF size distortions under CSD.
    """
    countries = panel_df["iso3"].unique()
    w_stats: List[float] = []
    t_samples: List[int] = []
    n_differenced_cause = 0
    n_differenced_effect = 0
    n_tested = 0

    diff_cause_globally = (cause_order == "I(1)")
    diff_effect_globally = (effect_order == "I(1)")

    for c in countries:
        c_data = panel_df[panel_df["iso3"] == c].sort_values("year")[[cause_col, effect_col]].dropna()
        if len(c_data) < max(min_obs_per_country, max_lag * 3):
            continue

        y_s = c_data[effect_col].values.astype(np.float64)
        x_s = c_data[cause_col].values.astype(np.float64)

        if cause_order is not None or effect_order is not None:
            # CIPS-governed global integration orders
            if diff_cause_globally and diff_effect_globally:
                x_s = np.diff(x_s)
                y_s = np.diff(y_s)
                n_differenced_cause += 1
                n_differenced_effect += 1
            elif diff_cause_globally:
                x_s = np.diff(x_s)
                y_s = y_s[1:]
                n_differenced_cause += 1
            elif diff_effect_globally:
                y_s = np.diff(y_s)
                x_s = x_s[1:]
                n_differenced_effect += 1

            if len(y_s) < max(min_obs_per_country - 2, max_lag * 3) or len(x_s) != len(y_s):
                continue
        elif difference_if_nonstationary:
            # Univariate ADF unit-root pre-test on original cause & effect series
            cause_diff_needed = False
            effect_diff_needed = False

            if len(x_s) >= 20 and np.std(x_s) > 1e-10:
                try:
                    adf_x = adfuller(x_s, maxlag=min(max_lag, int(len(x_s) / 5)), autolag="AIC")
                    if adf_x[1] > 0.05:  # Fail to reject unit root -> I(1)
                        cause_diff_needed = True
                except Exception:
                    pass

            if len(y_s) >= 20 and np.std(y_s) > 1e-10:
                try:
                    adf_y = adfuller(y_s, maxlag=min(max_lag, int(len(y_s) / 5)), autolag="AIC")
                    if adf_y[1] > 0.05:  # Fail to reject unit root -> I(1)
                        effect_diff_needed = True
                except Exception:
                    pass

            if cause_diff_needed and effect_diff_needed:
                x_s = np.diff(x_s)
                y_s = np.diff(y_s)
                n_differenced_cause += 1
                n_differenced_effect += 1
            elif cause_diff_needed:
                x_s = np.diff(x_s)
                y_s = y_s[1:]
                n_differenced_cause += 1
            elif effect_diff_needed:
                y_s = np.diff(y_s)
                x_s = x_s[1:]
                n_differenced_effect += 1

            # After differencing, re-check minimum length
            if len(y_s) < max(min_obs_per_country - 2, max_lag * 3) or len(x_s) != len(y_s):
                continue

        n_tested += 1
        res = individual_granger_wald(y_s, x_s, lags=max_lag)
        if res is not None:
            w_i, t_i = res
            if np.isfinite(w_i):
                w_stats.append(w_i)
                t_samples.append(t_i)

    n_countries = len(w_stats)
    if n_countries < 5:
        return {
            "w_bar": 0.0,
            "z_tilde": 0.0,
            "z_asymptotic": 0.0,
            "p_value": 1.0,
            "n_countries": n_countries,
            "n_tested": n_tested,
            "n_differenced_cause": n_differenced_cause,
            "n_differenced_effect": n_differenced_effect,
            "significant": False
        }

    k = float(max_lag)
    w_bar = float(np.mean(w_stats))
    t_bar = float(np.mean(t_samples))
    z_asymp = np.sqrt(n_countries / (2.0 * k)) * (w_bar - k)

    # EXACT Unbalanced Fixed-T Standardized Z_tilde (Dumitrescu & Hurlin 2012, Section 2.3, eq. 9)
    # E[W_i] = K * (T_i - 2K - 1) / (T_i - 2K - 3)
    # Var[W_i] = 2K * (T_i - 2K - 1)^2 * (T_i - K - 3) / (((T_i - 2K - 3)^2) * (T_i - 2K - 5))
    sum_diff = 0.0
    sum_var = 0.0
    valid_unbalanced = 0

    for w_i, t_i in zip(w_stats, t_samples):
        if t_i > 2 * k + 5:
            e_wi = k * (t_i - 2 * k - 1) / (t_i - 2 * k - 3)
            var_wi = 2.0 * k * ((t_i - 2 * k - 1) ** 2) * (t_i - k - 3) / (((t_i - 2 * k - 3) ** 2) * (t_i - 2 * k - 5))
            sum_diff += (w_i - e_wi)
            sum_var += var_wi
            valid_unbalanced += 1

    if valid_unbalanced >= 5 and sum_var > 0:
        z_tilde = sum_diff / np.sqrt(sum_var)
    else:
        z_tilde = z_asymp

    p_val_tilde = float(2.0 * (1.0 - norm.cdf(np.abs(z_tilde))))

    return {
        "w_bar": round(w_bar, 4),
        "z_tilde": round(float(z_tilde), 4),
        "z_asymptotic": round(float(z_asymp), 4),
        "p_value": p_val_tilde,
        "n_countries": n_countries,
        "n_tested": n_tested,
        "n_differenced_cause": n_differenced_cause,
        "n_differenced_effect": n_differenced_effect,
        "t_avg": round(t_bar, 1),
        "significant": bool(p_val_tilde < 0.05)
    }


def _fast_ols_ssr(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
    """Fast OLS solver and SSR computation for small dimensional parameter spaces."""
    XtX = X.T @ X
    Xty = X.T @ y
    try:
        b = np.linalg.solve(XtX, Xty)
        pred = X @ b
        ssr = float(np.sum((y - pred) ** 2))
        return b, ssr
    except np.linalg.LinAlgError:
        b, ssr_arr, _, _ = np.linalg.lstsq(X, y, rcond=None)
        ssr = float(ssr_arr[0]) if len(ssr_arr) > 0 else float(np.sum((y - X @ b) ** 2))
        return b, ssr


def dumitrescu_hurlin_bootstrap_test(
    panel_df: pd.DataFrame,
    cause_col: str,
    effect_col: str,
    max_lag: int = 2,
    min_obs_per_country: int = 20,
    n_boot: int = 1000,
    cause_order: Optional[str] = None,
    effect_order: Optional[str] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    r"""
    Cross-Sectional Residual Resampling Panel Bootstrap for Dumitrescu-Hurlin (2012) Test.

    Directly addresses Cross-Sectional Dependence (Pesaran CD > 85) by vector-resampling
    the cross-sectional error matrix under H0, preserving the empirical contemporaneous
    covariance structure Sigma_N = E[e_t e_t'] across sovereigns (Emirmahmutoglu & Kose 2011;
    Lopez & Weber 2017).

    Parameters
    ----------
    panel_df : pd.DataFrame
        Panel dataframe with columns ['iso3', 'year', cause_col, effect_col].
    cause_col : str
        Potential Granger-causing variable.
    effect_col : str
        Target dependent variable (e.g. GDP growth).
    max_lag : int
        Lag order (K).
    min_obs_per_country : int
        Minimum time series length per sovereign.
    n_boot : int
        Number of bootstrap replications (default 1000).
    cause_order, effect_order : Optional[str]
        CIPS-governed integration orders ('I(0)' or 'I(1)').
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    Dict containing observed Wald & Z_tilde, bootstrap mean/SD of Z_tilde, empirical
    bootstrap critical values (90%, 95%, 99%), and exact bootstrap p-value under CSD.
    """
    rng = np.random.default_rng(random_state)
    clean = panel_df[["iso3", "year", cause_col, effect_col]].dropna().sort_values(["iso3", "year"])
    years_set = sorted(clean["year"].unique())
    year_to_idx = {yr: i for i, yr in enumerate(years_set)}
    T_total = len(years_set)

    diff_cause = (cause_order == "I(1)")
    diff_effect = (effect_order == "I(1)")

    c_list = []
    for iso, grp in clean.groupby("iso3"):
        if len(grp) < min_obs_per_country:
            continue
        y = grp[effect_col].values.astype(np.float64)
        x = grp[cause_col].values.astype(np.float64)
        yrs = grp["year"].values.astype(int)

        if diff_cause and diff_effect:
            x = np.diff(x)
            y = np.diff(y)
            yrs = yrs[1:]
        elif diff_cause:
            x = np.diff(x)
            y = y[1:]
            yrs = yrs[1:]
        elif diff_effect:
            y = np.diff(y)
            x = x[1:]
            yrs = yrs[1:]

        T_i = len(y)
        if T_i <= 2 * max_lag + 5:
            continue

        N_eff = T_i - max_lag
        X_res = np.ones((N_eff, max_lag + 1))
        for k_idx in range(1, max_lag + 1):
            X_res[:, k_idx] = y[max_lag - k_idx : T_i - k_idx]
        y_dep = y[max_lag:]

        beta_res, ssr_res = _fast_ols_ssr(X_res, y_dep)
        resids = y_dep - X_res @ beta_res
        resids = resids - np.mean(resids)

        X_unres = np.zeros((N_eff, 2 * max_lag + 1))
        X_unres[:, 0] = 1.0
        for k_idx in range(1, max_lag + 1):
            X_unres[:, k_idx] = y[max_lag - k_idx : T_i - k_idx]
            X_unres[:, max_lag + k_idx] = x[max_lag - k_idx : T_i - k_idx]

        beta_unres, ssr_unres = _fast_ols_ssr(X_unres, y_dep)
        df_unres = N_eff - (2 * max_lag + 1)

        if df_unres > 0 and ssr_unres > 1e-12:
            f_stat = ((ssr_res - ssr_unres) / max_lag) / (ssr_unres / df_unres)
            w_obs = max_lag * f_stat
        else:
            w_obs = 0.0

        c_list.append({
            "iso": iso, "y": y, "x": x, "yrs": yrs, "T_i": T_i,
            "beta_res": beta_res, "resids": resids, "w_obs": w_obs,
            "N_eff": N_eff
        })

    n_countries = len(c_list)
    if n_countries < 5:
        return {
            "w_bar_obs": 0.0, "z_tilde_obs": 0.0, "p_value_boot": 1.0,
            "boot_mean_z": 0.0, "boot_sd_z": 1.0, "cv_10_boot": 1.645, "cv_05_boot": 1.645, "cv_01_boot": 2.326,
            "n_countries": n_countries, "significant_boot_05": False
        }

    w_obs_arr = np.array([c["w_obs"] for c in c_list])
    w_bar_obs = float(np.mean(w_obs_arr))

    k = float(max_lag)
    sum_diff, sum_var = 0.0, 0.0
    for c in c_list:
        t_i = c["T_i"]
        e_wi = k * (t_i - 2 * k - 1) / (t_i - 2 * k - 3)
        var_wi = 2.0 * k * ((t_i - 2 * k - 1) ** 2) * (t_i - k - 3) / (((t_i - 2 * k - 3) ** 2) * (t_i - 2 * k - 5))
        sum_diff += (c["w_obs"] - e_wi)
        sum_var += var_wi

    z_tilde_obs = float(sum_diff / np.sqrt(sum_var)) if sum_var > 0 else 0.0

    # Form (T_total x N_countries) residual matrix for vector cross-sectional resampling
    res_mat = np.full((T_total, n_countries), np.nan)
    for j, c in enumerate(c_list):
        for t_idx, yr in enumerate(c["yrs"][max_lag:]):
            res_mat[year_to_idx[yr], j] = c["resids"][t_idx]

    boot_z_tildes = []
    boot_w_bars = []

    for b in range(n_boot):
        boot_year_idx = rng.choice(T_total, size=T_total, replace=True)
        sum_diff_b = 0.0
        boot_w_list = []

        for j, c in enumerate(c_list):
            T_i = c["T_i"]
            N_eff = c["N_eff"]
            c_boot_resids = np.zeros(N_eff)
            for t_idx, yr in enumerate(c["yrs"][max_lag:]):
                row = boot_year_idx[year_to_idx[yr]]
                val = res_mat[row, j]
                if np.isnan(val):
                    val = rng.choice(c["resids"])
                c_boot_resids[t_idx] = val

            # Generate bootstrap series recursively under H0
            y_boot = np.zeros(T_i)
            y_boot[:max_lag] = c["y"][:max_lag]
            for t in range(max_lag, T_i):
                y_boot[t] = c["beta_res"][0] + sum(c["beta_res"][lag] * y_boot[t - lag] for lag in range(1, max_lag + 1)) + c_boot_resids[t - max_lag]

            y_boot_dep = y_boot[max_lag:]

            # Restricted fit
            X_res = np.ones((N_eff, max_lag + 1))
            for lag in range(1, max_lag + 1):
                X_res[:, lag] = y_boot[max_lag - lag : T_i - lag]
            _, ssr_res_b = _fast_ols_ssr(X_res, y_boot_dep)

            # Unrestricted fit with original x
            X_unres = np.zeros((N_eff, 2 * max_lag + 1))
            X_unres[:, 0] = 1.0
            for lag in range(1, max_lag + 1):
                X_unres[:, lag] = y_boot[max_lag - lag : T_i - lag]
                X_unres[:, max_lag + lag] = c["x"][max_lag - lag : T_i - lag]

            _, ssr_unres_b = _fast_ols_ssr(X_unres, y_boot_dep)
            df_unres_b = N_eff - (2 * max_lag + 1)

            if df_unres_b > 0 and ssr_unres_b > 1e-12:
                f_b = ((ssr_res_b - ssr_unres_b) / max_lag) / (ssr_unres_b / df_unres_b)
                w_b = max_lag * f_b
            else:
                w_b = 0.0

            boot_w_list.append(w_b)
            e_wi = k * (T_i - 2 * k - 1) / (T_i - 2 * k - 3)
            sum_diff_b += (w_b - e_wi)

        boot_w_bars.append(float(np.mean(boot_w_list)))
        boot_z_tildes.append(float(sum_diff_b / np.sqrt(sum_var)) if sum_var > 0 else 0.0)

    p_boot_z = float(np.mean(np.array(boot_z_tildes) >= z_tilde_obs))
    cv_10_z = float(np.percentile(boot_z_tildes, 90))
    cv_05_z = float(np.percentile(boot_z_tildes, 95))
    cv_01_z = float(np.percentile(boot_z_tildes, 99))
    mean_boot_z = float(np.mean(boot_z_tildes))
    sd_boot_z = float(np.std(boot_z_tildes))

    return {
        "w_bar_obs": round(w_bar_obs, 4),
        "z_tilde_obs": round(z_tilde_obs, 4),
        "p_value_boot": p_boot_z,
        "boot_mean_z": round(mean_boot_z, 3),
        "boot_sd_z": round(sd_boot_z, 3),
        "cv_10_boot": round(cv_10_z, 3),
        "cv_05_boot": round(cv_05_z, 3),
        "cv_01_boot": round(cv_01_z, 3),
        "n_countries": n_countries,
        "significant_boot_05": bool(p_boot_z < 0.05)
    }


def cross_sectionally_augmented_granger_test(
    panel_df: pd.DataFrame,
    cause_col: str,
    effect_col: str,
    max_lag: int = 2,
    min_obs_per_country: int = 25,
    cause_order: Optional[str] = None,
    effect_order: Optional[str] = None,
) -> Dict[str, Any]:
    r"""
    Cross-Sectionally Augmented Panel Granger Non-Causality Test (Chudik & Pesaran 2015, 2016).

    Filters out unobserved global common factors (global business cycles, commodity shocks,
    global warming trends, international democratic waves) by augmenting country-level regressions
    with contemporaneous and lagged cross-sectional averages:
      y_{i,t} = \alpha_i + \sum_{k=1}^K \gamma_{i,k} y_{i,t-k}
                         + \sum_{k=1}^K \beta_{i,k} x_{i,t-k}
                         + \sum_{k=0}^K \delta_{i,k} \bar{y}_{t-k}
                         + \sum_{k=0}^K \psi_{i,k} \bar{x}_{t-k} + \varepsilon_{i,t}

    Tests H0: \beta_{i,1} = ... = \beta_{i,K} = 0 for all i.
    """
    clean = panel_df[["iso3", "year", cause_col, effect_col]].dropna().sort_values(["iso3", "year"])

    if cause_order == "I(1)":
        clean[f"{cause_col}_touse"] = clean.groupby("iso3")[cause_col].diff()
    else:
        clean[f"{cause_col}_touse"] = clean[cause_col]

    if effect_order == "I(1)":
        clean[f"{effect_col}_touse"] = clean.groupby("iso3")[effect_col].diff()
    else:
        clean[f"{effect_col}_touse"] = clean[effect_col]

    clean = clean.dropna(subset=[f"{cause_col}_touse", f"{effect_col}_touse"])
    year_means_y = clean.groupby("year")[f"{effect_col}_touse"].mean()
    year_means_x = clean.groupby("year")[f"{cause_col}_touse"].mean()
    clean["y_bar"] = clean["year"].map(year_means_y)
    clean["x_bar"] = clean["year"].map(year_means_x)

    w_stats_cs = []
    t_samples_cs = []

    for iso, grp in clean.groupby("iso3"):
        if len(grp) < min_obs_per_country:
            continue
        y = grp[f"{effect_col}_touse"].values.astype(np.float64)
        x = grp[f"{cause_col}_touse"].values.astype(np.float64)
        y_bar = grp["y_bar"].values.astype(np.float64)
        x_bar = grp["x_bar"].values.astype(np.float64)

        T_i = len(y)
        if T_i <= 3 * max_lag + 7:
            continue

        N_eff = T_i - max_lag
        y_dep = y[max_lag:]

        common_cols = [np.ones(N_eff)]
        for k in range(1, max_lag + 1):
            common_cols.append(y[max_lag - k : T_i - k])
        for k in range(0, max_lag + 1):
            common_cols.append(y_bar[max_lag - k : T_i - k])
            common_cols.append(x_bar[max_lag - k : T_i - k])

        X_res_cs = np.column_stack(common_cols)
        unres_cols = list(common_cols)
        for k in range(1, max_lag + 1):
            unres_cols.append(x[max_lag - k : T_i - k])

        X_unres_cs = np.column_stack(unres_cols)
        p_res = X_res_cs.shape[1]
        p_unres = X_unres_cs.shape[1]

        if N_eff <= p_unres + 2:
            continue

        try:
            b_res, ssr_res = _fast_ols_ssr(X_res_cs, y_dep)
            b_unres, ssr_unres = _fast_ols_ssr(X_unres_cs, y_dep)
            df_unres = N_eff - p_unres

            if df_unres > 0 and ssr_unres > 1e-12:
                f_stat = ((ssr_res - ssr_unres) / max_lag) / (ssr_unres / df_unres)
                w_i = max_lag * f_stat
                if np.isfinite(w_i):
                    w_stats_cs.append(w_i)
                    t_samples_cs.append(T_i)
        except Exception:
            continue

    n_cs = len(w_stats_cs)
    if n_cs < 5:
        return {
            "w_bar_cs": 0.0, "z_tilde_cs": 0.0, "p_value_cs": 1.0,
            "n_countries": n_cs, "significant_cs_05": False
        }

    w_bar_cs = float(np.mean(w_stats_cs))
    z_asymp_cs = float(np.sqrt(n_cs / (2.0 * max_lag)) * (w_bar_cs - max_lag))

    k = float(max_lag)
    sum_diff, sum_var = 0.0, 0.0
    for w_i, t_i in zip(w_stats_cs, t_samples_cs):
        p_extra = 2 * (max_lag + 1)
        t_adj = t_i - p_extra
        if t_adj > 2 * k + 5:
            e_wi = k * (t_adj - 2 * k - 1) / (t_adj - 2 * k - 3)
            var_wi = 2.0 * k * ((t_adj - 2 * k - 1) ** 2) * (t_adj - k - 3) / (((t_adj - 2 * k - 3) ** 2) * (t_adj - 2 * k - 5))
            sum_diff += (w_i - e_wi)
            sum_var += var_wi

    z_tilde_cs = float(sum_diff / np.sqrt(sum_var)) if sum_var > 0 else z_asymp_cs
    p_cs = float(2.0 * (1.0 - norm.cdf(np.abs(z_tilde_cs))))

    return {
        "w_bar_cs": round(w_bar_cs, 4),
        "z_tilde_cs": round(z_tilde_cs, 4),
        "p_value_cs": p_cs,
        "n_countries": n_cs,
        "significant_cs_05": bool(p_cs < 0.05)
    }


def holm_bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[Dict[str, Any]]:
    """Apply Holm-Bonferroni FWER stepdown correction across m hypotheses."""
    m = len(p_values)
    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
    results = [None] * m

    for rank, (orig_idx, p_val) in enumerate(indexed_p):
        threshold = alpha / (m - rank)
        is_sig = p_val <= threshold
        results[orig_idx] = {
            "rank": rank + 1,
            "p_value_raw": p_val,
            "p_value_adj": min(1.0, p_val * (m - rank)),
            "threshold": threshold,
            "significant": is_sig
        }
    return results


def pesaran_cadf_individual(
    y_i: np.ndarray,
    y_bar: np.ndarray,
    lags: int = 1,
) -> Optional[float]:
    r"""
    Individual Cross-Sectionally Augmented Dickey-Fuller (CADF) regression (Pesaran 2007).

    Estimates:
      \Delta y_{i,t} = \alpha_i + b_i y_{i,t-1} + c_i \bar{y}_{t-1}
                       + \sum_{j=1}^p d_{ij} \Delta y_{i,t-j}
                       + \sum_{j=0}^p f_{ij} \Delta \bar{y}_{t-j} + \varepsilon_{i,t}

    Returns the t-ratio on b_i: CADF_i = \hat{b}_i / SE(\hat{b}_i).
    """
    T = len(y_i)
    if T <= 2 * lags + 6 or len(y_bar) != T:
        return None

    dy_i = np.diff(y_i)
    dy_bar = np.diff(y_bar)
    y_i_lag1 = y_i[:-1]
    y_bar_lag1 = y_bar[:-1]

    # Aligned range: from lags to T-1 (which corresponds to length T - 1 - lags)
    start = lags
    end = T - 1

    dep = dy_i[start:end]
    n_reg = len(dep)
    if n_reg < lags + 5:
        return None

    # Regressors:
    # 1. Constant
    # 2. y_{i, t-1}
    # 3. \bar{y}_{t-1}
    # 4. \Delta \bar{y}_t
    # 5. \Delta y_{i, t-j} for j=1..lags
    # 6. \Delta \bar{y}_{t-j} for j=1..lags
    X_list = [
        np.ones(n_reg),
        y_i_lag1[start:end],
        y_bar_lag1[start:end],
        dy_bar[start:end],
    ]

    for j in range(1, lags + 1):
        X_list.append(dy_i[start - j : end - j])
        X_list.append(dy_bar[start - j : end - j])

    X_mat = np.column_stack(X_list)
    p_dim = X_mat.shape[1]

    if n_reg <= p_dim:
        return None

    try:
        beta, residuals, rank, _ = np.linalg.lstsq(X_mat, dep, rcond=None)
        if rank < p_dim:
            return None

        if len(residuals) > 0:
            ssr = float(residuals[0])
        else:
            pred = X_mat @ beta
            ssr = float(np.sum((dep - pred) ** 2))

        df = n_reg - p_dim
        if df <= 0 or ssr <= 1e-12:
            return None

        sigma2 = ssr / df
        cov_matrix = np.linalg.inv(X_mat.T @ X_mat) * sigma2
        se_b1 = np.sqrt(max(1e-12, cov_matrix[1, 1]))

        t_stat = float(beta[1] / se_b1)
        if not np.isfinite(t_stat):
            return None
        return t_stat
    except Exception:
        return None


def pesaran_cips_test(
    panel_df: pd.DataFrame,
    variable_col: str,
    iso_col: str = "iso3",
    year_col: str = "year",
    lags: int = 1,
    min_obs_per_country: int = 20,
    truncated: bool = True,
) -> Dict[str, Any]:
    r"""
    Pesaran (2007) Cross-Sectionally Augmented IPS (CIPS) Panel Unit Root Test.

    Tests the null hypothesis:
      H0: b_i = 0 for all i (unit root / non-stationary across all cross-sections)
    against the heterogeneous alternative:
      H1: b_i < 0 for at least a fraction of cross-sections (stationary).

    Parameters
    ----------
    panel_df : pd.DataFrame
        Panel dataframe carrying iso_col, year_col, and variable_col.
    variable_col : str
        Column to test.
    lags : int
        Number of augmented lags (p).
    min_obs_per_country : int
        Minimum time periods required per country.
    truncated : bool
        If True, applies Pesaran's CADF truncation [-6.19, 2.61] to prevent finite-sample outlier distortion.

    Returns
    -------
    Dict containing:
      cips_stat : float (mean CADF statistic across countries)
      n_countries : int
      critical_values_1pct, 5pct, 10pct : float
      is_stationary_5pct : bool (True if CIPS < CV_5pct)
      order_of_integration : str ('I(0)' or 'I(1)')
    """
    clean = panel_df[[iso_col, year_col, variable_col]].dropna().sort_values([iso_col, year_col])
    if len(clean) < 100:
        return {"cips_stat": 0.0, "n_countries": 0, "order_of_integration": "Inconclusive"}

    # Compute cross-sectional average for each year: \bar{y}_t
    year_means = clean.groupby(year_col)[variable_col].mean()
    clean = clean.copy()
    clean["y_bar"] = clean[year_col].map(year_means)

    cadf_stats = []
    countries = clean[iso_col].unique()

    for iso in countries:
        c_sub = clean[clean[iso_col] == iso].sort_values(year_col)
        if len(c_sub) < min_obs_per_country:
            continue

        y_i = c_sub[variable_col].to_numpy(dtype=np.float64)
        y_bar = c_sub["y_bar"].to_numpy(dtype=np.float64)

        stat = pesaran_cadf_individual(y_i, y_bar, lags=lags)
        if stat is not None and np.isfinite(stat):
            if truncated:
                # Pesaran (2007) truncation bounds for Case II (with intercept):
                # k1 = -6.19 (for T=30,50) to avoid extreme sample leverage
                stat = float(np.clip(stat, -6.19, 2.61))
            cadf_stats.append(stat)

    n_valid = len(cadf_stats)
    if n_valid < 5:
        return {"cips_stat": 0.0, "n_countries": n_valid, "order_of_integration": "Inconclusive"}

    cips_stat = float(np.mean(cadf_stats))

    # Pesaran (2007) Table IIb Critical Values (Case II: Intercept only, N > 100, T >= 30-50):
    # 10% CV: -2.04
    #  5% CV: -2.10
    #  1% CV: -2.22
    cv_10 = -2.04
    cv_05 = -2.10
    cv_01 = -2.22

    # Rejection rule: CIPS < Critical Value -> Reject H0 (Series is Stationary I(0))
    # Fail to reject: CIPS >= Critical Value -> Retain H0 (Series is Non-Stationary I(1))
    is_stationary = bool(cips_stat < cv_05)
    order = "I(0) Stationary" if is_stationary else "I(1) Unit Root"

    return {
        "variable": variable_col,
        "cips_stat": round(cips_stat, 4),
        "n_countries": n_valid,
        "cv_10pct": cv_10,
        "cv_05pct": cv_05,
        "cv_01pct": cv_01,
        "is_stationary_05pct": is_stationary,
        "order_of_integration": order,
        "p_val_approx": "< 0.01" if cips_stat < cv_01 else ("< 0.05" if cips_stat < cv_05 else (
            "< 0.10" if cips_stat < cv_10 else ">= 0.10 (Unit Root)"
        ))
    }


def pedroni_panel_cointegration_test(
    panel_df: pd.DataFrame,
    y_col: str,
    x_col: str,
    lags: int = 2,
    iso_col: str = "iso3",
    year_col: str = "year",
    min_obs: int = 20,
) -> Dict[str, Any]:
    r"""
    Pedroni (1999, 2004) Residual-Based Panel Cointegration Test (Group-Mean ADF Statistic).

    Tests the null hypothesis of No Cointegration (H0: e_{i,t} is I(1)) against the heterogeneous
    alternative of Cointegration across sovereigns.

    Procedure:
      1. Estimate country-by-country static cointegrating regressions:
         y_{i,t} = \alpha_i + \beta_i x_{i,t} + e_{i,t}
      2. Run ADF regression on residuals:
         \Delta e_{i,t} = \rho_i e_{i,t-1} + \sum_{k=1}^K \gamma_{ik} \Delta e_{i,t-k} + v_{i,t}
      3. Compute individual ADF t-ratios t_i = \hat{\rho}_i / SE(\hat{\rho}_i)
      4. Compute standardized Group-Mean ADF statistic:
         Z_{P-ADF} = \frac{1}{\sqrt{N}} \sum_{i=1}^N t_i \sim \mathcal{N}(0, 1)

    Rejection Rule:
      Z_{P-ADF} < -1.645 -> Reject H0 (Series are Cointegrated, Error Correction Term needed)
      Z_{P-ADF} >= -1.645 -> Fail to reject H0 (No Cointegration, Differencing is econometrically valid)
    """
    clean = panel_df[[iso_col, year_col, y_col, x_col]].dropna().sort_values([iso_col, year_col])
    countries = clean[iso_col].unique()

    t_stats = []
    for iso in countries:
        c_sub = clean[clean[iso_col] == iso].sort_values(year_col)
        if len(c_sub) < min_obs:
            continue

        y_i = c_sub[y_col].to_numpy(dtype=np.float64)
        x_i = c_sub[x_col].to_numpy(dtype=np.float64)
        if np.std(x_i) < 1e-10 or np.std(y_i) < 1e-10:
            continue

        # 1. Static cointegrating regression
        X_static = np.column_stack([np.ones(len(x_i)), x_i])
        beta_static, res_static, rank, _ = np.linalg.lstsq(X_static, y_i, rcond=None)
        if rank < 2:
            continue
        e_i = y_i - X_static @ beta_static

        # 2. ADF on residuals (no constant since e_i is mean zero by OLS)
        T_e = len(e_i)
        if T_e <= lags + 3:
            continue

        delta_e = np.diff(e_i)
        M = T_e - 1 - lags
        if M <= lags + 2:
            continue

        dep_delta = delta_e[lags : lags + M]
        e_lag1 = e_i[lags : lags + M]

        reg_cols = [e_lag1]
        for k in range(1, lags + 1):
            reg_cols.append(delta_e[lags - k : lags - k + M])

        X_adf = np.column_stack(reg_cols)
        if len(dep_delta) <= X_adf.shape[1]:
            continue

        beta_adf, res_adf, rank_adf, _ = np.linalg.lstsq(X_adf, dep_delta, rcond=None)
        if rank_adf < X_adf.shape[1]:
            continue

        y_fit = X_adf @ beta_adf
        ssr = float(np.sum((dep_delta - y_fit) ** 2))
        df_adf = len(dep_delta) - X_adf.shape[1]
        if df_adf <= 0:
            continue

        s2 = ssr / df_adf
        try:
            cov_mat = s2 * np.linalg.inv(X_adf.T @ X_adf)
            se_rho = np.sqrt(max(1e-12, cov_mat[0, 0]))
            t_i = float(beta_adf[0] / se_rho)
            if np.isfinite(t_i):
                t_stats.append(t_i)
        except np.linalg.LinAlgError:
            continue

    n_countries = len(t_stats)
    if n_countries < 5:
        return {
            "y_variable": y_col,
            "x_variable": x_col,
            "z_group_adf": 0.0,
            "n_countries": n_countries,
            "p_value": 1.0,
            "cointegrated_5pct": False,
            "verdict": "Inconclusive (Insufficient Countries)"
        }

    # Mean individual ADF and standardized group statistic
    t_bar = float(np.mean(t_stats))
    # Pedroni asymptotic adjustment parameters for single regressor with intercept (Pedroni 1999 Table 2):
    # E[t_i] \approx -2.0, Var[t_i] \approx 1.0
    z_stat = float(np.sqrt(n_countries) * (t_bar - (-2.0)) / 1.0)
    p_value = float(norm.cdf(z_stat))  # One-sided lower tail test

    cointegrated = bool(z_stat < -1.645)

    return {
        "y_variable": y_col,
        "x_variable": x_col,
        "t_bar": round(t_bar, 4),
        "z_group_adf": round(z_stat, 4),
        "n_countries": n_countries,
        "p_value": float(p_value),
        "cointegrated_5pct": cointegrated,
        "verdict": "Reject H0: Cointegrated" if cointegrated else "Fail to Reject H0: No Cointegration (Differencing Valid)"
    }


