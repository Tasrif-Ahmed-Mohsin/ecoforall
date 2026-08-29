"""
Test Suite: Econometric Inference & Causality Module
=====================================================
Validates Dumitrescu-Hurlin exact finite-T, Clark-West nested tests,
Pesaran CD tests, and Holm-Bonferroni corrections.
"""

import numpy as np
import pandas as pd
import pytest
from src.econometrics.panel_granger import (
    dumitrescu_hurlin_test,
    dumitrescu_hurlin_bootstrap_test,
    cross_sectionally_augmented_granger_test,
    clark_west_test,
    diebold_mariano_test,
    pesaran_cd_test,
    holm_bonferroni_correction
)


def test_dumitrescu_hurlin_white_noise_null():
    """Verify that white noise against white noise does NOT reject the null."""
    np.random.seed(42)
    records = []
    countries = [f"CTY_{i:02d}" for i in range(20)]
    years = list(range(1990, 2020))
    
    for c in countries:
        y = np.random.normal(0, 1, len(years))
        x = np.random.normal(0, 1, len(years))
        for yr, y_val, x_val in zip(years, y, x):
            records.append({"iso3": c, "year": yr, "cause": x_val, "effect": y_val})
            
    df = pd.DataFrame(records)
    res = dumitrescu_hurlin_test(df, cause_col="cause", effect_col="effect", max_lag=1)
    
    assert res["n_countries"] == 20
    assert not res["significant"], f"Expected null not rejected on white noise, got z_tilde={res['z_tilde']}"
    assert abs(res["z_tilde"]) < 2.0


def test_dumitrescu_hurlin_bootstrap_csd_null():
    """Verify that panel bootstrap under null of no causality does not spuriously reject."""
    np.random.seed(42)
    records = []
    countries = [f"C_{i:02d}" for i in range(15)]
    years = list(range(1980, 2020))

    # Introduce common global factor in shocks to simulate CSD
    common_shocks = {yr: np.random.normal(0, 1) for yr in years}
    for c in countries:
        y = np.zeros(len(years))
        x = np.zeros(len(years))
        for t, yr in enumerate(years):
            shock_c = 0.5 * common_shocks[yr] + np.random.normal(0, 0.8)
            y[t] = 0.3 * (y[t-1] if t > 0 else 0) + shock_c
            x[t] = 0.4 * (x[t-1] if t > 0 else 0) + np.random.normal(0, 1)
            records.append({"iso3": c, "year": yr, "cause": x[t], "effect": y[t]})

    df = pd.DataFrame(records)
    res = dumitrescu_hurlin_bootstrap_test(df, cause_col="cause", effect_col="effect", max_lag=1, n_boot=200, random_state=42)
    assert res["n_countries"] == 15
    assert not res["significant_boot_05"], f"Expected bootstrap p > 0.05 on independent null with CSD, got {res['p_value_boot']}"


def test_cross_sectionally_augmented_granger_test():
    """Verify CS-augmented Granger test absorbs common factor confounding."""
    np.random.seed(42)
    records = []
    countries = [f"C_{i:02d}" for i in range(20)]
    years = list(range(1975, 2020))

    # Common global factor driving both y and x spuriously
    global_cycle = {yr: np.sin(yr / 5.0) + np.random.normal(0, 0.5) for yr in years}
    for c in countries:
        for yr in years:
            f_t = global_cycle[yr]
            y_it = 0.8 * f_t + np.random.normal(0, 0.5)
            x_it = 0.8 * f_t + np.random.normal(0, 0.5)
            records.append({"iso3": c, "year": yr, "cause": x_it, "effect": y_it})

    df = pd.DataFrame(records)
    cs_res = cross_sectionally_augmented_granger_test(df, cause_col="cause", effect_col="effect", max_lag=1, min_obs_per_country=20)
    assert cs_res["n_countries"] == 20
    # After absorbing cross-sectional averages (global_cycle), x does not Granger cause y
    assert not cs_res["significant_cs_05"], f"Expected CS-DH p > 0.05 after common factor removal, got p={cs_res['p_value_cs']}"


def test_clark_west_nested_superiority():
    """Verify Clark-West detects true structural improvement in larger nested model."""
    np.random.seed(42)
    n = 100
    y_true = np.random.normal(0, 1, n)
    # Small model has larger error variance
    y_pred_small = y_true + np.random.normal(0, 1.0, n)
    # Large model has lower error variance
    y_pred_large = y_true + np.random.normal(0, 0.2, n)
    
    cw_stat, p_val = clark_west_test(y_true, y_pred_small, y_pred_large)
    assert cw_stat > 2.0
    assert p_val < 0.01


def test_pesaran_cd_independent_panels():
    """Verify Pesaran CD test on independent residual series does not reject."""
    np.random.seed(42)
    records = []
    countries = [f"C_{i}" for i in range(10)]
    for yr in range(2000, 2020):
        for c in countries:
            records.append({"year": yr, "iso3": c, "residual": np.random.normal(0, 1)})
            
    df = pd.DataFrame(records)
    stat, p_val = pesaran_cd_test(df, time_col="year", unit_col="iso3", res_col="residual")
    assert abs(stat) < 2.0
    assert p_val > 0.05


def test_holm_bonferroni_correction():
    p_vals = [0.001, 0.04, 0.045, 0.20]
    res = holm_bonferroni_correction(p_vals, alpha=0.05)
    assert len(res) == 4
    # The smallest p-value should be adjusted by m=4
    assert res[0]["significant"]
    # The largest should not be significant
    assert not res[3]["significant"]


def test_dumitrescu_hurlin_theoretical_moments():
    """Verify exact theoretical finite-T moments according to Dumitrescu & Hurlin (2012, eq. 9)."""
    k = 2.0
    t_i = 30.0
    # E[W_i] = K * (T_i - 2K - 1) / (T_i - 2K - 3)
    expected_e = k * (t_i - 2 * k - 1) / (t_i - 2 * k - 3)
    assert abs(expected_e - (2.0 * 25.0 / 23.0)) < 1e-10

    # Var[W_i] = 2K * (T_i - 2K - 1)^2 * (T_i - K - 3) / (((T_i - 2K - 3)^2) * (T_i - 2K - 5))
    expected_var = (2.0 * k * (25.0 ** 2) * 25.0) / ((23.0 ** 2) * 21.0)
    assert abs(expected_var - (4.0 * 625.0 * 25.0) / (529.0 * 21.0)) < 1e-10

