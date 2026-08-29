"""
Test Suite: Macroeconomic Baselines Module
==========================================
Validates PerCountryARForecaster, EqualWeightCombinationForecaster,
SingleDomainSpecialist, and DynamicFactorForecaster.

The AR tests are deliberately weighted toward the regressor contract. The previous
implementation fit on pct_change(gdp_pc_real) and predicted with gdp_pc_real_logret5,
applying rho_i to a series at 4.3x its fitted scale; that inflated baseline MAE by
20-55% and every reported lift was measured against it. The old test only called
predict_country() directly, so it could not observe the defect.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.macro_baselines import (
    PerCountryARForecaster,
    EqualWeightCombinationForecaster,
    SingleDomainSpecialist,
    DynamicFactorForecaster,
    add_lagged_growth,
    DEFAULT_LAG_GROWTH_COL,
)

RNG = np.random.default_rng(20260826)


def _panel(countries=("USA", "GBR", "DEU"), start=1980, end=2010) -> pd.DataFrame:
    rows = []
    for c in countries:
        gdp = 100.0
        for yr in range(start, end):
            gdp *= 1.0 + RNG.normal(0.02, 0.01)
            rows.append({"iso3": c, "year": yr, "gdp_pc_real": gdp})
    df = pd.DataFrame(rows)
    df = add_lagged_growth(df, level_col="gdp_pc_real")
    df["gdp_pc_growth_1y_fwd"] = (
        df.groupby("iso3")["gdp_pc_real"].shift(-1) / df["gdp_pc_real"] - 1.0
    )
    return df


# --------------------------------------------------------- regressor construction
def test_add_lagged_growth_matches_one_year_growth():
    df = _panel(countries=("USA",))
    d = df.dropna(subset=[DEFAULT_LAG_GROWTH_COL])
    expected = d["gdp_pc_real"].to_numpy() / d["gdp_pc_real"].shift(1).to_numpy()[
        : len(d)
    ]
    manual = df["gdp_pc_real"].pct_change().to_numpy()[1:]
    np.testing.assert_allclose(d[DEFAULT_LAG_GROWTH_COL].to_numpy(), manual, atol=1e-12)
    assert np.isnan(df[DEFAULT_LAG_GROWTH_COL].iloc[0])


def test_add_lagged_growth_nulls_non_consecutive_years():
    """A gap in the calendar must not be reported as a one-year growth rate."""
    df = pd.DataFrame({
        "iso3": ["USA"] * 3,
        "year": [1990, 1991, 1997],   # 6-year jump
        "gdp_pc_real": [100.0, 110.0, 200.0],
    })
    out = add_lagged_growth(df)
    vals = out[DEFAULT_LAG_GROWTH_COL].tolist()
    assert np.isnan(vals[0])
    assert vals[1] == pytest.approx(0.10)
    assert np.isnan(vals[2]), "growth across a 6-year gap must be NaN, not 0.818"


# ------------------------------------------------------------ regressor contract
def test_ar_predict_panel_rejects_a_different_regressor():
    """The exact defect that produced the +40% headline must now raise."""
    df = _panel()
    df["gdp_pc_real_logret5"] = np.log(
        df["gdp_pc_real"] / df.groupby("iso3")["gdp_pc_real"].shift(5)
    )
    model = PerCountryARForecaster(horizon=1).fit(df, target_col="gdp_pc_growth_1y_fwd")

    with pytest.raises(ValueError, match="regressor mismatch"):
        model.predict_panel(df, lag_growth_col="gdp_pc_real_logret5")


def test_ar_fit_requires_the_regressor_to_be_precomputed():
    df = _panel().drop(columns=[DEFAULT_LAG_GROWTH_COL])
    with pytest.raises(KeyError, match="add_lagged_growth"):
        PerCountryARForecaster(horizon=1).fit(df, target_col="gdp_pc_growth_1y_fwd")


def test_ar_predict_panel_requires_the_fitted_column_in_the_test_frame():
    df = _panel()
    model = PerCountryARForecaster(horizon=1).fit(df, target_col="gdp_pc_growth_1y_fwd")
    with pytest.raises(KeyError, match=DEFAULT_LAG_GROWTH_COL):
        model.predict_panel(df.drop(columns=[DEFAULT_LAG_GROWTH_COL]))


def test_ar_predict_panel_before_fit_raises():
    with pytest.raises(RuntimeError, match="before fit"):
        PerCountryARForecaster().predict_panel(_panel())


def test_ar_records_the_regressor_it_fitted():
    df = _panel()
    model = PerCountryARForecaster(horizon=1).fit(df, target_col="gdp_pc_growth_1y_fwd")
    assert model.lag_growth_col_ == DEFAULT_LAG_GROWTH_COL
    # Naming the fitted column explicitly is allowed; it asserts caller intent.
    np.testing.assert_allclose(
        model.predict_panel(df, lag_growth_col=DEFAULT_LAG_GROWTH_COL),
        model.predict_panel(df),
    )


# ----------------------------------------------------------------- AR behaviour
def test_ar_fit_predict_is_finite_and_clipped():
    df = _panel()
    model = PerCountryARForecaster(horizon=1).fit(df, target_col="gdp_pc_growth_1y_fwd")
    assert "USA" in model.country_models_
    preds = model.predict_panel(df)
    assert len(preds) == len(df)
    assert np.all(np.isfinite(preds))
    assert np.all(np.abs(preds) <= 0.5)


def test_ar_recovers_a_known_autoregression():
    """With y_{t+1} = 0.01 + 0.6 g_t and no noise, the pooled fit must recover (0.01, 0.6)."""
    rows = []
    for c in ("AAA", "BBB", "CCC"):
        gdp = 100.0
        for yr in range(1960, 2010):
            g = RNG.normal(0.02, 0.02)
            gdp *= 1.0 + g
            rows.append({"iso3": c, "year": yr, "gdp_pc_real": gdp})
    df = add_lagged_growth(pd.DataFrame(rows))
    df["y"] = 0.01 + 0.6 * df[DEFAULT_LAG_GROWTH_COL]

    model = PerCountryARForecaster(horizon=1, shrinkage_weight=0.0).fit(df, target_col="y")
    assert model.global_alpha_ == pytest.approx(0.01, abs=1e-8)
    assert model.global_rho_ == pytest.approx(0.60, abs=1e-8)


def test_ar_beats_a_scale_mismatched_regressor():
    """
    Regression guard for the fixed defect: predicting with a 5-year cumulative return
    where a 1-year rate was fitted must be materially worse. If this ever fails, the
    contract has been bypassed somewhere.
    """
    df = _panel(countries=("USA", "GBR", "DEU", "FRA", "ITA"), start=1960, end=2015)
    df["logret5"] = np.log(df["gdp_pc_real"] / df.groupby("iso3")["gdp_pc_real"].shift(5))
    d = df.dropna(subset=["gdp_pc_growth_1y_fwd", DEFAULT_LAG_GROWTH_COL, "logret5"])

    model = PerCountryARForecaster(horizon=1).fit(d, target_col="gdp_pc_growth_1y_fwd")
    y = d["gdp_pc_growth_1y_fwd"].to_numpy()

    correct = np.mean(np.abs(y - model.predict_panel(d)))
    # Reproduce the old code path by hand, since the class now refuses to do it.
    alpha = np.array([model.country_models_[i]["alpha"] for i in d["iso3"]])
    rho = np.array([model.country_models_[i]["rho"] for i in d["iso3"]])
    mismatched = np.mean(np.abs(y - np.clip(alpha + rho * d["logret5"].to_numpy(), -0.5, 0.5)))
    assert mismatched > correct * 1.2, (
        f"mismatched regressor should be far worse; got {mismatched:.5f} vs {correct:.5f}"
    )


# ------------------------------------------------------------- other baselines
def test_equal_weight_combination():
    p1 = np.array([0.02, 0.04, 0.06])
    p2 = np.array([0.04, 0.02, 0.00])
    np.testing.assert_allclose(
        EqualWeightCombinationForecaster.combine([p1, p2]), np.array([0.03, 0.03, 0.03])
    )


def test_single_domain_quarantine():
    df = pd.DataFrame({
        "eco_gdp": RNG.normal(0, 1, 50),
        "eco_cpi": RNG.normal(0, 1, 50),
        "pol_vdem": RNG.normal(0, 1, 50),
    })
    y = RNG.normal(0, 1, 50)
    spec = SingleDomainSpecialist(domain_name="Economy", domain_prefix="eco_").fit(df, y)
    assert set(spec.selected_columns_) == {"eco_gdp", "eco_cpi"}
    assert len(spec.predict(df)) == 50


def test_dynamic_factor_forecaster_runs_and_clips():
    X = RNG.normal(0, 1, (80, 12))
    y = X[:, 0] * 0.05 + RNG.normal(0, 0.01, 80)
    dfm = DynamicFactorForecaster(n_factors=3).fit(X, y)
    preds = dfm.predict(X)
    assert preds.shape == (80,)
    assert np.all(np.abs(preds) <= 0.5)
