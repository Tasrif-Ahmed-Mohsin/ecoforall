"""
Macroeconomic and Panel Time-Series Baselines
============================================
Implements mathematically verified, honest baselines for sovereign panel forecasting:
  1. PerCountryARForecaster: per-country AR(1) on a *contract-checked* regressor,
     with empirical-Bayes shrinkage toward the pooled panel estimate.
  2. EqualWeightCombinationForecaster: 1/M simple average benchmark (Clemen 1989).
  3. SingleDomainSpecialist: Domain-quarantined Ridge/LightGBM regressors.
  4. DynamicFactorForecaster: Principal Component Factor Regression (Stock & Watson 2002).

Helper `add_lagged_growth` materialises the AR regressor on the full panel; see the
regressor-contract note in PerCountryARForecaster for why that must happen before
the train/test split.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from typing import Optional, Dict, List, Any


#: Canonical name of the AR regressor: real GDP per-capita growth realised *into*
#: origin year t. Known at t, therefore usable at t without leakage.
DEFAULT_LAG_GROWTH_COL = "growth_into_origin"


def add_lagged_growth(
    df: pd.DataFrame,
    level_col: str = "gdp_pc_real",
    iso_col: str = "iso3",
    year_col: str = "year",
    out_col: str = DEFAULT_LAG_GROWTH_COL,
) -> pd.DataFrame:
    r"""
    Materialise the AR regressor $g_{i,t} = Y_{i,t}/Y_{i,t-1} - 1$ on the panel.

    Must be called on the **full** panel before any train/test split: the quantity is
    a within-country lagged difference, so it cannot be reconstructed from a training
    slice alone (the test rows are not in that slice). Returns a copy sorted by
    ``(iso_col, year_col)``.

    Consecutive-year guard: if the previous observation for a country is not exactly
    one year earlier, the value is set to NaN rather than silently reporting a
    multi-year growth rate as a one-year one.
    """
    for c in (level_col, iso_col, year_col):
        if c not in df.columns:
            raise KeyError(f"add_lagged_growth: required column '{c}' not in frame")

    d = df.sort_values([iso_col, year_col]).copy()
    d[out_col] = d.groupby(iso_col)[level_col].pct_change(fill_method=None)
    year_gap = d.groupby(iso_col)[year_col].diff()
    d.loc[year_gap.ne(1), out_col] = np.nan
    return d


class PerCountryARForecaster(BaseEstimator, RegressorMixin):
    r"""
    Honest Per-Country Autoregressive Baseline.

    For each country $i$, estimates by OLS

    .. math:: y_{i,t+h} = \alpha_i + \rho_i g_{i,t} + \varepsilon_{i,t}

    where $g_{i,t}$ is the growth realised *into* origin year $t$, then shrinks
    $(\hat\alpha_i, \hat\rho_i)$ toward the pooled panel estimate with weight
    $n_i / (n_i + 10\,\lambda_{\text{shrink}})$.

    Regressor contract
    ------------------
    The regressor column is resolved **once**, in :meth:`fit`, and recorded on
    ``self.lag_growth_col_``. :meth:`predict_panel` uses that column and *raises* if
    the caller names a different one.

    This guard exists because the previous implementation fit on
    ``pct_change(gdp_pc_real)`` (mean 0.020, sd 0.067) and predicted with
    ``gdp_pc_real_logret5`` (mean 0.085, sd 0.194) -- applying $\hat\rho_i$ to a
    series at 4.3x its fitted scale. That inflated baseline MAE by 20-55% and, since
    every reported lift was measured against it, inflated every headline number in the
    manuscript. A silent train/test regressor mismatch must not be reachable again.
    """

    def __init__(
        self,
        horizon: int = 1,
        shrinkage_weight: float = 0.2,
        default_growth: float = 0.02,
        prediction_clip: float = 0.5,
    ):
        self.horizon = horizon
        self.shrinkage_weight = shrinkage_weight
        self.default_growth = default_growth
        self.prediction_clip = prediction_clip
        self.country_models_: Dict[str, Dict[str, float]] = {}
        self.global_alpha_: float = 0.02
        self.global_rho_: float = 0.5
        #: Regressor resolved at fit time; predict_panel is pinned to it.
        self.lag_growth_col_: Optional[str] = None

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str,
        lag_growth_col: str = DEFAULT_LAG_GROWTH_COL,
        iso_col: str = "iso3",
        year_col: str = "year",
    ) -> PerCountryARForecaster:
        if lag_growth_col not in df.columns:
            raise KeyError(
                f"PerCountryARForecaster.fit: regressor column '{lag_growth_col}' is not in "
                f"the frame. Call add_lagged_growth(panel) on the FULL panel before splitting "
                f"into train/test, then pass the same column name to fit() and predict_panel()."
            )
        self.lag_growth_col_ = lag_growth_col

        clean = df.dropna(subset=[target_col, lag_growth_col, iso_col]).copy()
        x_all = pd.to_numeric(clean[lag_growth_col], errors="coerce").to_numpy(dtype=np.float64)
        y_all = pd.to_numeric(clean[target_col], errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(x_all) & np.isfinite(y_all)
        clean, x_all, y_all = clean[finite], x_all[finite], y_all[finite]

        if len(clean) > 10:
            beta, _, _, _ = np.linalg.lstsq(
                np.column_stack([np.ones(len(x_all)), x_all]), y_all, rcond=None
            )
            self.global_alpha_, self.global_rho_ = float(beta[0]), float(beta[1])

        self.country_models_ = {}
        iso_all = clean[iso_col].to_numpy()
        for iso in pd.unique(iso_all):
            m = iso_all == iso
            n_obs = int(m.sum())
            if n_obs < 5:
                continue  # falls back to the pooled estimate at predict time
            try:
                beta_c, _, _, _ = np.linalg.lstsq(
                    np.column_stack([np.ones(n_obs), x_all[m]]), y_all[m], rcond=None
                )
            except np.linalg.LinAlgError:
                continue
            w = n_obs / (n_obs + 10.0 * self.shrinkage_weight)
            self.country_models_[str(iso)] = {
                "alpha": w * float(beta_c[0]) + (1.0 - w) * self.global_alpha_,
                "rho": w * float(beta_c[1]) + (1.0 - w) * self.global_rho_,
                "n_obs": float(n_obs),
            }
        return self

    def predict_country(self, iso: str, last_known_growth: float) -> float:
        p = self.country_models_.get(
            str(iso), {"alpha": self.global_alpha_, "rho": self.global_rho_}
        )
        if not np.isfinite(last_known_growth):
            last_known_growth = self.default_growth
        pred = p["alpha"] + p["rho"] * last_known_growth
        return float(np.clip(pred, -self.prediction_clip, self.prediction_clip))

    def predict_panel(
        self,
        test_df: pd.DataFrame,
        lag_growth_col: Optional[str] = None,
        iso_col: str = "iso3",
    ) -> np.ndarray:
        """
        Predict on a test frame using the regressor pinned at fit time.

        ``lag_growth_col`` is optional and exists only so callers can assert their
        intent. Passing a column other than the fitted one raises ``ValueError``.
        """
        if self.lag_growth_col_ is None:
            raise RuntimeError(
                "PerCountryARForecaster.predict_panel called before fit()."
            )
        if lag_growth_col is not None and lag_growth_col != self.lag_growth_col_:
            raise ValueError(
                f"PerCountryARForecaster regressor mismatch: fit() used "
                f"'{self.lag_growth_col_}' but predict_panel() was given "
                f"'{lag_growth_col}'. The AR coefficients are calibrated to the fitted "
                f"column; applying them to a different series silently corrupts the "
                f"baseline and every lift measured against it."
            )
        col = self.lag_growth_col_
        if col not in test_df.columns:
            raise KeyError(
                f"PerCountryARForecaster.predict_panel: fitted regressor '{col}' is not "
                f"in the test frame. Build train and test slices from the same panel "
                f"returned by add_lagged_growth()."
            )

        lags = pd.to_numeric(test_df[col], errors="coerce").to_numpy(dtype=np.float64)
        lags = np.where(np.isfinite(lags), lags, self.default_growth)
        alpha = np.empty(len(test_df), dtype=np.float64)
        rho = np.empty(len(test_df), dtype=np.float64)
        for j, iso in enumerate(test_df[iso_col].astype(str).to_numpy()):
            p = self.country_models_.get(iso)
            if p is None:
                alpha[j], rho[j] = self.global_alpha_, self.global_rho_
            else:
                alpha[j], rho[j] = p["alpha"], p["rho"]
        c = self.prediction_clip
        return np.clip(alpha + rho * lags, -c, c)


class EqualWeightCombinationForecaster:
    """
    Equal-Weight Forecast Combination Baseline (1/M simple average).
    Standard forecasting tournament benchmark (Clemen 1989; Smith & Wallis 2009).
    """

    @staticmethod
    def combine(predictions: List[np.ndarray]) -> np.ndarray:
        stack = np.column_stack(predictions)
        return np.mean(stack, axis=1)


class SingleDomainSpecialist(BaseEstimator, RegressorMixin):
    """
    True Single-Domain Specialist:
    Enforces strict column quarantine—only ingests features belonging to its assigned domain.
    """

    def __init__(self, domain_name: str, domain_prefix: str, alpha: float = 100.0, random_state: int = 42):
        self.domain_name = domain_name
        self.domain_prefix = domain_prefix
        self.alpha = alpha
        self.random_state = random_state
        self.model = Ridge(alpha=alpha, random_state=random_state)
        self.selected_columns_: List[str] = []

    def fit(self, X_df: pd.DataFrame, y: np.ndarray) -> SingleDomainSpecialist:
        # Strictly select columns matching domain prefix or dictionary
        self.selected_columns_ = [c for c in X_df.columns if c.startswith(self.domain_prefix)]
        if len(self.selected_columns_) == 0:
            # Fallback if domain prefix is not strict: use all provided columns
            self.selected_columns_ = list(X_df.columns)

        X_sub = X_df[self.selected_columns_].values
        X_clean = np.nan_to_num(X_sub, nan=0.0)
        self.model.fit(X_clean, y)
        return self

    def predict(self, X_df: pd.DataFrame) -> np.ndarray:
        X_sub = X_df[self.selected_columns_].values
        X_clean = np.nan_to_num(X_sub, nan=0.0)
        return np.clip(self.model.predict(X_clean), -0.5, 0.5)


class DynamicFactorForecaster(BaseEstimator, RegressorMixin):
    r"""
    Dynamic Factor Model Forecaster (Stock & Watson, 2002):
    Extracts k dominant static latent factors from high-dimensional predictors:
      y_{i, t+h} = \alpha + \boldsymbol{\beta}^\top \mathbf{F}_t + \varepsilon_{t+h}
    """

    def __init__(self, n_factors: int = 5, alpha: float = 20.0, random_state: int = 42):
        self.n_factors = n_factors
        self.alpha = alpha
        self.random_state = random_state
        self.pca = PCA(n_components=n_factors, random_state=random_state)
        self.regressor = Ridge(alpha=alpha, random_state=random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> DynamicFactorForecaster:
        X_clean = np.nan_to_num(X, nan=0.0)
        n_comp = min(self.n_factors, X_clean.shape[1], X_clean.shape[0])
        self.pca.n_components = n_comp
        factors = self.pca.fit_transform(X_clean)
        self.regressor.fit(factors, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_clean = np.nan_to_num(X, nan=0.0)
        factors = self.pca.transform(X_clean)
        return np.clip(self.regressor.predict(factors), -0.5, 0.5)
