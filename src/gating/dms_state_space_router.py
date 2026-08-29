"""
State-Space Dynamic Model Averaging & Selection (DMA / DMS) Online Router
=========================================================================
Implements an online recursive Bayesian expert-weighting rule inspired by Gary Koop & Dimitris
Korobilis (International Economic Review, 2012; Economic Modelling, 2011).

Methodological Architecture:
----------------------------
1. Operates over fixed point predictions from M candidate functional specialists.
2. Maintains recursive Bayesian state-space probabilities pi_{t|t-1, m}^{(c)} for each
   country c over time, discounted via forgetting factor lambda in (0, 1].
3. Updates predictive likelihoods under an online decaying country-level residual variance
   sigma_c^2 (shared across specialists).
4. Produces either a probability-weighted convex combination (Dynamic Model Averaging,
   mode="dma") or a hard argmax selection (Dynamic Model Selection, mode="dms").

Real-time feedback discipline:
------------------------------
An h-step-ahead forecast made at origin t targets y_{t+h}, which is not realised until
t + h. The filter may therefore only condition on targets from origins t0 with
t0 + h <= t. `route_panel` enforces this by queueing each origin's realisation and
releasing it into the filter only once the calendar has caught up.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


class DynamicModelSelectionRouter:
    """
    Online State-Space DMA/DMS Router for Sovereign Macroeconomic Panels.

    Combines M domain specialists via online recursive Bayesian probability discounting
    (Koop & Korobilis 2012) with a country-level residual variance.
    """

    def __init__(self, n_experts: int = 4, forgetting_factor: float = 0.92,
                 initial_prior: np.ndarray | None = None, mode: str = "dma",
                 init_variance: float = 1e-3, variance_decay: float = 0.90):
        if not (0.0 < forgetting_factor <= 1.0):
            raise ValueError("forgetting_factor must lie in (0, 1]")
        self.n_experts = n_experts
        self.forgetting_factor = forgetting_factor
        self.mode = mode.lower()

        self.init_variance = init_variance
        self.variance_decay = variance_decay
        if initial_prior is None:
            self.initial_prior = np.ones(n_experts, dtype=np.float64) / n_experts
        else:
            self.initial_prior = np.asarray(initial_prior, dtype=np.float64) / np.sum(initial_prior)

        # Country posterior state: iso3 -> pi_post (M,)
        self.country_states: dict[str, np.ndarray] = {}
        # Country shared variance estimate: iso3 -> sigma_sq
        self.country_variances: dict[str, float] = {}
        # Realisations queued but not yet observable: iso3 -> [(origin_year, preds, actual)]
        self._pending: dict[str, list[tuple[int, np.ndarray, float]]] = {}
        # Last origin year processed per country, for the elapsed-time discount
        self._last_origin: dict[str, int] = {}

    def reset(self):
        """Reset all sovereign state histories."""
        self.country_states.clear()
        self.country_variances.clear()
        self._pending.clear()
        self._last_origin.clear()

    def n_pending(self) -> int:
        """Realisations queued but never released -- diagnostic for warm-up sizing."""
        return sum(len(v) for v in self._pending.values())

    def _discount(self, pi: np.ndarray, periods: int = 1) -> np.ndarray:
        r"""State-space prediction step: $\pi^{\lambda^k}$, renormalised (Koop-Korobilis 2012)."""
        if periods <= 0:
            return pi
        p = np.power(np.maximum(pi, 1e-12), self.forgetting_factor ** periods)
        total = np.sum(p)
        return p / total if total > 0 else self.initial_prior.copy()

    def _measurement_update(self, pi_prior: np.ndarray, preds: np.ndarray,
                            actual: float, sigma_sq: float) -> tuple[np.ndarray, float]:
        """Gaussian predictive-density update for one realised observation."""
        sq_errors = (actual - preds) ** 2
        sigma_sq = (self.variance_decay * sigma_sq
                    + (1.0 - self.variance_decay) * max(float(np.mean(sq_errors)), 1e-5))
        log_lik = -0.5 * (sq_errors / sigma_sq)
        log_lik -= np.max(log_lik)
        pi_post = pi_prior * np.exp(log_lik)
        total = np.sum(pi_post)
        if not np.isfinite(total) or total <= 1e-300:
            return pi_prior.copy(), sigma_sq
        return pi_post / total, sigma_sq

    def route_panel(self, df: pd.DataFrame, expert_preds: np.ndarray,
                    y_true: np.ndarray | None = None, horizon: int = 1,
                    year_col: str = "year", iso_col: str = "iso3"
                    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sequential state-space DMS/DMA routing across a chronological country-year panel.

        Parameters
        ----------
        df
            Frame carrying ``iso_col`` and ``year_col``; ``year_col`` is the forecast
            **origin** year, not the target year.
        expert_preds
            ``(N, M)`` matrix of specialist point forecasts, row-aligned to ``df``.
        y_true
            Realised targets, row-aligned to ``df``. Each entry is released into the
            filter only at origins ``>= origin_year + horizon``. Pass ``None`` to run
            with no feedback at all (the weights then stay at the initial prior, so DMA
            degenerates to the 1/M average).
        horizon
            Forecast horizon in years. Sets the release delay. ``horizon=1`` gives the
            classical one-step filter that updates from the immediately preceding origin.

        Returns
        -------
        (y_gated, weights)
            Both row-aligned to ``df`` in its original order.
        """
        expert_preds = np.asarray(expert_preds, dtype=np.float64)
        if expert_preds.ndim != 2:
            raise ValueError(f"expert_preds must be 2-D (N, M); got shape {expert_preds.shape}")
        N, M = expert_preds.shape
        if M != self.n_experts:
            raise ValueError(f"expert_preds has {M} experts but router was built for {self.n_experts}")
        if len(df) != N:
            raise ValueError(f"df has {len(df)} rows but expert_preds has {N}")
        if int(horizon) < 1:
            raise ValueError(f"horizon must be >= 1; got {horizon}")
        horizon = int(horizon)
        for c in (iso_col, year_col):
            if c not in df.columns:
                raise KeyError(
                    f"route_panel: column '{c}' is required. The router needs the forecast "
                    f"origin year to gate feedback by realisation date."
                )

        iso_arr = df[iso_col].astype(str).to_numpy()
        year_arr = pd.to_numeric(df[year_col], errors="coerce").to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(year_arr)):
            raise ValueError("route_panel: non-numeric or missing values in the origin-year column")
        year_arr = year_arr.astype(np.int64)

        y_arr = None
        if y_true is not None:
            y_arr = np.asarray(y_true, dtype=np.float64).ravel()
            if len(y_arr) != N:
                raise ValueError(f"y_true has {len(y_arr)} entries but df has {N} rows")

        y_gated = np.zeros(N, dtype=np.float64)
        weights = np.zeros((N, M), dtype=np.float64)

        # Chronological within country: primary key iso, secondary key origin year.
        order = np.lexsort((year_arr, iso_arr))

        for pos in order:
            iso = iso_arr[pos]
            t = int(year_arr[pos])

            if iso not in self.country_states:
                self.country_states[iso] = self.initial_prior.copy()
                self.country_variances[iso] = self.init_variance
                self._pending[iso] = []

            # 1. Prediction step, once per elapsed year since this country was last seen.
            last_t = self._last_origin.get(iso)
            if last_t is not None and t > last_t:
                self.country_states[iso] = self._discount(self.country_states[iso], t - last_t)
            self._last_origin[iso] = t

            # 2. Release only those realisations already observable at origin t.
            still_pending = []
            for origin_t, preds_t, actual_t in self._pending[iso]:
                if origin_t + horizon <= t:
                    self.country_states[iso], self.country_variances[iso] = self._measurement_update(
                        self.country_states[iso], preds_t, actual_t, self.country_variances[iso]
                    )
                else:
                    still_pending.append((origin_t, preds_t, actual_t))
            self._pending[iso] = still_pending

            # 3. Gated forecast from the information set available at origin t.
            pi = self.country_states[iso]
            weights[pos] = pi
            if self.mode == "dms":
                y_gated[pos] = expert_preds[pos, int(np.argmax(pi))]
            else:
                y_gated[pos] = float(np.dot(pi, expert_preds[pos]))

            # 4. Queue this origin's own realisation; unusable until t + horizon.
            if y_arr is not None and np.isfinite(y_arr[pos]):
                self._pending[iso].append((t, expert_preds[pos].copy(), float(y_arr[pos])))

        return y_gated, weights
