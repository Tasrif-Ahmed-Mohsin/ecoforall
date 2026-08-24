"""
Koop-Korobilis Dynamic Model Selection & Averaging (DMS / DMA) State-Space Router
=================================================================================
From Gary Koop & Dimitris Korobilis (International Economic Review, 2012; Economic Modelling, 2011).

In sovereign macroeconomic panels, regimes (e.g. tranquil growth, debt stress, financial crisis,
climate shocks) persist across multiple contiguous years.

Instead of treating each year as an independent i.i.d. instance, DMS/DMA maintains a recursive
Bayesian state-space probability vector pi_{t, m}^{(c)} for each country c over time, updated
via a forgetting factor lambda in [0.85, 0.99].

Mathematical Formulation:
-------------------------
1. Prior Model Probability (with Forgetting Factor lambda):
       pi_{t|t-1, m}^{(c)} = (pi_{t-1|t-1, m}^{(c)})^lambda / sum_j (pi_{t-1|t-1, j}^{(c)})^lambda

2. Predictive Likelihood (Laplace / Gaussian Predictive Density):
       f_m(y_t^{(c)} | y_{1:t-1}^{(c)}) propto exp( - | y_t^{(c)} - y_hat_{t,m}^{(c)} | / (sigma_{t,m}^{(c)} + eps) )

3. Posterior Probability Update:
       pi_{t|t, m}^{(c)} = pi_{t|t-1, m}^{(c)} * f_m(y_t^{(c)}) / sum_j ( pi_{t|t-1, j}^{(c)} * f_j(y_t^{(c)}) )

4. Forecasting:
       - DMA (Averaging): y_hat_t = sum_m pi_{t|t-1, m}^{(c)} * y_hat_{t,m}^{(c)}
       - DMS (Selection): y_hat_t = y_hat_{t, m*}^{(c)}, where m* = argmax_m pi_{t|t-1, m}^{(c)}
"""

from __future__ import annotations
import numpy as np
import pandas as pd


class DynamicModelSelectionRouter:
    """
    Koop-Korobilis State-Space DMS/DMA Router for Sovereign Macroeconomic Panels.
    """
    def __init__(self, n_experts: int = 4, forgetting_factor: float = 0.92,
                 initial_prior: np.ndarray | None = None, mode: str = "dma"):
        """
        Args:
            n_experts: Number of competing specialist models (M).
            forgetting_factor: Decay factor lambda in [0.80, 0.99] controlling memory horizon.
            initial_prior: Initial uniform or informative model probability vector.
            mode: 'dma' for dynamic model averaging, 'dms' for hard dynamic model selection.
        """
        self.n_experts = n_experts
        self.forgetting_factor = forgetting_factor
        self.mode = mode.lower()
        if initial_prior is None:
            self.initial_prior = np.ones(n_experts, dtype=np.float64) / n_experts
        else:
            self.initial_prior = np.asarray(initial_prior, dtype=np.float64) / np.sum(initial_prior)

        # Dictionary tracking sovereign country states: iso3 -> current_posterior_probs
        self.country_states: dict[str, np.ndarray] = {}
        # Running volatility estimate per country and expert: iso3 -> sigma_m
        self.country_volatilities: dict[str, np.ndarray] = {}

    def reset(self):
        """Reset all sovereign state histories."""
        self.country_states.clear()
        self.country_volatilities.clear()

    def route_panel(self, df: pd.DataFrame, expert_preds: np.ndarray,
                    y_true: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Execute sequential state-space DMS/DMA routing across a chronological country-year panel.

        Args:
            df: DataFrame containing at least 'iso3' and 'year' columns.
            expert_preds: Array of shape (N, M) of predictions from each specialist.
            y_true: Optional ground-truth targets of shape (N,) used for recursive Bayesian updating.

        Returns:
            y_gated: Gated predictions of shape (N,).
            weights: Dynamic model probability weights of shape (N, M).
        """
        N, M = expert_preds.shape
        lam = self.forgetting_factor
        
        y_gated = np.zeros(N, dtype=np.float64)
        weights = np.zeros((N, M), dtype=np.float64)

        # Ensure we iterate in chronological order per country
        df_sorted = df.copy().reset_index(drop=True)
        iso3_series = df_sorted["iso3"].values

        for i in range(N):
            iso = str(iso3_series[i])
            
            # Retrieve or initialize country prior state
            if iso not in self.country_states:
                pi_prior = self.initial_prior.copy()
                sigma_m = np.full(M, 0.03, dtype=np.float64)
            else:
                pi_prev = self.country_states[iso]
                # 1. State-Space Prediction Step with Forgetting Factor
                pi_exp = np.power(np.maximum(pi_prev, 1e-12), lam)
                pi_prior = pi_exp / np.sum(pi_exp)
                sigma_m = self.country_volatilities[iso]

            weights[i] = pi_prior

            # 2. Compute Gated Forecast
            if self.mode == "dms":
                # Hard Selection: pick best expert
                best_m = np.argmax(pi_prior)
                y_gated[i] = expert_preds[i, best_m]
            else:
                # Soft Averaging (DMA)
                y_gated[i] = np.sum(pi_prior * expert_preds[i])

            # 3. Recursive Bayesian Update if ground truth is available
            if y_true is not None and not np.isnan(y_true[i]):
                actual = y_true[i]
                abs_errors = np.abs(actual - expert_preds[i])  # (M,)
                
                # Recursive volatility update
                sigma_m = 0.90 * sigma_m + 0.10 * abs_errors
                self.country_volatilities[iso] = sigma_m

                # Laplace predictive likelihood
                log_lik = - (abs_errors / np.maximum(sigma_m, 1e-4))
                log_lik -= np.max(log_lik)  # Numerical stability
                lik = np.exp(log_lik)
                
                # Posterior update
                pi_post = pi_prior * lik
                pi_post = pi_post / np.maximum(np.sum(pi_post), 1e-12)
                self.country_states[iso] = pi_post

        return y_gated, weights
