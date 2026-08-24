"""
Hybrid DMS-Sparsemax Router: State-Space Memory & Sparse Simplex Gating
========================================================================
A unified mathematical synthesis that combines:
  1. Koop-Korobilis Dynamic Model Selection (DMS) state-space regime tracking per country.
  2. Instantaneous cross-domain feature conditioning via a neural/linear routing layer.
  3. Sparsemax projection onto the probability simplex, enabling exact boundary selection.

Mathematical Formulation:
-------------------------
At country c and year t with feature vector x_t^{(c)} and state-space prior pi_{t|t-1}^{(c)}:
    z_m(x_t^{(c)}) = W_m^T x_t^{(c)} + b_m + beta * log( pi_{t|t-1, m}^{(c)} + eps )
    w_t^{(c)} = sparsemax( z(x_t^{(c)}) )
    y_hat_t^{(c)} = sum_{m=1}^M w_{t, m}^{(c)} * y_hat_{t, m}^{(c)}
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from src.gating.sparsemax_router import sparsemax, SparsemaxDirectMAERouter
from src.gating.dms_state_space_router import DynamicModelSelectionRouter


class HybridDMSSparsemaxRouter:
    """
    Hybrid Router combining State-Space Regime Memory with Sparsemax Feature Gating.
    """
    def __init__(self, n_features: int, n_experts: int = 4, forgetting_factor: float = 0.92,
                 prior_weight_beta: float = 1.5, reg_lambda: float = 1e-4, lr: float = 0.05,
                 max_iter: int = 350, random_state: int = 42):
        self.n_features = n_features
        self.n_experts = n_experts
        self.forgetting_factor = forgetting_factor
        self.prior_weight_beta = prior_weight_beta
        self.reg_lambda = reg_lambda
        self.lr = lr
        self.max_iter = max_iter
        self.random_state = random_state

        self.sparsemax_model = SparsemaxDirectMAERouter(
            n_features=n_features, n_experts=n_experts, reg_lambda=reg_lambda,
            lr=lr, max_iter=max_iter, random_state=random_state
        )
        self.dms_engine = DynamicModelSelectionRouter(
            n_experts=n_experts, forgetting_factor=forgetting_factor, mode="dma"
        )

    def fit(self, X: np.ndarray, expert_preds: np.ndarray, y: np.ndarray):
        """Fit the underlying Sparsemax feature routing parameters."""
        self.sparsemax_model.fit(X, expert_preds, y)
        return self

    def route_panel(self, df: pd.DataFrame, X: np.ndarray, expert_preds: np.ndarray,
                    y_true: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Execute Hybrid State-Space + Sparsemax routing across panel observations.

        Args:
            df: DataFrame containing at least 'iso3' and 'year' columns.
            X: Feature matrix of shape (N, D).
            expert_preds: Array of shape (N, M) containing predictions from each expert.
            y_true: Optional ground truth vector for state-space Bayesian updates.

        Returns:
            y_gated: Gated predictions of shape (N,).
            weights: Sparse routing weights of shape (N, M).
        """
        N, M = expert_preds.shape
        X_arr = np.asarray(X, dtype=np.float64)
        
        # 1. Compute baseline feature logits from Sparsemax layer
        feature_logits = X_arr @ self.sparsemax_model.W + self.sparsemax_model.b  # (N, M)

        y_gated = np.zeros(N, dtype=np.float64)
        weights = np.zeros((N, M), dtype=np.float64)

        df_sorted = df.copy().reset_index(drop=True)
        iso3_series = df_sorted["iso3"].values
        lam = self.forgetting_factor
        beta = self.prior_weight_beta

        for i in range(N):
            iso = str(iso3_series[i])
            
            # State-space prior
            if iso not in self.dms_engine.country_states:
                pi_prior = np.ones(M, dtype=np.float64) / M
                sigma_m = np.full(M, 0.03, dtype=np.float64)
            else:
                pi_prev = self.dms_engine.country_states[iso]
                pi_exp = np.power(np.maximum(pi_prev, 1e-12), lam)
                pi_prior = pi_exp / np.sum(pi_exp)
                sigma_m = self.dms_engine.country_volatilities[iso]

            # Fuse feature logits with state-space log-prior
            log_prior = np.log(np.maximum(pi_prior, 1e-12))
            combined_logits = feature_logits[i] + beta * log_prior
            
            # Project onto simplex via Sparsemax
            p_sparse = sparsemax(combined_logits)  # (M,)
            weights[i] = p_sparse
            y_gated[i] = np.sum(p_sparse * expert_preds[i])

            # Recursive Bayesian update
            if y_true is not None and not np.isnan(y_true[i]):
                actual = y_true[i]
                abs_errors = np.abs(actual - expert_preds[i])
                sigma_m = 0.90 * sigma_m + 0.10 * abs_errors
                self.dms_engine.country_volatilities[iso] = sigma_m

                log_lik = - (abs_errors / np.maximum(sigma_m, 1e-4))
                log_lik -= np.max(log_lik)
                lik = np.exp(log_lik)
                
                pi_post = pi_prior * lik
                pi_post = pi_post / np.maximum(np.sum(pi_post), 1e-12)
                self.dms_engine.country_states[iso] = pi_post

        return y_gated, weights
