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
                    y_true: np.ndarray | None = None, horizon: int = 1,
                    year_col: str = "year", iso_col: str = "iso3"
                    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Execute Hybrid State-Space + Sparsemax routing across panel observations.

        Args:
            df: DataFrame containing at least 'iso3' and 'year' (the forecast ORIGIN year).
            X: Feature matrix of shape (N, D).
            expert_preds: Array of shape (N, M) containing predictions from each expert.
            y_true: Optional realised targets for state-space Bayesian updates. Each is
                released into the filter only at origins >= origin_year + horizon.
            horizon: Forecast horizon in years, setting the release delay.

        Returns:
            y_gated: Gated predictions of shape (N,).
            weights: Sparse routing weights of shape (N, M).

        Feedback discipline mirrors DynamicModelSelectionRouter.route_panel: this class
        previously carried its own copy of the update loop and applied every realisation
        immediately, so at horizon h the weights depended on up to h-1 years of future
        data. Duplicated gating logic is how that defect survived, so the release rule is
        enforced here identically.
        """
        expert_preds = np.asarray(expert_preds, dtype=np.float64)
        N, M = expert_preds.shape
        if len(df) != N:
            raise ValueError(f"df has {len(df)} rows but expert_preds has {N}")
        if int(horizon) < 1:
            raise ValueError(f"horizon must be >= 1; got {horizon}")
        horizon = int(horizon)
        for c in (iso_col, year_col):
            if c not in df.columns:
                raise KeyError(
                    f"route_panel: column '{c}' is required to gate feedback by "
                    f"realisation date."
                )

        X_arr = np.asarray(X, dtype=np.float64)

        # 1. Compute baseline feature logits from Sparsemax layer
        feature_logits = X_arr @ self.sparsemax_model.W + self.sparsemax_model.b  # (N, M)

        y_gated = np.zeros(N, dtype=np.float64)
        weights = np.zeros((N, M), dtype=np.float64)

        iso_arr = df[iso_col].astype(str).to_numpy()
        year_arr = pd.to_numeric(df[year_col], errors="coerce").to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(year_arr)):
            raise ValueError("route_panel: non-numeric or missing origin years")
        year_arr = year_arr.astype(np.int64)

        y_arr = None
        if y_true is not None:
            y_arr = np.asarray(y_true, dtype=np.float64).ravel()

        lam = self.forgetting_factor
        beta = self.prior_weight_beta
        states: dict[str, np.ndarray] = self.dms_engine.country_states
        vols: dict[str, np.ndarray] = {}
        pending: dict[str, list[tuple[int, np.ndarray, float]]] = {}
        last_origin: dict[str, int] = {}

        def measurement_update(pi_prior, preds, actual, sigma_m):
            abs_errors = np.abs(actual - preds)
            sigma_m = 0.90 * sigma_m + 0.10 * abs_errors
            log_lik = -(abs_errors / np.maximum(sigma_m, 1e-4))
            log_lik -= np.max(log_lik)
            pi_post = pi_prior * np.exp(log_lik)
            total = np.sum(pi_post)
            if not np.isfinite(total) or total <= 1e-300:
                return pi_prior.copy(), sigma_m
            return pi_post / total, sigma_m

        # Chronological within country: primary key iso, secondary key origin year.
        for pos in np.lexsort((year_arr, iso_arr)):
            iso = iso_arr[pos]
            t = int(year_arr[pos])

            if iso not in states:
                states[iso] = np.ones(M, dtype=np.float64) / M
                vols[iso] = np.full(M, 0.03, dtype=np.float64)
                pending[iso] = []

            # State-space prediction step, once per elapsed year for this country.
            prev_t = last_origin.get(iso)
            if prev_t is not None and t > prev_t:
                p = np.power(np.maximum(states[iso], 1e-12), lam ** (t - prev_t))
                states[iso] = p / np.sum(p)
            last_origin[iso] = t

            # Release only realisations already observable at origin t.
            still = []
            for origin_t, preds_t, actual_t in pending[iso]:
                if origin_t + horizon <= t:
                    states[iso], vols[iso] = measurement_update(
                        states[iso], preds_t, actual_t, vols[iso]
                    )
                else:
                    still.append((origin_t, preds_t, actual_t))
            pending[iso] = still

            # Fuse feature logits with state-space log-prior, project via Sparsemax.
            log_prior = np.log(np.maximum(states[iso], 1e-12))
            p_sparse = sparsemax(feature_logits[pos] + beta * log_prior)
            weights[pos] = p_sparse
            y_gated[pos] = float(np.sum(p_sparse * expert_preds[pos]))

            # Queue this origin's own realisation; unusable until t + horizon.
            if y_arr is not None and np.isfinite(y_arr[pos]):
                pending[iso].append((t, expert_preds[pos].copy(), float(y_arr[pos])))

        self.dms_engine.country_volatilities = vols
        self.dms_engine.country_variances = {k: float(np.mean(v ** 2)) for k, v in vols.items()}
        return y_gated, weights


