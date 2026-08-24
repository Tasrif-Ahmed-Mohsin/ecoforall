"""
Cost-Sensitive Regret Minimization Router (Learning-to-Defer Paradigm)
======================================================================
From Mozannar & Sontag (ICML 2020, NeurIPS 2024 / ICML 2024).

Instead of treating expert selection as a discrete multiclass classification problem
(which discards the continuous severity of errors), Cost-Sensitive Gating optimizes
a surrogate of the continuous empirical Regret Matrix:

    C_{i, m} = | y_i - y_hat_{i, m} |

Objective:
    min_{W, b} (1/N) * sum_{i=1}^N sum_{m=1}^M C_{i, m} * P_m(x_i; W, b) + (lambda/2) * ||W||_F^2
"""

from __future__ import annotations
import numpy as np


def softmax(z: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Compute numerically stable row-wise softmax with temperature scaling."""
    z_scaled = z / max(1e-4, temperature)
    z_max = np.max(z_scaled, axis=-1, keepdims=True)
    exp_z = np.exp(z_scaled - z_max)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


class CostSensitiveRegretRouter:
    """
    Cost-Sensitive Surrogate Router optimizing continuous regret over expert losses.
    """
    def __init__(self, n_features: int, n_experts: int = 4, temperature: float = 0.5,
                 reg_lambda: float = 1e-4, lr: float = 0.05, max_iter: int = 350,
                 random_state: int = 42):
        self.n_features = n_features
        self.n_experts = n_experts
        self.temperature = temperature
        self.reg_lambda = reg_lambda
        self.lr = lr
        self.max_iter = max_iter
        self.random_state = random_state

        rng = np.random.RandomState(random_state)
        self.W = rng.randn(n_features, n_experts) * 0.01
        self.b = np.zeros(n_experts, dtype=np.float64)

    def fit(self, X: np.ndarray, expert_preds: np.ndarray, y: np.ndarray):
        """
        Fit the cost-sensitive router weights.

        Args:
            X: Feature matrix of shape (N, D).
            expert_preds: Array of shape (N, M) of predictions from each specialist.
            y: Ground truth target vector of shape (N,).
        """
        X_arr = np.asarray(X, dtype=np.float64)
        E_arr = np.asarray(expert_preds, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)

        N, D = X_arr.shape
        M = self.n_experts

        # Compute empirical Cost Matrix C_{i, m} = | y_i - E_{i, m} |
        C = np.abs(y_arr[:, None] - E_arr)  # Shape (N, M)

        W = self.W.copy()
        b = self.b.copy()

        best_loss = float("inf")
        best_W = W.copy()
        best_b = b.copy()

        mW, vW = np.zeros_like(W), np.zeros_like(W)
        mb, vb = np.zeros_like(b), np.zeros_like(b)
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8

        T = self.temperature

        for it in range(1, self.max_iter + 1):
            logits = X_arr @ W + b  # (N, M)
            P = softmax(logits, temperature=T)  # (N, M)

            # Expected cost under current routing distribution: sum_m P_im * C_im
            expected_cost_per_instance = np.sum(P * C, axis=1)  # (N,)
            total_cost = np.mean(expected_cost_per_instance)
            loss = total_cost + 0.5 * self.reg_lambda * np.sum(W ** 2)

            if loss < best_loss:
                best_loss = loss
                best_W = W.copy()
                best_b = b.copy()

            # Exact gradient of expected cost with respect to logits z_im:
            # d/dz_im [ sum_j P_ij C_ij ] = (1/T) * P_im * ( C_im - sum_j P_ij C_ij )
            mean_cost_vec = expected_cost_per_instance[:, None]  # (N, 1)
            dL_dz = (1.0 / (N * T)) * P * (C - mean_cost_vec)  # (N, M)

            grad_W = X_arr.T @ dL_dz + self.reg_lambda * W
            grad_b = np.sum(dL_dz, axis=0)

            # Adam update
            mW = beta1 * mW + (1 - beta1) * grad_W
            vW = beta2 * vW + (1 - beta2) * (grad_W ** 2)
            mW_hat = mW / (1 - beta1 ** it)
            vW_hat = vW / (1 - beta2 ** it)
            W -= self.lr * mW_hat / (np.sqrt(vW_hat) + eps)

            mb = beta1 * mb + (1 - beta1) * grad_b
            vb = beta2 * vb + (1 - beta2) * (grad_b ** 2)
            mb_hat = mb / (1 - beta1 ** it)
            vb_hat = vb / (1 - beta2 ** it)
            b -= self.lr * mb_hat / (np.sqrt(vb_hat) + eps)

        self.W = best_W
        self.b = best_b
        return self

    def predict_weights(self, X: np.ndarray) -> np.ndarray:
        """Compute routing weights for query instances."""
        X_arr = np.asarray(X, dtype=np.float64)
        logits = X_arr @ self.W + self.b
        return softmax(logits, temperature=self.temperature)

    def route(self, X: np.ndarray, expert_preds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute gated prediction and routing weights."""
        P = self.predict_weights(X)
        y_gated = np.sum(P * expert_preds, axis=1)
        return y_gated, P
