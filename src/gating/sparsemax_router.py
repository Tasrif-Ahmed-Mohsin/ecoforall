"""
Sparsemax Direct MAE Optimization Router
========================================
Implements the Sparsemax probability simplex projection operator (Martins & Astudillo, JMLR 2016)
and trains a routing network by directly minimizing the out-of-fold Mean Absolute Error (MAE)
of the specialist mixture.

Unlike Softmax, Sparsemax produces exact zeros and ones (hard/soft hybrid selection),
enabling the network to perform hard Oracle expert selection when confidence is high.
"""

from __future__ import annotations
import numpy as np


def sparsemax(z: np.ndarray) -> np.ndarray:
    """
    Project input vector z onto the probability simplex:
        sparsemax(z) = argmin_{p in Delta^{M-1}} ||p - z||^2

    Args:
        z: 2D array of shape (N, M) of unnormalized router logits.

    Returns:
        p: 2D array of shape (N, M) of sparse probabilities summing to 1 along axis 1.
    """
    z_arr = np.asarray(z, dtype=np.float64)
    if z_arr.ndim == 1:
        z_arr = z_arr.reshape(1, -1)
        single_dim = True
    else:
        single_dim = False

    N, M = z_arr.shape
    # Sort logits in descending order
    z_sorted = np.sort(z_arr, axis=1)[:, ::-1]
    
    # Cumulative sums along rows
    z_cumsum = np.cumsum(z_sorted, axis=1)
    
    # Indices 1, 2, ..., M
    k_indices = np.arange(1, M + 1, dtype=np.float64).reshape(1, -1)
    
    # Condition: 1 + k * z_{(k)} > sum_{j<=k} z_{(j)}
    condition = (1.0 + k_indices * z_sorted) > z_cumsum
    
    # Find k(z) for each row (the largest index satisfying the condition)
    # Since condition is true up to k(z) and false after, argmax of condition[::-1] gives k(z)
    k_z = np.sum(condition, axis=1, keepdims=True)  # Shape (N, 1)
    
    # Compute threshold tau(z) = (sum_{j<=k(z)} z_{(j)} - 1) / k(z)
    # Extract the cumulative sum at k(z)
    row_indices = np.arange(N)
    k_z_flat = k_z.ravel().astype(int) - 1
    tau_z = (z_cumsum[row_indices, k_z_flat].reshape(N, 1) - 1.0) / k_z
    
    # Sparsemax output: max(0, z - tau(z))
    p = np.maximum(0.0, z_arr - tau_z)
    
    if single_dim:
        return p[0]
    return p


class SparsemaxDirectMAERouter:
    """
    End-to-End Sparsemax Router trained via direct MAE subgradient descent.
    
    Objective:
        min_{W, b} (1/N) * sum_{i=1}^N | y_i - sum_{m=1}^M p_m(x_i; W, b) * y_hat_{i,m} | + (lambda/2) * ||W||_F^2
    """
    def __init__(self, n_features: int, n_experts: int = 4, reg_lambda: float = 1e-4,
                 lr: float = 0.05, max_iter: int = 350, random_state: int = 42):
        self.n_features = n_features
        self.n_experts = n_experts
        self.reg_lambda = reg_lambda
        self.lr = lr
        self.max_iter = max_iter
        self.random_state = random_state
        
        # Initialize weights
        rng = np.random.RandomState(random_state)
        self.W = rng.randn(n_features, n_experts) * 0.01
        self.b = np.zeros(n_experts, dtype=np.float64)

    def fit(self, X: np.ndarray, expert_preds: np.ndarray, y: np.ndarray):
        """
        Fit the Sparsemax router parameters W, b to minimize MAE.

        Args:
            X: Input feature matrix of shape (N, D).
            expert_preds: Array of shape (N, M) containing predictions from each expert.
            y: Ground truth targets of shape (N,).
        """
        X_arr = np.asarray(X, dtype=np.float64)
        E_arr = np.asarray(expert_preds, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        
        N, D = X_arr.shape
        M = self.n_experts

        W = self.W.copy()
        b = self.b.copy()
        
        best_loss = float("inf")
        best_W = W.copy()
        best_b = b.copy()

        # Adam optimizer parameters
        mW, vW = np.zeros_like(W), np.zeros_like(W)
        mb, vb = np.zeros_like(b), np.zeros_like(b)
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8

        for it in range(1, self.max_iter + 1):
            # Forward pass
            logits = X_arr @ W + b  # (N, M)
            P = sparsemax(logits)  # (N, M)
            
            # Gated prediction
            y_gated = np.sum(P * E_arr, axis=1)  # (N,)
            errors = y_gated - y_arr  # (N,)
            
            # Current MAE loss
            mae = np.mean(np.abs(errors))
            loss = mae + 0.5 * self.reg_lambda * np.sum(W ** 2)
            
            if loss < best_loss:
                best_loss = loss
                best_W = W.copy()
                best_b = b.copy()

            # Subgradient of MAE loss: sign(y_gated - y)
            sign_err = np.sign(errors) / N  # (N,)
            
            # Gradient w.r.t P: dL/dP_im = sign_err_i * E_im
            dL_dP = sign_err[:, None] * E_arr  # (N, M)
            
            # Vectorized Jacobian-vector product for Sparsemax:
            # J_sparsemax(z) v = S * (v - (S v / |S|_1)) where S is the support indicator I(P > 0)
            S = (P > 0.0).astype(np.float64)  # (N, M)
            S_sum = np.sum(S, axis=1, keepdims=True)  # (N, 1)
            S_sum = np.maximum(S_sum, 1.0)
            
            v_masked = dL_dP * S
            mean_v = np.sum(v_masked, axis=1, keepdims=True) / S_sum
            dL_dz = S * (dL_dP - mean_v)  # (N, M)
            
            # Gradients w.r.t W and b
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
        """Compute sparse routing weights for query instances."""
        X_arr = np.asarray(X, dtype=np.float64)
        logits = X_arr @ self.W + self.b
        return sparsemax(logits)

    def route(self, X: np.ndarray, expert_preds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute gated prediction and routing weights."""
        P = self.predict_weights(X)
        y_gated = np.sum(P * expert_preds, axis=1)
        return y_gated, P
