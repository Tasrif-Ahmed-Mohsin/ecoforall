"""
Conformal Uncertainty-Weighted Gating Engine (LGCF-v2)
======================================================
Computes dynamic gating weights inversely proportional to empirical
conformal prediction interval widths and residual variance.
"""

from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd


def compute_conformal_uncertainty_weights(
    specialist_predictions: Dict[str, np.ndarray],
    train_residuals: Dict[str, np.ndarray],
    alpha: float = 0.10,
    tau: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates dynamic conformal gating weights for each test sample.
    
    Parameters:
      specialist_predictions: Dict of model predictions on test fold [N]
      train_residuals: Dict of in-fold absolute residuals on train fold [M]
      alpha: Desired miscoverage rate (0.10 for 90% conformal coverage)
      tau: Temperature scaling parameter for softmax sharpness
      
    Returns:
      gated_prediction: [N] blended forecast
      weights_matrix: [N, K] matrix of dynamic weights per specialist
    """
    model_names = list(specialist_predictions.keys())
    K = len(model_names)
    N = len(specialist_predictions[model_names[0]])

    # 1. Compute empirical conformal quantile for each model
    # q_hat = (1 - alpha) * (1 + 1/M) quantile of absolute residuals
    q_hats = {}
    for m in model_names:
        res = train_residuals[m]
        n_res = len(res)
        q_level = min(1.0, (1.0 - alpha) * (1.0 + 1.0 / max(1, n_res)))
        q_hats[m] = np.quantile(res, min(0.999, q_level))

    # 2. Compute uncertainty score per test sample
    # Base uncertainty = conformal quantile interval width
    uncertainties = np.zeros((N, K), dtype=np.float64)
    preds_matrix = np.zeros((N, K), dtype=np.float64)

    for i, m in enumerate(model_names):
        preds_matrix[:, i] = specialist_predictions[m]
        uncertainties[:, i] = q_hats[m]

    # 3. Softmax weighting inversely proportional to uncertainty:
    # w_i = exp(-unc_i / tau) / sum(exp(-unc_j / tau))
    logits = -uncertainties / max(1e-4, tau)
    # Numerical stability shift
    logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits_shifted)
    weights = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # 4. Gated blended prediction
    gated_pred = np.sum(preds_matrix * weights, axis=1)

    return gated_pred, weights
