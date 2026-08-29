"""
Model Confidence Set (MCS) Algorithm
===================================
Reference:
  - Hansen, P. R., Lunde, A., & Nason, J. M. (2011). The model confidence set.
    Econometrica, 79(2), 453-497.

Implements the iterative Model Confidence Set procedure with stationary/moving-block bootstrap
to determine the subset of superior models M^*_{1-alpha} from an initial set of models M_0.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd


def block_bootstrap_indices(n_obs: int, block_size: int, n_bootstraps: int, rng: np.random.Generator) -> np.ndarray:
    """Generate moving-block bootstrap sample indices of shape (n_bootstraps, n_obs)."""
    n_blocks = int(np.ceil(n_obs / block_size))
    max_start = n_obs - block_size + 1
    if max_start <= 0:
        # Fallback to standard i.i.d. bootstrap if n_obs < block_size
        return rng.integers(0, n_obs, size=(n_bootstraps, n_obs))

    indices = np.empty((n_bootstraps, n_obs), dtype=np.int64)
    for b in range(n_bootstraps):
        starts = rng.integers(0, max_start, size=n_blocks)
        sampled_blocks = np.concatenate([np.arange(s, s + block_size) for s in starts])
        indices[b] = sampled_blocks[:n_obs]
    return indices


def model_confidence_set(
    losses: np.ndarray,
    model_names: List[str],
    alpha: float = 0.10,
    n_boot: int = 1000,
    block_size: int = 2,
    seed: int = 42,
    stat_type: str = "t_max",
) -> pd.DataFrame:
    r"""
    Compute Hansen, Lunde & Nason (2011) Model Confidence Set.

    Parameters
    ----------
    losses : np.ndarray
        Shape (T, M) matrix of evaluation losses for T observations and M candidate models.
    model_names : List[str]
        List of M model names matching columns of `losses`.
    alpha : float
        Significance level for the MCS (e.g. 0.10 for 90% confidence set).
    n_boot : int
        Number of bootstrap replications.
    block_size : int
        Block size for moving block bootstrap (capturing multi-step horizon serial correlation).
    seed : int
        Random seed for reproducibility.
    stat_type : str
        't_max' (max t-statistic over model pairwise averages) or 't_r' (range statistic).

    Returns
    -------
    pd.DataFrame
        Table with Model, Mean_Loss, Rank, MCS_P_Value, In_MCS_90pct, In_MCS_75pct.
    """
    losses = np.asarray(losses, dtype=np.float64)
    T, M = losses.shape
    if len(model_names) != M:
        raise ValueError(f"Length of model_names ({len(model_names)}) must match losses columns ({M})")

    rng = np.random.default_rng(seed)
    boot_indices = block_bootstrap_indices(T, block_size=max(1, block_size), n_bootstraps=n_boot, rng=rng)

    active_models = list(range(M))
    p_values: Dict[int, float] = {}
    elimination_order: List[int] = []

    # Precompute mean losses
    mean_losses = np.mean(losses, axis=0)

    # Iterative elimination loop
    prev_pval = 0.0

    while len(active_models) > 1:
        k = len(active_models)
        idx_arr = np.array(active_models)
        sub_losses = losses[:, idx_arr]  # Shape (T, k)

        # Compute sample loss differentials: d_{u,v,t} = L_{u,t} - L_{v,t}
        # Shape (T, k, k)
        d_mat = sub_losses[:, :, None] - sub_losses[:, None, :]
        d_bar = np.mean(d_mat, axis=0)  # Shape (k, k)

        # Model u relative to group average: d_{u., t} = (1 / (k-1)) sum_{v != u} d_{u,v,t}
        d_u_dot = np.sum(d_bar, axis=1) / (k - 1)  # Shape (k,)

        # Bootstrap centered loss differentials
        # Sampled d_mat: shape (n_boot, T, k, k)
        d_boot = d_mat[boot_indices]
        d_boot_mean = np.mean(d_boot, axis=1)  # Shape (n_boot, k, k)

        # Center bootstrap samples under null hypothesis: \tilde{d} = d^* - \bar{d}
        d_boot_centered = d_boot_mean - d_bar[None, :, :]

        # Bootstrap variances
        var_d_u = np.var(np.sum(d_boot_centered, axis=2) / (k - 1), axis=0, ddof=1)
        se_d_u = np.sqrt(np.maximum(1e-12, var_d_u))

        if stat_type == "t_max":
            t_stats = d_u_dot / se_d_u
            test_stat = float(np.max(t_stats))
            worst_model_local_idx = int(np.argmax(t_stats))

            # Bootstrap test statistics
            t_boot = (np.sum(d_boot_centered, axis=2) / (k - 1)) / se_d_u[None, :]
            test_stat_boot = np.max(t_boot, axis=1)
        else:  # 't_r' Range statistic
            var_d_uv = np.var(d_boot_centered, axis=0, ddof=1)
            se_d_uv = np.sqrt(np.maximum(1e-12, var_d_uv))
            se_d_uv[np.diag_indices(k)] = 1.0
            t_stats_matrix = np.abs(d_bar) / se_d_uv
            np.fill_diagonal(t_stats_matrix, 0.0)
            test_stat = float(np.max(t_stats_matrix))
            worst_model_local_idx = int(np.argmax(d_u_dot))

            t_boot_matrix = np.abs(d_boot_centered) / se_d_uv[None, :, :]
            test_stat_boot = np.max(t_boot_matrix, axis=(1, 2))

        # Empirical p-value under null
        pval = float(np.mean(test_stat_boot >= test_stat))
        # Ensure monotonicity of MCS p-values
        curr_pval = max(pval, prev_pval)
        prev_pval = curr_pval

        worst_model_global_idx = active_models[worst_model_local_idx]
        p_values[worst_model_global_idx] = curr_pval
        elimination_order.append(worst_model_global_idx)

        active_models.remove(worst_model_global_idx)

    # Last surviving model has p-value 1.0
    survivor = active_models[0]
    p_values[survivor] = 1.0
    elimination_order.append(survivor)

    records = []
    for rank, m_idx in enumerate(reversed(elimination_order), start=1):
        m_name = model_names[m_idx]
        p_val = p_values[m_idx]
        records.append({
            "Rank": rank,
            "Model": m_name,
            "Mean_Loss": round(float(mean_losses[m_idx]), 5),
            "MCS_P_Value": round(p_val, 4),
            "In_MCS_90pct": bool(p_val >= 0.10),
            "In_MCS_75pct": bool(p_val >= 0.25),
        })

    return pd.DataFrame(records)
