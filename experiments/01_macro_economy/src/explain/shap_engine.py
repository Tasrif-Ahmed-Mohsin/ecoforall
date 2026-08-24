"""SHAP Feature Driver Attribution Engine for LightGBM Macro Models."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.explain.llm_narrative import _humanize_indicator


def compute_feature_drivers(
    lgbm_model: Any,
    feature_meta: dict[str, Any],
    query_row_full: pd.Series,
    top_k: int = 4,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compute exact feature contributions (positive drivers & negative drags) for a single query row.
    
    Uses LightGBM's fast `pred_contrib=True` feature attribution method.
    Returns: (positive_drivers, negative_drags)
    """
    full_cols = feature_meta.get("full_cols", [])
    if not full_cols or lgbm_model is None:
        return [], []

    # Build numeric vector for full_cols
    x_vec = np.array([float(query_row_full.get(c, np.nan)) for c in full_cols], dtype=np.float32).reshape(1, -1)
    
    try:
        # LightGBM booster pred_contrib
        booster = lgbm_model.booster_ if hasattr(lgbm_model, "booster_") else lgbm_model
        contribs = booster.predict(x_vec, pred_contrib=True)[0]  # shape (num_features + 1,)
        feat_contribs = contribs[:-1]  # last element is bias / base value
        
        drivers = []
        for col, val, contrib in zip(full_cols, x_vec[0], feat_contribs):
            if np.isnan(val) or np.isnan(contrib) or abs(contrib) < 1e-6:
                continue
            drivers.append({
                "feature": col,
                "human_name": _humanize_indicator(col),
                "val": float(val),
                "contribution": float(contrib),
            })
            
        pos_drivers = sorted([d for d in drivers if d["contribution"] > 0], key=lambda d: -d["contribution"])[:top_k]
        neg_drags = sorted([d for d in drivers if d["contribution"] < 0], key=lambda d: d["contribution"])[:top_k]
        
        return pos_drivers, neg_drags
    except Exception:
        return [], []
