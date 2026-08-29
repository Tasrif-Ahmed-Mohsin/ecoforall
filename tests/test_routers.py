"""
Test Suite: Dynamic Gating & Model Selection Routers
=====================================================
Validates DynamicModelSelectionRouter (DMS/DMA) and SovereignSegmentedAdaptiveRouter.
"""

import numpy as np
import pandas as pd
import pytest
from src.gating.dms_state_space_router import DynamicModelSelectionRouter
from src.gating.sovereign_segmentation_router import SovereignSegmentedAdaptiveRouter


def test_dms_router_sequential_adaptation():
    """Verify DMS router adapts model probabilities to better expert."""
    np.random.seed(42)
    N = 50
    years = list(range(1970, 1970 + N))
    df = pd.DataFrame({"iso3": ["USA"] * N, "year": years})
    
    y_true = np.zeros(N)
    # Expert 0 is accurate (noise = 0.01); Expert 1 is inaccurate (noise = 0.5)
    p0 = np.random.normal(0, 0.01, N)
    p1 = np.random.normal(0, 0.50, N)
    expert_preds = np.column_stack([p0, p1])
    
    router = DynamicModelSelectionRouter(n_experts=2, forgetting_factor=0.90, mode="dma")
    y_gated, weights = router.route_panel(df, expert_preds, y_true)
    
    # By the end of 50 steps, weight on Expert 0 should dominate
    final_weight_expert0 = weights[-1, 0]
    assert final_weight_expert0 > 0.70, f"Expected adaptation toward expert 0, got {final_weight_expert0}"


def test_sovereign_segmented_router_convex_fit():
    """Verify SovereignSegmentedAdaptiveRouter learns valid convex combination weights."""
    np.random.seed(42)
    N = 100
    df = pd.DataFrame({
        "iso3": ["USA"] * 50 + ["DEU"] * 50,
        "year": list(range(1970, 2020)) + list(range(1970, 2020))
    })
    y_true = np.random.normal(0.02, 0.01, N)
    
    # 4 specialist predictions
    e1 = y_true + np.random.normal(0, 0.005, N)
    e2 = np.random.normal(0, 0.1, N)
    e3 = np.random.normal(0, 0.1, N)
    e4 = np.random.normal(0, 0.1, N)
    expert_preds = np.column_stack([e1, e2, e3, e4])
    
    router = SovereignSegmentedAdaptiveRouter(horizon=1)
    router.fit_from_predictions(df, expert_preds, y_true)
    
    # Check that weights sum to 1 and weight on e1 dominates
    assert np.isclose(np.sum(router.global_weights_), 1.0)
    assert router.global_weights_[0] > 0.50
    
    preds = router.predict_blend(df["iso3"].tolist(), expert_preds)
    assert len(preds) == N
    assert np.all(np.isfinite(preds))
