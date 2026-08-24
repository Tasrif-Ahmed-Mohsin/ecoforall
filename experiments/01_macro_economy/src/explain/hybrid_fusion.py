"""Horizon-Adaptive LLM-ML Hybrid Fusion & Scenario Weighting Engine with Outlier Protection."""
from __future__ import annotations

import math
from typing import Any

from src.explain.deepseek_handler import logret_to_human


def get_hybrid_fusion_weight(horizon: int) -> float:
    """Return the ML weight w(h) for horizon h in [0, 1].
    
    w(h=1) = 0.50 (50% ML + 50% LLM)
    w(h=3) = 0.75 (75% ML + 25% LLM)
    w(h=5) = 0.90 (90% ML + 10% LLM)
    w(h=10) = 1.00 (100% ML)
    """
    weights = {1: 0.50, 3: 0.75, 5: 0.90, 10: 1.00}
    return weights.get(horizon, 0.90)


def compute_scenario_weighted_forecast(
    scenario_tree: dict[str, Any],
    ml_ensemble_logret: float | None = None,
    horizon: int = 5,
) -> dict[str, Any]:
    """Compute a data-grounded scenario-weighted point estimate based on historical analog outcomes."""
    scenarios = scenario_tree.get("scenarios", [])
    if not scenarios:
        return {
            "scenario_weighted_logret": ml_ensemble_logret,
            "scenario_weighted_human": logret_to_human(ml_ensemble_logret, horizon),
            "dominant_scenario": "N/A",
            "scenario_count": 0,
        }
    
    weighted_logret = sum(s["mean_growth"] * s["probability"] for s in scenarios)
    dominant_scenario = max(scenarios, key=lambda s: s["probability"])["label"]
    
    return {
        "scenario_weighted_logret": weighted_logret,
        "scenario_weighted_human": logret_to_human(weighted_logret, horizon),
        "dominant_scenario": dominant_scenario,
        "scenario_count": len(scenarios),
        "ml_ensemble_logret": ml_ensemble_logret,
        "ml_ensemble_human": logret_to_human(ml_ensemble_logret, horizon),
    }


def compute_hybrid_forecast(
    ml_ensemble_logret: float | None,
    llm_estimated_logret: float | None,
    horizon: int = 5,
) -> dict[str, Any]:
    """Compute weighted hybrid point estimate with outlier hallucination rejection."""
    w_ml = get_hybrid_fusion_weight(horizon)
    w_llm = 1.0 - w_ml
    
    if ml_ensemble_logret is None and llm_estimated_logret is None:
        return {"hybrid_logret": None, "hybrid_human": "N/A", "weight_ml": w_ml, "weight_llm": w_llm}
        
    if llm_estimated_logret is None:
        hybrid_logret = float(ml_ensemble_logret)
    elif ml_ensemble_logret is None:
        hybrid_logret = float(llm_estimated_logret)
    else:
        # Check if LLM estimate is a wild outlier (annualized growth > 25%/yr or > 0.25 log ret diff from ML)
        llm_annual_rate = abs(math.exp(llm_estimated_logret / max(horizon, 1)) - 1.0)
        diff_from_ml = abs(llm_estimated_logret - ml_ensemble_logret) / max(horizon, 1)
        
        if llm_annual_rate > 0.25 or diff_from_ml > 0.20:
            # Reject LLM hallucination and rely 100% on Grounded ML Ensemble
            hybrid_logret = float(ml_ensemble_logret)
            w_ml = 1.0
            w_llm = 0.0
        else:
            hybrid_logret = w_ml * float(ml_ensemble_logret) + w_llm * float(llm_estimated_logret)
        
    return {
        "hybrid_logret": hybrid_logret,
        "hybrid_human": logret_to_human(hybrid_logret, horizon),
        "weight_ml": w_ml,
        "weight_llm": w_llm,
    }

