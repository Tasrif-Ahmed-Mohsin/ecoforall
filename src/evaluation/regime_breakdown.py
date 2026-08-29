"""
Regime-Conditional Forecast Evaluation & Dilution Audit
======================================================
Evaluates sovereign growth forecasting performance disaggregated across:
  1. Institutional Transition Regimes (|Delta V-Dem| >= 0.05 in 3yr window) vs Stable Regimes
  2. Macro-Financial Shock Regimes (GFC 2008-09, COVID 2020, Growth < -3%) vs Tranquil Regimes

Directly tests Proposition 1 (Information Dilution in Tranquil Regimes vs Stress Utility).
"""

from __future__ import annotations
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd


def identify_regimes(
    df: pd.DataFrame,
    vdem_col: str = "vdem_electoral_democracy",
    lag_growth_col: str = "growth_into_origin",
    year_col: str = "year",
    iso_col: str = "iso3",
) -> pd.DataFrame:
    r"""
    Label each country-year with strictly backward-looking regime indicators (no future leakage).

    - institutional_regime: 'Transition / Stress' (|Delta_3y VDem| >= 0.05) vs 'Stable' (< 0.02)
    - macro_regime: 'Crisis / Shock' (Global GFC 2008-09, COVID 2020, or Growth < -3%) vs 'Tranquil Expansion' (Growth > 1%)
    """
    d = df.sort_values([iso_col, year_col]).copy()

    # 3-year trailing change in political stability/democracy (backward looking only)
    if vdem_col in d.columns:
        vdem_lag3 = d.groupby(iso_col)[vdem_col].shift(3)
        d["vdem_delta_3y"] = (d[vdem_col] - vdem_lag3).abs()
        d["is_inst_transition"] = d["vdem_delta_3y"] >= 0.05
        d["is_inst_stable"] = d["vdem_delta_3y"] < 0.02
    else:
        d["is_inst_transition"] = False
        d["is_inst_stable"] = True

    # Trailing growth into origin
    if lag_growth_col in d.columns:
        growth = pd.to_numeric(d[lag_growth_col], errors="coerce")
    else:
        growth = pd.Series(0.02, index=d.index)

    years = pd.to_numeric(d[year_col], errors="coerce")

    # Global shock years or severe national contraction
    is_global_shock = years.isin([2008, 2009, 2020])
    is_severe_drop = growth < -0.03
    is_tranquil_growth = growth > 0.01

    d["is_macro_crisis"] = is_global_shock | is_severe_drop
    d["is_macro_tranquil"] = is_tranquil_growth & (~is_global_shock)

    return d


def evaluate_regime_performance(
    df: pd.DataFrame,
    target_col: str,
    model_pred_cols: Dict[str, str],
    regime_mask: pd.Series,
    regime_name: str,
    horizon: int = 1,
) -> Dict[str, Any]:
    """Calculate performance metrics across models for a specific regime subset."""
    sub = df[regime_mask].dropna(subset=[target_col]).copy()
    n_obs = len(sub)
    if n_obs < 10:
        return {"Regime": regime_name, "N_obs": n_obs, "Horizon": horizon}

    y_true = sub[target_col].to_numpy(dtype=np.float64)

    res = {
        "Regime": regime_name,
        "N_obs": n_obs,
        "Horizon": horizon,
    }

    # AR(1) benchmark MAE for relative lift calculation
    ar1_col = model_pred_cols.get("AR(1) Baseline")
    mae_ar1 = float(np.mean(np.abs(y_true - sub[ar1_col].to_numpy(dtype=np.float64)))) if ar1_col in sub else None

    # Eco Ridge for dilution calculation
    eco_col = model_pred_cols.get("Economy-Only Ridge")
    mae_eco = float(np.mean(np.abs(y_true - sub[eco_col].to_numpy(dtype=np.float64)))) if eco_col in sub else None

    for model_name, pred_col in model_pred_cols.items():
        if pred_col not in sub.columns:
            continue
        preds = sub[pred_col].to_numpy(dtype=np.float64)
        mae = float(np.mean(np.abs(y_true - preds)))
        rmse = float(np.sqrt(np.mean((y_true - preds) ** 2)))
        res[f"MAE_{model_name}"] = round(mae, 5)
        res[f"RMSE_{model_name}"] = round(rmse, 5)

        if mae_ar1 is not None and mae_ar1 > 1e-12:
            res[f"Lift_vs_AR1_{model_name}_pct"] = round(((mae_ar1 - mae) / mae_ar1) * 100.0, 2)
        else:
            res[f"Lift_vs_AR1_{model_name}_pct"] = 0.0

    # Dilution penalty check: All-Domain Ridge MAE - Eco-Only Ridge MAE
    all_col = model_pred_cols.get("All-Domain Ridge (Concat)")
    if eco_col in sub and all_col in sub and mae_eco is not None:
        mae_all = float(np.mean(np.abs(y_true - sub[all_col].to_numpy(dtype=np.float64))))
        if mae_eco > 1e-12:
            res["Ridge_Concat_Penalty_pct"] = round(((mae_all - mae_eco) / mae_eco) * 100.0, 2)
        else:
            res["Ridge_Concat_Penalty_pct"] = 0.0

    return res
