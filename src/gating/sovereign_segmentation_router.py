"""
Sovereign Segmented Dynamic Router (Principled Cross-Validation Weight Learning)
================================================================================
Replaces hardcoded heuristics with inner-fold optimization:
Estimates optimal specialist combination weights per sovereign typology
strictly using inner-fold historical out-of-fold training data.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler


def get_wb_region(iso3: str) -> str:
    """Official World Bank regional classification mapping dynamically loaded from official data."""
    wb_csv = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "world_bank_country_metadata.csv"
    if wb_csv.exists():
        try:
            wb_df = pd.read_csv(wb_csv)
            mapping = dict(zip(wb_df["iso3"], wb_df["region"]))
            if iso3 in mapping and pd.notna(mapping[iso3]):
                return str(mapping[iso3])
        except Exception:
            pass

    # Standard fallback for historical entities or non-standard ISO3
    REGION_MAP: Dict[str, str] = {
        "USA": "North America", "CAN": "North America", "BMU": "North America", "GRL": "North America",
        "DEU": "Europe & Central Asia", "GBR": "Europe & Central Asia", "FRA": "Europe & Central Asia",
        "ITA": "Europe & Central Asia", "ESP": "Europe & Central Asia", "NLD": "Europe & Central Asia",
        "CHE": "Europe & Central Asia", "SWE": "Europe & Central Asia", "NOR": "Europe & Central Asia",
        "POL": "Europe & Central Asia", "AUT": "Europe & Central Asia", "BEL": "Europe & Central Asia",
        "DNK": "Europe & Central Asia", "FIN": "Europe & Central Asia", "IRL": "Europe & Central Asia",
        "PRT": "Europe & Central Asia", "GRC": "Europe & Central Asia", "CZE": "Europe & Central Asia",
        "ROU": "Europe & Central Asia", "HUN": "Europe & Central Asia", "RUS": "Europe & Central Asia",
        "TUR": "Europe & Central Asia", "UKR": "Europe & Central Asia", "KAZ": "Europe & Central Asia",
        "CHN": "East Asia & Pacific", "JPN": "East Asia & Pacific", "KOR": "East Asia & Pacific",
        "AUS": "East Asia & Pacific", "NZL": "East Asia & Pacific", "IDN": "East Asia & Pacific",
        "MYS": "East Asia & Pacific", "THA": "East Asia & Pacific", "VNM": "East Asia & Pacific",
        "PHL": "East Asia & Pacific", "SGP": "East Asia & Pacific", "HKG": "East Asia & Pacific",
        "IND": "South Asia", "PAK": "South Asia", "BGD": "South Asia", "LKA": "South Asia", "NPL": "South Asia",
        "BRA": "Latin America & Caribbean", "MEX": "Latin America & Caribbean", "ARG": "Latin America & Caribbean",
        "COL": "Latin America & Caribbean", "CHL": "Latin America & Caribbean", "PER": "Latin America & Caribbean",
        "SAU": "Middle East & North Africa", "ARE": "Middle East & North Africa", "EGY": "Middle East & North Africa",
        "IRN": "Middle East & North Africa", "ISR": "Middle East & North Africa", "QAT": "Middle East & North Africa",
        "ZAF": "Sub-Saharan Africa", "NGA": "Sub-Saharan Africa", "KEN": "Sub-Saharan Africa", "ETH": "Sub-Saharan Africa",
        "GHA": "Sub-Saharan Africa", "TZA": "Sub-Saharan Africa", "AGO": "Sub-Saharan Africa"
    }
    return REGION_MAP.get(iso3, "Other / Emerging")


class SovereignSegmentedAdaptiveRouter:
    """
    Learns dynamic combination weights per region/typology strictly within inner training folds.
    Enforces simplex constraint: sum(w_m) = 1, w_m >= 0.
    """

    def __init__(self, horizon: int = 1):
        self.horizon = horizon
        self.learned_weights_: Dict[str, np.ndarray] = {}
        self.global_weights_: np.ndarray = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)

    def fit_from_predictions(self, inner_train_df: pd.DataFrame, expert_preds: np.ndarray, y_true: np.ndarray, iso_col: str = "iso3") -> SovereignSegmentedAdaptiveRouter:
        """
        Fit convex combination weights on training data to minimize MAE loss.
        """
        N, M = expert_preds.shape
        if N < 10:
            return self

        # Objective function: MAE loss with L2 regularization toward uniform prior
        def loss_fn(w: np.ndarray, P: np.ndarray, y: np.ndarray) -> float:
            pred = P @ w
            mae = np.mean(np.abs(y - pred))
            reg = 0.01 * np.sum((w - 1.0 / M) ** 2)
            return float(mae + reg)

        bounds = [(0.0, 1.0)] * M
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        w0 = np.ones(M) / M

        # 1. Global optimal weights
        res_glob = minimize(loss_fn, w0, args=(expert_preds, y_true), method="SLSQP", bounds=bounds, constraints=constraints)
        if res_glob.success:
            self.global_weights_ = np.array(res_glob.x, dtype=np.float64)

        # 2. Per-region optimal weights if sample size permits
        inner_train_df = inner_train_df.reset_index(drop=True)
        regions = inner_train_df[iso_col].apply(get_wb_region)
        
        for reg, group_idx in regions.groupby(regions).groups.items():
            idx = np.array(group_idx)
            if len(idx) >= 20:
                P_sub = expert_preds[idx]
                y_sub = y_true[idx]
                res_reg = minimize(loss_fn, self.global_weights_, args=(P_sub, y_sub), method="SLSQP", bounds=bounds, constraints=constraints)
                if res_reg.success:
                    self.learned_weights_[reg] = np.array(res_reg.x, dtype=np.float64)
                else:
                    self.learned_weights_[reg] = self.global_weights_.copy()
            else:
                self.learned_weights_[reg] = self.global_weights_.copy()

        return self

    def predict_blend(self, iso3_list: List[str], expert_preds: np.ndarray) -> np.ndarray:
        N, M = expert_preds.shape
        out = np.zeros(N, dtype=np.float64)
        
        for i, iso in enumerate(iso3_list):
            reg = get_wb_region(iso)
            w = self.learned_weights_.get(reg, self.global_weights_)
            out[i] = np.clip(np.dot(w, expert_preds[i]), -0.5, 0.5)
            
        return out
