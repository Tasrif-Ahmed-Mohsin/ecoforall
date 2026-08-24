"""
Sovereign Segmentation & Heterogeneous Regime Router
===================================================
Solves the fundamental challenge of macroeconomic heterogeneity across global panels:
  - Low-Volatility / Institutional Steady-State Economies (e.g., USA, Western Europe)
    benefit from strong low-variance autoregressive regularizers in calm periods.
  - Emerging / High-Growth / Structural Transition Economies (e.g., Asia, LatAm, Africa)
    derive massive predictive lift from non-linear Quad-Domain specialists (Politics, Climate, Society).
  - Resource & Commodity Transition Economies (e.g., MENA)
    require balanced stress and commodity-shock routing.

This module dynamically classifies countries into sovereign typologies using historical
training-slice volatility, institutional variance, and geographic region, adapting routing priors.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

# World Bank Geographic Region Mapping
REGION_MAP = {
    "USA": "North America", "CAN": "North America",
    "DEU": "Europe & Central Asia", "FRA": "Europe & Central Asia", "GBR": "Europe & Central Asia",
    "ITA": "Europe & Central Asia", "ESP": "Europe & Central Asia", "NLD": "Europe & Central Asia",
    "CHE": "Europe & Central Asia", "SWE": "Europe & Central Asia", "NOR": "Europe & Central Asia",
    "POL": "Europe & Central Asia", "AUT": "Europe & Central Asia", "BEL": "Europe & Central Asia",
    "RUS": "Europe & Central Asia", "TUR": "Europe & Central Asia", "UKR": "Europe & Central Asia",
    "GRC": "Europe & Central Asia", "PRT": "Europe & Central Asia", "IRL": "Europe & Central Asia",
    "FIN": "Europe & Central Asia", "DNK": "Europe & Central Asia", "CZE": "Europe & Central Asia",
    "ROU": "Europe & Central Asia", "HUN": "Europe & Central Asia", "KAZ": "Europe & Central Asia",
    "CHN": "East Asia & Pacific", "JPN": "East Asia & Pacific", "KOR": "East Asia & Pacific",
    "AUS": "East Asia & Pacific", "IDN": "East Asia & Pacific", "MYS": "East Asia & Pacific",
    "PHL": "East Asia & Pacific", "SGP": "East Asia & Pacific", "THA": "East Asia & Pacific",
    "VNM": "East Asia & Pacific", "NZL": "East Asia & Pacific", "MMR": "East Asia & Pacific",
    "BRA": "Latin America & Caribbean", "MEX": "Latin America & Caribbean", "ARG": "Latin America & Caribbean",
    "CHL": "Latin America & Caribbean", "COL": "Latin America & Caribbean", "PER": "Latin America & Caribbean",
    "VEN": "Latin America & Caribbean", "ECU": "Latin America & Caribbean", "URY": "Latin America & Caribbean",
    "BOL": "Latin America & Caribbean", "PRY": "Latin America & Caribbean", "PAN": "Latin America & Caribbean",
    "CRI": "Latin America & Caribbean", "DOM": "Latin America & Caribbean", "GTM": "Latin America & Caribbean",
    "IND": "South Asia", "PAK": "South Asia", "BGD": "South Asia",
    "LKA": "South Asia", "NPL": "South Asia", "AFG": "South Asia",
    "NGA": "Sub-Saharan Africa", "ZAF": "Sub-Saharan Africa", "KEN": "Sub-Saharan Africa",
    "ETH": "Sub-Saharan Africa", "GHA": "Sub-Saharan Africa", "TZA": "Sub-Saharan Africa",
    "UGA": "Sub-Saharan Africa", "AGO": "Sub-Saharan Africa", "CIV": "Sub-Saharan Africa",
    "SEN": "Sub-Saharan Africa", "CMR": "Sub-Saharan Africa", "ZMB": "Sub-Saharan Africa",
    "ZWE": "Sub-Saharan Africa", "MOZ": "Sub-Saharan Africa", "RWA": "Sub-Saharan Africa",
    "SAU": "Middle East & North Africa", "EGY": "Middle East & North Africa", "IRN": "Middle East & North Africa",
    "ISR": "Middle East & North Africa", "ARE": "Middle East & North Africa", "IRQ": "Middle East & North Africa",
    "DZA": "Middle East & North Africa", "MAR": "Middle East & North Africa", "QAT": "Middle East & North Africa",
    "KWT": "Middle East & North Africa", "OMN": "Middle East & North Africa", "JOR": "Middle East & North Africa",
    "TUN": "Middle East & North Africa", "LBN": "Middle East & North Africa",
}


def get_wb_region(iso3: str) -> str:
    return REGION_MAP.get(iso3, "Other / Emerging")


class SovereignSegmentedAdaptiveRouter(BaseEstimator, RegressorMixin):
    """
    Sovereign-Segmented Adaptive Gating Router.
    
    Dynamically blends specialist predictions based on sovereign typologies:
      - w_ar1: Weight on autoregressive baseline (for steady-state variance reduction)
      - w_eco: Weight on regularized single-domain economic specialist
      - w_quad: Weight on non-linear quad-domain specialist (LightGBM)
      - w_huber: Weight on robust tail-shock specialist (Huber)
    """

    def __init__(self, horizon: int = 1):
        self.horizon = horizon
        self.vol_map_: dict[str, float] = {}
        self.med_vol_: float = 0.03

    def fit(self, train_df: pd.DataFrame, target_col: str) -> SovereignSegmentedAdaptiveRouter:
        """
        Fit sovereign typology statistics strictly on training data slice.
        """
        self.vol_map_ = {}
        for iso, grp in train_df.groupby("iso3"):
            vol = grp[target_col].std()
            if np.isfinite(vol) and vol > 0:
                self.vol_map_[iso] = float(vol)
            else:
                self.vol_map_[iso] = 0.03

        vols = list(self.vol_map_.values())
        self.med_vol_ = float(np.median(vols)) if len(vols) > 0 else 0.03
        return self

    def predict_blend(
        self,
        iso3_list: list[str],
        pred_ar1: np.ndarray,
        pred_eco: np.ndarray,
        pred_quad: np.ndarray,
        pred_huber: np.ndarray,
    ) -> np.ndarray:
        """
        Compute typology-adapted predictions for test country-years.
        """
        out = np.zeros(len(iso3_list), dtype=np.float64)
        h = self.horizon

        for i, iso in enumerate(iso3_list):
            reg = get_wb_region(iso)
            vol = self.vol_map_.get(iso, self.med_vol_)
            is_low_vol = vol < self.med_vol_

            # Horizon-calibrated typology routing weights
            if h == 1:
                # 1-Year Horizon: High AR(1) utility in calm periods, selective quad in emerging
                if reg in ["North America"] or (reg in ["Europe & Central Asia"] and is_low_vol):
                    # Developed Steady-State
                    w = [0.70, 0.20, 0.10, 0.00]
                elif reg in ["East Asia & Pacific", "South Asia", "Sub-Saharan Africa", "Latin America & Caribbean"]:
                    # Emerging / High Volatility
                    w = [0.15, 0.35, 0.35, 0.15]
                else:
                    # Mixed Transition
                    w = [0.35, 0.35, 0.20, 0.10]

            elif h == 3:
                # 3-Year Horizon: Medium horizon balance
                if reg in ["North America"] or (reg in ["Europe & Central Asia"] and is_low_vol):
                    w = [0.55, 0.25, 0.15, 0.05]
                elif reg in ["East Asia & Pacific", "South Asia", "Sub-Saharan Africa", "Latin America & Caribbean"]:
                    w = [0.10, 0.30, 0.45, 0.15]
                else:
                    w = [0.20, 0.35, 0.30, 0.15]

            else:  # h >= 5
                # 5-Year Horizon: Structural non-linear features dominate emerging; AR(1) regularizes developed
                if reg in ["North America"] or (reg in ["Europe & Central Asia"] and is_low_vol):
                    w = [0.60, 0.25, 0.10, 0.05]
                elif reg in ["East Asia & Pacific", "South Asia", "Sub-Saharan Africa", "Latin America & Caribbean"]:
                    w = [0.05, 0.30, 0.45, 0.20]
                else:
                    w = [0.15, 0.35, 0.35, 0.15]

            pred_val = (
                w[0] * pred_ar1[i]
                + w[1] * pred_eco[i]
                + w[2] * pred_quad[i]
                + w[3] * pred_huber[i]
            )
            out[i] = np.clip(pred_val, -0.5, 0.5)

        return out
