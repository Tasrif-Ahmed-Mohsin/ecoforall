"""
4D Country-Year Historical Twin Retrieval Engine
================================================
Rank-Euclidean FAISS similarity search across multi-dimensional developmental
trajectories (Economy, Politics/GDELT, Environment, Society/Psychology).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import faiss


@dataclass
class TwinMatch:
    query_iso3: str
    query_year: int
    twin_iso3: str
    twin_year: int
    similarity_score: float
    distance: float
    future_growth_1y: Optional[float] = None
    future_growth_5y: Optional[float] = None


class QuadDomainTwinEngine:
    def __init__(self, feature_cols: List[str]):
        self.feature_cols = feature_cols
        self.index: Optional[faiss.IndexFlatL2] = None
        self.meta_df: Optional[pd.DataFrame] = None
        self.col_sorted_values: Dict[int, np.ndarray] = {}
        self.avail_cols: List[str] = []
        self.is_fitted = False

    def fit(self, panel_df: pd.DataFrame) -> QuadDomainTwinEngine:
        """Build Rank-Euclidean FAISS Index on normalized country-year features."""
        df_clean = panel_df.dropna(subset=["iso3", "year"]).copy()
        
        # Available feature intersection
        self.avail_cols = [c for c in self.feature_cols if c in df_clean.columns]
        X = df_clean[self.avail_cols].fillna(0.0).values.astype(np.float32)

        # Scale-invariant rank transformation per column
        X_ranked = np.zeros_like(X)
        self.col_sorted_values = {}
        for j in range(X.shape[1]):
            col = X[:, j]
            sorted_vals = np.sort(col)
            self.col_sorted_values[j] = sorted_vals
            # Percentile rank [0, 1]
            ranks = np.searchsorted(sorted_vals, col).astype(np.float32) / max(1.0, float(len(col) - 1))
            X_ranked[:, j] = ranks

        # Build FAISS L2 Euclidean index on ranked coordinates
        d = X_ranked.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(X_ranked)

        self.index = index
        self.meta_df = df_clean[["iso3", "year"]].reset_index(drop=True)
        if "gdp_pc_growth_1y_fwd" in df_clean.columns:
            self.meta_df["growth_1y"] = df_clean["gdp_pc_growth_1y_fwd"].values
        if "gdp_pc_growth_5y_fwd" in df_clean.columns:
            self.meta_df["growth_5y"] = df_clean["gdp_pc_growth_5y_fwd"].values

        self.is_fitted = True
        return self

    def find_twins(
        self,
        query_row: pd.Series | dict,
        k: int = 5,
        exclude_same_country: bool = True
    ) -> List[TwinMatch]:
        """Retrieve top-K historical country-year analog twins."""
        if not self.is_fitted or self.index is None or self.meta_df is None:
            raise ValueError("Twin Engine is not fitted.")

        # Transform query row into ranked coordinates
        q_ranked = np.zeros((1, len(self.avail_cols)), dtype=np.float32)
        for j, col_name in enumerate(self.avail_cols):
            val = query_row.get(col_name, 0.0)
            if np.isnan(val) or not np.isfinite(val):
                val = 0.0
            sorted_vals = self.col_sorted_values[j]
            rank = float(np.searchsorted(sorted_vals, val)) / max(1.0, float(len(sorted_vals) - 1))
            q_ranked[0, j] = rank
        
        # Query FAISS
        search_k = min(len(self.meta_df), max(k * 8, 50) if exclude_same_country else k)
        distances, indices = self.index.search(q_ranked, search_k)

        matches: List[TwinMatch] = []
        q_iso3 = query_row.get("iso3", "")
        q_year = int(query_row.get("year", 0))

        dim = float(len(self.avail_cols))
        max_dist = np.sqrt(dim)

        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.meta_df):
                continue
            t_row = self.meta_df.iloc[idx]
            t_iso3 = t_row["iso3"]
            t_year = int(t_row["year"])

            if exclude_same_country and t_iso3 == q_iso3:
                continue

            # Normalized Euclidean distance to similarity percentage
            dist_norm = np.sqrt(max(0.0, float(dist))) / max_dist
            similarity = max(0.0, min(100.0, (1.0 - dist_norm) * 100.0))
            
            matches.append(TwinMatch(
                query_iso3=q_iso3,
                query_year=q_year,
                twin_iso3=t_iso3,
                twin_year=t_year,
                similarity_score=round(similarity, 1),
                distance=round(float(dist), 3),
                future_growth_1y=t_row.get("growth_1y") if pd.notna(t_row.get("growth_1y")) else None,
                future_growth_5y=t_row.get("growth_5y") if pd.notna(t_row.get("growth_5y")) else None
            ))

            if len(matches) >= k:
                break

        return matches
