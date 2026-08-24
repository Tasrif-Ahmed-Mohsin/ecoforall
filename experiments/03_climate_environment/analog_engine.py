"""
Rank-Euclidean Similarity & Historical Trajectory Analog Engine.

Retrieves top-K historical environmental trajectory twins using scale-invariant
Rank-Euclidean state distance over 65 years of longitudinal data (1960-2025).
"""

import os
import numpy as np
import pandas as pd
import yaml
from scipy.spatial.distance import cdist

class EnvironmentalAnalogEngine:
    def __init__(self, harmonized_df, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.top_k = self.config["retrieval"].get("top_k", 8)
        self.horizons = self.config["forecasting"]["horizons"]
        self.df = harmonized_df.copy()
        
        # Identify feature columns for similarity matching
        self.feature_cols = [c for c in self.df.columns if c.endswith("_rank") or c.endswith("_roll_mean_10y")]
        self.valid_df = self.df.dropna(subset=self.feature_cols).reset_index(drop=True)
        
        # Normalize feature matrix for L2 distance matching
        self.feature_matrix = self.valid_df[self.feature_cols].values
        
    def find_analogs(self, query_iso3, query_year, top_k=None):
        """Finds top-K historical analogs for a specified country and year."""
        if top_k is None:
            top_k = self.top_k
            
        # Locate query index
        query_mask = (self.valid_df["iso3"] == query_iso3) & (self.valid_df["year"] == query_year)
        if not query_mask.any():
            raise ValueError(f"Query entity '{query_iso3}' at year {query_year} not found in valid index.")
            
        query_idx = self.valid_df[query_mask].index[0]
        query_vec = self.feature_matrix[query_idx:query_idx+1]
        
        # Compute Rank-Euclidean distances across all entity-year slices
        distances = cdist(query_vec, self.feature_matrix, metric="euclidean")[0]
        
        # Filter out self-match (same iso3 and same year)
        valid_indices = []
        for idx, dist in enumerate(distances):
            row = self.valid_df.iloc[idx]
            # Exclude exact query match and immediate +-1 year of same entity
            if row["iso3"] == query_iso3 and abs(row["year"] - query_year) <= 1:
                continue
            valid_indices.append((idx, dist))
            
        # Sort by distance (ascending)
        valid_indices.sort(key=lambda x: x[1])
        top_matches = valid_indices[:top_k]
        
        results = []
        max_dist = valid_indices[-1][1] if valid_indices else 1.0
        
        for idx, dist in top_matches:
            matched_row = self.valid_df.iloc[idx]
            match_confidence = max(0.0, 1.0 - (dist / (max_dist + 1e-5)))
            
            # Extract forward trajectories for primary target
            forward_path = {}
            for h in self.horizons:
                col = f"co2_emissions_per_capita_target_h{h}"
                val = matched_row[col] if col in matched_row else np.nan
                forward_path[f"h_{h}y"] = val
                
            results.append({
                "analog_iso3": matched_row["iso3"],
                "analog_year": int(matched_row["year"]),
                "distance": float(np.round(dist, 4)),
                "similarity_score": float(np.round(match_confidence, 4)),
                "current_co2": float(matched_row["co2_emissions_per_capita"]),
                "current_temp_anomaly": float(matched_row["temp_anomaly_celsius"]),
                "forward_trajectories": forward_path
            })
            
        return pd.DataFrame(results)

def main():
    from cross_domain_dataset_harmonizer import EnvironmentDatasetHarmonizer
    print("Testing Environmental Analog Retrieval Engine...")
    harmonizer = EnvironmentDatasetHarmonizer()
    df_featured = harmonizer.build_features_and_targets()
    
    engine = EnvironmentalAnalogEngine(df_featured)
    analogs = engine.find_analogs("USA", 2015, top_k=5)
    print("\nTop Historical Analogs for USA (2015):")
    print(analogs[["analog_iso3", "analog_year", "similarity_score", "current_co2", "current_temp_anomaly"]])

if __name__ == "__main__":
    main()
