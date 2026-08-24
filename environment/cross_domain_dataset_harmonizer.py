"""
Cross-Domain Dataset Harmonizer & Feature Pipeline for Environmental Time Series.

Handles canonical schema conversion, percentile rank transformations, lag generation,
rolling statistics calculation, and multi-horizon target assembly across ALL indicators
including individual disaster breakdowns (floods, droughts, wildfires, storms, extreme temps).
"""

import os
import numpy as np
import pandas as pd
import yaml

INDICATOR_COLS = [
    "co2_emissions_per_capita",
    "temp_anomaly_celsius",
    "forest_area_pct_land",
    "floods_count",
    "droughts_count",
    "wildfires_count",
    "extreme_temp_count",
    "storms_count",
    "disaster_economic_damage_usd",
    "renewable_energy_pct_share",
    "greenhouse_gas_total_kt",
    "energy_use_per_capita",
    "protected_area_pct"
]

class EnvironmentDatasetHarmonizer:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.horizons = self.config["forecasting"]["horizons"]
        self.target_indicators = self.config["forecasting"].get("target_indicators", INDICATOR_COLS)
        self.primary_target = self.config["forecasting"].get("primary_target", self.target_indicators[0])
        self.lookback_window = self.config["retrieval"]["lookback_window"]
        
    def load_raw_data(self, data_path=None):
        if data_path is None:
            data_path = os.path.join("data", "environment_65yr_panel.parquet")
            if not os.path.exists(data_path):
                data_path = os.path.join("data", "environment_65yr_panel.csv")
                
        if data_path.endswith(".parquet"):
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_csv(data_path)
            
        df = df.sort_values(["iso3", "year"]).reset_index(drop=True)
        return df

    def compute_percentile_ranks(self, df):
        """Scale continuous indicator vectors to [0, 1] uniform rank percentiles per year slice."""
        df_ranked = df.copy()
        
        for col in INDICATOR_COLS:
            if col in df.columns:
                df_ranked[f"{col}_rank"] = df_ranked.groupby("year")[col].rank(pct=True)
                
        return df_ranked

    def build_features_and_targets(self, df=None):
        """Constructs lag features, rolling statistics, rank features, and multi-horizon targets."""
        if df is None:
            df = self.load_raw_data()
            
        df = self.compute_percentile_ranks(df)
        df_featured = df.copy()
        
        present_indicators = [c for c in INDICATOR_COLS if c in df_featured.columns]
        
        # 1. Multi-horizon targets: y_{t+h}
        new_cols = {}
        for h in self.horizons:
            for col in present_indicators:
                new_cols[f"{col}_target_h{h}"] = df_featured.groupby("iso3")[col].shift(-h)
                new_cols[f"{col}_diff_h{h}"] = new_cols[f"{col}_target_h{h}"] - df_featured[col]

        # 2. Historical Lags & Rolling Statistics per ISO3
        for col in present_indicators:
            for lag in [1, 3, 5]:
                new_cols[f"{col}_lag_{lag}"] = df_featured.groupby("iso3")[col].shift(lag)
                
            new_cols[f"{col}_roll_mean_10y"] = df_featured.groupby("iso3")[col].transform(
                lambda x: x.shift(1).rolling(window=10, min_periods=3).mean()
            )
            new_cols[f"{col}_roll_std_10y"] = df_featured.groupby("iso3")[col].transform(
                lambda x: x.shift(1).rolling(window=10, min_periods=3).std()
            )
            new_cols[f"{col}_velocity_3y"] = df_featured[col] - new_cols[f"{col}_lag_3"]
            
        df_featured = pd.concat([df_featured, pd.DataFrame(new_cols, index=df_featured.index)], axis=1)
        return df_featured

    def extract_state_vectors(self, df_featured):
        """Extracts scale-invariant rank state feature vectors for retrieval & similarity search."""
        rank_cols = [c for c in df_featured.columns if c.endswith("_rank")]
        velocity_cols = [c for c in df_featured.columns if c.endswith("_velocity_3y")]
        feature_cols = rank_cols + velocity_cols
        
        return df_featured[["iso3", "year"] + feature_cols].dropna().reset_index(drop=True), feature_cols

def main():
    print("Executing Cross-Domain Dataset Harmonizer for Environment Data...")
    harmonizer = EnvironmentDatasetHarmonizer()
    df_raw = harmonizer.load_raw_data()
    print(f"Loaded raw panel: {df_raw.shape}")
    
    df_featured = harmonizer.build_features_and_targets(df_raw)
    print(f"Constructed feature panel: {df_featured.shape}")
    
    output_path = os.path.join("data", "environment_harmonized_features.parquet")
    df_featured.to_parquet(output_path, index=False)
    print(f"Harmonized feature dataset saved to: {output_path}")

if __name__ == "__main__":
    main()
