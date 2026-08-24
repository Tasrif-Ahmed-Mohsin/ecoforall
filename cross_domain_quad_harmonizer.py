"""
Quad-Domain Dataset Harmonizer & Feature Pipeline (Economy, Politics, Environment, Human/Society)
--------------------------------------------------------------------------------------------------
Merges Economic, Political, Environmental, and Collective Psychology & Society panels on [iso3, year].
Constructs lag features, rolling statistics, percentile ranks, and multi-horizon targets across all 4 domains.
"""

import os
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

ECO_POL_PANEL_PATH = "data/joint_annual_eco_political_panel.parquet"
ENV_PANEL_PATH = "environment/data/environment_65yr_panel.parquet"
HUMAN_PANEL_PATH = "human/data/dataset_wide.parquet"
OUTPUT_QUAD_PANEL_PATH = "data/quad_domain_annual_panel.parquet"

ENV_INDICATORS = [
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

HUMAN_INDICATORS = [
    "psychology_trust",
    "psychology_fear",
    "psychology_optimism",
    "psychology_nationalism",
    "psychology_social_cohesion",
    "psychology_confidence",
    "society_education",
    "society_urbanization",
    "society_population",
    "society_age",
    "society_religion",
    "society_healthcare",
    "society_migration"
]


class QuadDomainDatasetHarmonizer:
    def __init__(self):
        pass

    def load_and_merge_datasets(self):
        logging.info("Loading Eco-Pol, Environment, and Human/Society panels...")
        if not os.path.exists(ECO_POL_PANEL_PATH):
            raise FileNotFoundError(f"Missing {ECO_POL_PANEL_PATH}")

        df_eco_pol = pd.read_parquet(ECO_POL_PANEL_PATH)

        if os.path.exists(ENV_PANEL_PATH):
            df_env = pd.read_parquet(ENV_PANEL_PATH)
        elif os.path.exists("environment/data/environment_65yr_panel.csv"):
            df_env = pd.read_csv("environment/data/environment_65yr_panel.csv")
        else:
            raise FileNotFoundError("Missing Environment dataset")

        if os.path.exists(HUMAN_PANEL_PATH):
            df_human_raw = pd.read_parquet(HUMAN_PANEL_PATH)
        elif os.path.exists("human/data/dataset_canonical.parquet"):
            df_human_raw = pd.read_parquet("human/data/dataset_canonical.parquet")
        else:
            raise FileNotFoundError("Missing Human dataset")

        # Standardize formatting for join keys
        df_eco_pol["iso3"] = df_eco_pol["iso3"].astype(str).str.upper().str.strip()
        df_env["iso3"] = df_env["iso3"].astype(str).str.upper().str.strip()
        df_human_raw["iso3"] = df_human_raw["iso3"].astype(str).str.upper().str.strip()

        df_eco_pol["year"] = df_eco_pol["year"].astype(int)
        df_env["year"] = df_env["year"].astype(int)

        if "timestamp" in df_human_raw.columns:
            df_human_raw["year"] = pd.to_datetime(df_human_raw["timestamp"]).dt.year
        elif "year" not in df_human_raw.columns:
            raise ValueError("Human dataset missing year column")
        df_human_raw["year"] = df_human_raw["year"].astype(int)

        # Aggregate Human indicators to annual panel [iso3, year]
        human_cols = [c for c in HUMAN_INDICATORS if c in df_human_raw.columns]
        df_human_annual = df_human_raw.groupby(["iso3", "year"])[human_cols].mean().reset_index()

        logging.info(f"Eco-Pol panel: {df_eco_pol.shape} | Env panel: {df_env.shape} | Human annual panel: {df_human_annual.shape}")

        # 3-way outer merge on [iso3, year]
        df_merged = pd.merge(df_eco_pol, df_env, on=["iso3", "year"], how="outer")
        df_merged = pd.merge(df_merged, df_human_annual, on=["iso3", "year"], how="outer")
        df_merged = df_merged.sort_values(["iso3", "year"]).reset_index(drop=True)

        logging.info(f"Merged Quad-Domain dataset shape: {df_merged.shape}")
        return df_merged

    def compute_percentile_ranks(self, df):
        logging.info("Computing yearly percentile rank scaling for key indicators...")
        df_ranked = df.copy()

        rank_targets = [c for c in (ENV_INDICATORS + HUMAN_INDICATORS) if c in df_ranked.columns]
        for col in rank_targets:
            df_ranked[f"{col}_rank"] = df_ranked.groupby("year")[col].rank(pct=True)

        return df_ranked

    def build_features_and_targets(self, df):
        logging.info("Constructing lag features, rolling statistics, and multi-horizon targets...")
        df_featured = self.compute_percentile_ranks(df)

        present_env = [c for c in ENV_INDICATORS if c in df_featured.columns]
        present_human = [c for c in HUMAN_INDICATORS if c in df_featured.columns]
        target_indicators = present_env + present_human

        new_cols = {}

        # Multi-horizon Targets (1, 3, 5 years forward)
        for h in [1, 3, 5]:
            for col in target_indicators:
                new_cols[f"{col}_target_h{h}"] = df_featured.groupby("iso3")[col].shift(-h)
                new_cols[f"{col}_diff_h{h}"] = new_cols[f"{col}_target_h{h}"] - df_featured[col]

        # Historical Lags & Rolling Statistics
        for col in target_indicators:
            for lag in [1, 3, 5]:
                new_cols[f"{col}_lag_{lag}"] = df_featured.groupby("iso3")[col].shift(lag)

            new_cols[f"{col}_roll_mean_10y"] = df_featured.groupby("iso3")[col].transform(
                lambda x: x.shift(1).rolling(window=10, min_periods=3).mean()
            )
            new_cols[f"{col}_roll_std_10y"] = df_featured.groupby("iso3")[col].transform(
                lambda x: x.shift(1).rolling(window=10, min_periods=3).std()
            )
            new_cols[f"{col}_velocity_3y"] = df_featured[col] - new_cols.get(f"{col}_lag_3", df_featured.groupby("iso3")[col].shift(3))

        df_concat = pd.concat([df_featured, pd.DataFrame(new_cols, index=df_featured.index)], axis=1)
        return df_concat

    def run(self):
        df_merged = self.load_and_merge_datasets()
        df_final = self.build_features_and_targets(df_merged)

        os.makedirs("data", exist_ok=True)
        df_final.to_parquet(OUTPUT_QUAD_PANEL_PATH, index=False)
        logging.info(f"Quad-Domain annual panel saved to {OUTPUT_QUAD_PANEL_PATH} with {df_final.shape[1]} columns.")
        return df_final


def main():
    harmonizer = QuadDomainDatasetHarmonizer()
    df_panel = harmonizer.run()
    print(f"Quad-Domain Harmonizer complete! Dataset shape: {df_panel.shape}")


if __name__ == "__main__":
    main()
