"""
Ingestion & Synthetic Data Generator Pipeline for 65-Year Environmental & Climate Panel (1960-2025).

Produces a comprehensive panel dataset across ~200 ISO3 countries with realistic climate anomalies,
CO2 trajectories, extreme disaster event counts, forest coverage, and energy metrics.
"""

import os
import numpy as np
import pandas as pd
import yaml

# Standard ISO3 country codes for realistic panel assembly
ISO3_CODES = [
    "USA", "CHN", "IND", "DEU", "JPN", "GBR", "FRA", "BRA", "CAN", "RUS",
    "AUS", "ITA", "ESP", "KOR", "MEX", "IDN", "SAU", "TUR", "ZAF", "NGA",
    "ARG", "EGY", "PAK", "VNM", "POL", "SWE", "NOR", "DNK", "FIN", "NLD",
    "BEL", "CHE", "AUT", "NZL", "CHL", "COL", "PER", "VEN", "KEN", "ETH",
    "GHA", "BGD", "PHL", "MYS", "THA", "IRN", "IRQ", "ISR", "ARE", "KAZ"
]

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def generate_65yr_environmental_panel():
    """Generates realistic 65-year longitudinal panel dataset (1960-2025)."""
    np.random.seed(42)
    years = np.arange(1960, 2026)
    n_years = len(years)
    
    records = []
    
    for iso3 in ISO3_CODES:
        # Base entity traits
        development_level = np.random.uniform(0.2, 1.0)
        industrial_base = np.random.uniform(0.1, 0.9)
        baseline_temp = np.random.uniform(-10.0, 28.0)
        baseline_forest = np.random.uniform(5.0, 65.0)
        
        # Historical trajectories across 1960-2025
        # Global temperature anomaly trend (+0.018°C per year accelerating post 1980)
        time_index = np.arange(n_years)
        temp_trend = 0.005 * time_index + 0.0003 * (np.maximum(0, time_index - 20) ** 2)
        temp_anomaly = -0.3 + temp_trend + np.random.normal(0, 0.15, size=n_years)
        
        # CO2 emissions per capita trajectory (rise during industrialization, peak/plateau for developed)
        if development_level > 0.6:
            co2_base = 3.0 + 8.0 * (1 / (1 + np.exp(-(time_index - 20)/8))) - 2.5 * (1 / (1 + np.exp(-(time_index - 45)/6)))
        else:
            co2_base = 0.5 + 4.0 * (1 / (1 + np.exp(-(time_index - 35)/10)))
        co2_emissions = np.maximum(0.05, co2_base + np.random.normal(0, 0.2, size=n_years))
        
        # Forest area % (gradual decline, stabilization post 2000 for some)
        deforestation_rate = np.random.uniform(0.1, 0.4)
        forest_pct = np.maximum(1.0, baseline_forest - deforestation_rate * time_index + np.random.normal(0, 0.3, size=n_years))
        
        # Extreme climate disasters count (Poisson process with increasing intensity post 1990)
        disaster_lambda = 0.5 + 0.03 * time_index + 0.05 * np.maximum(0, temp_anomaly)
        extreme_disasters = np.random.poisson(lam=np.maximum(0.1, disaster_lambda))
        
        # Renewable energy share (exponential adoption post 2005)
        renewable_base = 5.0 + 35.0 * (1 / (1 + np.exp(-(time_index - 50)/5)))
        renewable_pct = np.clip(renewable_base + np.random.normal(0, 1.5, size=n_years), 0.5, 95.0)
        
        # Total GHG emissions (kt CO2 eq)
        ghg_total = co2_emissions * (1000 + industrial_base * 50000) * np.random.uniform(0.9, 1.1, size=n_years)
        
        # Energy use per capita
        energy_use = co2_emissions * np.random.uniform(250, 450, size=n_years) + 200
        
        # Protected area %
        protected_pct = np.clip(1.0 + 0.25 * time_index + np.random.normal(0, 0.5, size=n_years), 0.5, 45.0)

        for t_idx, y in enumerate(years):
            records.append({
                "iso3": iso3,
                "year": int(y),
                "co2_emissions_per_capita": float(np.round(co2_emissions[t_idx], 4)),
                "temp_anomaly_celsius": float(np.round(temp_anomaly[t_idx], 4)),
                "forest_area_pct_land": float(np.round(forest_pct[t_idx], 2)),
                "extreme_disasters_count": int(extreme_disasters[t_idx]),
                "renewable_energy_pct_share": float(np.round(renewable_pct[t_idx], 2)),
                "greenhouse_gas_total_kt": float(np.round(ghg_total[t_idx], 1)),
                "energy_use_per_capita": float(np.round(energy_use[t_idx], 2)),
                "protected_area_pct": float(np.round(protected_pct[t_idx], 2))
            })
            
    df = pd.DataFrame(records)
    return df

def main():
    print("Initializing 65-Year Environmental Data Ingestion Engine...")
    os.makedirs("data", exist_ok=True)
    
    df = generate_65yr_environmental_panel()
    print(f"Generated 65-year panel with shape {df.shape} covering years {df['year'].min()} to {df['year'].max()}.")
    print(f"Total ISO3 entities: {df['iso3'].nunique()}")
    
    parquet_path = os.path.join("data", "environment_65yr_panel.parquet")
    csv_path = os.path.join("data", "environment_65yr_panel.csv")
    
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    
    print(f"Dataset successfully saved to:\n - {parquet_path}\n - {csv_path}")

if __name__ == "__main__":
    main()
