import os
import yaml
import numpy as np
import pandas as pd
from datetime import datetime

# Set seed for reproducibility
np.random.seed(42)

# List of 105 major ISO3 country codes across regions
ISO3_COUNTRIES = [
    # North America
    'USA', 'CAN', 'MEX',
    # Europe (West, North, South, East)
    'DEU', 'FRA', 'GBR', 'ITA', 'ESP', 'NLD', 'BEL', 'CHE', 'AUT', 'SWE', 'NOR', 'DNK', 'FIN',
    'POL', 'CZE', 'HUN', 'ROU', 'GRC', 'PRT', 'IRL', 'UKR', 'RUS', 'SVK', 'BGR', 'HRV', 'SRB',
    # East Asia & Pacific
    'CHN', 'JPN', 'KOR', 'TWN', 'AUS', 'NZL', 'SGP', 'HKG', 'MNG',
    # Southeast Asia
    'IDN', 'VNM', 'THA', 'MYS', 'PHL', 'MMR', 'KHM', 'LAO',
    # South Asia
    'IND', 'PAK', 'BGD', 'LKA', 'NPL',
    # Latin America & Caribbean
    'BRA', 'ARG', 'CHL', 'COL', 'PER', 'VEN', 'ECU', 'DOM', 'GTM', 'CRI', 'PAN', 'PRY', 'URY',
    # Middle East & North Africa
    'SAU', 'ARE', 'ISR', 'TUR', 'EGY', 'DZA', 'MAR', 'QAT', 'KWT', 'JOR', 'LBN', 'IRQ', 'IRN', 'OMN',
    # Sub-Saharan Africa
    'ZAF', 'NGA', 'KEN', 'ETH', 'GHA', 'EGY', 'TZA', 'UGA', 'AGO', 'MOZ', 'ZWE', 'SEN', 'CIV', 'RWA',
    # Central Asia
    'KAZ', 'UZB', 'AZE', 'GEO', 'ARM'
]
# Remove duplicates if any
ISO3_COUNTRIES = sorted(list(set(ISO3_COUNTRIES)))

REGION_MAP = {
    'USA': 'North America', 'CAN': 'North America', 'MEX': 'Latin America',
    'DEU': 'Europe', 'FRA': 'Europe', 'GBR': 'Europe', 'ITA': 'Europe', 'ESP': 'Europe',
    'NLD': 'Europe', 'BEL': 'Europe', 'CHE': 'Europe', 'AUT': 'Europe', 'SWE': 'Europe',
    'NOR': 'Europe', 'DNK': 'Europe', 'FIN': 'Europe', 'POL': 'Europe', 'CZE': 'Europe',
    'HUN': 'Europe', 'ROU': 'Europe', 'GRC': 'Europe', 'PRT': 'Europe', 'IRL': 'Europe',
    'UKR': 'Europe', 'RUS': 'Europe', 'SVK': 'Europe', 'BGR': 'Europe', 'HRV': 'Europe', 'SRB': 'Europe',
    'CHN': 'East Asia', 'JPN': 'East Asia', 'KOR': 'East Asia', 'TWN': 'East Asia',
    'AUS': 'Oceania', 'NZL': 'Oceania', 'SGP': 'Southeast Asia', 'HKG': 'East Asia', 'MNG': 'East Asia',
    'IDN': 'Southeast Asia', 'VNM': 'Southeast Asia', 'THA': 'Southeast Asia',
    'MYS': 'Southeast Asia', 'PHL': 'Southeast Asia', 'MMR': 'Southeast Asia',
    'KHM': 'Southeast Asia', 'LAO': 'Southeast Asia',
    'IND': 'South Asia', 'PAK': 'South Asia', 'BGD': 'South Asia', 'LKA': 'South Asia', 'NPL': 'South Asia',
    'BRA': 'Latin America', 'ARG': 'Latin America', 'CHL': 'Latin America', 'COL': 'Latin America',
    'PER': 'Latin America', 'VEN': 'Latin America', 'ECU': 'Latin America', 'DOM': 'Latin America',
    'GTM': 'Latin America', 'CRI': 'Latin America', 'PAN': 'Latin America', 'PRY': 'Latin America', 'URY': 'Latin America',
    'SAU': 'MENA', 'ARE': 'MENA', 'ISR': 'MENA', 'TUR': 'MENA', 'EGY': 'MENA',
    'DZA': 'MENA', 'MAR': 'MENA', 'QAT': 'MENA', 'KWT': 'MENA', 'JOR': 'MENA',
    'LBN': 'MENA', 'IRQ': 'MENA', 'IRN': 'MENA', 'OMN': 'MENA',
    'ZAF': 'Sub-Saharan Africa', 'NGA': 'Sub-Saharan Africa', 'KEN': 'Sub-Saharan Africa',
    'ETH': 'Sub-Saharan Africa', 'GHA': 'Sub-Saharan Africa', 'TZA': 'Sub-Saharan Africa',
    'UGA': 'Sub-Saharan Africa', 'AGO': 'Sub-Saharan Africa', 'MOZ': 'Sub-Saharan Africa',
    'ZWE': 'Sub-Saharan Africa', 'SEN': 'Sub-Saharan Africa', 'CIV': 'Sub-Saharan Africa', 'RWA': 'Sub-Saharan Africa',
    'KAZ': 'Central Asia', 'UZB': 'Central Asia', 'AZE': 'Central Asia', 'GEO': 'Central Asia', 'ARM': 'Central Asia'
}

INDICATORS_PSYCHOLOGY = [
    "psychology_trust",
    "psychology_fear",
    "psychology_optimism",
    "psychology_nationalism",
    "psychology_social_cohesion",
    "psychology_confidence"
]

INDICATORS_SOCIETY = [
    "society_education",
    "society_urbanization",
    "society_population",
    "society_age",
    "society_migration",
    "society_religion",
    "society_healthcare"
]

ALL_INDICATORS = INDICATORS_PSYCHOLOGY + INDICATORS_SOCIETY

def generate_synthetic_panel(start_date="1995-01-01", end_date="2026-06-01"):
    """
    Generates a realistic multi-country panel dataset of Collective Psychology & Society indicators.
    """
    print(f"Generating panel data for {len(ISO3_COUNTRIES)} countries from {start_date} to {end_date}...")
    dates = pd.date_range(start=start_date, end=end_date, freq='MS')
    num_dates = len(dates)
    
    panel_rows = []
    
    for iso3 in ISO3_COUNTRIES:
        region = REGION_MAP.get(iso3, 'Other')
        
        # Country baseline offsets (0 to 1 scale)
        base_trust = np.random.uniform(0.35, 0.85) if region in ['Europe', 'North America', 'Oceania'] else np.random.uniform(0.20, 0.65)
        base_education = np.random.uniform(0.60, 0.95) if region in ['Europe', 'North America', 'East Asia'] else np.random.uniform(0.25, 0.70)
        base_healthcare = np.random.uniform(0.55, 0.95) if region in ['Europe', 'North America', 'East Asia'] else np.random.uniform(0.20, 0.65)
        base_urban = np.random.uniform(0.50, 0.95) if iso3 in ['SGP', 'HKG', 'USA', 'JPN', 'GBR'] else np.random.uniform(0.20, 0.75)
        base_pop = np.random.uniform(0.2, 0.9) if iso3 in ['CHN', 'IND', 'USA', 'IDN', 'BRA', 'NGA', 'PAK'] else np.random.uniform(0.01, 0.4)
        base_age = np.random.uniform(0.50, 0.90) if region in ['Europe', 'East Asia'] else np.random.uniform(0.20, 0.55)
        base_nationalism = np.random.uniform(0.25, 0.75)
        base_religion = np.random.uniform(0.30, 0.80)
        
        # Initialize AR(1) state variables
        state_trust = base_trust
        state_fear = 1.0 - base_trust + np.random.uniform(-0.1, 0.1)
        state_optimism = base_trust * 0.8 + np.random.uniform(-0.05, 0.05)
        state_nationalism = base_nationalism
        state_cohesion = base_trust * 0.7 + np.random.uniform(-0.05, 0.05)
        state_confidence = base_trust * 0.85 + np.random.uniform(-0.05, 0.05)
        
        state_education = base_education
        state_urbanization = base_urban
        state_population = base_pop
        state_age = base_age
        state_migration = np.random.uniform(0.4, 0.6)
        state_religion = base_religion
        state_healthcare = base_healthcare
        
        for t_idx, dt in enumerate(dates):
            yr = dt.year
            mth = dt.month
            
            # Historical Shock Injections
            global_fear_shock = 0.0
            global_confidence_shock = 0.0
            global_trust_shock = 0.0
            global_health_shock = 0.0
            global_mig_shock = 0.0
            
            # 2001 Post-9/11 Shock
            if yr == 2001 and mth >= 9:
                global_fear_shock += 0.12
                global_confidence_shock -= 0.08
                
            # 2008 Financial Crisis
            if yr in [2008, 2009]:
                global_confidence_shock -= 0.25
                global_fear_shock += 0.20
                global_trust_shock -= 0.15
                
            # 2015 European Migration Wave
            if yr in [2015, 2016] and region in ['Europe', 'MENA']:
                global_mig_shock += 0.25
                state_nationalism += 0.005
                
            # 2020 COVID Pandemic Shock
            if yr in [2020, 2021]:
                global_health_shock -= 0.20
                global_fear_shock += 0.30
                global_confidence_shock -= 0.20
                global_trust_shock += 0.05 if region in ['Europe', 'East Asia'] else -0.10
                
            # 2022 Inflation & Geopolitical Shock
            if yr in [2022, 2023]:
                global_fear_shock += 0.15
                global_confidence_shock -= 0.12
                
            # Structural drift for slow societal indicators
            state_education = min(0.99, state_education + 0.0003 + np.random.normal(0, 0.0002))
            state_urbanization = min(0.99, state_urbanization + 0.00025 + np.random.normal(0, 0.0001))
            state_healthcare = np.clip(state_healthcare + 0.0002 + global_health_shock * 0.02 + np.random.normal(0, 0.001), 0.05, 0.99)
            state_age = np.clip(state_age + 0.0002 + np.random.normal(0, 0.0001), 0.1, 0.95)
            state_population = np.clip(state_population + 0.0001 + np.random.normal(0, 0.00005), 0.01, 1.0)
            state_religion = np.clip(state_religion - 0.0001 + np.random.normal(0, 0.0005), 0.05, 0.95)
            state_migration = np.clip(state_migration + global_mig_shock * 0.01 + np.random.normal(0, 0.002), 0.0, 1.0)
            
            # Dynamic AR(1) update with cross-indicator feedback for psychology
            trust_noise = np.random.normal(0, 0.015)
            fear_noise = np.random.normal(0, 0.018)
            opt_noise = np.random.normal(0, 0.02)
            
            state_trust = 0.92 * state_trust + 0.08 * (base_trust + global_trust_shock) - 0.1 * (state_fear - 0.5) + trust_noise
            state_fear = 0.88 * state_fear + 0.12 * (1.0 - base_trust + global_fear_shock) + fear_noise
            state_optimism = 0.90 * state_optimism + 0.10 * (state_trust * 0.7 + state_confidence * 0.3 + global_confidence_shock) + opt_noise
            state_confidence = 0.89 * state_confidence + 0.11 * (state_trust * 0.6 + state_optimism * 0.4 + global_confidence_shock) + np.random.normal(0, 0.02)
            state_nationalism = 0.95 * state_nationalism + 0.05 * (base_nationalism + 0.3 * (state_fear - 0.4)) + np.random.normal(0, 0.01)
            state_cohesion = 0.93 * state_cohesion + 0.07 * (state_trust * 0.8 - 0.2 * (state_nationalism - 0.5)) + np.random.normal(0, 0.015)
            
            # Clip psychological metrics to realistic 0-100 index values
            val_trust = np.clip(state_trust * 100, 5.0, 95.0)
            val_fear = np.clip(state_fear * 100, 5.0, 95.0)
            val_optimism = np.clip(state_optimism * 100, 5.0, 95.0)
            val_nationalism = np.clip(state_nationalism * 100, 5.0, 95.0)
            val_cohesion = np.clip(state_cohesion * 100, 5.0, 95.0)
            val_confidence = np.clip(state_confidence * 100, 5.0, 95.0)
            
            val_edu = np.clip(state_education * 100, 10.0, 99.0)
            val_urb = np.clip(state_urbanization * 100, 10.0, 99.0)
            val_pop = np.clip(state_population * 100, 1.0, 100.0)
            val_age = np.clip(state_age * 100, 15.0, 90.0)
            val_mig = np.clip(state_migration * 100, 0.0, 100.0)
            val_rel = np.clip(state_religion * 100, 5.0, 95.0)
            val_hc = np.clip(state_healthcare * 100, 10.0, 99.0)
            
            row_dict = {
                'iso3': iso3,
                'region': region,
                'timestamp': dt,
                'psychology_trust': round(float(val_trust), 2),
                'psychology_fear': round(float(val_fear), 2),
                'psychology_optimism': round(float(val_optimism), 2),
                'psychology_nationalism': round(float(val_nationalism), 2),
                'psychology_social_cohesion': round(float(val_cohesion), 2),
                'psychology_confidence': round(float(val_confidence), 2),
                'society_education': round(float(val_edu), 2),
                'society_urbanization': round(float(val_urb), 2),
                'society_population': round(float(val_pop), 2),
                'society_age': round(float(val_age), 2),
                'society_migration': round(float(val_mig), 2),
                'society_religion': round(float(val_rel), 2),
                'society_healthcare': round(float(val_hc), 2)
            }
            panel_rows.append(row_dict)

    df_wide = pd.DataFrame(panel_rows)
    print(f"Generated wide panel DataFrame: {df_wide.shape[0]} rows x {df_wide.shape[1]} columns.")
    
    # Create canonical long format: [iso3, timestamp, indicator_id, value, domain_category]
    long_rows = []
    for ind in ALL_INDICATORS:
        cat = 'psychology' if ind.startswith('psychology') else 'society'
        sub = df_wide[['iso3', 'region', 'timestamp', ind]].copy()
        sub.rename(columns={ind: 'value'}, inplace=True)
        sub['indicator_id'] = ind
        sub['domain_category'] = cat
        long_rows.append(sub)
        
    df_long = pd.concat(long_rows, ignore_index=True)
    print(f"Generated canonical long format: {df_long.shape[0]} records.")
    
    # Ensure output directory exists
    os.makedirs("data", exist_ok=True)
    
    df_wide.to_parquet("data/dataset_wide.parquet", index=False)
    df_long.to_parquet("data/dataset_canonical.parquet", index=False)
    df_wide.to_csv("data/dataset_wide.csv", index=False)
    df_long.to_csv("data/dataset_canonical.csv", index=False)
    
    print("Dataset generation completed and saved to 'data/' folder successfully!")
    return df_wide, df_long

if __name__ == "__main__":
    generate_synthetic_panel()
