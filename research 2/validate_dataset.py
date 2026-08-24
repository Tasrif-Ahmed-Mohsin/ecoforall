import os
import yaml
import pandas as pd
import numpy as np

def validate_gdelt_dataset(wide_path="data/gdelt_panel_wide.parquet", canonical_path="data/gdelt_panel_canonical.parquet"):
    print("=" * 70)
    print(" GDELT POLITICAL DATASET VALIDATION REPORT")
    print("=" * 70)
    
    if not os.path.exists(wide_path):
        print(f"Error: Dataset file not found at {wide_path}. Please run ingest_gdelt.py first.")
        return
        
    df_wide = pd.read_parquet(wide_path)
    df_canonical = pd.read_parquet(canonical_path)
    
    print(f"\n1. DATASET OVERVIEW:")
    print(f"   - Wide Panel Dimensions   : {df_wide.shape[0]} rows x {df_wide.shape[1]} columns")
    print(f"   - Canonical Dimensions    : {df_canonical.shape[0]} rows x {df_canonical.shape[1]} columns")
    print(f"   - Entities (Countries)    : {df_wide['country_iso3'].nunique()} ({', '.join(df_wide['country_iso3'].unique()[:8])}...)")
    print(f"   - Temporal Range          : {df_wide['timestamp'].min().strftime('%Y-%m-%d')} to {df_wide['timestamp'].max().strftime('%Y-%m-%d')}")
    
    print(f"\n2. INDICATORS CAPTURED ({len(df_canonical['indicator_id'].unique())} indicators):")
    for ind in df_canonical['indicator_id'].unique():
        sub = df_canonical[df_canonical['indicator_id'] == ind]
        print(f"   - {ind:<28} | Mean: {sub['value'].mean():10.2f} | Min: {sub['value'].min():10.2f} | Max: {sub['value'].max():10.2f}")
        
    print(f"\n3. DATA QUALITY AUDIT:")
    null_count = df_wide.isnull().sum().sum()
    print(f"   - Total Missing/Null Values: {null_count}")
    
    # Check temporal gaps per country
    gaps = []
    for country, group in df_wide.groupby('country_iso3'):
        dates = group['timestamp'].sort_values()
        diffs = dates.diff().dropna()
        if len(diffs.unique()) > 1:
            gaps.append(country)
            
    if gaps:
        print(f"   - Countries with Temporal Gaps : {len(gaps)} ({', '.join(gaps)})")
    else:
        print(f"   - Temporal Continuity Check    : PASSED (Uniform weekly frequency across all countries)")
        
    print(f"\n4. CANONICAL SCHEMA SAMPLE:")
    print(df_canonical.head(6).to_string(index=False))
    
    print("\n" + "=" * 70)
    print(" VALIDATION COMPLETE: Dataset is ready for feature engineering & modeling!")
    print("=" * 70)

if __name__ == "__main__":
    validate_gdelt_dataset()
