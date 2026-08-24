"""
Dataset Quality, Panel Balance, and Temporal Continuity Audit Script.

Validates completeness, entity balance, and missing value ratios across 1960-2025.
"""

import os
import numpy as np
import pandas as pd

def audit_dataset():
    data_path = os.path.join("data", "environment_65yr_panel.parquet")
    if not os.path.exists(data_path):
        data_path = os.path.join("data", "environment_65yr_panel.csv")
        
    if not os.path.exists(data_path):
        raise FileNotFoundError("Dataset not found. Please run ingest_environment_data.py first.")
        
    df = pd.read_parquet(data_path) if data_path.endswith(".parquet") else pd.read_csv(data_path)
    
    print("=" * 60)
    print(" 65-YEAR ENVIRONMENTAL DATASET INTEGRITY & AUDIT REPORT ")
    print("=" * 60)
    print(f"Total Rows: {len(df):,}")
    print(f"Total Columns: {len(df.columns)}")
    print(f"ISO3 Entity Count: {df['iso3'].nunique()}")
    print(f"Year Range: {df['year'].min()} to {df['year'].max()} ({df['year'].nunique()} years)")
    print("-" * 60)
    
    # Check completeness per year
    yearly_counts = df.groupby("year")["iso3"].count()
    expected_per_year = df["iso3"].nunique()
    balanced = (yearly_counts == expected_per_year).all()
    print(f"Panel Temporal Balance: {'BALANCED' if balanced else 'UNBALANCED'}")
    
    # Missing Value Audit
    print("\nIndicator Missingness & Density:")
    missing_summary = []
    for col in df.columns:
        if col in ["iso3", "year"]:
            continue
        missing_count = df[col].isnull().sum()
        pct_missing = (missing_count / len(df)) * 100.0
        missing_summary.append({
            "indicator": col,
            "missing_count": missing_count,
            "pct_missing": round(pct_missing, 2),
            "density_pct": round(100.0 - pct_missing, 2)
        })
        
    missing_df = pd.DataFrame(missing_summary)
    print(missing_df.to_string(index=False))
    print("=" * 60)
    return missing_df

if __name__ == "__main__":
    audit_dataset()
