import os
import json
import numpy as np
import pandas as pd

def run_data_audit():
    print("Running comprehensive Data Quality & Audit Pipeline...")
    
    wide_path = "data/dataset_wide.parquet"
    if not os.path.exists(wide_path):
        raise FileNotFoundError(f"Dataset wide panel file not found at {wide_path}")
        
    df_wide = pd.read_parquet(wide_path)
    
    indicators = [c for c in df_wide.columns if c not in ['iso3', 'region', 'timestamp']]
    
    num_countries = df_wide['iso3'].nunique()
    min_date = str(df_wide['timestamp'].min().date())
    max_date = str(df_wide['timestamp'].max().date())
    total_rows = len(df_wide)
    expected_rows = num_countries * df_wide['timestamp'].nunique()
    balance_ratio = round(total_rows / float(expected_rows), 4)
    
    # Missingness check
    missing_stats = {}
    for col in indicators:
        null_cnt = int(df_wide[col].isnull().sum())
        missing_stats[col] = {
            'null_count': null_cnt,
            'null_percentage': round(null_cnt / float(total_rows) * 100, 2)
        }
        
    # Summary statistics
    summary_stats = {}
    for col in indicators:
        ser = df_wide[col].dropna()
        summary_stats[col] = {
            'mean': round(float(ser.mean()), 2),
            'std': round(float(ser.std()), 2),
            'min': round(float(ser.min()), 2),
            'q25': round(float(ser.quantile(0.25)), 2),
            'median': round(float(ser.median()), 2),
            'q75': round(float(ser.quantile(0.75)), 2),
            'max': round(float(ser.max()), 2),
            'skewness': round(float(ser.skew()), 2)
        }
        
    # Correlation matrix
    corr_df = df_wide[indicators].corr()
    corr_dict = {}
    for r in indicators:
        corr_dict[r] = {c: round(float(corr_df.loc[r, c]), 3) for c in indicators}
        
    # Autocorrelation (lag-1) per indicator across countries
    autocorr_stats = {}
    for col in indicators:
        lags = []
        for iso3, group in df_wide.groupby('iso3'):
            ser = group.sort_values('timestamp')[col]
            if len(ser) > 5:
                lags.append(ser.autocorr(lag=1))
        autocorr_stats[col] = round(float(np.nanmean(lags)), 3)
        
    # Outlier counts (z-score > 3.0)
    outlier_stats = {}
    for col in indicators:
        ser = df_wide[col]
        z = np.abs((ser - ser.mean()) / (ser.std() + 1e-9))
        outlier_cnt = int((z > 3.0).sum())
        outlier_stats[col] = {
            'outlier_count': outlier_cnt,
            'outlier_percentage': round(outlier_cnt / float(total_rows) * 100, 2)
        }
        
    audit_report = {
        'audit_metadata': {
            'dataset_name': 'Collective Psychology & Society Global Panel',
            'entity_count': num_countries,
            'total_panel_rows': total_rows,
            'total_canonical_records': total_rows * len(indicators),
            'start_date': min_date,
            'end_date': max_date,
            'balance_ratio': balance_ratio,
            'audit_status': 'PASS' if balance_ratio >= 0.99 else 'WARNING'
        },
        'missingness': missing_stats,
        'summary_statistics': summary_stats,
        'correlation_matrix': corr_dict,
        'lag1_autocorrelation': autocorr_stats,
        'outliers': outlier_stats
    }
    
    with open("data/audit_report.json", "w") as f:
        json.dump(audit_report, f, indent=2)
        
    print(f"Data Audit Complete! Report saved to 'data/audit_report.json'. Balance Ratio: {balance_ratio}, Total Records: {audit_report['audit_metadata']['total_canonical_records']}")
    return audit_report

if __name__ == "__main__":
    run_data_audit()
