import os
import yaml
import logging
import pandas as pd
import numpy as np
from scipy.stats import rankdata

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def build_dynamic_features(df, horizons=[1, 4, 12, 26], lookback_windows=[4, 12]):
    """
    Build frequency-aware dynamic features:
    1. Lags for all target indicators.
    2. Rolling aggregations (mean, std, min, max).
    3. Cyclical temporal encodings (Sine/Cosine for week of year, month of year).
    """
    logging.info("Constructing dynamic feature matrix...")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["country_iso3", "timestamp"]).reset_index(drop=True)
    
    numeric_cols = [
        "goldstein_score_mean", "avg_tone_mean", "event_count_total",
        "verbal_coop_count", "material_coop_count", "verbal_conflict_count",
        "material_conflict_count", "protest_count", "conflict_cooperation_ratio", "conflict_intensity_pct"
    ]
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    feature_dfs = []
    
    for country, group in df.groupby("country_iso3"):
        group = group.copy().sort_values("timestamp")
        new_cols = {}
        
        # 1. Generate Lags
        for col in numeric_cols:
            for h in horizons:
                new_cols[f"{col}_lag_{h}"] = group[col].shift(h)
                
        # 2. Generate Rolling Statistics
        for col in numeric_cols:
            for w in lookback_windows:
                rolling = group[col].rolling(window=w, min_periods=1)
                new_cols[f"{col}_roll_mean_{w}"] = rolling.mean()
                new_cols[f"{col}_roll_std_{w}"] = rolling.std().fillna(0)
                new_cols[f"{col}_roll_min_{w}"] = rolling.min()
                new_cols[f"{col}_roll_max_{w}"] = rolling.max()
                
        # 3. Cyclical Encodings
        weeks_in_year = group["timestamp"].dt.isocalendar().week.astype(float)
        months_in_year = group["timestamp"].dt.month.astype(float)
        
        new_cols["sin_week_of_year"] = np.sin(2 * np.pi * weeks_in_year / 52.0)
        new_cols["cos_week_of_year"] = np.cos(2 * np.pi * weeks_in_year / 52.0)
        new_cols["sin_month_of_year"] = np.sin(2 * np.pi * months_in_year / 12.0)
        new_cols["cos_month_of_year"] = np.cos(2 * np.pi * months_in_year / 12.0)
        
        new_features_df = pd.DataFrame(new_cols, index=group.index)
        full_group = pd.concat([group, new_features_df], axis=1)
        feature_dfs.append(full_group)
        
    full_features_df = pd.concat(feature_dfs, ignore_index=True)
    logging.info(f"Feature matrix built: {full_features_df.shape[0]} rows x {full_features_df.shape[1]} columns!")
    return full_features_df

def apply_percentile_rank_mapping(features_df, exclude_cols=["timestamp", "country_iso3"]):
    """
    Normalize feature vectors per timestamp slice to [0, 1] uniform rank percentiles across entities.
    Enforces scale-invariant vector similarity matching in FAISS.
    """
    logging.info("Applying scale-invariant percentile rank normalization across entity-timestamp slices...")
    rank_df = features_df.copy()
    feature_cols = [col for col in rank_df.columns if col not in exclude_cols]
    
    # Cast all feature columns to float64 to support rank values
    for col in feature_cols:
        rank_df[col] = rank_df[col].astype(float)
    
    ranked_slices = []
    for ts, slice_df in rank_df.groupby("timestamp"):
        slice_df = slice_df.copy()
        for col in feature_cols:
            vals = slice_df[col].values
            non_nan_mask = ~np.isnan(vals)
            if non_nan_mask.sum() > 1:
                ranks = rankdata(vals[non_nan_mask], method="average") / float(non_nan_mask.sum())
                slice_df.loc[non_nan_mask, col] = ranks
            else:
                slice_df[col] = 0.5
        ranked_slices.append(slice_df)
        
    final_rank_df = pd.concat(ranked_slices, ignore_index=True)
    logging.info("Percentile rank mapping complete!")
    return final_rank_df

def main():
    config = load_config()
    horizons = config["forecasting"]["horizons"]
    lookback = config["retrieval"]["lookback_window"]
    
    data_path = "data/massive_65year_political_dataset.csv"
    if not os.path.exists(data_path):
        data_path = "data/massive_40year_political_dataset.csv"
    if not os.path.exists(data_path):
        data_path = "data/gdelt_panel_wide.csv"
        
    logging.info(f"Loading raw panel data from {data_path}...")
    df_raw = pd.read_csv(data_path)
    
    # 1. Build Dynamic Feature Matrix
    df_features = build_dynamic_features(df_raw, horizons=horizons, lookback_windows=[4, lookback])
    
    # 2. Apply Percentile Rank Normalization
    df_features_rank = apply_percentile_rank_mapping(df_features)
    
    # 3. Save Feature Matrices
    os.makedirs("data", exist_ok=True)
    
    raw_feature_path = "data/feature_matrix.parquet"
    rank_feature_path = "data/feature_matrix_rank.parquet"
    
    df_features.to_parquet(raw_feature_path, index=False)
    df_features_rank.to_parquet(rank_feature_path, index=False)
    
    logging.info(f"Saved Raw Feature Matrix: {raw_feature_path} ({df_features.shape[0]} rows x {df_features.shape[1]} cols)")
    logging.info(f"Saved Rank Normalized Matrix: {rank_feature_path} ({df_features_rank.shape[0]} rows x {df_features_rank.shape[1]} cols)")
    print("\n" + "="*70)
    print(" FEATURE PIPELINE COMPLETE: Dynamic & Rank Feature Matrices Generated!")
    print("="*70)

if __name__ == "__main__":
    main()
