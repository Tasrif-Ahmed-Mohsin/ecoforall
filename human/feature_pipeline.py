import os
import yaml
import numpy as np
import pandas as pd
from scipy.stats import rankdata

class DynamicFeaturePipeline:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.entity_col = self.config['domain']['entity_label']
        self.time_col = self.config['domain']['time_col']
        self.target_ind = self.config['forecasting']['target_indicator']
        self.target_transform = self.config['forecasting']['target_transform']
        self.horizons = self.config['forecasting']['horizons']
        self.lookback = self.config['retrieval']['lookback_window']
        
        self.psychology_inds = self.config['indicators']['psychology']
        self.society_inds = self.config['indicators']['society']
        self.all_indicators = self.psychology_inds + self.society_inds

    def create_features(self, df_wide):
        """
        Engineers lags, rolling stats, time cyclical features, uniform rank percentiles,
        and dynamic target horizons y_{t+h}.
        """
        df = df_wide.copy()
        df[self.time_col] = pd.to_datetime(df[self.time_col])
        df = df.sort_values([self.entity_col, self.time_col]).reset_index(drop=True)
        
        # 1. Cyclical Time Embeddings
        month_ser = df[self.time_col].dt.month
        df['sin_month'] = np.sin(2 * np.pi * month_ser / 12.0)
        df['cos_month'] = np.cos(2 * np.pi * month_ser / 12.0)
        
        feature_cols = ['sin_month', 'cos_month']
        
        # 2. Indicator Lags and Rolling Aggregations per Entity
        print("Generating entity-level lags and rolling window features...")
        for ind in self.all_indicators:
            # Lags
            for l in [1, 3, 6, 12]:
                lag_col = f"{ind}_lag_{l}"
                df[lag_col] = df.groupby(self.entity_col)[ind].shift(l)
                feature_cols.append(lag_col)
                
            # Rolling statistics over lookback window (e.g. 12 months)
            roll_mean_col = f"{ind}_roll_mean_{self.lookback}"
            roll_std_col = f"{ind}_roll_std_{self.lookback}"
            roll_min_col = f"{ind}_roll_min_{self.lookback}"
            roll_max_col = f"{ind}_roll_max_{self.lookback}"
            
            df[roll_mean_col] = df.groupby(self.entity_col)[ind].transform(lambda x: x.rolling(self.lookback, min_periods=3).mean())
            df[roll_std_col] = df.groupby(self.entity_col)[ind].transform(lambda x: x.rolling(self.lookback, min_periods=3).std())
            df[roll_min_col] = df.groupby(self.entity_col)[ind].transform(lambda x: x.rolling(self.lookback, min_periods=3).min())
            df[roll_max_col] = df.groupby(self.entity_col)[ind].transform(lambda x: x.rolling(self.lookback, min_periods=3).max())
            
            feature_cols.extend([roll_mean_col, roll_std_col, roll_min_col, roll_max_col])

        # Raw current indicator values are also features
        feature_cols.extend(self.all_indicators)
        
        # 3. Dynamic Multi-Horizon Forecast Target Creation y_{t+h}
        print("Building dynamic multi-horizon targets y_{t+h}...")
        target_cols = []
        for h in self.horizons:
            target_h_col = f"target_h_{h}"
            shift_val = df.groupby(self.entity_col)[self.target_ind].shift(-h)
            
            if self.target_transform == "absolute_change":
                df[target_h_col] = shift_val - df[self.target_ind]
            elif self.target_transform == "log_return":
                df[target_h_col] = np.log((shift_val + 1e-5) / (df[self.target_ind] + 1e-5))
            else: # raw_level
                df[target_h_col] = shift_val
                
            target_cols.append(target_h_col)

        # 4. Percentile Rank Mapping Across Entities per Timestamp
        print("Applying Uniform Percentile Rank Mapping across entity slices...")
        rank_feature_cols = []
        for col in feature_cols:
            rank_col = f"rank_{col}"
            # Rank transform per timestamp slice
            df[rank_col] = df.groupby(self.time_col)[col].transform(
                lambda x: rankdata(x, method='average') / float(len(x)) if len(x.dropna()) > 0 else np.nan
            )
            rank_feature_cols.append(rank_col)
            
        print(f"Feature engineering complete! Raw features: {len(feature_cols)}, Rank features: {len(rank_feature_cols)}")
        return df, feature_cols, rank_feature_cols, target_cols

if __name__ == "__main__":
    df_wide = pd.read_parquet("data/dataset_wide.parquet")
    pipeline = DynamicFeaturePipeline()
    df_feat, f_cols, r_cols, t_cols = pipeline.create_features(df_wide)
    df_feat.to_parquet("data/dataset_features.parquet", index=False)
    print("Saved feature dataset to 'data/dataset_features.parquet'.")
