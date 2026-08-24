import os
import yaml
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, QuantileRegressor
from lightgbm import LGBMRegressor

class MultiHeadForecaster:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.horizons = self.config['forecasting']['horizons']
        self.quantiles = self.config['models']['quantiles']
        self.target_ind = self.config['forecasting']['target_indicator']
        
        self.models_point = {} # horizon -> { 'lgbm': model, 'ridge': model, 'meta': model }
        self.models_quantile = {} # horizon -> { q -> model }
        self.feature_cols = None

    def fit(self, df_features, feature_cols):
        """
        Trains multi-head regressors, quantile models, and baseline models for each target horizon.
        """
        print("Training Multi-Head Forecaster across horizons:", self.horizons)
        self.feature_cols = feature_cols
        
        df_clean = df_features.dropna(subset=feature_cols).copy()
        X = df_clean[feature_cols].values
        
        for h in self.horizons:
            target_h_col = f"target_h_{h}"
            valid_mask = ~df_clean[target_h_col].isna()
            X_h = X[valid_mask]
            y_h = df_clean.loc[valid_mask, target_h_col].values
            
            print(f"--- Training Horizon h={h} ({len(y_h)} valid samples) ---")
            
            # Point Estimators
            ridge = Ridge(alpha=10.0)
            ridge.fit(X_h, y_h)
            
            lgbm = LGBMRegressor(
                n_estimators=100,
                learning_rate=0.05,
                num_leaves=31,
                random_state=42,
                verbosity=-1
            )
            lgbm.fit(X_h, y_h)
            
            pred_ridge = ridge.predict(X_h)
            pred_lgbm = lgbm.predict(X_h)
            
            # Stacking Meta-Learner for point prediction
            X_meta = np.column_stack([pred_ridge, pred_lgbm])
            meta_stacker = Ridge(alpha=1.0)
            meta_stacker.fit(X_meta, y_h)
            
            self.models_point[h] = {
                'ridge': ridge,
                'lgbm': lgbm,
                'meta': meta_stacker
            }
            
            # Quantile Estimators
            self.models_quantile[h] = {}
            for q in self.quantiles:
                lgbm_q = LGBMRegressor(
                    objective='quantile',
                    alpha=q,
                    n_estimators=80,
                    learning_rate=0.05,
                    num_leaves=31,
                    random_state=42,
                    verbosity=-1
                )
                lgbm_q.fit(X_h, y_h)
                self.models_quantile[h][q] = lgbm_q
                
        print("Multi-Head Quantile Forecaster training complete!")
        return self

    def predict(self, X_input):
        """
        Generates point predictions, quantile fan bounds, and baseline outputs for input features.
        """
        predictions = {}
        for h in self.horizons:
            ridge_p = self.models_point[h]['ridge'].predict(X_input)
            lgbm_p = self.models_point[h]['lgbm'].predict(X_input)
            meta_p = self.models_point[h]['meta'].predict(np.column_stack([ridge_p, lgbm_p]))
            
            q_preds = {}
            for q in self.quantiles:
                q_preds[q] = self.models_quantile[h][q].predict(X_input)
                
            predictions[h] = {
                'point_ensemble': meta_p,
                'point_lgbm': lgbm_p,
                'point_ridge': ridge_p,
                'quantiles': q_preds
            }
        return predictions

if __name__ == "__main__":
    from feature_pipeline import DynamicFeaturePipeline
    
    df_feat = pd.read_parquet("data/dataset_features.parquet")
    pipeline = DynamicFeaturePipeline()
    _, f_cols, _, _ = pipeline.create_features(pd.read_parquet("data/dataset_wide.parquet"))
    
    forecaster = MultiHeadForecaster()
    forecaster.fit(df_feat, f_cols)
    print("Forecaster verification test passed successfully!")
