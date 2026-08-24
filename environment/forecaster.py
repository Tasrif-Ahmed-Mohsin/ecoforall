"""
Multi-Horizon Multi-Target Quantile Forecaster & Split-Conformal Prediction Engine.

Fits multi-head estimators (Ridge, LightGBM, Quantile Regressors) across ALL 8 targets
and dynamic horizons h in {1, 3, 5, 10} years with conformal prediction intervals.
"""

import os
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    from sklearn.ensemble import GradientBoostingRegressor

class EnvironmentMultiHorizonForecaster:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.horizons = self.config["forecasting"]["horizons"]
        self.target_indicators = self.config["forecasting"]["target_indicators"]
        self.quantiles = self.config["models"]["quantiles"]
        self.alpha_conformal = self.config["models"].get("alpha_conformal", 0.10)
        
        # Structure: self.models[target_indicator][horizon]
        self.models = {}
        self.conformal_residuals = {}
        
    def get_feature_columns(self, df):
        exclude = ["iso3", "year"] + [c for c in df.columns if "target" in c or "diff" in c]
        return [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]

    def train_horizon_head(self, X_train, y_train):
        """Fits multi-quantile LightGBM or GradientBoosting models for a target & horizon."""
        head_models = {}
        
        if HAS_LGB:
            point_model = lgb.LGBMRegressor(n_estimators=30, learning_rate=0.08, max_depth=4, n_jobs=-1, random_state=42, verbose=-1)
        else:
            point_model = GradientBoostingRegressor(n_estimators=30, learning_rate=0.08, max_depth=4, random_state=42)
            
        point_model.fit(X_train, y_train)
        head_models["point"] = point_model
        
        ridge_model = Ridge(alpha=1.0)
        ridge_model.fit(X_train, y_train)
        head_models["ridge"] = ridge_model
        
        for q in self.quantiles:
            if HAS_LGB:
                q_model = lgb.LGBMRegressor(objective="quantile", alpha=q, n_estimators=30, learning_rate=0.08, max_depth=4, n_jobs=-1, random_state=42, verbose=-1)
            else:
                q_model = GradientBoostingRegressor(loss="quantile", alpha=q, n_estimators=30, learning_rate=0.08, max_depth=4, random_state=42)
                
            q_model.fit(X_train, y_train)
            head_models[f"q_{q}"] = q_model
            
        return head_models

    def fit(self, df_train, df_calib=None, target_list=None):
        """Fits forecasters across ALL target indicators and dynamic horizons."""
        if target_list is None:
            target_list = self.target_indicators
            
        feature_cols = self.get_feature_columns(df_train)
        
        for target in target_list:
            self.models[target] = {}
            self.conformal_residuals[target] = {}
            
            for h in self.horizons:
                target_col = f"{target}_target_h{h}"
                if target_col not in df_train.columns:
                    continue
                    
                valid_train = df_train.dropna(subset=feature_cols + [target_col])
                if len(valid_train) == 0:
                    continue
                    
                X_tr = valid_train[feature_cols]
                y_tr = valid_train[target_col]
                
                self.models[target][h] = self.train_horizon_head(X_tr, y_tr)
                
                # Calibrate conformal bounds if calibration set provided
                if df_calib is not None:
                    valid_calib = df_calib.dropna(subset=feature_cols + [target_col])
                    if len(valid_calib) > 0:
                        y_pred_point = self.models[target][h]["point"].predict(valid_calib[feature_cols])
                        residuals = np.abs(valid_calib[target_col] - y_pred_point)
                        self.conformal_residuals[target][h] = np.quantile(residuals, 1.0 - self.alpha_conformal)
                    else:
                        self.conformal_residuals[target][h] = 0.5
                else:
                    y_tr_pred = self.models[target][h]["point"].predict(X_tr)
                    self.conformal_residuals[target][h] = np.quantile(np.abs(y_tr - y_tr_pred), 1.0 - self.alpha_conformal)

    def predict(self, df_query, target_indicator="co2_emissions_per_capita"):
        """Generates multi-horizon point forecasts, quantiles, and conformal bounds for a specified target."""
        if target_indicator not in self.models:
            raise KeyError(f"Target indicator '{target_indicator}' not fitted. Available: {list(self.models.keys())}")
            
        feature_cols = self.get_feature_columns(df_query)
        X_query = df_query[feature_cols]
        
        results = {}
        
        for h in self.horizons:
            if h not in self.models[target_indicator]:
                continue
                
            head = self.models[target_indicator][h]
            point_pred = head["point"].predict(X_query)
            ridge_pred = head["ridge"].predict(X_query)
            
            quantiles_pred = {}
            for q in self.quantiles:
                quantiles_pred[f"q_{q}"] = head[f"q_{q}"].predict(X_query)
                
            conf_val = self.conformal_residuals[target_indicator].get(h, 0.5)
            conformal_lower = point_pred - conf_val
            conformal_upper = point_pred + conf_val
            
            results[h] = {
                "point_ensemble": point_pred,
                "ridge_baseline": ridge_pred,
                "quantiles": quantiles_pred,
                "conformal_lower_90": conformal_lower,
                "conformal_upper_90": conformal_upper
            }
            
        return results

def main():
    from cross_domain_dataset_harmonizer import EnvironmentDatasetHarmonizer
    print("Testing Multi-Target Multi-Horizon Quantile Forecaster across ALL 8 targets...")
    harmonizer = EnvironmentDatasetHarmonizer()
    df_featured = harmonizer.build_features_and_targets()
    
    df_train = df_featured[df_featured["year"] < 2015]
    df_calib = df_featured[(df_featured["year"] >= 2015) & (df_featured["year"] <= 2020)]
    df_test = df_featured[df_featured["year"] > 2020]
    
    forecaster = EnvironmentMultiHorizonForecaster()
    forecaster.fit(df_train, df_calib)
    
    query_sample = df_test[df_test["iso3"] == "USA"].head(1)
    
    for target in forecaster.target_indicators:
        preds = forecaster.predict(query_sample, target_indicator=target)
        print(f"\n--- Forecast Results for USA (Post-2020) Target: {target} ---")
        for h in forecaster.horizons:
            res = preds[h]
            print(f"  Horizon h={h}y: Point={res['point_ensemble'][0]:.4f}, "
                  f"Conformal 90% Bounds=[{res['conformal_lower_90'][0]:.4f}, {res['conformal_upper_90'][0]:.4f}]")

if __name__ == "__main__":
    main()
