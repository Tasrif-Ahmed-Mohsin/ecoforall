import os
import yaml
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from feature_pipeline import DynamicFeaturePipeline
from validation import diebold_mariano_test

def run_5x3_cross_validation():
    """
    Executes a formal 5x3 Nested Cross-Validation (5 Folds x 3 Lookback Windows W in {6, 12, 24} months).
    """
    print("================================================================================")
    print("        EXECUTING FORMAL 5x3 CROSS-VALIDATION MATRIX EVALUATION                 ")
    print("================================================================================\n")
    
    df_wide = pd.read_parquet("data/dataset_wide.parquet")
    lookback_windows = [6, 12, 24]
    cv_folds = 5
    horizons = [1, 3, 6, 12]
    
    matrix_results = {}
    
    for w in lookback_windows:
        print(f"--- Running 5-Fold CV for Lookback Window W = {w} Months ---")
        pipeline = DynamicFeaturePipeline()
        pipeline.lookback = w
        df_feat, f_cols, r_cols, t_cols = pipeline.create_features(df_wide)
        
        df = df_feat.dropna(subset=f_cols).sort_values(pipeline.time_col).reset_index(drop=True)
        unique_times = df[pipeline.time_col].drop_duplicates().sort_values().values
        n_times = len(unique_times)
        fold_size = n_times // (cv_folds + 1)
        
        fold_metrics = {}
        for h in horizons:
            target_col = f"target_h_{h}"
            y_all_true, y_all_ml, y_all_ar1 = [], [], []
            
            for f in range(1, cv_folds + 1):
                train_cutoff_idx = fold_size * (f + 1)
                if train_cutoff_idx >= n_times - 5:
                    break
                    
                train_times = unique_times[:train_cutoff_idx]
                test_times = unique_times[train_cutoff_idx:train_cutoff_idx + fold_size]
                
                train_mask = df[pipeline.time_col].isin(train_times) & (~df[target_col].isna())
                test_mask = df[pipeline.time_col].isin(test_times) & (~df[target_col].isna())
                
                df_train = df[train_mask]
                df_test = df[test_mask]
                
                if len(df_train) < 50 or len(df_test) < 10:
                    continue
                    
                X_tr = df_train[f_cols].values
                y_tr = df_train[target_col].values
                X_te = df_test[f_cols].values
                y_te = df_test[target_col].values
                
                # Fit ML Regressor
                lgbm = LGBMRegressor(n_estimators=50, learning_rate=0.05, num_leaves=31, random_state=42, verbosity=-1)
                lgbm.fit(X_tr, y_tr)
                p_ml = lgbm.predict(X_te)
                
                # Fit AR(1) Regressor
                lag1_col = f"{pipeline.target_ind}_lag_1"
                ar1_model = Ridge(alpha=10.0)
                ar1_model.fit(df_train[[lag1_col]].values, y_tr)
                p_ar1 = ar1_model.predict(df_test[[lag1_col]].values)
                
                y_all_true.extend(y_te)
                y_all_ml.extend(p_ml)
                y_all_ar1.extend(p_ar1)
                
            y_all_true = np.array(y_all_true)
            y_all_ml = np.array(y_all_ml)
            y_all_ar1 = np.array(y_all_ar1)
            
            rmse_ml = np.sqrt(np.mean((y_all_true - y_all_ml)**2))
            rmse_ar1 = np.sqrt(np.mean((y_all_true - y_all_ar1)**2))
            dm_stat, dm_p = diebold_mariano_test(y_all_true, y_all_ml, y_all_ar1, h=h)
            
            fold_metrics[f"h_{h}m"] = {
                'rmse_ml': round(float(rmse_ml), 4),
                'rmse_ar1': round(float(rmse_ar1), 4),
                'dm_pvalue': dm_p,
                'ml_statistically_superior': bool(dm_p < 0.05)
            }
            print(f"  [W={w}m | 5-Fold Aggregate] Horizon h={h}m -> ML RMSE: {rmse_ml:.4f} vs AR(1) RMSE: {rmse_ar1:.4f} | DM p-val: {dm_p}")
            
        matrix_results[f"W_{w}m"] = fold_metrics

    with open("data/cv_5x3_results.json", "w") as f:
        json.dump(matrix_results, f, indent=2)
        
    print("\n5x3 Cross-Validation Complete! Results exported to 'data/cv_5x3_results.json'.")
    return matrix_results

if __name__ == "__main__":
    run_5x3_cross_validation()
