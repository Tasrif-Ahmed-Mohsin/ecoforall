import os
import yaml
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor

def diebold_mariano_test(y_true, y_pred1, y_pred2, h=1):
    """
    Diebold-Mariano (DM) test for comparing predictive accuracy of two forecasts with Newey-West HAC variance adjustment.
    """
    e1 = y_true - y_pred1
    e2 = y_true - y_pred2
    
    # Loss differential (Squared Error Loss)
    d = (e1 ** 2) - (e2 ** 2)
    n = len(d)
    if n < 10:
        return 0.0, 1.0
        
    mean_d = np.mean(d)
    
    # Autocovariance up to lag h-1 for Newey-West
    gamma_0 = np.var(d, ddof=0)
    gamma_sum = 0.0
    for k in range(1, h):
        cov_k = np.cov(d[k:], d[:-k])[0, 1] if len(d[k:]) > 0 else 0.0
        gamma_sum += (1.0 - k / float(h)) * cov_k
        
    var_d = (gamma_0 + 2.0 * gamma_sum) / float(n)
    if var_d <= 1e-12:
        return 0.0, 1.0
        
    dm_stat = mean_d / np.sqrt(var_d)
    p_value = 2.0 * (1.0 - norm.cdf(np.abs(dm_stat)))
    return round(float(dm_stat), 4), round(float(p_value), 4)

class AntiLeakageValidator:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.entity_col = self.config['domain']['entity_label']
        self.time_col = self.config['domain']['time_col']
        self.target_ind = self.config['forecasting']['target_indicator']
        self.horizons = self.config['forecasting']['horizons']
        self.cv_folds = self.config['models']['cv_folds']

    def run_walk_forward_cv(self, df_features, feature_cols):
        """
        Executes expanding-window 5-fold cross-validation and performs DM test vs AR(1) & Naive baselines.
        """
        print(f"Executing {self.cv_folds}-Fold Expanding Window Walk-Forward CV...")
        df = df_features.dropna(subset=feature_cols).sort_values(self.time_col).reset_index(drop=True)
        
        unique_times = df[self.time_col].drop_duplicates().sort_values().values
        n_times = len(unique_times)
        
        fold_size = n_times // (self.cv_folds + 1)
        
        cv_results = {}
        
        for h in self.horizons:
            target_col = f"target_h_{h}"
            rmse_ml, rmse_ar1, rmse_naive = [], [], []
            y_all_true, y_all_ml, y_all_ar1, y_all_naive = [], [], [], []
            
            for f in range(1, self.cv_folds + 1):
                train_cutoff_idx = fold_size * (f + 1)
                if train_cutoff_idx >= n_times - 5:
                    break
                    
                train_times = unique_times[:train_cutoff_idx]
                test_times = unique_times[train_cutoff_idx:train_cutoff_idx + fold_size]
                
                train_mask = df[self.time_col].isin(train_times) & (~df[target_col].isna())
                test_mask = df[self.time_col].isin(test_times) & (~df[target_col].isna())
                
                df_train = df[train_mask]
                df_test = df[test_mask]
                
                if len(df_train) < 50 or len(df_test) < 10:
                    continue
                    
                X_tr = df_train[feature_cols].values
                y_tr = df_train[target_col].values
                
                X_te = df_test[feature_cols].values
                y_te = df_test[target_col].values
                
                # ML Regressor
                lgbm = LGBMRegressor(n_estimators=60, learning_rate=0.05, num_leaves=31, random_state=42, verbosity=-1)
                lgbm.fit(X_tr, y_tr)
                p_ml = lgbm.predict(X_te)
                
                # AR(1) Baseline: use lag_1 of target indicator
                lag1_col = f"{self.target_ind}_lag_1"
                ar1_model = Ridge(alpha=10.0)
                ar1_model.fit(df_train[[lag1_col]].values, y_tr)
                p_ar1 = ar1_model.predict(df_test[[lag1_col]].values)
                
                # Naive Baseline (0 forecast for change)
                p_naive = np.zeros_like(y_te)
                
                y_all_true.extend(y_te)
                y_all_ml.extend(p_ml)
                y_all_ar1.extend(p_ar1)
                y_all_naive.extend(p_naive)
                
            y_all_true = np.array(y_all_true)
            y_all_ml = np.array(y_all_ml)
            y_all_ar1 = np.array(y_all_ar1)
            y_all_naive = np.array(y_all_naive)
            
            rmse_ml_val = np.sqrt(np.mean((y_all_true - y_all_ml)**2))
            rmse_ar1_val = np.sqrt(np.mean((y_all_true - y_all_ar1)**2))
            rmse_naive_val = np.sqrt(np.mean((y_all_true - y_all_naive)**2))
            
            dm_ar1_stat, dm_ar1_p = diebold_mariano_test(y_all_true, y_all_ml, y_all_ar1, h=h)
            dm_naive_stat, dm_naive_p = diebold_mariano_test(y_all_true, y_all_ml, y_all_naive, h=h)
            
            cv_results[h] = {
                'rmse_ml': round(float(rmse_ml_val), 4),
                'rmse_ar1': round(float(rmse_ar1_val), 4),
                'rmse_naive': round(float(rmse_naive_val), 4),
                'dm_stat_vs_ar1': dm_ar1_stat,
                'dm_pvalue_vs_ar1': dm_ar1_p,
                'dm_stat_vs_naive': dm_naive_stat,
                'dm_pvalue_vs_naive': dm_naive_p,
                'ml_statistically_superior': bool(dm_ar1_p < 0.05 and dm_naive_p < 0.05)
            }
            print(f"Horizon h={h}: ML RMSE={rmse_ml_val:.4f} vs AR(1) RMSE={rmse_ar1_val:.4f} | DM Test p-val vs AR1: {dm_ar1_p}")
            
        return cv_results

if __name__ == "__main__":
    from feature_pipeline import DynamicFeaturePipeline
    
    df_feat = pd.read_parquet("data/dataset_features.parquet")
    pipeline = DynamicFeaturePipeline()
    _, f_cols, _, _ = pipeline.create_features(pd.read_parquet("data/dataset_wide.parquet"))
    
    validator = AntiLeakageValidator()
    results = validator.run_walk_forward_cv(df_feat, f_cols)
    print("Validation Walk-Forward CV Results:", results)
