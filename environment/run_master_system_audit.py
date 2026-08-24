"""
Master System Audit & Diebold-Mariano Statistical Falsification Suite across ALL Indicators.

Executes expanding-window cross validation across dynamic horizons (1, 3, 5, 10 years)
calculating Raw RMSE, Normalized RMSE (NRMSE = RMSE / StdDev), MAE, and Diebold-Mariano tests.
"""

import os
import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from cross_domain_dataset_harmonizer import EnvironmentDatasetHarmonizer
from forecaster import EnvironmentMultiHorizonForecaster

def diebold_mariano_test(e1, e2, h=1, power=2):
    """Calculates Diebold-Mariano test statistic comparing forecast errors e1 (ML) and e2 (Baseline)."""
    d = np.abs(e1)**power - np.abs(e2)**power
    mean_d = np.mean(d)
    n = len(d)
    
    if n < 5 or np.std(d) < 1e-8:
        return 0.0, 1.0
        
    gamma0 = np.var(d, ddof=0)
    gamma_sum = 0.0
    for lag in range(1, h):
        if lag < n:
            gamma_lag = np.cov(d[lag:], d[:-lag])[0, 1]
            weight = 1.0 - (lag / h)
            gamma_sum += 2.0 * weight * gamma_lag
            
    var_d = (gamma0 + gamma_sum) / n
    if var_d <= 0:
        return 0.0, 1.0
        
    dm_stat = mean_d / np.sqrt(var_d)
    p_value = 2.0 * (1.0 - stats.norm.cdf(np.abs(dm_stat)))
    return dm_stat, p_value

def run_system_audit():
    print("==========================================================================")
    print(" MASTER SYSTEM AUDIT: ALL ENVIRONMENTAL TARGETS (RAW & NORMALIZED RMSE) ")
    print("==========================================================================")
    
    harmonizer = EnvironmentDatasetHarmonizer()
    df_featured = harmonizer.build_features_and_targets()
    
    config = harmonizer.config
    horizons = config["forecasting"]["horizons"]
    targets = config["forecasting"]["target_indicators"]
    
    feature_cols = harmonizer.extract_state_vectors(df_featured)[1]
    
    audit_results = []
    
    start_train_year = 1990
    test_window_size = 5
    
    for target in targets:
        print(f"\n==================== Target: {target} ====================")
        for h in horizons:
            target_col = f"{target}_target_h{h}"
            naive_baseline_col = target
            
            if target_col not in df_featured.columns:
                continue
                
            ml_errors = []
            ridge_errors = []
            naive_errors = []
            y_actuals = []
            
            n_folds = config["models"].get("cv_folds", 5)
            for fold in range(n_folds):
                split_year = start_train_year + fold * test_window_size
                df_tr = df_featured[df_featured["year"] <= split_year]
                df_te = df_featured[(df_featured["year"] > split_year) & (df_featured["year"] <= split_year + test_window_size)]
                
                valid_tr = df_tr.dropna(subset=feature_cols + [target_col])
                valid_te = df_te.dropna(subset=feature_cols + [target_col, naive_baseline_col])
                
                if len(valid_tr) < 50 or len(valid_te) < 10:
                    continue
                    
                forecaster = EnvironmentMultiHorizonForecaster()
                forecaster.fit(valid_tr, target_list=[target])
                
                preds_dict = forecaster.predict(valid_te, target_indicator=target)
                ml_pred = preds_dict[h]["point_ensemble"]
                ridge_pred = preds_dict[h]["ridge_baseline"]
                
                y_actual = valid_te[target_col].values
                naive_pred = valid_te[naive_baseline_col].values
                
                ml_errors.extend(y_actual - ml_pred)
                ridge_errors.extend(y_actual - ridge_pred)
                naive_errors.extend(y_actual - naive_pred)
                y_actuals.extend(y_actual)
                
            if len(ml_errors) == 0:
                continue
                
            ml_errors = np.array(ml_errors)
            ridge_errors = np.array(ridge_errors)
            naive_errors = np.array(naive_errors)
            y_actuals = np.array(y_actuals)
            
            ml_rmse = np.sqrt(np.mean(ml_errors**2))
            ml_mae = np.mean(np.abs(ml_errors))
            ridge_rmse = np.sqrt(np.mean(ridge_errors**2))
            naive_rmse = np.sqrt(np.mean(naive_errors**2))
            
            # Target standard deviation and Normalized RMSE (NRMSE)
            target_std = np.std(y_actuals) if np.std(y_actuals) > 1e-8 else 1.0
            ml_nrmse = ml_rmse / target_std
            naive_nrmse = naive_rmse / target_std
            
            dm_vs_naive_stat, dm_vs_naive_p = diebold_mariano_test(ml_errors, naive_errors, h=h)
            
            print(f" Horizon h={h}y: ML RMSE={ml_rmse:.4f} (NRMSE={ml_nrmse:.2%}, MAE={ml_mae:.4f}) | "
                  f"Naive RMSE={naive_rmse:.4f} | DM Stat={dm_vs_naive_stat:.2f} (p={dm_vs_naive_p:.4f})")
            
            audit_results.append({
                "target_indicator": target,
                "horizon_years": h,
                "sample_count": len(ml_errors),
                "ml_ensemble_rmse": float(np.round(ml_rmse, 4)),
                "ml_ensemble_mae": float(np.round(ml_mae, 4)),
                "normalized_rmse_nrmse": float(np.round(ml_nrmse, 4)),
                "nrmse_pct": f"{ml_nrmse*100:.1f}%",
                "naive_baseline_rmse": float(np.round(naive_rmse, 4)),
                "dm_vs_naive_stat": float(np.round(dm_vs_naive_stat, 3)),
                "dm_vs_naive_pvalue": float(np.round(dm_vs_naive_p, 4)),
                "statistically_superior": dm_vs_naive_p < 0.05 and dm_vs_naive_stat < 0
            })
            
    audit_df = pd.DataFrame(audit_results)
    out_path = os.path.join("data", "audit_tournament_results.csv")
    audit_df.to_csv(out_path, index=False)
    print("\n==========================================================================")
    print(f"Master Audit completed across ALL targets. Saved to: {out_path}")
    print("==========================================================================")
    return audit_df

if __name__ == "__main__":
    run_system_audit()
