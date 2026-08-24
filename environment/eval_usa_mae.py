"""
Fast USA-Specific 5-Fold Expanding Window Cross-Validation Evaluation Script.

Executes a 5-fold temporal expanding window cross-validation specifically for ISO3='USA'
across all 13 environmental and disaster indicators.
"""

import os
import numpy as np
import pandas as pd
from cross_domain_dataset_harmonizer import EnvironmentDatasetHarmonizer
from forecaster import EnvironmentMultiHorizonForecaster

def evaluate_usa_5fold_mae():
    harmonizer = EnvironmentDatasetHarmonizer()
    df_featured = harmonizer.build_features_and_targets()
    
    config = harmonizer.config
    horizons = config["forecasting"]["horizons"]
    targets = config["forecasting"]["target_indicators"]
    
    start_train_year = 1990
    test_window_size = 5
    n_folds = 5
    
    # Store errors per target and horizon: errors_dict[target][h] = (list of actuals, list of preds)
    results_store = {t: {h: {"actuals": [], "preds": []} for h in horizons} for t in targets}
    
    for fold in range(n_folds):
        split_year = start_train_year + fold * test_window_size
        df_tr = df_featured[df_featured["year"] <= split_year]
        df_te_usa = df_featured[(df_featured["year"] > split_year) & 
                                (df_featured["year"] <= split_year + test_window_size) & 
                                (df_featured["iso3"] == "USA")]
        
        if len(df_tr) < 50 or len(df_te_usa) == 0:
            continue
            
        forecaster = EnvironmentMultiHorizonForecaster()
        forecaster.fit(df_tr, target_list=targets)
        
        for target in targets:
            target_cols = [f"{target}_target_h{h}" for h in horizons]
            valid_te_usa = df_te_usa.dropna(subset=[f for f in target_cols if f in df_te_usa.columns])
            if len(valid_te_usa) == 0:
                continue
                
            preds_dict = forecaster.predict(valid_te_usa, target_indicator=target)
            
            for h in horizons:
                if h in preds_dict:
                    actuals = valid_te_usa[f"{target}_target_h{h}"].values
                    preds = preds_dict[h]["point_ensemble"]
                    
                    results_store[target][h]["actuals"].extend(actuals)
                    results_store[target][h]["preds"].extend(preds)
                    
    results = []
    for target in targets:
        for h in horizons:
            acts = np.array(results_store[target][h]["actuals"])
            prds = np.array(results_store[target][h]["preds"])
            
            if len(acts) == 0:
                continue
                
            errors = acts - prds
            mae = np.mean(np.abs(errors))
            rmse = np.sqrt(np.mean(errors**2))
            mean_actual = np.mean(acts)
            mae_pct = (mae / mean_actual * 100.0) if abs(mean_actual) > 1e-5 else np.nan
            
            results.append({
                "target_indicator": target,
                "horizon_years": f"+{h} Year(s)",
                "eval_mode": "5-Fold CV (USA)",
                "sample_count": len(acts),
                "usa_mean_actual": float(np.round(mean_actual, 4)),
                "usa_mae": float(np.round(mae, 4)),
                "usa_rmse": float(np.round(rmse, 4)),
                "mae_relative_pct": f"{mae_pct:.1f}%" if not np.isnan(mae_pct) else "N/A"
            })
            
    res_df = pd.DataFrame(results)
    out_csv = os.path.join("data", "usa_mae_metrics.csv")
    res_df.to_csv(out_csv, index=False)
    
    print("==========================================================================")
    print(" USA 5-FOLD EXPANDING WINDOW CROSS-VALIDATION RESULTS ")
    print("==========================================================================")
    print(res_df.to_string(index=False))
    print("==========================================================================")
    return res_df

if __name__ == "__main__":
    evaluate_usa_5fold_mae()
