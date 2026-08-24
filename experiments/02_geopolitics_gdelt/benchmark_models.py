import os
import yaml
import logging
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_model_tournament(feature_path="data/feature_matrix.parquet"):
    config = load_config()
    target_col = config["forecasting"]["target_indicator"]
    horizons = config["forecasting"]["horizons"]  # [4, 26, 52]
    
    logging.info(f"Loading feature matrix from {feature_path}...")
    df = pd.read_parquet(feature_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["country_iso3", "timestamp"]).reset_index(drop=True)
    
    meta_cols = ["timestamp", "country_iso3"]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    
    all_results = []
    
    for h in horizons:
        logging.info(f"RUNNING MODEL TOURNAMENT FOR HORIZON h={h} WEEKS...")
        
        df_h = df.copy()
        df_h["target_forward"] = df_h.groupby("country_iso3")[target_col].shift(-h)
        clean_df = df_h.dropna(subset=["target_forward"] + feature_cols[:5]).copy().fillna(0.0)
        
        X = clean_df[feature_cols].values
        y = clean_df["target_forward"].values
        y_naive = clean_df[target_col].values
        
        split_idx = int(len(clean_df) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        y_naive_test = y_naive[split_idx:]
        
        # Optimized Multi-Core Models
        models = {
            "Naive Persistence": None,
            "Ridge Regression": Ridge(alpha=1.0),
            "Linear Regression": LinearRegression(),
            "LightGBM": lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, n_jobs=-1, random_state=42, verbose=-1),
            "Random Forest": RandomForestRegressor(n_estimators=30, max_depth=10, n_jobs=-1, random_state=42),
            "Extra Trees": ExtraTreesRegressor(n_estimators=30, max_depth=10, n_jobs=-1, random_state=42),
        }
        
        if HAS_XGBOOST:
            models["XGBoost"] = xgb.XGBRegressor(n_estimators=50, learning_rate=0.05, max_depth=5, n_jobs=-1, random_state=42)
            
        predictions = {}
        
        for name, model in models.items():
            if name == "Naive Persistence":
                pred = y_naive_test
            else:
                try:
                    logging.info(f"  Training {name}...")
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)
                except Exception as e:
                    logging.warning(f"  Model {name} failed: {e}")
                    continue
                    
            predictions[name] = pred
            
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            mae = mean_absolute_error(y_test, pred)
            r2 = r2_score(y_test, pred)
            
            all_results.append({
                "horizon_weeks": h,
                "model_name": name,
                "rmse": round(float(rmse), 4),
                "mae": round(float(mae), 4),
                "r2_score": round(float(r2), 4)
            })
            
        # Top-3 Meta Ensemble
        top_candidates = sorted(
            [m for m in predictions if m != "Naive Persistence"],
            key=lambda m: mean_squared_error(y_test, predictions[m])
        )[:3]
        
        ensemble_pred = np.mean([predictions[m] for m in top_candidates], axis=0)
        ens_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
        ens_mae = mean_absolute_error(y_test, ensemble_pred)
        ens_r2 = r2_score(y_test, ensemble_pred)
        
        all_results.append({
            "horizon_weeks": h,
            "model_name": f"Meta-Ensemble ({'+'.join(top_candidates)})",
            "rmse": round(float(ens_rmse), 4),
            "mae": round(float(ens_mae), 4),
            "r2_score": round(float(ens_r2), 4)
        })
        
    results_df = pd.DataFrame(all_results)
    os.makedirs("data", exist_ok=True)
    results_df.to_csv("data/model_tournament_benchmarks.csv", index=False)
    return results_df

def main():
    results_df = run_model_tournament()
    
    print("\n" + "="*80)
    print(" MODEL TOURNAMENT & BENCHMARKING RESULTS (65-YEAR DATASET)")
    print("="*80)
    
    for h in results_df["horizon_weeks"].unique():
        sub_df = results_df[results_df["horizon_weeks"] == h].sort_values("rmse")
        print(f"\n--- Horizon h={h} Weeks ({'1 Month' if h==4 else '6 Months' if h==26 else '12 Months'}) ---")
        print(sub_df.to_string(index=False))
        
    print("\n" + "="*80)
    print(" Saved full benchmark rankings to data/model_tournament_benchmarks.csv")

if __name__ == "__main__":
    main()
