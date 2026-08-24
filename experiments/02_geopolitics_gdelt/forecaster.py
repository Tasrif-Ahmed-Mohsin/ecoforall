import os
import yaml
import logging
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import norm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def diebold_mariano_test(e1, e2, h=1):
    """
    Diebold-Mariano test for predictive accuracy equality with Newey-West HAC variance adjustment.
    """
    d = e1**2 - e2**2
    n = len(d)
    mean_d = np.mean(d)
    
    autocov = np.var(d)
    for lag in range(1, h):
        gamma = np.cov(d[lag:], d[:-lag])[0, 1]
        autocov += 2 * (1 - lag / h) * gamma
        
    var_d = max(1e-8, autocov / n)
    dm_stat = mean_d / np.sqrt(var_d)
    p_value = 2 * (1 - norm.cdf(abs(dm_stat)))
    return float(dm_stat), float(p_value)

class MultiHeadQuantileForecaster:
    """
    Winning Meta-Ensemble Forecaster with 5-Fold Expanding Window Walk-Forward CV.
    """
    def __init__(self, feature_path="data/feature_matrix.parquet", config_path="config.yaml"):
        self.config = load_config(config_path)
        self.feature_path = feature_path
        self.load_data()
        
    def load_data(self):
        logging.info(f"Loading feature matrix from {self.feature_path}...")
        self.df = pd.read_parquet(self.feature_path)
        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])
        self.df = self.df.sort_values(["country_iso3", "timestamp"]).reset_index(drop=True)
        
        self.target_col = self.config["forecasting"]["target_indicator"]
        self.horizons = self.config["forecasting"]["horizons"]
        self.cv_folds = self.config["models"].get("cv_folds", 5)
        
        meta_cols = ["timestamp", "country_iso3"]
        self.feature_cols = [c for c in self.df.columns if c not in meta_cols]
        
    def train_horizon_5fold_cv(self, horizon):
        logging.info(f"\n=======================================================")
        logging.info(f" 5-FOLD WALK-FORWARD CV FOR HORIZON h={horizon} WEEKS")
        logging.info(f"=======================================================")
        
        df_h = self.df.copy()
        df_h["target_forward"] = df_h.groupby("country_iso3")[self.target_col].shift(-horizon)
        clean_df = df_h.dropna(subset=["target_forward"] + self.feature_cols[:5]).copy().fillna(0.0)
        clean_df = clean_df.sort_values("timestamp").reset_index(drop=True)
        
        X = clean_df[self.feature_cols].values
        y = clean_df["target_forward"].values
        y_naive = clean_df[self.target_col].values
        n_total = len(clean_df)
        
        fold_results = []
        all_oof_y_true = []
        all_oof_pred_ens = []
        all_oof_y_naive = []
        
        # 5 Expanding Folds: Train on [0, train_end], Test on [train_end, test_end]
        min_train_pct = 0.50
        step_pct = (1.0 - min_train_pct) / self.cv_folds
        
        for fold in range(1, self.cv_folds + 1):
            train_end_pct = min_train_pct + (fold - 1) * step_pct
            test_end_pct = min_train_pct + fold * step_pct
            
            train_idx = int(n_total * train_end_pct)
            test_idx = int(n_total * test_end_pct)
            
            X_tr, y_tr = X[:train_idx], y[:train_idx]
            X_val, y_val = X[train_idx:test_idx], y[train_idx:test_idx]
            y_naive_val = y_naive[train_idx:test_idx]
            
            if len(y_val) == 0:
                continue
                
            # Fit Models on Expanding Training Set
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_tr, y_tr)
            pred_ridge = ridge.predict(X_val)
            
            lr = LinearRegression()
            lr.fit(X_tr, y_tr)
            pred_lr = lr.predict(X_val)
            
            lgb_point = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, n_jobs=-1, random_state=42, verbose=-1)
            lgb_point.fit(X_tr, y_tr)
            pred_lgb = lgb_point.predict(X_val)
            
            # Meta-Ensemble (40% Ridge + 40% Linear + 20% LightGBM)
            pred_ens = 0.4 * pred_ridge + 0.4 * pred_lr + 0.2 * pred_lgb
            
            rmse_f = np.sqrt(mean_squared_error(y_val, pred_ens))
            rmse_n_f = np.sqrt(mean_squared_error(y_val, y_naive_val))
            mae_f = mean_absolute_error(y_val, pred_ens)
            r2_f = r2_score(y_val, pred_ens)
            
            fold_results.append({
                "horizon_weeks": horizon,
                "fold": fold,
                "train_samples": train_idx,
                "val_samples": len(y_val),
                "rmse_ensemble": round(float(rmse_f), 4),
                "rmse_naive": round(float(rmse_n_f), 4),
                "mae_ensemble": round(float(mae_f), 4),
                "r2_score": round(float(r2_f), 4)
            })
            
            all_oof_y_true.extend(y_val)
            all_oof_pred_ens.extend(pred_ens)
            all_oof_y_naive.extend(y_naive_val)
            
            logging.info(f"  Fold {fold}/5 | Train: {train_idx:,} | Test: {len(y_val):,} | Ensemble RMSE: {rmse_f:.2f} (R2: {r2_f:.4f}) vs Naive RMSE: {rmse_n_f:.2f}")
            
        # Overall 5-Fold Cross-Validated Metrics
        oof_y_true = np.array(all_oof_y_true)
        oof_pred_ens = np.array(all_oof_pred_ens)
        oof_y_naive = np.array(all_oof_y_naive)
        
        total_rmse = np.sqrt(mean_squared_error(oof_y_true, oof_pred_ens))
        total_naive_rmse = np.sqrt(mean_squared_error(oof_y_true, oof_y_naive))
        total_mae = mean_absolute_error(oof_y_true, oof_pred_ens)
        total_r2 = r2_score(oof_y_true, oof_pred_ens)
        
        dm_stat, p_val = diebold_mariano_test(oof_y_true - oof_pred_ens, oof_y_true - oof_y_naive, h=horizon)
        
        overall_summary = {
            "horizon_weeks": horizon,
            "5fold_cv_rmse_ensemble": round(float(total_rmse), 4),
            "5fold_cv_rmse_naive": round(float(total_naive_rmse), 4),
            "5fold_cv_mae": round(float(total_mae), 4),
            "5fold_cv_r2_score": round(float(total_r2), 4),
            "dm_stat": round(float(dm_stat), 4),
            "p_value": round(float(p_val), 4),
            "significant_outperformance": bool(p_val < 0.05 and total_rmse < total_naive_rmse)
        }
        
        logging.info(f"--> HORIZON {horizon}w OVERALL 5-FOLD CV RESULT: RMSE={total_rmse:.2f} (R2={total_r2:.4f}) vs Naive RMSE={total_naive_rmse:.2f} | DM p-val={p_val:.4f}")
        return overall_summary, fold_results

    def run_all_horizons_cv(self):
        overall_summaries = []
        detailed_folds = []
        
        for h in self.horizons:
            summary, Folds = self.train_horizon_5fold_cv(h)
            overall_summaries.append(summary)
            detailed_folds.extend(Folds)
            
        summary_df = pd.DataFrame(overall_summaries)
        folds_df = pd.DataFrame(detailed_folds)
        
        os.makedirs("data", exist_ok=True)
        summary_df.to_csv("data/forecast_evaluation_results.csv", index=False)
        folds_df.to_csv("data/5fold_cv_detailed_results.csv", index=False)
        return summary_df, folds_df

def main():
    forecaster = MultiHeadQuantileForecaster()
    summary_df, folds_df = forecaster.run_all_horizons_cv()
    
    print("\n" + "="*85)
    print(" 5-FOLD WALK-FORWARD CROSS-VALIDATION SUMMARY (65-YEAR DATASET)")
    print("="*85)
    print(summary_df.to_string(index=False))
    print("="*85)
    print("\n Saved 5-Fold summary to data/forecast_evaluation_results.csv")
    print(" Saved detailed fold logs to data/5fold_cv_detailed_results.csv")

if __name__ == "__main__":
    main()
