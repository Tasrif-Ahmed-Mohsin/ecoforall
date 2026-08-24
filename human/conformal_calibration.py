import os
import yaml
import json
import numpy as np
import pandas as pd
from feature_pipeline import DynamicFeaturePipeline
from forecaster import MultiHeadForecaster

class SplitConformalCalibrator:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.horizons = self.config['forecasting']['horizons']
        self.target_coverage = 0.90 # 90% target coverage interval (between q05 and q95)

    def calibrate_intervals(self):
        """
        Executes out-of-time split-conformal calibration and calculates interval widening factors.
        """
        print("Executing Split-Conformal Prediction Band Calibration...")
        
        df_wide = pd.read_parquet("data/dataset_wide.parquet")
        pipeline = DynamicFeaturePipeline()
        df_feat, f_cols, r_cols, t_cols = pipeline.create_features(df_wide)
        
        df_clean = df_feat.dropna(subset=f_cols).sort_values(pipeline.time_col).reset_index(drop=True)
        
        # Split train / calibration split (80% train, 20% calibration)
        split_idx = int(len(df_clean) * 0.80)
        df_train = df_clean.iloc[:split_idx].copy()
        df_calib = df_clean.iloc[split_idx:].copy()
        
        forecaster = MultiHeadForecaster()
        forecaster.fit(df_train, f_cols)
        
        X_calib = df_calib[f_cols].values
        preds = forecaster.predict(X_calib)
        
        calibration_results = {}
        
        for h in self.horizons:
            target_col = f"target_h_{h}"
            valid_mask = ~df_calib[target_col].isna()
            y_val = df_calib.loc[valid_mask, target_col].values
            
            q05_val = preds[h]['quantiles'][0.05][valid_mask]
            q95_val = preds[h]['quantiles'][0.95][valid_mask]
            
            # Empirical coverage check: proportion of y_true inside [q05, q95]
            inside_mask = (y_val >= q05_val) & (y_val <= q95_val)
            empirical_coverage = float(np.mean(inside_mask))
            
            # Calculate non-parametric conformity scores s_i
            lower_residuals = q05_val - y_val
            upper_residuals = y_val - q95_val
            conformity_scores = np.maximum(lower_residuals, upper_residuals)
            
            # Conformal quantile adjustment at (1 - alpha) level
            alpha = 1.0 - self.target_coverage
            n_calib = len(conformity_scores)
            q_level = float(np.ceil((n_calib + 1) * (1.0 - alpha)) / float(n_calib))
            q_level = np.clip(q_level, 0.5, 0.99)
            
            conformal_margin = float(np.quantile(conformity_scores, q_level))
            widen_factor = round(max(1.0, 1.0 + (self.target_coverage - empirical_coverage)), 3)
            
            acceptable = bool(empirical_coverage >= 0.85)
            
            calibration_results[h] = {
                'empirical_coverage': round(empirical_coverage * 100.0, 2),
                'target_coverage': round(self.target_coverage * 100.0, 2),
                'calibration_acceptable': acceptable,
                'conformal_margin_adjustment': round(conformal_margin, 4),
                'recommended_widen_factor': widen_factor,
                'eval_sample_size': n_calib
            }
            print(f"Horizon h={h}m | Empirical Coverage: {empirical_coverage*100:.2f}% | Acceptable: {acceptable} | Margin Adj: {conformal_margin:.4f}")

        calibration_report = {
            'calibration_metadata': {
                'target_confidence_interval': '90% (q05 to q95)',
                'minimum_acceptable_threshold': '85%',
                'status': 'CALIBRATED'
            },
            'horizon_calibrations': calibration_results
        }
        
        with open("data/conformal_calibration.json", "w") as f:
            json.dump(calibration_report, f, indent=2)
            
        print("Split-Conformal Calibration Complete! Saved report to 'data/conformal_calibration.json'.")
        return calibration_report

if __name__ == "__main__":
    calibrator = SplitConformalCalibrator()
    calibrator.calibrate_intervals()
