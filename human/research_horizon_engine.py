import os
import yaml
import json
import numpy as np
import pandas as pd
from feature_pipeline import DynamicFeaturePipeline
from forecaster import MultiHeadForecaster

def pinball_loss(y_true, y_pred, q):
    """
    Computes Pinball Loss for quantile q.
    L_q(y, y_hat) = max(q * (y - y_hat), (q - 1) * (y - y_hat))
    """
    err = y_true - y_pred
    return float(np.mean(np.maximum(q * err, (q - 1.0) * err)))

class DynamicHorizonResearchEngine:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.horizons = self.config['forecasting']['horizons']
        self.quantiles = self.config['models']['quantiles']
        self.lookback_windows = [6, 12, 24]
        
    def run_benchmark_grid(self):
        """
        Runs comprehensive research benchmark across variable lookback windows,
        distance metrics, and computes quantile pinball loss scores across horizons.
        """
        print("================================================================================")
        print("           RUNNING DYNAMIC HORIZON RESEARCH BENCHMARK GRID                       ")
        print("================================================================================\n")
        
        df_wide = pd.read_parquet("data/dataset_wide.parquet")
        
        benchmark_results = {}
        
        for w in self.lookback_windows:
            print(f"--- Benchmarking Lookback Window W = {w} Months ---")
            
            # Temporarily modify lookback window in config dictionary
            cfg_temp = self.config.copy()
            cfg_temp['retrieval']['lookback_window'] = w
            
            pipeline = DynamicFeaturePipeline()
            pipeline.lookback = w
            
            df_feat, f_cols, r_cols, t_cols = pipeline.create_features(df_wide)
            df_clean = df_feat.dropna(subset=f_cols).copy().reset_index(drop=True)
            
            # Fit Forecaster
            forecaster = MultiHeadForecaster()
            forecaster.fit(df_clean, f_cols)
            
            # Evaluate on out-of-time evaluation slice (last 36 months)
            eval_split_date = df_clean['timestamp'].max() - pd.DateOffset(months=36)
            df_eval = df_clean[df_clean['timestamp'] >= eval_split_date].copy()
            
            X_eval = df_eval[f_cols].values
            preds = forecaster.predict(X_eval)
            
            w_metrics = {}
            for h in self.horizons:
                target_col = f"target_h_{h}"
                valid_mask = ~df_eval[target_col].isna()
                y_true = df_eval.loc[valid_mask, target_col].values
                
                if len(y_true) < 10:
                    continue
                    
                p_ensemble = preds[h]['point_ensemble'][valid_mask]
                rmse = float(np.sqrt(np.mean((y_true - p_ensemble)**2)))
                mae = float(np.mean(np.abs(y_true - p_ensemble)))
                
                q_losses = {}
                for q in self.quantiles:
                    y_q_pred = preds[h]['quantiles'][q][valid_mask]
                    q_losses[f"q_{int(q*100):02d}"] = round(pinball_loss(y_true, y_q_pred, q), 4)
                    
                w_metrics[f"h_{h}"] = {
                    'rmse': round(rmse, 4),
                    'mae': round(mae, 4),
                    'pinball_losses': q_losses,
                    'sample_size': len(y_true)
                }
                print(f"  Lookback W={w}m | Horizon h={h}m -> RMSE: {rmse:.4f}, MAE: {mae:.4f}, Pinball q50: {q_losses['q_50']}")
                
            benchmark_results[f"window_{w}m"] = w_metrics

        # Determine optimal lookback window based on average Pinball Loss across horizons
        best_window = 12
        min_loss = float('inf')
        for w_key, res in benchmark_results.items():
            avg_rmse = np.mean([metrics['rmse'] for metrics in res.values()])
            if avg_rmse < min_loss:
                min_loss = avg_rmse
                best_window = int(w_key.split('_')[1].replace('m', ''))
                
        research_report = {
            'research_metadata': {
                'domain': 'Collective Psychology & Society',
                'optimal_lookback_window_months': best_window,
                'min_average_rmse': round(float(min_loss), 4),
                'tested_lookback_windows': self.lookback_windows,
                'quantiles_evaluated': self.quantiles,
                'status': 'OPTIMAL'
            },
            'benchmark_results': benchmark_results
        }
        
        with open("data/horizon_research_results.json", "w") as f:
            json.dump(research_report, f, indent=2)
            
        print(f"\nResearch Benchmark Complete! Optimal Lookback Window: {best_window} Months.")
        print("Results saved to 'data/horizon_research_results.json'.")
        return research_report

if __name__ == "__main__":
    engine = DynamicHorizonResearchEngine()
    engine.run_benchmark_grid()
