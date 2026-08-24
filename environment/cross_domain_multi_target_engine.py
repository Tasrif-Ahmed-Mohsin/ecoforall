"""
Cross-Domain Multi-Target Synergy & Inter-Indicator Predictability Engine.

Evaluates how environmental drivers (CO2, Energy, Deforestation) interact to predict
secondary targets (Temperature Anomalies & Extreme Climate Disasters).
"""

import os
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

class CrossDomainMultiTargetEngine:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.targets = self.config["forecasting"]["target_indicators"]
        
    def evaluate_inter_target_predictability(self, df_featured):
        """Measures predictive power (R^2 & RMSE) when forecasting each target using all other drivers."""
        results = []
        
        feature_cols = [c for c in df_featured.columns if c.endswith("_rank") or c.endswith("_roll_mean_10y") or c.endswith("_velocity_3y")]
        
        for target in self.targets:
            target_col = f"{target}_target_h5" # 5-year forward horizon
            if target_col not in df_featured.columns:
                continue
                
            valid_df = df_featured.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)
            if len(valid_df) < 100:
                continue
                
            X = valid_df[feature_cols]
            y = valid_df[target_col]
            
            # 80/20 temporal train/test split
            split_idx = int(len(valid_df) * 0.8)
            X_tr, X_te = X.iloc[:split_idx], X.iloc[split_idx:]
            y_tr, y_te = y.iloc[:split_idx], y.iloc[split_idx:]
            
            model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
            model.fit(X_tr, y_tr)
            
            preds = model.predict(X_te)
            r2 = r2_score(y_te, preds)
            rmse = np.sqrt(mean_squared_error(y_te, preds))
            
            # Extract top feature importances
            importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
            top_3_drivers = ", ".join(importances.head(3).index)
            
            results.append({
                "target_indicator": target,
                "horizon": "5_years",
                "r2_score": float(np.round(r2, 4)),
                "rmse": float(np.round(rmse, 4)),
                "top_predictive_drivers": top_3_drivers
            })
            
        return pd.DataFrame(results)

def main():
    from cross_domain_dataset_harmonizer import EnvironmentDatasetHarmonizer
    print("Running Multi-Target Synergy Analysis...")
    harmonizer = EnvironmentDatasetHarmonizer()
    df_featured = harmonizer.build_features_and_targets()
    
    engine = CrossDomainMultiTargetEngine()
    synergy_df = engine.evaluate_inter_target_predictability(df_featured)
    
    print("\nMulti-Target Predictability & Driver Synergies (5-Year Horizon):")
    print(synergy_df.to_string(index=False))
    
    output_path = os.path.join("data", "multi_target_synergy_results.csv")
    synergy_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
