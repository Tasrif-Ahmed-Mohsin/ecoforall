import os
import yaml
import numpy as np
import pandas as pd
from feature_pipeline import DynamicFeaturePipeline
from retrieval_engine import AnalogTrajectoryEngine
from forecaster import MultiHeadForecaster

class ScenarioStressEngine:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.entity_col = self.config['domain']['entity_label']
        self.time_col = self.config['domain']['time_col']
        self.horizons = self.config['forecasting']['horizons']
        self.quantiles = self.config['models']['quantiles']

    def simulate_counterfactual_shock(
        self,
        entity_id='USA',
        trust_shock_pct=0.0,
        polarization_shock_pct=0.0,
        aging_shock_pct=0.0
    ):
        """
        Simulates counterfactual socio-psychological shocks on an entity's latest state vector.
        Returns baseline vs counterfactual forecasts and post-shock FAISS historical analogs.
        """
        df_wide = pd.read_parquet("data/dataset_wide.parquet")
        pipeline = DynamicFeaturePipeline()
        df_feat, f_cols, r_cols, t_cols = pipeline.create_features(df_wide)
        
        # Get latest state for target entity
        df_c = df_feat[df_feat[self.entity_col] == entity_id].sort_values(self.time_col).reset_index(drop=True)
        latest_row = df_c.iloc[-1:].copy()
        
        # Apply shocks to raw indicators
        shocked_row = latest_row.copy()
        if trust_shock_pct != 0.0:
            val = shocked_row['psychology_trust'].values[0]
            shocked_row['psychology_trust'] = np.clip(val * (1.0 + trust_shock_pct / 100.0), 5.0, 95.0)
            
        if polarization_shock_pct != 0.0:
            val = shocked_row['psychology_social_cohesion'].values[0]
            shocked_row['psychology_social_cohesion'] = np.clip(val * (1.0 + polarization_shock_pct / 100.0), 5.0, 95.0)
            
        if aging_shock_pct != 0.0:
            val = shocked_row['society_age'].values[0]
            shocked_row['society_age'] = np.clip(val * (1.0 + aging_shock_pct / 100.0), 5.0, 95.0)

        # Fit forecaster and analog engine
        df_clean = df_feat.dropna(subset=f_cols).copy()
        forecaster = MultiHeadForecaster()
        forecaster.fit(df_clean, f_cols)
        
        analog_engine = AnalogTrajectoryEngine()
        analog_engine.fit_index(df_feat, r_cols)
        
        # Predict baseline
        base_preds = forecaster.predict(latest_row[f_cols].values)
        
        # Predict shocked counterfactual
        shock_preds = forecaster.predict(shocked_row[f_cols].values)
        
        # Search FAISS analogs for shocked state
        latest_ts = str(latest_row[self.time_col].values[0])[:10]
        post_shock_analogs = analog_engine.find_analogs(entity_id, latest_ts, k=5)
        
        simulation_summary = {
            'entity_id': entity_id,
            'timestamp': latest_ts,
            'shocks_applied': {
                'trust_shock_pct': trust_shock_pct,
                'polarization_shock_pct': polarization_shock_pct,
                'aging_shock_pct': aging_shock_pct
            },
            'baseline_forecasts': {h: round(float(base_preds[h]['point_ensemble'][0]), 2) for h in self.horizons},
            'counterfactual_forecasts': {h: round(float(shock_preds[h]['point_ensemble'][0]), 2) for h in self.horizons},
            'counterfactual_quantiles': {
                h: {f"q_{int(q*100):02d}": round(float(shock_preds[h]['quantiles'][q][0]), 2) for q in self.quantiles}
                for h in self.horizons
            },
            'post_shock_analogs': post_shock_analogs
        }
        
        print(f"Counterfactual simulation completed for {entity_id}!")
        print("Baseline vs Shocked Forecasts:", simulation_summary['counterfactual_forecasts'])
        return simulation_summary

if __name__ == "__main__":
    engine = ScenarioStressEngine()
    summary = engine.simulate_counterfactual_shock(
        entity_id='USA',
        trust_shock_pct=-15.0,
        polarization_shock_pct=-20.0
    )
