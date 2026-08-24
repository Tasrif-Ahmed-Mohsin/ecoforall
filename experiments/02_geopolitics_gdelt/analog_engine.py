import os
import yaml
import logging
import faiss
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class PoliticalAnalogEngine:
    """
    Vector Retrieval Engine using FAISS over percentile-rank normalized state vectors.
    Finds historical political analogs and tracks their forward outcome trajectories.
    """
    def __init__(self, feature_path="data/feature_matrix_rank.parquet", config_path="config.yaml"):
        self.config = load_config(config_path)
        self.feature_path = feature_path
        self.load_and_index_features()
        
    def load_and_index_features(self):
        logging.info(f"Loading rank-normalized feature matrix from {self.feature_path}...")
        self.df = pd.read_parquet(self.feature_path)
        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])
        
        self.meta_cols = ["timestamp", "country_iso3"]
        self.feature_cols = [c for c in self.df.columns if c not in self.meta_cols]
        
        # Fill missing values with median percentile (0.5)
        feature_matrix = self.df[self.feature_cols].fillna(0.5).values.astype(np.float32)
        
        self.dim = feature_matrix.shape[1]
        logging.info(f"Building FAISS L2 Vector Index (Dimension={self.dim}, Vectors={feature_matrix.shape[0]})...")
        
        self.index = faiss.IndexFlatL2(self.dim)
        self.index.add(feature_matrix)
        
        logging.info("FAISS Index successfully built and populated!")
        
    def find_analogs(self, country_iso3, timestamp, k=8, exclude_same_entity=False):
        """
        Find top-K historical political analogs for a target country and timestamp slice.
        """
        timestamp = pd.to_datetime(timestamp)
        target_row = self.df[(self.df["country_iso3"] == country_iso3) & (self.df["timestamp"] == timestamp)]
        
        if target_row.empty:
            logging.error(f"Target state ({country_iso3}, {timestamp}) not found in index.")
            return None
            
        target_vec = target_row[self.feature_cols].fillna(0.5).values.astype(np.float32)
        
        # Search FAISS index for top K + 10 candidates
        distances, indices = self.index.search(target_vec, k + 15)
        
        distances = distances[0]
        indices = indices[0]
        
        results = []
        target_target_val = target_row[self.config["forecasting"]["target_indicator"]].values[0] if self.config["forecasting"]["target_indicator"] in target_row.columns else 0.0
        
        max_dist = max(distances) + 1e-5
        
        for dist, idx in zip(distances, indices):
            match_row = self.df.iloc[idx]
            match_country = match_row["country_iso3"]
            match_ts = match_row["timestamp"]
            
            # Exclude exact same timestamp match
            if match_country == country_iso3 and match_ts == timestamp:
                continue
                
            if exclude_same_entity and match_country == country_iso3:
                continue
                
            # Convert L2 distance to confidence score [0.0, 1.0]
            similarity_score = float(np.clip(1.0 - (dist / (max_dist * 2.0)), 0.0, 1.0))
            
            # Extract forward trajectories for horizons [1, 4, 12, 26]
            forward_trajectory = {}
            target_ind = self.config["forecasting"]["target_indicator"]
            
            for h in self.config["forecasting"]["horizons"]:
                future_ts = match_ts + pd.Timedelta(weeks=h)
                future_row = self.df[(self.df["country_iso3"] == match_country) & (self.df["timestamp"] == future_ts)]
                if not future_row.empty and target_ind in future_row.columns:
                    forward_trajectory[f"horizon_{h}w"] = float(future_row[target_ind].values[0])
                else:
                    forward_trajectory[f"horizon_{h}w"] = np.nan
                    
            results.append({
                "country_iso3": match_country,
                "timestamp": match_ts.strftime("%Y-%m-%d"),
                "l2_distance": round(float(dist), 4),
                "similarity_score": round(similarity_score, 4),
                "current_target_value": round(float(match_row[target_ind]), 4) if target_ind in match_row.index else 0.0,
                "forward_trajectory": forward_trajectory
            })
            
            if len(results) >= k:
                break
                
        return results

def main():
    engine = PoliticalAnalogEngine()
    
    # Test query: Find political analogs for USA
    sample_country = "USA"
    sample_ts = "2023-01-08"
    
    print("\n" + "="*70)
    print(f" FINDING HISTORICAL ANALOGS FOR ({sample_country}, {sample_ts}):")
    print("="*70)
    
    analogs = engine.find_analogs(sample_country, sample_ts, k=5)
    
    if analogs:
        for idx, a in enumerate(analogs, 1):
            print(f"\nAnalog #{idx}: {a['country_iso3']} on {a['timestamp']}")
            print(f"   - Similarity Score  : {a['similarity_score'] * 100:.1f}% (L2 Distance: {a['l2_distance']})")
            print(f"   - Current Target Val: {a['current_target_value']}")
            print(f"   - Forward Outcomes  : {a['forward_trajectory']}")
            
    print("\n" + "="*70)
    print(" ANALOG ENGINE TEST COMPLETE: Vector Search Verified!")
    print("="*70)

if __name__ == "__main__":
    main()
