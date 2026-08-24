import os
import yaml
import numpy as np
import pandas as pd
import faiss

class AnalogTrajectoryEngine:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.entity_col = self.config['domain']['entity_label']
        self.time_col = self.config['domain']['time_col']
        self.target_ind = self.config['forecasting']['target_indicator']
        self.horizons = self.config['forecasting']['horizons']
        self.min_overlap = self.config['retrieval']['min_overlap_ratio']
        self.index = None
        self.metadata_df = None
        self.feature_cols = None

    def fit_index(self, df_features, rank_feature_cols):
        """
        Builds an L2 FAISS index over rank-transformed feature state vectors.
        """
        print("Constructing FAISS Index for Analog Trajectory Engine...")
        self.feature_cols = rank_feature_cols
        
        # Drop rows with excessive NaNs in rank features
        df_clean = df_features.dropna(subset=rank_feature_cols).copy().reset_index(drop=True)
        self.metadata_df = df_clean
        
        vectors = df_clean[rank_feature_cols].values.astype(np.float32)
        dimension = vectors.shape[1]
        
        # Build FAISS L2 index
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(vectors)
        
        print(f"FAISS Index built successfully! Total indexed state vectors: {self.index.ntotal}, Vector dim: {dimension}")
        return self

    def find_analogs(self, entity_id, timestamp, k=8):
        """
        Searches for top-K historical analogs for a specified entity and timestamp.
        Excludes exact self-matches at the same timestamp.
        """
        if self.index is None or self.metadata_df is None:
            raise ValueError("Index is not fitted. Call fit_index first.")
            
        ts = pd.to_datetime(timestamp)
        match_query = self.metadata_df[
            (self.metadata_df[self.entity_col] == entity_id) & 
            (pd.to_datetime(self.metadata_df[self.time_col]) == ts)
        ]
        
        if len(match_query) == 0:
            # Fallback to latest available record for entity
            match_query = self.metadata_df[self.metadata_df[self.entity_col] == entity_id].sort_values(self.time_col).iloc[-1:]
            if len(match_query) == 0:
                raise ValueError(f"No records found for entity {entity_id}")
                
        query_vector = match_query[self.feature_cols].values.astype(np.float32)
        
        # Retrieve extra neighbors to exclude self-match
        distances, indices = self.index.search(query_vector, k + 10)
        
        distances = distances[0]
        indices = indices[0]
        
        results = []
        max_d = max(np.max(distances), 1e-5)
        
        for d, idx in zip(distances, indices):
            row = self.metadata_df.iloc[idx]
            match_entity = row[self.entity_col]
            match_ts = row[self.time_col]
            
            # Exclude self exact match
            if match_entity == entity_id and pd.to_datetime(match_ts) == ts:
                continue
                
            # Confidence score scaling
            confidence = max(0.0, round(float(1.0 - (d / (max_d * 1.5))), 4)) * 100.0
            
            # Forward realized trajectory
            forward_trajectory = {}
            for h in self.horizons:
                h_col = f"target_h_{h}"
                if h_col in row and not pd.isna(row[h_col]):
                    forward_trajectory[f"h_{h}"] = round(float(row[h_col]), 2)
                else:
                    forward_trajectory[f"h_{h}"] = None
                    
            results.append({
                'entity_id': match_entity,
                'region': row.get('region', 'N/A'),
                'timestamp': str(pd.to_datetime(match_ts).date()),
                'distance': round(float(d), 4),
                'similarity_score_pct': round(confidence, 2),
                'forward_trajectory': forward_trajectory,
                'current_target_value': round(float(row[self.target_ind]), 2) if self.target_ind in row else None
            })
            
            if len(results) >= k:
                break
                
        return results

if __name__ == "__main__":
    from feature_pipeline import DynamicFeaturePipeline
    
    df_feat = pd.read_parquet("data/dataset_features.parquet")
    pipeline = DynamicFeaturePipeline()
    _, f_cols, r_cols, t_cols = pipeline.create_features(pd.read_parquet("data/dataset_wide.parquet"))
    
    engine = AnalogTrajectoryEngine()
    engine.fit_index(df_feat, r_cols)
    
    analogs = engine.find_analogs('USA', '2022-01-01', k=5)
    print(f"Top 5 Historical Analogs for USA at 2022-01-01:")
    for a in analogs:
        print(a)
