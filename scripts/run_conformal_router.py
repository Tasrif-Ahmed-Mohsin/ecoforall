"""
Execute LGCF-v2 Conformal Uncertainty-Weighted Router
=====================================================
Evaluates 5-fold Walk-Forward cross validation of the 4-specialist
mixture across horizons h in {1, 3, 5} years.
"""

from __future__ import annotations
import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from src.models.specialists import train_specialist_suite, predict_specialist_suite
from src.gating.conformal_router import compute_conformal_uncertainty_weights
from src.econometrics.panel_granger import diebold_mariano_test


def run_conformal_router_benchmark():
    panel_path = ROOT / "data" / "processed_panels" / "quad_domain_annual_panel.parquet"
    if not panel_path.exists():
        panel_path = ROOT / "data" / "quad_domain_annual_panel.parquet"
    
    print(f"Loading Quad Panel: {panel_path}")
    df = pd.read_parquet(panel_path)
    
    # Feature selection
    exclude = {"iso3", "country", "year", "region", "income_level"}
    feature_cols = [c for c in df.columns if c not in exclude and not c.startswith("gdp_pc_growth_")]
    
    horizons = [1, 3, 5]
    print(f"Evaluating {len(feature_cols)} features across {len(df)} country-years.")
    print("=" * 70)

    for h in horizons:
        target_col = f"gdp_pc_growth_{h}y_fwd"
        if target_col not in df.columns:
            continue
            
        clean = df.dropna(subset=[target_col]).copy()
        years = sorted(clean["year"].unique())
        
        # 5-fold walk-forward split
        fold_results = []
        n_folds = 5
        test_window = 4
        
        for fold in range(n_folds):
            test_end = years[-1] - fold * test_window
            test_start = test_end - test_window + 1
            train_end = test_start - 1 - (h - 1)  # Strict anti-leakage purge
            
            tr = clean[clean["year"] <= train_end]
            te = clean[(clean["year"] >= test_start) & (clean["year"] <= test_end)]
            
            if len(tr) < 200 or len(te) < 20:
                continue
                
            X_tr = tr[feature_cols]
            y_tr = tr[target_col]
            X_te = te[feature_cols]
            y_te = te[target_col]
            
            # Train orthogonal specialist suite
            suite = train_specialist_suite(X_tr, y_tr, seed=42 + fold)
            preds = predict_specialist_suite(suite, X_te)
            
            # Conformal Uncertainty Routing
            gated_pred, weights = compute_conformal_uncertainty_weights(
                preds, suite.train_residuals, alpha=0.10, tau=0.05
            )
            
            # Baseline (Ridge Trend)
            mae_base = np.mean(np.abs(y_te - preds["ridge"]))
            mae_gated = np.mean(np.abs(y_te - gated_pred))
            lift = (mae_base - mae_gated) / mae_base * 100.0
            
            fold_results.append({
                "fold": fold,
                "n_test": len(te),
                "mae_base": mae_base,
                "mae_gated": mae_gated,
                "lift_pct": lift
            })
            
        res_df = pd.DataFrame(fold_results)
        mean_base = res_df["mae_base"].mean()
        mean_gated = res_df["mae_gated"].mean()
        mean_lift = (mean_base - mean_gated) / mean_base * 100.0
        
        print(f"Horizon h={h}Y: Baseline MAE = {mean_base:.4f} | LGCF-v2 Gated MAE = {mean_gated:.4f} | Lift = +{mean_lift:.2f}%")

    print("=" * 70)
    print("Conformal Specialist Router Benchmark Complete.")


if __name__ == "__main__":
    run_conformal_router_benchmark()
