"""
LLM-Gated Cross-Domain Forecaster (LGCF) — Full Pipeline
==========================================================
Combines the LLM gate engine with domain-specific ML forecasters in a
mixture-of-experts architecture.

For each (country, year):
  1. LLM gate produces weights: g = [g_eco, g_pol, g_env, g_hum]
  2. Domain-specific models produce predictions: ŷ_eco, ŷ_eco+pol, ŷ_eco+env, ŷ_eco+hum, ŷ_full
  3. Gated prediction: ŷ = Σ g_d · ŷ_d (softmax-weighted mixture)

The ablation framework runs 6 configurations:
  A. Eco-Only baseline
  B. Uniform mixture (average all configs — the current null result)
  C. Random gating (proves LLM adds signal vs noise)
  D. Heuristic gating (proves LLM > hand-crafted rules)
  E. LLM-gated (the proposed contribution)
  F. Oracle gating (theoretical upper bound)
"""

from __future__ import annotations
import json
import logging
import time
import warnings
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

from llm_gate_engine import (
    GateWeights, build_context, compute_gate_weights,
    heuristic_gate, random_gate, build_gate_prompt, parse_gate_response,
)
from oracle_gating_analysis import (
    classify_features, build_domain_configs, make_target,
    rank_fit_transform, train_predict_models, diebold_mariano
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────── Config ───────────────────────────
ROOT = Path(r"e:\politics and economy")
QUAD_PANEL = ROOT / "data" / "quad_domain_annual_panel.parquet"
OUT_DIR = ROOT / "data" / "lgcf_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 3, 5]
N_FOLDS = 5
TEST_WINDOW = 4
MIN_TRAIN_ROWS = 200


# ─────────────────────────── Gated Mixture ───────────────────────────

def apply_gate_to_predictions(preds: dict[str, np.ndarray],
                               gate_weights: list[dict]) -> np.ndarray:
    """Apply gate weights to domain-specific predictions.
    
    Args:
        preds: {"eco_only": array, "eco_pol": array, "eco_env": array, "eco_human": array, "full_quad": array}
        gate_weights: list of dicts with keys {economy, politics, environment, human_society}
    
    Returns:
        Gated prediction array
    """
    n = len(gate_weights)
    gated_pred = np.zeros(n, dtype=np.float32)
    
    for i in range(n):
        gw = gate_weights[i]
        
        w_eco = gw.get("economy", 0.7)
        w_pol = gw.get("politics", 0.1)
        w_env = gw.get("environment", 0.1)
        w_hum = gw.get("human_society", 0.1)
        
        pred = 0.0
        total_w = 0.0
        
        if "eco_only" in preds:
            pred += w_eco * preds["eco_only"][i]
            total_w += w_eco
        if "eco_pol" in preds:
            pred += w_pol * preds["eco_pol"][i]
            total_w += w_pol
        if "eco_env" in preds:
            pred += w_env * preds["eco_env"][i]
            total_w += w_env
        if "eco_human" in preds:
            pred += w_hum * preds["eco_human"][i]
            total_w += w_hum
        
        if total_w > 0:
            gated_pred[i] = pred / total_w
        else:
            if "eco_only" in preds:
                gated_pred[i] = preds["eco_only"][i]
            elif "full_quad" in preds:
                gated_pred[i] = preds["full_quad"][i]
    
    return gated_pred


# ─────────────────────────── Main LGCF Pipeline ───────────────────────────

def run_lgcf_experiment(api_key: str | None = None,
                         llm_model: str = "deepseek-chat",
                         skip_llm: bool = False):
    """Run the full LGCF experiment with all ablation configurations."""
    t0 = time.time()
    
    log.info("=" * 70)
    log.info("  LLM-GATED CROSS-DOMAIN FORECASTING (LGCF)")
    log.info("  Full Ablation Experiment")
    log.info("=" * 70)
    
    df = pd.read_parquet(QUAD_PANEL)
    log.info(f"Loaded quad panel: {df.shape}")
    
    sectors = classify_features(df.columns.tolist())
    configs = build_domain_configs(sectors)
    
    for name, cols in configs.items():
        log.info(f"  {name}: {len(cols)} features")
    
    all_results = []
    
    for h in HORIZONS:
        log.info(f"\n{'#'*70}")
        log.info(f"  HORIZON h={h}")
        log.info(f"{'#'*70}")
        
        df_work = df.sort_values(["iso3", "year"]).reset_index(drop=True)
        target = make_target(df_work, h)
        df_work["target"] = target
        
        valid_mask = df_work["target"].notna() & np.isfinite(df_work["target"])
        df_valid = df_work[valid_mask].reset_index(drop=True)
        
        q01 = df_valid["target"].quantile(0.01)
        q99 = df_valid["target"].quantile(0.99)
        df_valid = df_valid[(df_valid["target"] >= q01) & (df_valid["target"] <= q99)].reset_index(drop=True)
        
        years = df_valid["year"].values
        y = df_valid["target"].values.astype(np.float32)
        
        shift = max(0, h - 5)
        anchor_end = 2022 - shift
        
        for fold in range(N_FOLDS):
            fold_test_end = anchor_end - fold * TEST_WINDOW
            fold_test_start = fold_test_end - TEST_WINDOW + 1
            fold_train_end = fold_test_start - 1
            
            if fold_train_end < 1970:
                continue
            
            train_mask = years <= fold_train_end
            test_mask = (years >= fold_test_start) & (years <= fold_test_end)
            
            n_train = train_mask.sum()
            n_test = test_mask.sum()
            
            if n_train < MIN_TRAIN_ROWS or n_test < 10:
                continue
            
            log.info(f"\n  Fold {fold}: train<={fold_train_end} ({n_train}), "
                     f"test {fold_test_start}-{fold_test_end} ({n_test})")
            
            y_test = y[test_mask]
            test_iso3 = df_valid.loc[test_mask, "iso3"].values
            test_years = df_valid.loc[test_mask, "year"].values
            
            preds = {}
            for cfg_name, cfg_cols in configs.items():
                avail_cols = [c for c in cfg_cols if c in df_valid.columns and c != "target"]
                if len(avail_cols) < 3:
                    continue
                
                X_all = df_valid[avail_cols].astype(np.float32).replace([np.inf, -np.inf], np.nan).values
                pred = train_predict_models(X_all, y, train_mask, test_mask, seed=fold)
                preds[cfg_name] = pred
                
                mae = float(np.mean(np.abs(y_test - pred)))
                log.info(f"    {cfg_name:15s}: MAE={mae:.5f}")
            
            if len(preds) < 2:
                continue
            
            eco_only_errors = np.abs(y_test - preds.get("eco_only", np.zeros(n_test)))
            
            # A: Eco-Only
            eco_mae = float(np.mean(eco_only_errors))
            all_results.append({
                "horizon": h, "fold": fold, "config": "A_EcoOnly",
                "mae": eco_mae, "n_test": n_test,
            })
            
            # B: Uniform Mixture
            pred_uniform = np.mean(np.column_stack(list(preds.values())), axis=1)
            uniform_errors = np.abs(y_test - pred_uniform)
            all_results.append({
                "horizon": h, "fold": fold, "config": "B_Uniform",
                "mae": float(np.mean(uniform_errors)), "n_test": n_test,
            })
            
            # C: Random Gating
            random_gates = [random_gate(seed=i + fold * 1000) for i in range(n_test)]
            pred_random = apply_gate_to_predictions(preds, random_gates)
            random_errors = np.abs(y_test - pred_random)
            all_results.append({
                "horizon": h, "fold": fold, "config": "C_RandomGate",
                "mae": float(np.mean(random_errors)), "n_test": n_test,
            })
            
            # D: Heuristic Gating
            heuristic_gates = []
            for i in range(n_test):
                ctx = build_context(df_valid, test_iso3[i], int(test_years[i]))
                hg = heuristic_gate(ctx, h)
                heuristic_gates.append(hg)
            pred_heuristic = apply_gate_to_predictions(preds, heuristic_gates)
            heuristic_errors = np.abs(y_test - pred_heuristic)
            all_results.append({
                "horizon": h, "fold": fold, "config": "D_HeuristicGate",
                "mae": float(np.mean(heuristic_errors)), "n_test": n_test,
            })
            
            # E: LLM-Gated
            if not skip_llm:
                llm_gates = []
                for i in range(n_test):
                    gate = compute_gate_weights(
                        df_valid, test_iso3[i], int(test_years[i]), h,
                        api_key=api_key, llm_model=llm_model, use_cache=True,
                    )
                    llm_gates.append({
                        "economy": gate.economy,
                        "politics": gate.politics,
                        "environment": gate.environment,
                        "human_society": gate.human_society,
                    })
                pred_llm = apply_gate_to_predictions(preds, llm_gates)
                llm_errors = np.abs(y_test - pred_llm)
                all_results.append({
                    "horizon": h, "fold": fold, "config": "E_LLMGate",
                    "mae": float(np.mean(llm_errors)), "n_test": n_test,
                })
                dm_stat, dm_p = diebold_mariano(eco_only_errors, llm_errors)
                log.info(f"    LLM Gate vs Eco-Only: DM={dm_stat:.3f}, p={dm_p:.4f}")
            
            # F: Oracle Gating
            error_matrix = np.column_stack([np.abs(y_test - preds[cfg]) for cfg in preds])
            pred_matrix = np.column_stack([preds[cfg] for cfg in preds])
            oracle_idx = np.argmin(error_matrix, axis=1)
            pred_oracle = np.array([pred_matrix[i, oracle_idx[i]] for i in range(n_test)])
            oracle_errors = np.abs(y_test - pred_oracle)
            all_results.append({
                "horizon": h, "fold": fold, "config": "F_OracleGate",
                "mae": float(np.mean(oracle_errors)), "n_test": n_test,
            })
            
            log.info(f"\n    FOLD {fold} SUMMARY:")
            for r in all_results[-6 if not skip_llm else -5:]:
                if r["horizon"] == h and r["fold"] == fold:
                    log.info(f"      {r['config']:20s}: MAE={r['mae']:.5f}")
    
    df_results = pd.DataFrame(all_results)
    
    summary = df_results.groupby(["horizon", "config"]).agg(
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        n_folds=("fold", "count"),
    ).reset_index()
    
    dm_rows = []
    for h in HORIZONS:
        eco_data = df_results[(df_results["horizon"] == h) & (df_results["config"] == "A_EcoOnly")]
        if eco_data.empty:
            continue
        eco_mae_avg = eco_data["mae"].mean()
        
        for cfg in ["B_Uniform", "C_RandomGate", "D_HeuristicGate", "E_LLMGate", "F_OracleGate"]:
            cfg_data = df_results[(df_results["horizon"] == h) & (df_results["config"] == cfg)]
            if cfg_data.empty:
                continue
            cfg_mae_avg = cfg_data["mae"].mean()
            improvement = (eco_mae_avg - cfg_mae_avg) / eco_mae_avg * 100
            
            dm_rows.append({
                "horizon": h,
                "config": cfg,
                "eco_mae": round(eco_mae_avg, 5),
                "config_mae": round(cfg_mae_avg, 5),
                "improvement_pct": round(improvement, 2),
            })
    
    df_dm = pd.DataFrame(dm_rows)
    
    df_results.to_csv(OUT_DIR / "lgcf_full_results.csv", index=False)
    summary.to_csv(OUT_DIR / "lgcf_summary.csv", index=False)
    df_dm.to_csv(OUT_DIR / "lgcf_dm_comparison.csv", index=False)
    
    elapsed = time.time() - t0
    
    print(f"\n{'='*80}")
    print(f"  LGCF EXPERIMENT RESULTS -- {elapsed:.0f}s")
    print(f"{'='*80}")
    print(f"\n  HEADLINE TABLE:")
    print(f"  {'Config':<20} {'h=1 MAE':<12} {'h=3 MAE':<12} {'h=5 MAE':<12}")
    print(f"  {'-'*56}")
    
    for cfg in ["A_EcoOnly", "B_Uniform", "C_RandomGate", "D_HeuristicGate", "E_LLMGate", "F_OracleGate"]:
        row_str = f"  {cfg:<20}"
        for h in HORIZONS:
            s = summary[(summary["horizon"] == h) & (summary["config"] == cfg)]
            if not s.empty:
                row_str += f" {s.iloc[0]['mae_mean']:.5f}     "
            else:
                row_str += " N/A          "
        print(row_str)
    
    if not df_dm.empty:
        print(f"\n  IMPROVEMENT vs ECO-ONLY:")
        print(df_dm.to_string(index=False))
    
    print(f"\n  Results saved to: {OUT_DIR}")
    
    return df_results, summary, df_dm


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip LLM gate (run only oracle/heuristic/random ablations)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="DeepSeek API key")
    args = parser.parse_args()
    
    run_lgcf_experiment(
        api_key=args.api_key,
        skip_llm=args.skip_llm,
    )
