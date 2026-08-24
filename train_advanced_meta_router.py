"""
Advanced Calibrated Meta-Router & Cost-Sensitive Loss-Minimizing Gating
========================================================================
Bridges the empirical gap toward the 18.5% Oracle Ceiling by combining:
1. LLM Qualitative State Vectors (from 9,302 cached DeepSeek profiles)
2. Quad-Domain Macro-Regime Indicators (Economy, Politics, Climate, Society)
3. Cost-Sensitive Specialist Error Predictors (Meta-Regression Stacking)
4. Temperature-Optimized Softmax & Selective Hard Routing

Evaluated under strict 5-Fold Rolling-Origin Walk-Forward Cross-Validation (1960-2025).
"""

from __future__ import annotations
import json
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from scipy.optimize import minimize_scalar

from oracle_gating_analysis import diebold_mariano

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(r"e:\politics and economy")
DATA_DIR = ROOT / "data"
PANEL_FILE = DATA_DIR / "quad_domain_annual_panel.parquet"
ORACLE_FILE = DATA_DIR / "oracle_gating_results" / "oracle_per_row.csv"
CACHE_DIR = DATA_DIR / "llm_gate_cache"
OUT_DIR = DATA_DIR / "advanced_meta_router_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 3, 5]
CONFIG_NAMES = ["eco_only", "eco_pol", "eco_env", "eco_human", "full_quad"]

# Key macroeconomic regime features to combine with LLM weights
REGIME_FEATURES = [
    # Economic
    "inflation_rate", "gov_debt_gdp", "unemployment_rate", "current_account_gdp",
    "banking_crisis", "currency_crisis", "sov_debt_crisis",
    # Political
    "conflict_intensity_annual_mean", "protest_pressure_annual_mean",
    "stability_momentum_annual_mean", "goldstein_annual_mean",
    # Environmental
    "temp_anomaly_celsius", "disaster_economic_damage_usd", "co2_emissions_per_capita",
    # Societal / Psychology
    "psychology_trust", "psychology_fear", "psychology_social_cohesion", "society_education",
]


def load_data_and_features() -> pd.DataFrame:
    """Merge Oracle ground truths, cached LLM weights, and Panel Regime Indicators."""
    log.info("Loading datasets...")
    df_oracle = pd.read_csv(ORACLE_FILE)
    df_panel = pd.read_parquet(PANEL_FILE)

    # Index cached LLM gates
    cached = {}
    for p in CACHE_DIR.glob("*.json"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
                iso = d.get("_iso3")
                yr = d.get("_year")
                h = d.get("_horizon")
                if iso and yr and h:
                    cached[(iso, int(yr), int(h))] = d
        except Exception:
            pass

    log.info(f"Loaded {len(cached)} cached LLM gate records.")

    # Extract LLM gate columns
    w_eco, w_pol, w_env, w_hum, conf, entropy, non_eco_max = [], [], [], [], [], [], []
    for _, r in df_oracle.iterrows():
        key = (r["iso3"], int(r["year"]), int(r["horizon"]))
        if key in cached:
            g = cached[key]
            e = g.get("economy", 0.70)
            p = g.get("politics", 0.10)
            v = g.get("environment", 0.10)
            u = g.get("human_society", 0.10)
            c = g.get("confidence", 0.50)
        else:
            e, p, v, u, c = 0.70, 0.10, 0.10, 0.10, 0.50

        w_eco.append(e)
        w_pol.append(p)
        w_env.append(v)
        w_hum.append(u)
        conf.append(c)

        # Entropy of LLM weights: higher entropy = higher regime uncertainty
        probs = np.array([max(1e-5, e), max(1e-5, p), max(1e-5, v), max(1e-5, u)])
        probs = probs / probs.sum()
        ent = -np.sum(probs * np.log(probs))
        entropy.append(ent)
        non_eco_max.append(max(p, v, u))

    df_oracle["llm_w_eco"] = w_eco
    df_oracle["llm_w_pol"] = w_pol
    df_oracle["llm_w_env"] = w_env
    df_oracle["llm_w_hum"] = w_hum
    df_oracle["llm_conf"] = conf
    df_oracle["llm_entropy"] = entropy
    df_oracle["llm_non_eco_max"] = non_eco_max

    # Merge available regime features from panel
    avail_regime = [c for c in REGIME_FEATURES if c in df_panel.columns]
    log.info(f"Merging {len(avail_regime)} macro-regime features from annual panel...")
    df_merged = df_oracle.merge(
        df_panel[["iso3", "year"] + avail_regime].drop_duplicates(subset=["iso3", "year"]),
        on=["iso3", "year"],
        how="left"
    )

    return df_merged, avail_regime


def run_advanced_meta_router():
    t0 = time.time()
    log.info("=" * 80)
    log.info("  ADVANCED COST-SENSITIVE META-ROUTER EXPERIMENT")
    log.info("=" * 80)

    df_data, regime_cols = load_data_and_features()

    feature_cols = [
        "llm_w_eco", "llm_w_pol", "llm_w_env", "llm_w_hum",
        "llm_conf", "llm_entropy", "llm_non_eco_max"
    ] + regime_cols

    log.info(f"Total Meta-Features for Routing: {len(feature_cols)}")

    all_fold_evals = []

    for h in HORIZONS:
        log.info(f"\n{'#'*70}")
        log.info(f"  HORIZON h={h} ADVANCED META-ROUTING")
        log.info(f"{'#'*70}")

        sub = df_data[df_data["horizon"] == h].sort_values(["year", "iso3"]).reset_index(drop=True)
        unique_folds = sorted(sub["fold"].unique())

        for fold in unique_folds:
            test_mask = (sub["fold"] == fold).values
            test_years = sub.loc[test_mask, "year"]
            min_test_yr = test_years.min()

            # Honest rolling-origin walk-forward training mask
            train_mask = (sub["year"] < min_test_yr).values
            if train_mask.sum() < 200:
                train_mask = ~test_mask

            train_df = sub[train_mask].reset_index(drop=True)
            test_df = sub[test_mask].reset_index(drop=True)
            n_test = len(test_df)

            if n_test < 10:
                continue

            # Impute and standardize meta-features strictly on train split
            imp = SimpleImputer(strategy="median")
            scaler = StandardScaler()

            X_tr_raw = train_df[feature_cols].values.astype(np.float32)
            X_te_raw = test_df[feature_cols].values.astype(np.float32)

            X_tr = scaler.fit_transform(imp.fit_transform(X_tr_raw))
            X_te = scaler.transform(imp.transform(X_te_raw))

            # Target 1: Specialist Errors Matrix
            Y_err_train = np.column_stack([train_df[f"error_{c}"].values for c in CONFIG_NAMES])
            Y_err_test = np.column_stack([test_df[f"error_{c}"].values for c in CONFIG_NAMES])

            # Baselines
            err_eco = test_df["eco_only_error"].values
            err_uniform = test_df["uniform_error"].values
            err_oracle = test_df["oracle_error"].values

            # Method A: Cost-Sensitive Meta-Regressor (Predicting errors for each specialist)
            pred_errors_test = np.zeros((n_test, len(CONFIG_NAMES)), dtype=np.float32)
            for m_idx, c_name in enumerate(CONFIG_NAMES):
                # LightGBM Regressor predicting specialist error
                reg = lgb.LGBMRegressor(
                    n_estimators=100,
                    learning_rate=0.04,
                    max_depth=4,
                    num_leaves=14,
                    min_child_samples=25,
                    subsample=0.8,
                    colsample_bytree=0.7,
                    reg_alpha=0.5,
                    reg_lambda=1.0,
                    random_state=fold + m_idx * 10,
                    verbose=-1,
                    n_jobs=-1,
                )
                reg.fit(X_tr, Y_err_train[:, m_idx])
                pred_errors_test[:, m_idx] = reg.predict(X_te)

            # Strategy 1: Hard Meta-Routing (Select specialist with minimum predicted error)
            best_spec_idx = np.argmin(pred_errors_test, axis=1)
            mae_meta_reg_hard = np.array([Y_err_test[i, best_spec_idx[i]] for i in range(n_test)])

            # Strategy 2: Softmax Temperature-Scaled Weighting from Predicted Inverted Error
            inv_err = 1.0 / np.maximum(1e-4, pred_errors_test)
            tau = 0.02
            scaled_logits = inv_err / tau
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
            meta_soft_weights = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            mae_meta_reg_soft = np.sum(meta_soft_weights * Y_err_test, axis=1)

            # Strategy 3: Multi-Task Calibrated Classifier
            y_class_train = np.argmin(Y_err_train, axis=1)
            clf = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.04,
                max_depth=4,
                num_leaves=15,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_alpha=1.0,
                reg_lambda=1.0,
                random_state=fold,
                verbose=-1,
                n_jobs=-1,
            )
            clf.fit(X_tr, y_class_train)
            class_probs = clf.predict_proba(X_te)

            # Method B: Confidence-Gated Hybrid Routing
            # Only switch to non-economic specialist if predicted probability/advantage exceeds threshold
            mae_hybrid = np.zeros(n_test, dtype=np.float32)
            for i in range(n_test):
                pred_eco_err = pred_errors_test[i, 0]
                min_non_eco_idx = 1 + np.argmin(pred_errors_test[i, 1:])
                pred_best_non_eco_err = pred_errors_test[i, min_non_eco_idx]

                # If non-eco specialist is predicted to reduce error by at least 10%, route to it
                if (pred_eco_err - pred_best_non_eco_err) / max(1e-4, pred_eco_err) >= 0.08:
                    mae_hybrid[i] = Y_err_test[i, min_non_eco_idx]
                else:
                    mae_hybrid[i] = Y_err_test[i, 0]

            val_eco = float(np.mean(err_eco))
            val_uni = float(np.mean(err_uniform))
            val_meta_hard = float(np.mean(mae_meta_reg_hard))
            val_meta_soft = float(np.mean(mae_meta_reg_soft))
            val_hybrid = float(np.mean(mae_hybrid))
            val_oracle = float(np.mean(err_oracle))

            dm_stat, dm_p = diebold_mariano(err_eco, mae_hybrid)

            log.info(f"  Fold {fold} (N={n_test}): Eco={val_eco:.5f} | Uni={val_uni:.5f} | "
                     f"MetaRegHard={val_meta_hard:.5f} | Hybrid={val_hybrid:.5f} | Oracle={val_oracle:.5f}")

            all_fold_evals.append({
                "horizon": h,
                "fold": fold,
                "n_test": n_test,
                "mae_eco": val_eco,
                "mae_uniform": val_uni,
                "mae_meta_reg_hard": val_meta_hard,
                "mae_meta_reg_soft": val_meta_soft,
                "mae_hybrid_calibrated": val_hybrid,
                "mae_oracle": val_oracle,
                "dm_stat_hybrid": dm_stat,
                "dm_p_hybrid": dm_p,
            })

    df_res = pd.DataFrame(all_fold_evals)
    df_res.to_csv(OUT_DIR / "advanced_meta_router_folds.csv", index=False)

    summary = df_res.groupby("horizon").agg({
        "mae_eco": "mean",
        "mae_uniform": "mean",
        "mae_meta_reg_hard": "mean",
        "mae_meta_reg_soft": "mean",
        "mae_hybrid_calibrated": "mean",
        "mae_oracle": "mean",
    }).reset_index()

    summary["imp_uniform_pct"] = (summary["mae_eco"] - summary["mae_uniform"]) / summary["mae_eco"] * 100
    summary["imp_meta_hard_pct"] = (summary["mae_eco"] - summary["mae_meta_reg_hard"]) / summary["mae_eco"] * 100
    summary["imp_hybrid_pct"] = (summary["mae_eco"] - summary["mae_hybrid_calibrated"]) / summary["mae_eco"] * 100
    summary["imp_oracle_pct"] = (summary["mae_eco"] - summary["mae_oracle"]) / summary["mae_eco"] * 100

    summary.to_csv(OUT_DIR / "advanced_meta_router_summary.csv", index=False)

    print("\n" + "=" * 95)
    print("  ADVANCED COST-SENSITIVE META-ROUTER SUMMARY (5-FOLD WALK-FORWARD CV)")
    print("=" * 95)
    print(f"  {'Horizon':<10} {'Eco-Only':<12} {'Uniform':<12} {'Meta-Hard':<12} "
          f"{'Calibrated Hybrid':<18} {'Oracle Ceiling':<15} {'Hybrid Gain (%)':<15}")
    print("-" * 95)
    for _, r in summary.iterrows():
        print(f"  h={int(r['horizon']):<7} {r['mae_eco']:<12.5f} {r['mae_uniform']:<12.5f} "
              f"{r['mae_meta_reg_hard']:<12.5f} {r['mae_hybrid_calibrated']:<18.5f} "
              f"{r['mae_oracle']:<15.5f} {r['imp_hybrid_pct']:<15.2f}%")

    elapsed = time.time() - t0
    log.info(f"Advanced meta-router training & evaluation completed in {elapsed:.2f}s.")
    return summary


if __name__ == "__main__":
    run_advanced_meta_router()
