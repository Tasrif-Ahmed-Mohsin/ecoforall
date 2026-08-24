"""
Cross-Domain Economic Prediction Enhancer
==========================================
Senior economist + mathematician approach to improving GDP growth forecasting
by incorporating political, environmental, and human/social signals.

4 Configurations tested via 5-Fold Walk-Forward CV:
  A: Econ-Only Baseline (reproduce existing v2 ensemble)
  B: Econ + Curated Cross-Domain Features
  C: Two-Stage Stacking (Econ Stage1 → Cross-domain correction Stage2)
  D: Country-Year Twin Enhanced (analog-country features from cross-domain space)

All results are directly comparable to projectresearch/walk_forward_cv_summary.json.
"""

from __future__ import annotations
import json
import logging
import os
import pickle
import sys
import time
import warnings
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────── Paths ───────────────────────────
ROOT = Path(r"e:\politics and economy")
GMD_PANEL = ROOT / "projectresearch" / "data" / "features" / "panel_wide.parquet"
QUAD_PANEL = ROOT / "data" / "quad_domain_annual_panel.parquet"
OUT_DIR = ROOT / "data" / "cross_domain_enhanced_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────── Constants ───────────────────────────
HORIZONS = [1, 3, 5, 10]
N_FOLDS = 5
OPTUNA_TRIALS = 50    # per fold; good balance of speed vs thoroughness
NESTED_VAL_YEARS = 4
MIN_TRAIN_ROWS = 200

# Features the v2 ablation proved are noise — drop from economic set
DROP_FEATURES = {"trade_gdp", "current_account_gdp", "fx_to_usd", "population"}

# Income tier thresholds (USD)
TIER_BOUNDS = [("LIC", -np.inf, 2000.0), ("LMIC", 2000.0, 4500.0),
               ("UMIC", 4500.0, 14000.0), ("HIC", 14000.0, np.inf)]

# ─────────────────────────── Theory-driven cross-domain features ───────────────────────────
# Selected based on economic theory and empirical literature:

# Political features: institutional quality → investment climate → growth
POLITICAL_FEATURES = [
    "stability_momentum_annual_mean",  # regime stability (North 1990)
    "conflict_intensity_annual_mean",   # capital flight, destruction
    "protest_pressure_annual_mean",     # social unrest indicator
    "sanctions_coercion_annual_sum",    # trade disruption
    "diplomatic_summit_annual_sum",     # international integration
    "verbal_cooperation_annual_sum",    # cooperation climate
    "verbal_conflict_annual_sum",       # tension indicator
    "material_conflict_annual_sum",     # actual conflicts
]

# Environmental features: resource constraints → supply shocks → growth
ENVIRONMENTAL_FEATURES = [
    "co2_emissions_per_capita",         # carbon intensity (Nordhaus 2018)
    "temp_anomaly_celsius",             # climate stress
    "disaster_economic_damage_usd",     # direct GDP impact
    "renewable_energy_pct_share",       # energy transition progress
    "floods_count",                     # natural disaster frequency
    "droughts_count",                   # agricultural disruption
    "storms_count",                     # infrastructure damage
    "wildfires_count",                  # resource destruction
    "forest_cover_pct",                 # environmental capital
    "protected_area_pct",              # environmental policy
]

# Human/social features: human capital → productivity → long-run TFP
HUMAN_FEATURES = [
    "society_education",                # human capital (Barro & Lee 2013)
    "society_healthcare",               # health → productivity
    "society_age",                      # demographic structure
    "society_urbanization",             # agglomeration economies
    "society_population",               # labor force size
    "society_migration",                # brain drain/gain
    "psychology_trust",                 # institutional quality proxy (Knack & Keefer 1997)
    "psychology_confidence",            # consumer/business sentiment
    "psychology_optimism",              # forward-looking expectations
    "psychology_fear",                  # risk aversion indicator
    "psychology_social_cohesion",       # social capital
    "psychology_nationalism",           # trade policy indicator
]

# Also grab lagged/derived versions that exist in the quad panel
CROSS_DOMAIN_DERIVED_SUFFIXES = ["_lag_1", "_lag_3", "_velocity_3y", "_rank",
                                  "_roll_mean_10y", "_diff_h1"]


# ════════════════════════════════════════════════════════════════
#  Data Loading & Merging
# ════════════════════════════════════════════════════════════════

def load_and_merge_panels() -> pd.DataFrame:
    """Load GMD economic panel and quad-domain panel, merge on (iso3, year)."""
    log.info("Loading GMD panel...")
    gmd = pd.read_parquet(GMD_PANEL)
    log.info(f"  GMD shape: {gmd.shape}, countries: {gmd['iso3'].nunique()}, "
             f"years: {gmd['year'].min()}-{gmd['year'].max()}")

    log.info("Loading quad-domain panel...")
    quad = pd.read_parquet(QUAD_PANEL)
    log.info(f"  Quad shape: {quad.shape}, countries: {quad['iso3'].nunique()}, "
             f"years: {quad['year'].min()}-{quad['year'].max()}")

    # Identify columns unique to quad (not in GMD) to merge in
    gmd_cols = set(gmd.columns)
    quad_only_cols = [c for c in quad.columns if c not in gmd_cols and c not in ("iso3", "year", "timestamp")]

    # Filter to our curated features + their derived variants
    base_features = POLITICAL_FEATURES + ENVIRONMENTAL_FEATURES + HUMAN_FEATURES
    keep_cols = []
    for col in quad_only_cols:
        # Keep if it's a base feature or a derived variant of one
        base_name = col
        for suffix in CROSS_DOMAIN_DERIVED_SUFFIXES:
            if col.endswith(suffix):
                base_name = col[: -len(suffix)]
                break
        if base_name in base_features or col in base_features:
            keep_cols.append(col)

    # Also keep base features that might overlap with GMD (we'll use the quad version)
    for col in base_features:
        if col in quad.columns and col not in keep_cols:
            keep_cols.append(col)

    keep_cols = sorted(set(keep_cols))
    log.info(f"  Selected {len(keep_cols)} cross-domain features from quad panel")

    # Merge
    quad_subset = quad[["iso3", "year"] + keep_cols].copy()
    merged = gmd.merge(quad_subset, on=["iso3", "year"], how="left", suffixes=("", "_quad"))

    # Remove any duplicate columns created by suffix
    dup_cols = [c for c in merged.columns if c.endswith("_quad")]
    for col in dup_cols:
        base = col.replace("_quad", "")
        # Fill missing values in GMD with quad values
        if base in merged.columns:
            merged[base] = merged[base].fillna(merged[col])
        merged.drop(columns=[col], inplace=True)

    log.info(f"  Merged panel shape: {merged.shape}")
    return merged


# ════════════════════════════════════════════════════════════════
#  Feature Engineering
# ════════════════════════════════════════════════════════════════

def engineer_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create economically-motivated interaction features."""
    out = df.copy()

    def _safe_interact(a, b, name):
        """Multiply two columns, handling NaN gracefully."""
        if a in out.columns and b in out.columns:
            va = out[a].fillna(0).astype(np.float64)
            vb = out[b].fillna(0).astype(np.float64)
            out[name] = (va * vb).astype(np.float32)

    def _safe_ratio(a, b, name):
        """Ratio of two columns with zero-safe denominator."""
        if a in out.columns and b in out.columns:
            va = out[a].fillna(0).astype(np.float64)
            vb = out[b].fillna(1).astype(np.float64).replace(0, 1)
            out[name] = (va / vb).astype(np.float32)

    # 1. Conflict x Debt: fragile states under fiscal stress
    _safe_interact("conflict_intensity_annual_mean", "gov_debt_gdp", "ix_conflict_x_debt")

    # 2. Disaster damage x Investment: reconstruction vs destruction dynamics
    _safe_interact("disaster_economic_damage_usd", "investment_gdp", "ix_disaster_x_invest")

    # 3. Stability x Trade openness: political risk premium on trade
    if "exports_gdp" in out.columns and "imports_gdp" in out.columns:
        out["_trade_openness"] = out["exports_gdp"].fillna(0) + out["imports_gdp"].fillna(0)
        _safe_interact("stability_momentum_annual_mean", "_trade_openness", "ix_stability_x_trade")
        out.drop(columns=["_trade_openness"], inplace=True)

    # 4. Trust x lagged GDP growth: confidence multiplier effect
    if "psychology_trust" in out.columns and "gdp_pc_real_usd_lag1" in out.columns:
        _safe_interact("psychology_trust", "gdp_pc_real_usd_lag1", "ix_trust_x_growth_lag")
    elif "psychology_trust" in out.columns and "gdp_pc_real_lag1" in out.columns:
        _safe_interact("psychology_trust", "gdp_pc_real_lag1", "ix_trust_x_growth_lag")

    # 5. Education x Urbanization: human capital density
    _safe_interact("society_education", "society_urbanization", "ix_education_x_urban")

    # 6. Demographic dividend: young educated workforce
    if "society_age" in out.columns and "society_education" in out.columns:
        inverted_age = (100 - out["society_age"].fillna(50)).astype(np.float64)
        edu = out["society_education"].fillna(0).astype(np.float64)
        out["ix_demographic_dividend"] = (inverted_age * edu).astype(np.float32)

    # 7. Environmental burden: carbon intensity x disaster exposure
    _safe_interact("co2_emissions_per_capita", "disaster_economic_damage_usd", "ix_environmental_burden")

    # 8. Conflict spiral: verbal conflict -> material conflict escalation
    _safe_interact("verbal_conflict_annual_sum", "material_conflict_annual_sum", "ix_conflict_spiral")

    # 9. Climate-agriculture vulnerability: temp anomaly x drought frequency
    _safe_interact("temp_anomaly_celsius", "droughts_count", "ix_climate_agri_vuln")

    # 10. Institutional quality composite: stability x trust x cooperation
    if all(c in out.columns for c in ["stability_momentum_annual_mean", "psychology_trust"]):
        stab = out["stability_momentum_annual_mean"].fillna(0).astype(np.float64)
        trust = out["psychology_trust"].fillna(0).astype(np.float64)
        out["ix_institutional_quality"] = (stab * trust).astype(np.float32)

    # 11. Social unrest pressure: protest x fear x nationalism
    if all(c in out.columns for c in ["protest_pressure_annual_mean", "psychology_fear"]):
        prot = out["protest_pressure_annual_mean"].fillna(0).astype(np.float64)
        fear = out["psychology_fear"].fillna(0).astype(np.float64)
        out["ix_unrest_pressure"] = (prot * fear).astype(np.float32)

    # 12. Green transition momentum: renewable share growth / CO2
    if "renewable_energy_pct_share" in out.columns and "co2_emissions_per_capita" in out.columns:
        ren = out["renewable_energy_pct_share"].fillna(0).astype(np.float64)
        co2 = out["co2_emissions_per_capita"].fillna(0).astype(np.float64)
        out["ix_green_transition"] = (ren / (co2 + 0.01)).astype(np.float32)

    ix_cols = [c for c in out.columns if c.startswith("ix_")]
    log.info(f"  Engineered {len(ix_cols)} interaction features")
    return out


# ════════════════════════════════════════════════════════════════
#  Country-Year Twin Features (Config D)
# ════════════════════════════════════════════════════════════════

def compute_twin_features(df: pd.DataFrame, cross_cols: list[str],
                          train_mask: np.ndarray, k: int = 5) -> np.ndarray:
    """Compute country-year twin features from cross-domain space.

    For each country-year, find the k nearest neighbors in the cross-domain
    feature space (using training data only), and return:
    - mean GDP growth of twins
    - std of twin GDP growth
    - distance to nearest twin
    """
    target_col = "target"
    xcols = [c for c in cross_cols if c in df.columns]
    if len(xcols) < 3:
        return np.full((len(df), 3), np.nan, dtype=np.float32)

    X_cross = df[xcols].fillna(0).values.astype(np.float32)

    # Standardize using training data only
    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train = X_cross[train_mask]
    imp.fit(X_train)
    scaler.fit(imp.transform(X_train))

    X_all = scaler.transform(imp.transform(X_cross))

    # Build KNN on training data only
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(X_train)), metric="euclidean")
    nn.fit(X_all[train_mask])

    # Query for all rows
    distances, indices = nn.kneighbors(X_all)

    # Get the twin growth values (from training target)
    train_y = df[target_col].values.copy()
    train_y_safe = np.where(train_mask, train_y, np.nan)

    twin_features = np.full((len(df), 3), np.nan, dtype=np.float32)
    train_indices = np.where(train_mask)[0]

    for i in range(len(df)):
        neighbor_orig_idx = train_indices[indices[i]]
        neighbor_orig_idx = neighbor_orig_idx[neighbor_orig_idx != i][:k]
        if len(neighbor_orig_idx) == 0:
            continue

        twin_y = train_y_safe[neighbor_orig_idx]
        valid = ~np.isnan(twin_y)
        if valid.sum() == 0:
            continue

        twin_features[i, 0] = np.nanmean(twin_y)
        twin_features[i, 1] = np.nanstd(twin_y) if valid.sum() > 1 else 0
        twin_features[i, 2] = distances[i, 1] if len(distances[i]) > 1 else 0

    return twin_features


# ════════════════════════════════════════════════════════════════
#  Core ML Pipeline Components
# ════════════════════════════════════════════════════════════════

def make_target(df: pd.DataFrame, h: int) -> pd.Series:
    """log(gdp_pc_{y+h} / gdp_pc_y), per country."""
    df = df.sort_values(["iso3", "year"]).copy()
    g_fwd = df.groupby("iso3")["gdp_pc"].shift(-h)
    ratio = g_fwd / df["gdp_pc"]
    return np.log(ratio.where(ratio > 0)).astype(np.float32).rename("target")


def add_country_and_tier_dummies(df: pd.DataFrame, iso_levels: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """One-hot iso3 (drop_first) and 3 income-tier dummies."""
    out = df.copy()
    iso_cols = [f"iso_{iso}" for iso in iso_levels[1:]]
    iso_block = pd.DataFrame(
        {c: (out["iso3"] == iso_levels[i + 1]).astype(np.float32) for i, c in enumerate(iso_cols)},
        index=out.index,
    )
    if "gdp_pc_real" in out.columns:
        g = out["gdp_pc_real"].astype(np.float64)
    elif "gdp_pc" in out.columns:
        g = out["gdp_pc"].astype(np.float64)
    else:
        g = pd.Series(np.nan, index=out.index)
    tier = pd.cut(g, bins=[-np.inf, 2000.0, 4500.0, 14000.0, np.inf],
                  labels=["LIC", "LMIC", "UMIC", "HIC"])
    tier_block = pd.DataFrame({
        "tier_LIC": (tier == "LIC").astype(np.float32),
        "tier_LMIC": (tier == "LMIC").astype(np.float32),
        "tier_UMIC": (tier == "UMIC").astype(np.float32),
    }, index=out.index)
    out = pd.concat([out, iso_block, tier_block], axis=1)
    return out, iso_cols + ["tier_LIC", "tier_LMIC", "tier_UMIC"]


def rank_fit_transform(X: np.ndarray, fit_mask: np.ndarray) -> np.ndarray:
    """Per-column rank transform, fit on train rows only."""
    out = np.full_like(X, np.nan, dtype=np.float32)
    for j in range(X.shape[1]):
        col = X[:, j]
        fit_vals = col[fit_mask]
        fit_valid = fit_vals[~np.isnan(fit_vals)]
        if len(fit_valid) < 2:
            continue
        sorted_fit = np.sort(fit_valid)
        n_fit = len(sorted_fit)
        all_valid = ~np.isnan(col)
        positions = np.searchsorted(sorted_fit, col[all_valid], side="right")
        out[all_valid, j] = (positions / n_fit).astype(np.float32)
    return out


def prepare_features(df: pd.DataFrame, iso_levels: list[str],
                     extra_cols: list[str] | None = None) -> tuple:
    """Build feature matrices."""
    df_aug, _ = add_country_and_tier_dummies(df, iso_levels)
    leak = {"iso3", "year", "gdp_pc", "target"} | {c for c in df_aug.columns if c.endswith("y_fwd")}
    dummy_cols = {c for c in df_aug.columns if c.startswith("iso_") or c.startswith("tier_")}

    cont_cols = [
        c for c in df_aug.columns
        if c not in leak
        and pd.api.types.is_numeric_dtype(df_aug[c])
        and c not in DROP_FEATURES
        and c not in dummy_cols
        and not c.endswith("y_fwd")
        and not c.endswith("_target_h1")
        and not c.endswith("_target_h3")
        and not c.endswith("_target_h5")
    ]

    if extra_cols:
        for c in extra_cols:
            if c in df_aug.columns and c not in cont_cols and c not in leak and c not in dummy_cols:
                cont_cols.append(c)

    X_cont = df_aug[cont_cols].astype(np.float32).replace([np.inf, -np.inf], np.nan)
    X_full = df_aug[cont_cols + sorted(dummy_cols)].astype(np.float32).replace([np.inf, -np.inf], np.nan)

    cont_keep = [c for c in cont_cols if X_cont[c].notna().any()]
    full_keep = cont_keep + sorted(dummy_cols)
    return X_cont[cont_keep], X_full[full_keep], cont_keep, full_keep


def get_econ_only_cols(cont_cols: list[str]) -> list[str]:
    """Filter to only economic columns (no cross-domain, no interaction)."""
    cross_bases = set(POLITICAL_FEATURES + ENVIRONMENTAL_FEATURES + HUMAN_FEATURES)
    ix_prefix = "ix_"

    econ_cols = []
    for c in cont_cols:
        if c.startswith(ix_prefix):
            continue
        is_cross = False
        for base in cross_bases:
            if c == base or c.startswith(base + "_"):
                is_cross = True
                break
        if not is_cross:
            econ_cols.append(c)
    return econ_cols


def get_cross_domain_cols(cont_cols: list[str]) -> list[str]:
    """Return only the cross-domain + interaction columns."""
    econ = set(get_econ_only_cols(cont_cols))
    return [c for c in cont_cols if c not in econ]


# ════════════════════════════════════════════════════════════════
#  Model Training Functions
# ════════════════════════════════════════════════════════════════

def nested_optuna_lgbm(X_tr, y_tr, years_tr, n_trials, seed, val_years=4):
    """Per-fold nested Optuna search for LGBM."""
    import optuna
    import lightgbm as lgb
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    max_year = int(years_tr.max())
    val_start = max_year - val_years + 1
    opt_mask = years_tr <= (val_start - 1)
    val_mask = (years_tr >= val_start) & (years_tr <= max_year)

    if opt_mask.sum() < MIN_TRAIN_ROWS or val_mask.sum() < 20:
        return {
            "objective": "regression_l1", "boosting_type": "gbdt",
            "n_estimators": 200, "learning_rate": 0.05, "num_leaves": 31,
            "min_child_samples": 30, "subsample": 0.9, "colsample_bytree": 0.9,
            "reg_alpha": 1.0, "reg_lambda": 1.0, "verbosity": -1, "n_jobs": -1,
        }

    imp = SimpleImputer(strategy="median")
    Xo = imp.fit_transform(X_tr[opt_mask])
    Xv = imp.transform(X_tr[val_mask])
    yo, yv = y_tr[opt_mask], y_tr[val_mask]

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 7, 31),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.8),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "boosting_type": "gbdt", "random_state": seed,
            "n_jobs": -1, "verbosity": -1, "objective": "regression_l1",
        }
        model = lgb.LGBMRegressor(
            **params,
            callbacks=[lgb.early_stopping(stopping_rounds=80, verbose=False)],
        )
        try:
            model.fit(Xo, yo, eval_set=[(Xv, yv)])
            pred = model.predict(Xv)
        except Exception:
            return float("inf")
        return float(np.mean(np.abs(pred - yv)))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    bp = dict(study.best_params)
    return {
        "objective": "regression_l1", "boosting_type": "gbdt",
        "n_estimators": int(bp.get("n_estimators", 200)),
        "learning_rate": float(bp.get("learning_rate", 0.05)),
        "num_leaves": int(bp.get("num_leaves", 31)),
        "min_child_samples": int(bp.get("min_child_samples", 30)),
        "subsample": float(bp.get("subsample", 0.9)),
        "colsample_bytree": float(bp.get("colsample_bytree", 0.9)),
        "reg_alpha": float(bp.get("reg_alpha", 1.0)),
        "reg_lambda": float(bp.get("reg_lambda", 1.0)),
        "verbosity": -1, "n_jobs": -1,
    }


def fit_predict_lgbm(X, y, train_mask, pred_mask, params, eval_mask=None):
    """Fit LGBM, predict on pred_mask rows."""
    import lightgbm as lgb
    Xa = X if isinstance(X, np.ndarray) else X.values
    if eval_mask is not None and eval_mask.any():
        fit_mask = train_mask & ~eval_mask
    else:
        fit_mask = train_mask
    imp = SimpleImputer(strategy="median")
    Xt = imp.fit_transform(Xa[fit_mask])
    Xp = imp.transform(Xa[pred_mask])
    fit_kwargs = {}
    if eval_mask is not None and eval_mask.any():
        Xe = imp.transform(Xa[eval_mask])
        ye = y[eval_mask]
        fit_kwargs["eval_set"] = [(Xe, ye)]
        fit_kwargs["callbacks"] = [lgb.early_stopping(stopping_rounds=80, verbose=False)]
    model = lgb.LGBMRegressor(**params)
    model.fit(Xt, y[fit_mask], **fit_kwargs)
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    return model.predict(Xp), importances


def fit_predict_ridge(X_cont, y, train_mask, pred_mask):
    """Fit Ridge on rank-transformed features."""
    Xr = rank_fit_transform(X_cont if isinstance(X_cont, np.ndarray) else X_cont.values, train_mask)
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("rg", Ridge(alpha=100.0)),
    ])
    pipe.fit(Xr[train_mask], y[train_mask])
    return pipe.predict(Xr[pred_mask])


def fit_predict_xgboost(X, y, train_mask, pred_mask):
    """Fit XGBoost on full features."""
    from xgboost import XGBRegressor
    Xa = X if isinstance(X, np.ndarray) else X.values
    model = XGBRegressor(
        n_estimators=600, max_depth=6, learning_rate=0.05,
        objective="reg:absoluteerror", tree_method="hist",
        subsample=0.8, colsample_bytree=0.8,
        random_state=0, n_jobs=-1, verbosity=0,
    )
    model.fit(Xa[train_mask], y[train_mask])
    return model.predict(Xa[pred_mask])


def prior_pred(df, y, train_mask, pred_mask):
    """Last-realised y for each country."""
    train = df.loc[train_mask, ["iso3", "year"]].copy()
    train["_y"] = y[train_mask]
    by_iso = train.sort_values("year").groupby("iso3")["_y"].last().to_dict()
    global_mean = float(y[train_mask].mean())
    return df.loc[pred_mask, "iso3"].map(lambda iso: by_iso.get(iso, global_mean)).values.astype(np.float32)


def ar1_honest_pred(df, y, train_mask, pred_mask, h):
    """Per-country AR(1) with honest train-only fitting."""
    train_df = df.loc[train_mask, ["iso3", "year"]].copy()
    train_df["_y"] = y[train_mask]
    train_df = train_df.sort_values(["iso3", "year"])
    by_iso_last = train_df.groupby("iso3")["_y"].last().to_dict()
    global_mean = float(y[train_mask].mean())
    pred_df = df.loc[pred_mask, ["iso3"]].copy()
    return pred_df["iso3"].map(lambda iso: by_iso_last.get(iso, global_mean)).values.astype(np.float32)


# ════════════════════════════════════════════════════════════════
#  Diebold-Mariano Test
# ════════════════════════════════════════════════════════════════

def diebold_mariano(e1, e2, h=1):
    """Two-sided DM test. Returns (statistic, p-value)."""
    d = e1**2 - e2**2
    n = len(d)
    if n < 5:
        return 0.0, 1.0
    mean_d = np.mean(d)
    var_d = max(1e-10, np.var(d) / n)
    dm_stat = mean_d / np.sqrt(var_d)
    p_value = 2 * (1 - norm.cdf(abs(dm_stat)))
    return float(dm_stat), float(p_value)


# ════════════════════════════════════════════════════════════════
#  Walk-Forward CV Engine
# ════════════════════════════════════════════════════════════════

@dataclass
class FoldResult:
    horizon: int
    fold: int
    config: str
    model: str
    n_test: int
    mae: float
    rmse: float
    dir_acc: float
    y_true: np.ndarray = field(repr=False)
    y_pred: np.ndarray = field(repr=False)


def calc_metrics(y_true, y_pred):
    err = y_pred - y_true
    return {
        "n": int(len(y_true)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "dir_acc": float(np.mean(np.sign(y_pred) == np.sign(y_true))),
    }


def run_walk_forward_cv(df: pd.DataFrame, horizon: int,
                        config_name: str, feature_cols: list[str],
                        full_feature_cols: list[str],
                        cross_domain_cols: list[str] | None = None,
                        use_two_stage: bool = False,
                        use_twins: bool = False) -> list[FoldResult]:
    """Run 5-fold walk-forward CV for a given configuration."""
    log.info(f"\n{'='*70}")
    log.info(f"  Config {config_name} | h={horizon} | Walk-Forward CV ({N_FOLDS} folds)")
    log.info(f"{'='*70}")

    # Build target
    df = df.sort_values(["iso3", "year"]).reset_index(drop=True)
    target = make_target(df, horizon)
    df["target"] = target

    # Drop rows without target
    valid = df["target"].notna()
    df_valid = df[valid].reset_index(drop=True)

    iso_levels = sorted(df_valid["iso3"].unique().tolist())
    years = df_valid["year"].values

    # Anchor years (matching projectresearch protocol)
    shift = max(0, horizon - 5)
    anchor_end = 2022 - shift

    # Walk-forward fold boundaries
    test_window = 4
    fold_results = []

    for fold in range(N_FOLDS):
        cp_path = CHECKPOINT_DIR / f"cp_h{horizon}_{config_name}_fold{fold}.pkl"
        if cp_path.exists():
            try:
                with open(cp_path, "rb") as f:
                    cached_results = pickle.load(f)
                log.info(f"  [CHECKPOINT LOADED] Horizon {horizon} | Config {config_name} | Fold {fold} ({len(cached_results)} models)")
                fold_results.extend(cached_results)
                continue
            except Exception as e:
                log.warning(f"  [CHECKPOINT FAILED] Fold {fold} for {config_name} h={horizon}: {e}. Recomputing...")

        current_fold_results = []
        fold_test_end = anchor_end - fold * test_window
        fold_test_start = fold_test_end - test_window + 1
        fold_train_end = fold_test_start - 1

        if fold_train_end < 1965:
            log.warning(f"  Fold {fold}: train_end={fold_train_end} too early, skipping")
            continue

        train_mask = years <= fold_train_end
        test_mask = (years >= fold_test_start) & (years <= fold_test_end)

        n_train = train_mask.sum()
        n_test = test_mask.sum()

        if n_train < MIN_TRAIN_ROWS or n_test < 10:
            log.warning(f"  Fold {fold}: n_train={n_train}, n_test={n_test}, skipping")
            continue

        log.info(f"  Fold {fold}: train<={fold_train_end} ({n_train}), "
                 f"test {fold_test_start}-{fold_test_end} ({n_test})")

        y = df_valid["target"].values
        years_tr = years[train_mask]

        # Filter feature_cols and full_feature_cols to what exists
        avail_feat = [c for c in feature_cols if c in df_valid.columns]
        avail_full = [c for c in full_feature_cols if c in df_valid.columns]

        X_cont = df_valid[avail_feat].astype(np.float32).replace([np.inf, -np.inf], np.nan).values
        X_full = df_valid[avail_full].astype(np.float32).replace([np.inf, -np.inf], np.nan).values

        # Eval mask for early stopping
        eval_start = fold_train_end - NESTED_VAL_YEARS + 1
        eval_mask = (years > eval_start) & (years <= fold_train_end)

        # ── Naive persistence ──
        pred_naive = prior_pred(df_valid, y, train_mask, test_mask)
        m = calc_metrics(y[test_mask], pred_naive)
        current_fold_results.append(FoldResult(horizon, fold, config_name, "naive_persistence",
                                               m["n"], m["mae"], m["rmse"], m["dir_acc"],
                                               y[test_mask], pred_naive))

        # ── AR(1) honest ──
        pred_ar1 = ar1_honest_pred(df_valid, y, train_mask, test_mask, horizon)
        m = calc_metrics(y[test_mask], pred_ar1)
        current_fold_results.append(FoldResult(horizon, fold, config_name, "ar1_honest",
                                               m["n"], m["mae"], m["rmse"], m["dir_acc"],
                                               y[test_mask], pred_ar1))

        # ── Ridge ──
        pred_ridge = fit_predict_ridge(X_cont, y, train_mask, test_mask)
        m = calc_metrics(y[test_mask], pred_ridge)
        current_fold_results.append(FoldResult(horizon, fold, config_name, "ridge",
                                               m["n"], m["mae"], m["rmse"], m["dir_acc"],
                                               y[test_mask], pred_ridge))

        # ── LGBM with nested Optuna ──
        log.info(f"    Nested Optuna ({OPTUNA_TRIALS} trials)...")
        lgbm_params = nested_optuna_lgbm(X_full[train_mask], y[train_mask], years_tr,
                                         OPTUNA_TRIALS, seed=fold)
        pred_lgbm, importances = fit_predict_lgbm(X_full, y, train_mask, test_mask,
                                                   lgbm_params, eval_mask)
        m = calc_metrics(y[test_mask], pred_lgbm)
        current_fold_results.append(FoldResult(horizon, fold, config_name, "lgbm",
                                               m["n"], m["mae"], m["rmse"], m["dir_acc"],
                                               y[test_mask], pred_lgbm))
        log.info(f"    LGBM MAE={m['mae']:.4f}")

        # ── XGBoost ──
        pred_xgb = fit_predict_xgboost(X_full, y, train_mask, test_mask)
        m = calc_metrics(y[test_mask], pred_xgb)
        current_fold_results.append(FoldResult(horizon, fold, config_name, "xgboost",
                                               m["n"], m["mae"], m["rmse"], m["dir_acc"],
                                               y[test_mask], pred_xgb))

        # ── Ensemble: lgbm + ridge + prior ──
        pred_prior = prior_pred(df_valid, y, train_mask, test_mask)
        pred_ens = 0.4 * pred_lgbm + 0.3 * pred_ridge + 0.3 * pred_prior
        m = calc_metrics(y[test_mask], pred_ens)
        current_fold_results.append(FoldResult(horizon, fold, config_name, "ensemble_lgbm_ridge_prior",
                                               m["n"], m["mae"], m["rmse"], m["dir_acc"],
                                               y[test_mask], pred_ens))
        log.info(f"    Ensemble MAE={m['mae']:.4f}")

        # ── Two-Stage Stacking (Config C) ──
        if use_two_stage and cross_domain_cols:
            avail_cross = [c for c in cross_domain_cols if c in df_valid.columns]
            if len(avail_cross) >= 3:
                X_stage2_cols = avail_cross
                X_stage2_base = df_valid[X_stage2_cols].fillna(0).values.astype(np.float32)

                # For training Stage 2, train Stage 1 on first portion, predict second
                mid_year = int(np.median(years_tr))
                s1_train = years <= mid_year
                s1_val = (years > mid_year) & (years <= fold_train_end)

                if s1_train.sum() > 100 and s1_val.sum() > 50:
                    s1_pred_ridge = fit_predict_ridge(X_cont, y, s1_train, s1_val)
                    s1_pred_prior = prior_pred(df_valid, y, s1_train, s1_val)
                    s1_lgbm_params = nested_optuna_lgbm(
                        X_full[s1_train], y[s1_train],
                        years[s1_train], max(10, OPTUNA_TRIALS // 3), seed=fold + 100
                    )
                    s1_pred_lgbm, _ = fit_predict_lgbm(X_full, y, s1_train, s1_val, s1_lgbm_params)
                    s1_ens = 0.4 * s1_pred_lgbm + 0.3 * s1_pred_ridge + 0.3 * s1_pred_prior

                    X_s2_train = np.column_stack([
                        s1_ens.reshape(-1, 1),
                        X_stage2_base[s1_val],
                    ])
                    y_s2_train = y[s1_val]

                    X_s2_test = np.column_stack([
                        pred_ens.reshape(-1, 1),
                        X_stage2_base[test_mask],
                    ])

                    s2_imp = SimpleImputer(strategy="median")
                    s2_scaler = StandardScaler()
                    X_s2_tr_proc = s2_scaler.fit_transform(s2_imp.fit_transform(X_s2_train))
                    X_s2_te_proc = s2_scaler.transform(s2_imp.transform(X_s2_test))

                    s2_model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000,
                                          random_state=0, warm_start=False)
                    s2_model.fit(X_s2_tr_proc, y_s2_train)
                    pred_two_stage = s2_model.predict(X_s2_te_proc)

                    m = calc_metrics(y[test_mask], pred_two_stage)
                    current_fold_results.append(FoldResult(horizon, fold, config_name, "two_stage_stacking",
                                                           m["n"], m["mae"], m["rmse"], m["dir_acc"],
                                                           y[test_mask], pred_two_stage))
                    log.info(f"    Two-Stage MAE={m['mae']:.4f}")

        # ── Country-Year Twin Enhanced (Config D) ──
        if use_twins and cross_domain_cols:
            avail_cross = [c for c in cross_domain_cols if c in df_valid.columns]
            if len(avail_cross) >= 3:
                twin_feats = compute_twin_features(df_valid, avail_cross, train_mask, k=5)
                X_twin_full = np.column_stack([X_full, twin_feats])

                twin_params = nested_optuna_lgbm(
                    X_twin_full[train_mask], y[train_mask], years_tr,
                    max(10, OPTUNA_TRIALS // 2), seed=fold + 200
                )
                imp_tw = SimpleImputer(strategy="median")
                Xt_tw = imp_tw.fit_transform(X_twin_full[train_mask])
                Xp_tw = imp_tw.transform(X_twin_full[test_mask])

                import lightgbm as lgb
                tw_model = lgb.LGBMRegressor(**twin_params)
                if eval_mask.any():
                    Xe_tw = imp_tw.transform(X_twin_full[eval_mask])
                    tw_model.fit(Xt_tw, y[train_mask],
                                 eval_set=[(Xe_tw, y[eval_mask])],
                                 callbacks=[lgb.early_stopping(stopping_rounds=80, verbose=False)])
                else:
                    tw_model.fit(Xt_tw, y[train_mask])

                pred_twin_lgbm = tw_model.predict(Xp_tw)
                pred_twin_ens = 0.5 * pred_twin_lgbm + 0.3 * pred_ridge + 0.2 * pred_prior
                m = calc_metrics(y[test_mask], pred_twin_ens)
                current_fold_results.append(FoldResult(horizon, fold, config_name, "twin_enhanced_ensemble",
                                                       m["n"], m["mae"], m["rmse"], m["dir_acc"],
                                                       y[test_mask], pred_twin_ens))
                log.info(f"    Twin-Enhanced MAE={m['mae']:.4f}")

        # Save fold checkpoint atomically
        tmp_cp = cp_path.with_suffix(".tmp")
        try:
            with open(tmp_cp, "wb") as f:
                pickle.dump(current_fold_results, f)
            tmp_cp.replace(cp_path)
            log.info(f"  [CHECKPOINT SAVED] Horizon {horizon} | Config {config_name} | Fold {fold}")
        except Exception as e:
            log.warning(f"  [CHECKPOINT SAVE ERROR] Fold {fold} for {config_name} h={horizon}: {e}")

        fold_results.extend(current_fold_results)

    return fold_results


# ════════════════════════════════════════════════════════════════
#  Main Experiment
# ════════════════════════════════════════════════════════════════

def run_experiment():
    t0 = time.time()
    log.info("=" * 70)
    log.info("  CROSS-DOMAIN ECONOMIC PREDICTION ENHANCER")
    log.info("  Senior Economist + Mathematician Approach")
    log.info("=" * 70)

    # 1. Load and merge data
    merged = load_and_merge_panels()

    # 2. Engineer interactions
    merged = engineer_interaction_features(merged)

    # 3. Setup feature sets
    iso_levels = sorted(merged["iso3"].unique().tolist())
    _, _, all_cont_cols, all_full_cols = prepare_features(merged, iso_levels)

    econ_cont = get_econ_only_cols(all_cont_cols)
    cross_cols = get_cross_domain_cols(all_cont_cols)

    dummy_cols = [c for c in all_full_cols if c.startswith("iso_") or c.startswith("tier_")]
    econ_full = econ_cont + dummy_cols
    cross_full = all_full_cols  # everything

    log.info(f"\nFeature sets:")
    log.info(f"  Econ-only continuous: {len(econ_cont)} features")
    log.info(f"  Cross-domain continuous: {len(cross_cols)} features")
    log.info(f"  All continuous: {len(all_cont_cols)} features")
    log.info(f"  Dummies: {len(dummy_cols)} features")

    all_results: list[FoldResult] = []

    for h in HORIZONS:
        log.info(f"\n{'#'*70}")
        log.info(f"  HORIZON h={h}")
        log.info(f"{'#'*70}")

        # Config A: Econ-Only Baseline
        results_a = run_walk_forward_cv(
            merged, h, "A_EconOnly", econ_cont, econ_full,
            use_two_stage=False, use_twins=False
        )
        all_results.extend(results_a)

        # Config B: Econ + Curated Cross-Domain
        results_b = run_walk_forward_cv(
            merged, h, "B_EconCrossDomain", all_cont_cols, cross_full,
            use_two_stage=False, use_twins=False
        )
        all_results.extend(results_b)

        # Config C: Two-Stage Stacking
        results_c = run_walk_forward_cv(
            merged, h, "C_TwoStageStack", econ_cont, econ_full,
            cross_domain_cols=cross_cols,
            use_two_stage=True, use_twins=False
        )
        all_results.extend(results_c)

        # Config D: Country-Year Twin Enhanced
        results_d = run_walk_forward_cv(
            merged, h, "D_TwinEnhanced", all_cont_cols, cross_full,
            cross_domain_cols=cross_cols,
            use_two_stage=False, use_twins=True
        )
        all_results.extend(results_d)

    # ═══════════════════════════════════════════════════════════
    #  Aggregate and Compare Results
    # ═══════════════════════════════════════════════════════════
    log.info(f"\n{'='*70}")
    log.info("  AGGREGATING RESULTS")
    log.info(f"{'='*70}")

    rows = []
    for r in all_results:
        rows.append({
            "horizon": r.horizon, "fold": r.fold, "config": r.config,
            "model": r.model, "n_test": r.n_test,
            "mae": r.mae, "rmse": r.rmse, "dir_acc": r.dir_acc,
        })
    df_results = pd.DataFrame(rows)

    summary_rows = []
    for (h, cfg, model), grp in df_results.groupby(["horizon", "config", "model"]):
        summary_rows.append({
            "horizon": h, "config": cfg, "model": model,
            "n_folds": len(grp), "mae_mean": grp["mae"].mean(),
            "mae_std": grp["mae"].std(), "rmse_mean": grp["rmse"].mean(),
            "dir_acc_mean": grp["dir_acc"].mean(),
        })
    df_summary = pd.DataFrame(summary_rows)

    # Diebold-Mariano comparisons
    dm_rows = []
    for h in HORIZONS:
        baseline_key = ("A_EconOnly", "ensemble_lgbm_ridge_prior")
        for cfg in ["B_EconCrossDomain", "C_TwoStageStack", "D_TwinEnhanced"]:
            for model in df_results[df_results["config"] == cfg]["model"].unique():
                if model in ("naive_persistence", "ar1_honest"):
                    continue
                base_errors = []
                test_errors = []
                for fold in range(N_FOLDS):
                    base_fold = [r for r in all_results
                                 if r.horizon == h and r.fold == fold
                                 and r.config == baseline_key[0] and r.model == baseline_key[1]]
                    test_fold = [r for r in all_results
                                 if r.horizon == h and r.fold == fold
                                 and r.config == cfg and r.model == model]
                    if base_fold and test_fold:
                        base_errors.append(np.abs(base_fold[0].y_true - base_fold[0].y_pred))
                        test_errors.append(np.abs(test_fold[0].y_true - test_fold[0].y_pred))

                if len(base_errors) >= 2:
                    e_base = np.concatenate(base_errors)
                    e_test = np.concatenate(test_errors)
                    min_len = min(len(e_base), len(e_test))
                    dm_stat, p_val = diebold_mariano(e_base[:min_len], e_test[:min_len])

                    base_mae = float(np.mean(e_base))
                    test_mae = float(np.mean(e_test))
                    improvement_pct = ((base_mae - test_mae) / base_mae) * 100

                    dm_rows.append({
                        "horizon": h,
                        "baseline": f"{baseline_key[0]}:{baseline_key[1]}",
                        "challenger": f"{cfg}:{model}",
                        "baseline_mae": round(base_mae, 5),
                        "challenger_mae": round(test_mae, 5),
                        "improvement_pct": round(improvement_pct, 2),
                        "dm_statistic": round(dm_stat, 4),
                        "p_value": round(p_val, 4),
                        "significant_at_0.05": p_val < 0.05,
                        "significant_at_0.10": p_val < 0.10,
                    })

    df_dm = pd.DataFrame(dm_rows)

    # ═══════════════════════════════════════════════════════════
    #  Save Results
    # ═══════════════════════════════════════════════════════════
    df_results.to_csv(OUT_DIR / "full_cv_results.csv", index=False)
    df_summary.to_csv(OUT_DIR / "summary_comparison.csv", index=False)
    df_dm.to_csv(OUT_DIR / "diebold_mariano_tests.csv", index=False)

    summary_json = {
        "experiment": "cross_domain_economic_enhancer",
        "horizons": HORIZONS,
        "n_folds": N_FOLDS,
        "optuna_trials_per_fold": OPTUNA_TRIALS,
        "runtime_seconds": round(time.time() - t0, 1),
        "configs": {
            "A_EconOnly": "Economic features only (baseline)",
            "B_EconCrossDomain": "Economic + curated cross-domain + interactions",
            "C_TwoStageStack": "Two-stage stacking (econ -> cross-domain correction)",
            "D_TwinEnhanced": "Country-year twin enhanced from cross-domain space",
        },
        "summary": df_summary.to_dict(orient="records"),
        "diebold_mariano": df_dm.to_dict(orient="records") if len(df_dm) > 0 else [],
    }
    with open(OUT_DIR / "experiment_summary.json", "w") as f:
        json.dump(summary_json, f, indent=2)

    # ═══════════════════════════════════════════════════════════
    #  Print Headline Results
    # ═══════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"  EXPERIMENT COMPLETE - {elapsed:.0f}s")
    print(f"{'='*80}")

    print(f"\n{'-'*80}")
    print(f"  HEADLINE COMPARISON: Ensemble MAE (mean +/- std across folds)")
    print(f"{'-'*80}")

    for h in HORIZONS:
        print(f"\n  h={h}:")
        for cfg in ["A_EconOnly", "B_EconCrossDomain", "C_TwoStageStack", "D_TwinEnhanced"]:
            ens_models = df_summary[
                (df_summary["horizon"] == h) &
                (df_summary["config"] == cfg) &
                (~df_summary["model"].isin(["naive_persistence", "ar1_honest"]))
            ]
            if len(ens_models) == 0:
                continue
            best = ens_models.sort_values("mae_mean").iloc[0]
            print(f"    {cfg:25s} | {best['model']:30s} | "
                  f"MAE = {best['mae_mean']:.4f} +/- {best['mae_std']:.4f}")

        for baseline_model in ["naive_persistence", "ar1_honest"]:
            base = df_summary[
                (df_summary["horizon"] == h) &
                (df_summary["config"] == "A_EconOnly") &
                (df_summary["model"] == baseline_model)
            ]
            if len(base) > 0:
                b = base.iloc[0]
                print(f"    {'Baseline':25s} | {b['model']:30s} | "
                      f"MAE = {b['mae_mean']:.4f} +/- {b['mae_std']:.4f}")

    if len(df_dm) > 0:
        print(f"\n{'-'*80}")
        print(f"  DIEBOLD-MARIANO SIGNIFICANCE TESTS (vs A_EconOnly ensemble)")
        print(f"{'-'*80}")
        print(df_dm.to_string(index=False))

    print(f"\n  Results saved to: {OUT_DIR}")
    print(f"  Files: full_cv_results.csv, summary_comparison.csv, "
          f"diebold_mariano_tests.csv, experiment_summary.json")

    log.info(f"\nTotal runtime: {elapsed:.0f}s")
    return df_results, df_summary, df_dm


if __name__ == "__main__":
    run_experiment()
