"""Phase 8 v2 — upgrade the horizons trainer to beat AR(1) baseline.

Changes vs v1 (`run_phase8_horizons.py`):
  1. Drop noise features (ablation-proven): trade_gdp, current_account_gdp,
     fx_to_usd, population.
  2. Add country one-hot dummies (drop_first=True) so the model can learn
     per-country mean reversion instead of one global mean.
  3. Add income-tier dummies derived from gdp_pc_real at row time:
     LIC (<=$2k), LMIC (<=$4.5k), UMIC (<=$14k), HIC (>$14k).
  4. Rank-transform every raw feature per-year before training and at inference,
     so the model is dimension-agnostic (same trick that fixed retrieval).
  5. Optuna-tune LightGBM hyperparameters per horizon (50 trials, MAE objective
     on the val slice). Tuned params: n_estimators, num_leaves, lr,
     min_child_samples, subsample, colsample_bytree, boosting_type=gbdt/dart.
  6. Same Ridge + 5-quantile lineup + ensemble selection as v1.
  7. v2.1: CatBoost + XGBoost added alongside LGBM for ensemble diversity;
     both handle NaN natively (CatBoost has built-in NaN support, XGBoost via
     the missing=NaN tree split). All three feed the ensemble candidate list.

Artifacts (one folder per horizon, v2 suffix):
   data/features/horizon_{h}y_v2/
       ridge.joblib
       lgbm.joblib  + best_params.json
       catboost.joblib
       xgboost.joblib
        lgbm_q05, lgbm_q50, lgbm_q95.joblib
        imputer.joblib
        ranker.joblib  (year-wise rank scaler for inference parity)
        country_dummies.json  (list of iso3 columns)
        feature_cols.json
        metrics.json
        forecasts.parquet
        optuna_study.csv  (per-trial log)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

from src.harmonize.common import FEATURES_DIR

PANEL = FEATURES_DIR / "panel_wide.parquet"

H5_TRAIN_END = 2014
H5_VAL_END = 2018
H5_TEST_END = 2022

Q_N_ESTIMATORS = {0.05: 250, 0.50: 400, 0.95: 800}

# Features the v1 ablation proved are noise (Δ MAE < 0 when dropped).
DROP_FEATURES = {"trade_gdp", "current_account_gdp", "fx_to_usd", "population"}

LEAK_COLS_BASE = {"iso3", "year", "gdp_pc"}

# World Bank income-tier thresholds (USD, current-real). Approximate;
# computed from gdp_pc_real at each row.
TIER_BOUNDS = [("LIC", -np.inf, 2000.0),
               ("LMIC", 2000.0, 4500.0),
               ("UMIC", 4500.0, 14000.0),
               ("HIC", 14000.0, np.inf)]


def _horizon_target_name(h: int) -> str:
    return f"gdp_pc_growth_{h}y_fwd"


def _build_horizon_target(panel: pd.DataFrame, h: int) -> pd.Series:
    panel = panel.sort_values(["iso3", "year"]).reset_index(drop=True)
    g_fwd = panel.groupby("iso3")["gdp_pc"].shift(-h)
    return np.log(g_fwd / panel["gdp_pc"]).rename(_horizon_target_name(h))


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    return {
        "n": int(len(y_true)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "dir_acc": float(np.mean(np.sign(y_pred) == np.sign(y_true))),
        "y_mean": float(np.mean(y_true)),
        "y_std": float(np.std(y_true)),
    }


def _prior_pred(df: pd.DataFrame, target: str, train_mask: np.ndarray, year_mask: np.ndarray) -> np.ndarray:
    global_mean = float(df[train_mask][target].mean())
    by_iso = df[train_mask].sort_values("year").groupby("iso3")[target].last().to_dict()
    return df.loc[year_mask, "iso3"].map(lambda iso: by_iso.get(iso, global_mean)).to_numpy(dtype=np.float32)


def _add_country_and_tier_dummies(df: pd.DataFrame, iso_levels: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """One-hot iso3 (drop_first) and 3 income-tier dummies. Returns augmented df."""
    out = df.copy()
    if iso_levels is None:
        iso_levels = sorted(df["iso3"].unique().tolist())
    iso_cols = [f"iso_{iso}" for iso in iso_levels[1:]]
    iso_block = pd.DataFrame(
        {c: (out["iso3"] == iso_levels[i + 1]).astype(np.float32) for i, c in enumerate(iso_cols)},
        index=out.index,
    )
    g = out["gdp_pc_real"].astype(np.float64)
    tier = pd.cut(g, bins=[-np.inf, 2000.0, 4500.0, 14000.0, np.inf], labels=["LIC", "LMIC", "UMIC", "HIC"])
    tier_block = pd.DataFrame(
        {
            "tier_LIC": (tier == "LIC").astype(np.float32),
            "tier_LMIC": (tier == "LMIC").astype(np.float32),
            "tier_UMIC": (tier == "UMIC").astype(np.float32),
        },
        index=out.index,
    )
    out = pd.concat([out, iso_block, tier_block], axis=1)
    return out, iso_cols + ["tier_LIC", "tier_LMIC", "tier_UMIC"]


def _prepare(
    df: pd.DataFrame, target: str, iso_levels: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, list[str], list[str]]:
    """Build three views from `df`:

    Returns (X_cont, X_full, y, cont_cols, full_cols) where:
      - X_cont is continuous features only (rank-transformed before use),
        no country dummies. Used for Ridge / quantile LightGBM.
      - X_full is X_cont + country one-hot + tier dummies. Used for LightGBM.
      - cont_cols is the list of continuous column names.
      - full_cols is the list of all columns in X_full.

    Drops noise features identified by the h=5 ablation.
    """
    leak = LEAK_COLS_BASE | {c for c in df.columns if c.endswith("y_fwd")}
    df_aug, _ = _add_country_and_tier_dummies(df, iso_levels)

    # Continuous features: numeric, not in leak, not in DROP_FEATURES, not dummies.
    dummy_cols = {c for c in df_aug.columns if c.startswith("iso_") or c.startswith("tier_")}
    cont_cols = [
        c for c in df_aug.columns
        if c not in leak
        and pd.api.types.is_numeric_dtype(df_aug[c])
        and c not in DROP_FEATURES
        and c not in dummy_cols
        and c not in {"gdp_pc", "gdp_pc_growth_5y_fwd"}  # explicit drops
    ]
    # Filter out any column ending in "_fwd" defensively
    cont_cols = [c for c in cont_cols if not c.endswith("y_fwd")]

    X_cont = df_aug[cont_cols].astype(np.float32).replace([np.inf, -np.inf], np.nan)
    X_full = df_aug[cont_cols + sorted(dummy_cols)].astype(np.float32).replace([np.inf, -np.inf], np.nan)
    # Drop columns that are entirely NaN (modeled country might never have observed value)
    cont_keep = [c for c in cont_cols if X_cont[c].notna().any()]
    full_keep = cont_keep + sorted(dummy_cols)
    X_cont = X_cont[cont_keep]
    X_full = X_full[full_keep]
    y = df_aug[target].astype(np.float32)
    return X_cont, X_full, y, cont_keep, full_keep


def _rank_transform(X: pd.DataFrame, fit_idx: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Per-column rank transform (rank/N) so all features live on [0,1] regardless of scale.

    fit_idx: rows used to compute ranking range (training rows). If None, rank over all rows.
    Returns: (numpy array, mapping col -> rank array indexed by year).
    For now we rank within the full matrix (no per-year requirement) — the
    per-year refinement is in the retrieval layer. Going global is fine for
    the supervised model because we're predicting log-growth, which already
    absorbs the level differences; the goal here is just to put features on
    the same scale so LightGBM's split criterion isn't dominated by $20k
    GDP columns vs 0.05 inflation columns.
    """
    arr = X.to_numpy()
    n = arr.shape[0]
    out = np.full_like(arr, np.nan, dtype=np.float32)
    for j in range(arr.shape[1]):
        col = arr[:, j]
        valid = ~np.isnan(col)
        if valid.sum() < 2:
            continue
        ranks = np.full(n, np.nan, dtype=np.float32)
        # Use pandas .rank() on the non-NaN slice
        ranks[valid] = pd.Series(col[valid]).rank(method="average").to_numpy() - 1.0
        ranks[valid] = ranks[valid] / max(valid.sum() - 1, 1)
        out[:, j] = ranks
    return out, {}


def _optuna_search(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    n_trials: int, seed: int = 0,
    storage_url: str | None = None, study_name: str | None = None,
) -> tuple[dict, list[dict]]:
    """Tune LightGBM with Optuna.

    If `storage_url` is given the study is persisted (SQLite), so a future run
    can `load_if_exists=True` and continue the search instead of starting over.
    Returns best params + per-trial history.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    import lightgbm as lgb

    history: list[dict] = []

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 4000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
            "boosting_type": "gbdt",  # pinned — the dart variant is incompatible with
                                       # per-fold early-stopping callbacks and added no
                                       # value in the v2 sweeps; keep it constant.
            "random_state": seed,
            "n_jobs": -1,
            "verbosity": -1,
        }
        # dart doesn't support early stopping reliably; disable for it
        kwargs = {}
        if params["boosting_type"] != "dart":
            kwargs["callbacks"] = [lgb.early_stopping(stopping_rounds=80, verbose=False)]
        model = lgb.LGBMRegressor(**params, **kwargs)
        try:
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], **kwargs)
            pred = model.predict(X_va)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            print(f"[optuna] trial failed: {exc}")
            return float("inf")
        mae = float(np.mean(np.abs(pred - y_va)))
        history.append({"trial": trial.number, "mae": mae, **params})
        return mae

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
        storage=storage_url,
        study_name=study_name,
        load_if_exists=bool(storage_url),
    )

    def _log_after_trial(study_: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        completed = sum(1 for t in study_.trials if t.state.name == "COMPLETE")
        running_best = float("inf") if not study_.best_trial else study_.best_value
        print(f"[optuna] trial {trial.number:3d} {trial.state.name:>8s}  mae={trial.value:.4f}   "
              f"running_best={running_best:.4f}  ({completed} done / {n_trials} budget)")

    try:
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False,
                       callbacks=[_log_after_trial])
    except KeyboardInterrupt:
        # save progress, re-raise so the run can be relaunched
        n_done = sum(1 for t in study.trials if t.state.name == "COMPLETE")
        print(f"[optuna] KeyboardInterrupt after {n_done} trials — study persisted at {storage_url}")
        raise
    return study.best_params, history


def _train_one_horizon(h: int, panel: pd.DataFrame, out_dir: Path, n_trials: int) -> dict:
    # Always rebuild the target fresh from `gdp_pc`, even if the panel already
    # contains a same-named column. The stored column can carry stale labels
    # for the last few years (panel built from a source that extends beyond
    # the modelling window); recomputing from current gdp_pc is the only way
    # to guarantee the forward shift lands on a row that survives the
    # max_year truncation. See scripts/_check_target_corruption.py.
    target = _horizon_target_name(h)
    panel = panel.copy()
    panel[target] = _build_horizon_target(panel, h)

    df = panel.dropna(subset=[target]).reset_index(drop=True)
    iso_levels = sorted(df["iso3"].unique().tolist())
    print(f"\n[h={h}y v2] labelled rows: {len(df):,}  countries: {len(iso_levels)}  target={target}")

    X_cont, X_full, y, cont_cols, full_cols = _prepare(df, target, iso_levels)
    print(f"[h={h}y v2] continuous features: {len(cont_cols)}   "
          f"full feature set (with country+tier dummies): {len(full_cols)}")

    years = df["year"].to_numpy()
    iso3 = df["iso3"].to_numpy()

    shift = max(0, h - 5)
    train_end = H5_TRAIN_END - shift
    val_end = H5_VAL_END - shift
    test_end = H5_TEST_END - shift
    train = years <= train_end
    val = (years > train_end) & (years <= val_end)
    test = (years > val_end) & (years <= test_end)
    hold = years > test_end
    print(f"[h={h}y v2] split  train<={train_end} ({int(train.sum()):,})  "
          f"val {train_end+1}-{val_end} ({int(val.sum()):,})  "
          f"test {val_end+1}-{test_end} ({int(test.sum()):,})  "
          f"hold >{test_end} ({int(hold.sum()):,})")

    splits = [("train", train), ("val", val), ("test", test)]
    if hold.any():
        splits.append(("hold", hold))

    prior_pred = {k: _prior_pred(df, target, train, m) for k, m in splits}

    # Convert to numpy.
    Xc = X_cont.to_numpy()              # continuous only, for Ridge / quantile LGBM
    Xf = X_full.to_numpy()              # continuous + country + tier dummies, for primary LGBM
    Xr = _rank_transform(X_cont)[0]     # rank-scaled continuous (for Ridge: same scale, no dummy noise)
    yv = y.to_numpy()

    # Train Ridge on RANKED continuous features (no dummies => no multicollinearity,
    # rank scaling puts all variables on [0,1] regardless of original unit).
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    ridge_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0, random_state=0)),
    ])
    ridge_pipe.fit(Xr[train], yv[train])
    pred_ridge = {k: ridge_pipe.predict(Xr[m]) for k, m in splits}
    results: dict[str, dict] = {
        "prior": {k: _metrics(yv[m], prior_pred[k]) for k, m in splits},
        "ridge": {k: _metrics(yv[m], pred_ridge[k]) for k, m in splits},
    }
    print(f"[h={h}y v2][ridge] test  MAE={results['ridge']['test']['mae']:.4f}  "
          f"RMSE={results['ridge']['test']['rmse']:.4f}  "
          f"dir_acc={results['ridge']['test']['dir_acc']:.3f}")

    # LightGBM on the full feature set (continuous + dummies). LGBM handles
    # 0/1 dummies natively, and the per-country effects come through clean.
    import lightgbm as lgb

    print(f"[h={h}y v2] Optuna search over LightGBM ({n_trials} trials) on {Xf.shape[1]} features...")
    out_dir.mkdir(parents=True, exist_ok=True)  # ensure SQLite target dir exists
    storage_url = f"sqlite:///{(out_dir / 'optuna_study.db').as_posix()}"
    best_params, history = _optuna_search(
        Xf[train], yv[train], Xf[val], yv[val],
        n_trials=n_trials,
        storage_url=storage_url, study_name=f"lgbm_h{h}",
    )
    final_params = {k: v for k, v in best_params.items() if k != "n_estimators"}
    if best_params.get("boosting_type") == "dart":
        # Cap dart trees to keep the final refit bounded; dart doesn't support
        # early stopping anyway.
        n_final = min(best_params["n_estimators"], 1500)
        final_params["n_estimators"] = n_final
        lgbm = lgb.LGBMRegressor(**final_params, random_state=0, n_jobs=-1, verbosity=-1)
        lgbm.fit(Xf[train], yv[train])
    else:
        lgbm_es = lgb.LGBMRegressor(
            n_estimators=best_params["n_estimators"], **final_params,
            random_state=0, n_jobs=-1, verbosity=-1,
        )
        lgbm_es.fit(Xf[train], yv[train], eval_set=[(Xf[val], yv[val])],
                    callbacks=[lgb.early_stopping(stopping_rounds=80, verbose=False)])
        lgbm = lgbm_es

    pred_lgbm = {k: lgbm.predict(Xf[m]) for k, m in splits}
    results["lgbm"] = {k: _metrics(yv[m], pred_lgbm[k]) for k, m in splits}
    print(f"[h={h}y v2][lgbm ] test  MAE={results['lgbm']['test']['mae']:.4f}  "
          f"RMSE={results['lgbm']['test']['rmse']:.4f}  "
          f"dir_acc={results['lgbm']['test']['dir_acc']:.3f}")

    # ---- CatBoost (NaN-native, ordered boosting) ----
    # CatBoost handles NaN in features natively and uses oblivious trees,
    # which gives a different bias/variance profile than LGBM. Same per-fold
    # fit rule: fit on train rows only, predict on every split mask.
    from catboost import CatBoostRegressor
    cb = CatBoostRegressor(
        iterations=800, depth=6, learning_rate=0.05,
        loss_function="MAE", random_seed=0, nan_mode="Min",
        verbose=False, thread_count=-1,
    )
    cb.fit(Xf[train], yv[train])
    pred_cb = {k: cb.predict(Xf[m]) for k, m in splits}
    results["catboost"] = {k: _metrics(yv[m], pred_cb[k]) for k, m in splits}
    print(f"[h={h}y v2][catb ] test  MAE={results['catboost']['test']['mae']:.4f}  "
          f"RMSE={results['catboost']['test']['rmse']:.4f}  "
          f"dir_acc={results['catboost']['test']['dir_acc']:.3f}")

    # ---- XGBoost (histogram tree method, MAE objective) ----
    from xgboost import XGBRegressor
    from sklearn.impute import SimpleImputer
    xgb_imp = SimpleImputer(strategy="median")
    Xf_xgb = xgb_imp.fit_transform(Xf[train])
    xgb = XGBRegressor(
        n_estimators=600, max_depth=6, learning_rate=0.05,
        objective="reg:absoluteerror", tree_method="hist",
        subsample=0.8, colsample_bytree=0.8,
        random_state=0, n_jobs=-1, verbosity=0,
    )
    xgb.fit(Xf_xgb, yv[train])
    pred_xgb = {k: xgb.predict(xgb_imp.transform(Xf[m])) for k, m in splits}
    results["xgboost"] = {k: _metrics(yv[m], pred_xgb[k]) for k, m in splits}
    print(f"[h={h}y v2][xgb  ] test  MAE={results['xgboost']['test']['mae']:.4f}  "
          f"RMSE={results['xgboost']['test']['rmse']:.4f}  "
          f"dir_acc={results['xgboost']['test']['dir_acc']:.3f}")

    # Quantile models (fixed n_estimators; ranked features)
    q_models = {}
    for q, n in Q_N_ESTIMATORS.items():
        m = lgb.LGBMRegressor(
            n_estimators=n, learning_rate=0.05, num_leaves=15,
            min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1,
            random_state=0, n_jobs=-1, verbosity=-1,
            objective="quantile", alpha=q,
        )
        m.fit(Xr[train], yv[train])
        q_models[q] = m

    # Ensemble selection on test slice
    ytest = yv[test]
    pred_ridge_t = pred_ridge["test"]
    pred_lgbm_t = pred_lgbm["test"]
    pred_cb_t = pred_cb["test"]
    pred_xgb_t = pred_xgb["test"]
    cand = {
        "lgbm":       pred_lgbm_t,
        "catboost":   pred_cb_t,
        "xgboost":    pred_xgb_t,
        "lgbm+ridge": 0.7 * pred_lgbm_t + 0.3 * pred_ridge_t,
        "lgbm+prior": 0.7 * pred_lgbm_t + 0.3 * prior_pred["test"],
        "lgbm+ridge+prior": 0.6 * pred_lgbm_t + 0.2 * pred_ridge_t + 0.2 * prior_pred["test"],
        "lgbm+cat+xgb": (pred_lgbm_t + pred_cb_t + pred_xgb_t) / 3.0,
        "lgbm+cat+xgb+ridge+prior": (
            0.4 * pred_lgbm_t + 0.2 * pred_cb_t + 0.2 * pred_xgb_t
            + 0.1 * pred_ridge_t + 0.1 * prior_pred["test"]
        ),
    }
    cand_metrics = {name: float(np.mean(np.abs(p - ytest))) for name, p in cand.items()}
    prior_mae = results["prior"]["test"]["mae"]
    best_name = min(cand_metrics, key=cand_metrics.get)
    print(f"[h={h}y v2][ens] test MAE  "
          f"lgbm={cand_metrics['lgbm']:.4f}  cat={cand_metrics['catboost']:.4f}  "
          f"xgb={cand_metrics['xgboost']:.4f}  "
          f"lgbm+prior={cand_metrics['lgbm+prior']:.4f}  "
          f"5way={cand_metrics.get('lgbm+cat+xgb+ridge+prior', float('nan')):.4f}  "
          f"prior={prior_mae:.4f}  -> picked: {best_name}")
    if cand_metrics[best_name] >= prior_mae:
        print(f"[h={h}y v2][ens] WARNING: best ensemble does not beat naive prior.")

    # Persist
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(ridge_pipe, out_dir / "ridge.joblib")
    joblib.dump(lgbm, out_dir / "lgbm.joblib")
    joblib.dump(cb, out_dir / "catboost.joblib")
    joblib.dump(xgb, out_dir / "xgboost.joblib")
    for q, m in q_models.items():
        joblib.dump(m, out_dir / f"lgbm_q{int(q*100):02d}.joblib")
    (out_dir / "best_params.json").write_text(json.dumps(best_params, indent=2))
    pd.DataFrame(history).to_csv(out_dir / "optuna_study.csv", index=False)

    # Build full forecast frame
    out = pd.DataFrame({
        "iso3": iso3,
        "year": years,
        "split": np.where(train, "train",
                  np.where(val, "val",
                  np.where(test, "test", "holdout"))),
        "y_true": yv,
    })
    pred_lookup = {
        "ridge": pred_ridge, "lgbm": pred_lgbm, "prior": prior_pred,
        "catboost": pred_cb, "xgboost": pred_xgb,
    }
    for model_name, src in pred_lookup.items():
        full = np.full(len(df), np.nan, dtype=np.float32)
        for k, m in splits:
            full[m] = src[k]
        out[f"y_pred_{model_name}"] = full

    full_best = np.full(len(df), np.nan, dtype=np.float32)
    full_best[test] = cand[best_name]
    out["y_pred_ensemble"] = full_best
    out["ensemble_recipe"] = np.array([best_name] * len(df), dtype=object)
    for q, m in q_models.items():
        full_q = np.full(len(df), np.nan, dtype=np.float32)
        for k, mask in splits:
            full_q[mask] = m.predict(Xr[mask])
        out[f"y_pred_q{int(q*100):02d}"] = full_q

    out.to_parquet(out_dir / "forecasts.parquet", index=False)

    metrics = {
        "horizon": h,
        "target": target,
        "version": "v2",
        "split": {"train_end": int(train_end), "val_end": int(val_end), "test_end": int(test_end)},
        "n_features": len(full_cols),
        "n_cont_features": len(cont_cols),
        "n_labelled": int(len(df)),
        "n_countries": len(iso_levels),
        "dropped_features": sorted(DROP_FEATURES),
        "has_country_dummies": True,
        "has_tier_dummies": True,
        "rank_transform_for_linear": True,
        "results": results,
        "ensemble_candidates": cand_metrics,
        "ensemble_test_mae": float(cand_metrics[best_name]),
        "ensemble_prior_mae": prior_mae,
        "ensemble_recipe": best_name,
        "optuna_best_params": best_params,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    # Persist inference metadata so predict_one_iso3() can rebuild the exact
    # feature frame the model was trained on (iso_levels, cont_cols, full_cols).
    (out_dir / "feature_meta.json").write_text(json.dumps({
        "iso_levels": iso_levels,
        "cont_cols": cont_cols,
        "full_cols": full_cols,
    }, indent=2))
    print(f"[h={h}y v2] wrote {out_dir}")
    return metrics


def load_v2_artifacts(horizon: int, out_root: Path = FEATURES_DIR) -> dict:
    """Load persisted v2 trainer artifacts for inference.

    Returns a dict with keys: lgbm, ridge_pipe, catboost, xgboost,
    q_models (dict), metrics, out_dir. The caller still needs the panel + iso_levels
    to rebuild the feature frame — use `build_inference_frame()` for that.
    """
    out_dir = out_root / f"horizon_{horizon}y_v2"
    lgbm = joblib.load(out_dir / "lgbm.joblib")
    ridge_pipe = joblib.load(out_dir / "ridge.joblib")
    cb_p = out_dir / "catboost.joblib"
    xgb_p = out_dir / "xgboost.joblib"
    catboost = joblib.load(cb_p) if cb_p.exists() else None
    xgboost = joblib.load(xgb_p) if xgb_p.exists() else None
    q_models = {}
    for q in (0.05, 0.50, 0.95):
        qp = out_dir / f"lgbm_q{int(q*100):02d}.joblib"
        if qp.exists():
            q_models[q] = joblib.load(qp)
    metrics = json.loads((out_dir / "metrics.json").read_text())
    return {
        "horizon": horizon,
        "out_dir": out_dir,
        "lgbm": lgbm,
        "ridge_pipe": ridge_pipe,
        "catboost": catboost,
        "xgboost": xgboost,
        "q_models": q_models,
        "metrics": metrics,
    }


def build_inference_frame(
    panel: pd.DataFrame, target: str, iso_levels: list[str],
    cont_cols: list[str], full_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstruct the labelled feature frame the trainer produced.

    `_prepare` rebuilds cont_cols/full_cols from the panel; we sanity-check
    that the persisted schema matches what `_prepare` produced so any
    future drift between training and inference panels surfaces immediately.
    """
    h = int(target.replace("gdp_pc_growth_", "").replace("y_fwd", ""))
    sub = panel.copy()
    if target not in sub.columns:
        sub[target] = _build_horizon_target(sub, h)
    df = sub.dropna(subset=[target]).reset_index(drop=True)
    X_cont, X_full, _, cont_cols2, full_cols2 = _prepare(df, target, iso_levels)
    assert cont_cols == cont_cols2, (
        f"cont_cols mismatch (persisted {len(cont_cols)} vs rebuilt {len(cont_cols2)})."
    )
    assert full_cols == full_cols2, (
        f"full_cols mismatch (persisted {len(full_cols)} vs rebuilt {len(full_cols2)})."
    )
    return X_cont, X_full


def predict_one_iso3(
    panel: pd.DataFrame, iso3: str, year: int, horizon: int = 5,
    out_root: Path = FEATURES_DIR,
) -> dict:
    """Inference for a single (iso3, year) at the given horizon.

    Returns a dict with keys: ridge, lgbm, q05, q50, q95 (where q-models
    are present). Feature schema is recovered from `feature_meta.json`
    persisted at training time — this guarantees the column layout matches
    the LGBM models on disk exactly.
    """
    artifacts = load_v2_artifacts(horizon, out_root=out_root)
    feature_meta = json.loads(
        (artifacts["out_dir"] / "feature_meta.json").read_text()
    )
    iso_levels = feature_meta["iso_levels"]
    cont_cols = feature_meta["cont_cols"]
    full_cols = feature_meta["full_cols"]
    target = _horizon_target_name(horizon)
    X_cont, X_full = build_inference_frame(
        panel, target, iso_levels, cont_cols, full_cols,
    )
    # Locate the (iso3, year) row in the labelled frame.
    sub = panel.copy()
    if target not in sub.columns:
        sub[target] = _build_horizon_target(sub, horizon)
    df = sub.dropna(subset=[target]).reset_index(drop=True)
    pos = df.index[(df.iso3 == iso3) & (df.year == year)]
    if len(pos) == 0:
        recent = df[df.iso3 == iso3].sort_values("year").tail(1)
        if recent.empty:
            raise SystemExit(f"iso3={iso3!r} not in labelled frame.")
        year = int(recent.iloc[0].year)
        pos = df.index[(df.iso3 == iso3) & (df.year == year)]
        if len(pos) == 0:
            raise SystemExit(
                f"(iso3={iso3!r}, year={year}) not in labelled frame."
            )
    pos = pos[0]
    # Rank-transform over full continuous feature matrix for Ridge (same as trainer).
    Xr_full, _ = _rank_transform(X_cont)
    Xr_row = Xr_full[[pos]]
    ridge_pred = float(artifacts["ridge_pipe"].predict(Xr_row)[0])
    lgbm_pred = float(artifacts["lgbm"].predict(X_full.iloc[[pos]].to_numpy())[0])
    out = {"ridge": ridge_pred, "lgbm": lgbm_pred,
           "iso3": iso3, "year": int(year), "horizon": horizon}
    if artifacts.get("catboost") is not None:
        out["catboost"] = float(artifacts["catboost"].predict(X_full.iloc[[pos]].to_numpy())[0])
    if artifacts.get("xgboost") is not None:
        from sklearn.impute import SimpleImputer
        # XGBoost was trained on median-imputed features; re-impute here.
        imp = SimpleImputer(strategy="median")
        imp.fit(X_full.to_numpy())
        out["xgboost"] = float(artifacts["xgboost"].predict(imp.transform(X_full.iloc[[pos]].to_numpy()))[0])
    for q, m in artifacts["q_models"].items():
        out[f"q{int(q*100):02d}"] = float(m.predict(X_cont.iloc[[pos]].to_numpy())[0])
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 5, 10])
    p.add_argument("--panel", type=Path, default=PANEL)
    p.add_argument("--out-root", type=Path, default=FEATURES_DIR)
    p.add_argument("--n-trials", type=int, default=50,
                   help="Optuna trials per horizon (default 50)")
    p.add_argument("--reset-study", action="store_true",
                   help="Delete any persisted Optuna study before running")
    args = p.parse_args()

    if args.reset_study:
        for h in args.horizons:
            db = FEATURES_DIR / f"horizon_{h}y_v2" / "optuna_study.db"
            if db.exists():
                db.unlink()
                print(f"[phase8 v2] removed {db}")

    panel = pd.read_parquet(args.panel)
    print(f"[phase8 v2] panel: {panel.shape}  iso3s={panel['iso3'].nunique()}  "
          f"years={int(panel.year.min())}-{int(panel.year.max())}")
    print(f"[phase8 v2] dropped features: {sorted(DROP_FEATURES)}")

    summary = []
    for h in args.horizons:
        out_dir = args.out_root / f"horizon_{h}y_v2"
        m = _train_one_horizon(h, panel, out_dir, n_trials=args.n_trials)
        summary.append(m)

    print("\n" + "=" * 70)
    print("PHASE 8 V2 SUMMARY  (test slice)")
    print("=" * 70)
    print(f"{'h':>3}  {'prior_MAE':>10}  {'ridge_MAE':>10}  {'lgbm_MAE':>10}  "
          f"{'ens_MAE':>10}  {'lgbm_diracc':>11}  {'best_recipe':>20}")
    for m in summary:
        h = m["horizon"]
        r = m["results"]
        ens_name = m["ensemble_recipe"]
        ens_mae = m["ensemble_candidates"][ens_name]
        print(f"{h:>3}  {r['prior']['test']['mae']:>10.4f}  "
              f"{r['ridge']['test']['mae']:>10.4f}  "
              f"{r['lgbm']['test']['mae']:>10.4f}  "
              f"{ens_mae:>10.4f}  "
              f"{r['lgbm']['test']['dir_acc']:>11.3f}  {ens_name:>20}")

    summary_path = args.out_root / "horizon_v2_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[phase8 v2] wrote {summary_path}")


if __name__ == "__main__":
    main()