"""Walk-forward cross-validated backtest for the v2 horizon models.

For each horizon h in {1, 3, 5, 10} and each fold f in {0..F-1}:
  1.  Pick a year window of length W that ends at year E_f.
  2.  Train Ridge + LightGBM + naive-prior on years <= E_f (only).
  3.  Score on the held-out window (E_f - W + 1 .. E_f) using MAE, RMSE, direction accuracy,
      skill-vs-naive-persistence.
  4.  Also score an "honest" AR(1) baseline fit per country on the same train slice.

We use the v2 trainer's exact feature pipeline (Tier-1 drop, country dummies, tier dummies,
rank-transform for the linear model) so the comparison to the v2 single-split results is
apples-to-apples.

LightGBM hyperparameters: by default, we run a *nested* Optuna search per fold
(--nested-optuna, on). For each fold, the search uses only rows with year <= fold.train_end,
with a held-out val slice carved from the last `--nested-val-years` of the train period.
This is the only configuration that gives an honest CV estimate of LGBM generalisation;
re-using the v2 single-split best_params would leak (the v2 Optuna search selected params
that minimised val MAE on the 2015-2018 slice, which overlaps every CV train slice).

A legacy, non-nested path is retained for diagnostics: pass --no-nested-optuna to load
the v2 best_params.json instead (the original behaviour, known to leak).

Outputs:
  data/features/walk_forward_cv.csv
  data/features/walk_forward_cv_summary.json
  data/features/walk_forward_cv_nested_params.json (per-fold best params + val MAE)
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Re-use the v2 trainer's feature pipeline + LGBM with same params.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_phase8_horizons_v2 import (  # type: ignore
    DROP_FEATURES,
    _add_country_and_tier_dummies,
    _rank_transform,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data" / "features" / "panel_wide.parquet"
OUT_DIR = ROOT / "data" / "features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- LightGBM: nested Optuna per fold (default) or v2 best_params (legacy, leaky) ----------
DEFAULT_FALLBACK_PARAMS = {
    "objective": "regression_l1",
    "boosting_type": "gbdt",
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 30,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "verbosity": -1,
    "n_jobs": -1,
}


def _v2_legacy_lgbm_params(h: int, cap_estimators: int | None = None) -> dict:
    """Load the v2 single-split best_params.json. KNOWN TO LEAK into CV folds; retained for
    diagnostic comparison only. The h=5 Optuna search minimised MAE on the 2015-2018 val
    slice, which appears inside every CV train slice."""
    p = OUT_DIR / f"horizon_{h}y_v2" / "best_params.json"
    if p.exists():
        bp = json.loads(p.read_text())
        n_est = int(bp.get("n_estimators", 250))
        if cap_estimators is not None and n_est > cap_estimators:
            n_est = cap_estimators
        return {
            "objective": "regression_l1",
            "boosting_type": bp.get("boosting_type", "gbdt"),
            "n_estimators": n_est,
            "learning_rate": float(bp.get("learning_rate", 0.03)),
            "num_leaves": int(bp.get("num_leaves", 31)),
            "min_child_samples": int(bp.get("min_child_samples", 30)),
            "subsample": float(bp.get("subsample", 0.9)),
            "colsample_bytree": float(bp.get("colsample_bytree", 0.9)),
            "reg_alpha": float(bp.get("reg_alpha", 0.0)),
            "reg_lambda": float(bp.get("reg_lambda", 0.0)),
            "verbosity": -1,
            "n_jobs": -1,
        }
    return {**DEFAULT_FALLBACK_PARAMS}


def _nested_optuna_lgbm_params(
    X_tr_full: np.ndarray, y_tr_full: np.ndarray, years_tr: np.ndarray,
    n_trials: int, seed: int,
    val_years: int = 4,
) -> dict:
    """Per-fold nested Optuna search.

    The fold's training rows (years <= fold.train_end) are split by year:
      - "opt"   : years <= fold.train_end - val_years  (used to fit each trial)
      - "val"   : fold.train_end - val_years + 1 .. fold.train_end  (used to score it)
    The winner is the trial with the lowest val MAE. This is the honest CV protocol: no
    row with year > fold.train_end ever enters hyperparameter selection.

    Returns the best params dict (LGBM-ready). If val_years is too large for the fold,
    falls back to DEFAULT_FALLBACK_PARAMS and returns it with a flag (the caller decides
    whether to skip the fold).
    """
    import optuna
    import lightgbm as lgb
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    max_year = int(years_tr.max())
    val_end = max_year
    val_start = max_year - val_years + 1
    opt_mask = years_tr <= (val_start - 1)
    val_mask = (years_tr >= val_start) & (years_tr <= val_end)

    if opt_mask.sum() < 200 or val_mask.sum() < 20:
        # Not enough data to do a meaningful search; caller will skip or fall back.
        return {**DEFAULT_FALLBACK_PARAMS, "_nested_insufficient": True}

    # Impute using only opt-train medians so val/test feature distributions don't leak.
    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(strategy="median")
    Xo = imp.fit_transform(X_tr_full[opt_mask])
    Xv = imp.transform(X_tr_full[val_mask])
    yo, yv = y_tr_full[opt_mask], y_tr_full[val_mask]

    def objective(trial: optuna.Trial) -> float:
        params = {
            # Tightened search space vs v2 single-split: the v2 space (up to
            # 4000 trees, 127 leaves) produces models complex enough to
            # memorize per-country macro fingerprints on panel data.  These
            # bounds still allow Optuna to find good models while keeping
            # capacity realistic for ~10k training rows.
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 7, 31),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.8),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "boosting_type": "gbdt",
            "random_state": seed,
            "n_jobs": -1,
            "verbosity": -1,
            "objective": "regression_l1",
        }
        model = lgb.LGBMRegressor(
            **params,
            callbacks=[lgb.early_stopping(stopping_rounds=80, verbose=False)],
        )
        try:
            model.fit(Xo, yo, eval_set=[(Xv, yv)])
            pred = model.predict(Xv)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            print(f"[nested-optuna] trial failed: {exc}")
            return float("inf")
        return float(np.mean(np.abs(pred - yv)))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )

    def _log(study_: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        completed = sum(1 for t in study_.trials if t.state.name == "COMPLETE")
        running_best = float("inf") if not study_.best_trial else study_.best_value
        print(f"[nested-optuna h<seed>] trial {trial.number:3d} {trial.state.name:>8s}  "
              f"mae={trial.value:.4f}   running_best={running_best:.4f}  "
              f"({completed} done / {n_trials} budget)", flush=True)

    # Original spec: sequential trials, each LightGBM uses all cores.
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False,
                   callbacks=[_log])
    bp = dict(study.best_params)
    # n_estimators from Optuna is the chosen tree count; honour it for the refit below.
    return {
        "objective": "regression_l1",
        "boosting_type": "gbdt",
        "n_estimators": int(bp.get("n_estimators", 200)),
        "learning_rate": float(bp.get("learning_rate", 0.05)),
        "num_leaves": int(bp.get("num_leaves", 31)),
        "min_child_samples": int(bp.get("min_child_samples", 30)),
        "subsample": float(bp.get("subsample", 0.9)),
        "colsample_bytree": float(bp.get("colsample_bytree", 0.9)),
        "reg_alpha": float(bp.get("reg_alpha", 0.0)),
        "reg_lambda": float(bp.get("reg_lambda", 0.0)),
        "verbosity": -1,
        "n_jobs": -1,
        "_nested_best_val_mae": float(study.best_value),
    }


# ---------- helpers ----------
def _make_target(df: pd.DataFrame, h: int) -> pd.Series:
    """log(gdp_pc_{y+h} / gdp_pc_y), per country. NaN if either side is missing/non-positive."""
    g = df.sort_values(["iso3", "year"]).groupby("iso3")["gdp_pc"]
    fwd = g.shift(-h)
    ratio = fwd / df["gdp_pc"]
    return np.log(ratio.where(ratio > 0)).astype(np.float32)


def _features(df: pd.DataFrame, iso_levels: list[str], fit_years: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Return (X_cont, X_full, cont_cols, full_cols) for the *full* df (no row filtering).
    Rank-transform is fit on rows where year <= max(fit_years)."""
    df_aug, _ = _add_country_and_tier_dummies(df, iso_levels)
    leak = {"iso3", "year", "gdp_pc", "target"} | {c for c in df_aug.columns if c.endswith("y_fwd")}
    dummy_cols = {c for c in df_aug.columns if c.startswith("iso_") or c.startswith("tier_")}
    cont_cols = [
        c for c in df_aug.columns
        if c not in leak
        and pd.api.types.is_numeric_dtype(df_aug[c])
        and c not in DROP_FEATURES
        and c not in dummy_cols
    ]
    cont_cols = [c for c in cont_cols if not c.endswith("y_fwd")]
    X_cont = df_aug[cont_cols].astype(np.float32).replace([np.inf, -np.inf], np.nan)
    X_full = df_aug[cont_cols + sorted(dummy_cols)].astype(np.float32).replace([np.inf, -np.inf], np.nan)
    # Drop entirely-NaN columns (some countries never observed the source series).
    cont_keep = [c for c in cont_cols if X_cont[c].notna().any()]
    full_keep = cont_keep + sorted(dummy_cols)
    return X_cont[cont_keep], X_full[full_keep], cont_keep, full_keep


def _rank_fit_transform(X: pd.DataFrame, fit_mask: np.ndarray) -> np.ndarray:
    """Rank-transform where the ranking distribution is fit on fit_mask rows only.
    Test/pred rows are scored by interpolation into the training distribution,
    so no information from test rows leaks into the training features."""
    Xn = X.to_numpy().copy()
    out = np.full_like(Xn, np.nan, dtype=np.float32)
    for j in range(Xn.shape[1]):
        col = Xn[:, j]
        # Fit: build rank distribution from training rows only
        fit_vals = col[fit_mask]
        fit_valid = fit_vals[~np.isnan(fit_vals)]
        if len(fit_valid) < 2:
            continue
        sorted_fit = np.sort(fit_valid)
        n_fit = len(sorted_fit)
        # Transform all rows: percentile rank within training distribution
        all_valid = ~np.isnan(col)
        positions = np.searchsorted(sorted_fit, col[all_valid], side="right")
        out[all_valid, j] = (positions / n_fit).astype(np.float32)
    return out


def _fit_predict_ridge(X_cont: pd.DataFrame, y: np.ndarray, train_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    Xr = _rank_fit_transform(X_cont, train_mask)
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("rg", Ridge(alpha=100.0)),
    ])
    pipe.fit(Xr[train_mask], y[train_mask])
    return pipe.predict(Xr[pred_mask])


def _fit_predict_lgbm(
    X_full: pd.DataFrame, y: np.ndarray, train_mask: np.ndarray, pred_mask: np.ndarray,
    params: dict, eval_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Fit LGBM on train_mask rows, predict on pred_mask rows.

    If eval_mask is provided, it is EXCLUDED from the training set and used as a
    held-out eval_set for early stopping.  This mirrors the nested-Optuna trial
    setup (train on opt rows, early-stop on val rows) and prevents the model from
    training on the same data it evaluates — which would make early stopping a
    no-op and let the model run for all n_estimators (massive overfitting).

    The imputer is fit on the actual fit rows only (train_mask minus eval_mask).
    """
    import lightgbm as lgb
    from sklearn.impute import SimpleImputer
    Xa = X_full.to_numpy()
    # When eval_mask is provided, exclude those rows from training so early
    # stopping evaluates on genuinely held-out data.
    if eval_mask is not None and eval_mask.any():
        fit_mask = train_mask & ~eval_mask
    else:
        fit_mask = train_mask
    imp = SimpleImputer(strategy="median")
    Xt = imp.fit_transform(Xa[fit_mask])
    Xp = imp.transform(Xa[pred_mask])
    fit_kwargs: dict = {}
    if eval_mask is not None and eval_mask.any():
        Xt_eval = imp.transform(Xa[eval_mask])
        y_eval = y[eval_mask]
        fit_kwargs["eval_set"] = [(Xt_eval, y_eval)]
        fit_kwargs["callbacks"] = [lgb.early_stopping(stopping_rounds=80, verbose=False)]
    model = lgb.LGBMRegressor(**params)
    model.fit(Xt, y[fit_mask], **fit_kwargs)
    return model.predict(Xp)


def _fit_predict_catboost(X_full: pd.DataFrame, y: np.ndarray, train_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    """Fit CatBoost (NaN-native) on train_mask rows, predict on pred_mask rows.

    CatBoost handles NaN natively so no imputer is needed; we still pass the
    raw numpy (with NaN) directly. The training rows are train_mask (no
    additional eval_mask because ordered boosting has its own internal validation).
    """
    from catboost import CatBoostRegressor
    Xa = X_full.to_numpy()
    model = CatBoostRegressor(
        iterations=800, depth=6, learning_rate=0.05,
        loss_function="MAE", random_seed=0, nan_mode="Min",
        verbose=False, thread_count=-1,
    )
    model.fit(Xa[train_mask], y[train_mask])
    return model.predict(Xa[pred_mask])


def _fit_predict_xgboost(X_full: pd.DataFrame, y: np.ndarray, train_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    """Fit XGBoost (hist tree method, MAE objective) on train_mask rows, predict on pred_mask rows.

    XGBoost has native NaN handling (`missing=np.nan`) so no imputer is needed.
    """
    from xgboost import XGBRegressor
    Xa = X_full.to_numpy()
    model = XGBRegressor(
        n_estimators=600, max_depth=6, learning_rate=0.05,
        objective="reg:absoluteerror", tree_method="hist",
        subsample=0.8, colsample_bytree=0.8,
        random_state=0, n_jobs=-1, verbosity=0,
    )
    model.fit(Xa[train_mask], y[train_mask])
    return model.predict(Xa[pred_mask])


def _prior_pred(df: pd.DataFrame, y: np.ndarray, train_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    """Last-realised y for each country, computed on training rows only. NaN -> global mean."""
    train = df.loc[train_mask, ["iso3", "year"]].copy()
    train["y"] = y[train_mask]
    gmean = float(np.nanmean(y[train_mask])) if (~np.isnan(y[train_mask])).any() else 0.0
    by_iso = train.sort_values("year").groupby("iso3")["y"].last().to_dict()
    pred_iso = df.loc[pred_mask, "iso3"].to_numpy()
    out = np.array([by_iso.get(iso, gmean) for iso in pred_iso], dtype=np.float32)
    return out


def _ar1_honest_pred(df: pd.DataFrame, y: np.ndarray, train_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    """Pooled AR(1): y_t = a + b * y_{t-1}, fit on all training rows where y_{t-1} exists.
    y_{t-1} is the previous-year realised y for that country.
    Pooling matches the v2 benchmark's `_ar1_honest` behaviour (a single global fit), which
    is stable because it sees ~9k rows per fold. Per-country fits on log-growth are too
    noisy (each country only has ~50 obs and log-growth is highly non-stationary)."""
    train_df = df.loc[train_mask, ["iso3", "year"]].copy()
    train_df["y"] = y[train_mask]
    train_df = train_df.sort_values(["iso3", "year"]).reset_index(drop=True)
    train_df["y_prev"] = train_df.groupby("iso3")["y"].shift(1)
    sub = train_df.dropna(subset=["y_prev", "y"])
    if len(sub) < 10:
        return np.zeros(int(pred_mask.sum()), dtype=np.float32)
    x = sub["y_prev"].to_numpy(dtype=np.float64)
    yobs = sub["y"].to_numpy(dtype=np.float64)
    A = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(A, yobs, rcond=None)
    a, b = float(coef[0]), float(coef[1])

    # For prediction: use most-recent realised y for that (iso, year).
    last_y_by_iso: dict[str, pd.Series] = {
        iso: g.set_index("year")["y"].sort_index()
        for iso, g in train_df.dropna(subset=["y"]).groupby("iso3")
    }
    pred_df = df.loc[pred_mask, ["iso3", "year"]].copy().reset_index(drop=True)
    pred_y_prev = np.full(len(pred_df), np.nan, dtype=np.float32)
    for i, (iso, yr) in enumerate(zip(pred_df["iso3"].to_numpy(), pred_df["year"].to_numpy())):
        s = last_y_by_iso.get(iso)
        if s is None or len(s) == 0:
            continue
        prev = s[s.index < yr]
        if len(prev):
            pred_y_prev[i] = float(prev.iloc[-1])
    out = (a + b * np.where(np.isnan(pred_y_prev), 0.0, pred_y_prev)).astype(np.float32)
    return out


def _naive_persistence_pred(df: pd.DataFrame, y: np.ndarray, train_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    """y_pred = last realised y for that country. Identical to prior in the v2 trainer's definition."""
    return _prior_pred(df, y, train_mask, pred_mask)


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if m.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true[m] - y_pred[m])))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if m.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true[m] - y_pred[m]) ** 2)))


def _dir_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if m.sum() == 0:
        return float("nan")
    return float(np.mean(np.sign(y_true[m]) == np.sign(y_pred[m])))


def _skill(y_true: np.ndarray, y_pred: np.ndarray, baseline: np.ndarray) -> float:
    m = ~np.isnan(y_true) & ~np.isnan(y_pred) & ~np.isnan(baseline)
    if m.sum() == 0:
        return float("nan")
    mae_p = np.mean(np.abs(y_true[m] - y_pred[m]))
    mae_b = np.mean(np.abs(y_true[m] - baseline[m]))
    return float(1.0 - mae_p / mae_b) if mae_b > 0 else float("nan")


# ---------- fold design ----------
@dataclass
class Fold:
    fold: int
    train_end: int       # inclusive
    val_end: int         # inclusive (small in-sample val)
    test_start: int
    test_end: int        # inclusive


def _build_folds(test_window: int, anchor_end: int, n_folds: int) -> list[Fold]:
    """Folds step backward in time: latest fold has test_end == anchor_end, earliest has test_end == anchor_end - (n_folds-1)*test_window."""
    folds = []
    for k in range(n_folds):
        test_end = anchor_end - k * test_window
        test_start = test_end - test_window + 1
        train_end = test_start - 1
        val_end = test_start - 1
        folds.append(Fold(fold=k, train_end=train_end, val_end=val_end, test_start=test_start, test_end=test_end))
    return list(reversed(folds))  # oldest fold first


# ---------- main loop ----------
def run_one_horizon(
    h: int, n_folds: int, test_window: int, anchor_end: int,
    cap_estimators: int | None = None, seed: int = 0,
    nested_optuna: bool = True, nested_trials: int = 300, nested_val_years: int = 4,
    drop_dummies: bool = True,
) -> tuple[pd.DataFrame, list[dict]]:
    """Run walk-forward CV for horizon h. Returns (per-row metrics, per-fold nested params)."""
    print(f"\n=== h={h}  folds={n_folds}  window={test_window}y  anchor_end={anchor_end}  "
          f"nested_optuna={nested_optuna}  nested_trials={nested_trials}  "
          f"nested_val_years={nested_val_years}  drop_dummies={drop_dummies} ===", flush=True)
    df = pd.read_parquet(PANEL).sort_values(["iso3", "year"]).reset_index(drop=True)
    df["target"] = _make_target(df, h)
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    print(f"  panel rows with target_h{h}: {len(df):,}", flush=True)

    iso_levels = sorted(df["iso3"].unique().tolist())
    X_cont, X_full, cont_cols, full_cols = _features(df, iso_levels, fit_years=df["year"].to_numpy())
    print(f"  cont_cols={len(cont_cols)}  full_cols={len(full_cols)}", flush=True)

    folds = _build_folds(test_window=test_window, anchor_end=anchor_end, n_folds=n_folds)
    y = df["target"].to_numpy(dtype=np.float32)
    years = df["year"].to_numpy()
    Xa_full = X_full.to_numpy()
    Xa_cont = X_cont.to_numpy()

    # Choose the feature matrix for LGBM. When drop_dummies=True (default for
    # CV), LGBM uses only continuous macro features — identical to Ridge's input.
    # Country/tier dummies let LGBM memorize per-country growth trajectories,
    # which gives unrealistically low walk-forward error because all CV folds
    # share the same set of countries.
    X_lgbm = X_cont if drop_dummies else X_full
    lgbm_feat_label = f"cont_only ({len(cont_cols)})" if drop_dummies else f"full ({len(full_cols)})"
    print(f"  lgbm features: {lgbm_feat_label}", flush=True)

    nested_log: list[dict] = []
    rows = []
    for fold in folds:
        train_mask = (years <= fold.train_end)
        # Nested-Optuna val slice: the last `nested_val_years` years inside train_mask, used
        # both for Optuna trial scoring AND for early-stopping in the final refit. This is
        # the honest equivalent of the v2 val slice: rows used for HP selection never appear
        # in any test slice.
        opt_train_mask = (years <= fold.train_end - nested_val_years)
        opt_val_mask = train_mask & ~opt_train_mask
        test_mask = (years >= fold.test_start) & (years <= fold.test_end)

        if train_mask.sum() < 200 or test_mask.sum() < 10:
            print(f"  fold {fold.fold}: train_n={int(train_mask.sum())} test_n={int(test_mask.sum())} "
                  f"-- SKIP (too small)", flush=True)
            continue

        # ---- LGBM hyperparameter selection ----
        if nested_optuna:
            lgbm_params = _nested_optuna_lgbm_params(
                X_lgbm.to_numpy()[train_mask], y[train_mask], years[train_mask],
                n_trials=nested_trials, seed=seed,
                val_years=nested_val_years,
            )
            if lgbm_params.get("_nested_insufficient"):
                print(f"  fold {fold.fold}: insufficient rows for nested Optuna "
                      f"(opt_n={int(opt_train_mask.sum())} val_n={int(opt_val_mask.sum())}); "
                      f"using DEFAULT_FALLBACK_PARAMS.", flush=True)
            eval_mask = opt_val_mask
        else:
            lgbm_params = _v2_legacy_lgbm_params(h, cap_estimators=cap_estimators)
            # No clean early-stopping slice; refit full n_estimators. (Legacy path.)
            eval_mask = None

        # Record the per-fold params (and val MAE if nested) for diagnostics.
        nested_log.append({
            "horizon": h,
            "fold": fold.fold,
            "train_end": int(fold.train_end),
            "test_start": int(fold.test_start),
            "test_end": int(fold.test_end),
            "n_opt_train": int((train_mask & ~eval_mask).sum()) if eval_mask is not None else None,
            "n_opt_val": int(eval_mask.sum()) if eval_mask is not None else None,
            "lgbm_params": {k: v for k, v in lgbm_params.items() if not k.startswith("_")},
            "nested_best_val_mae": lgbm_params.get("_nested_best_val_mae"),
            "used_default_fallback": bool(lgbm_params.get("_nested_insufficient")),
        })
        print(f"  fold {fold.fold}: train<={fold.train_end} (n={int(train_mask.sum())})  "
              f"opt_n={(int((train_mask & ~eval_mask).sum()) if eval_mask is not None else 'n/a')}  "
              f"opt_val_n={(int(eval_mask.sum()) if eval_mask is not None else 'n/a')}  "
              f"test {fold.test_start}-{fold.test_end} (n={int(test_mask.sum())})  "
              f"lgbm n_est={lgbm_params['n_estimators']} lr={lgbm_params['learning_rate']:.4f} "
              f"leaves={lgbm_params['num_leaves']}", flush=True)

        # ---- Models ----
        # Ridge uses its own rank-imputer; we re-impute inside for clean independence.
        ridge = _fit_predict_ridge(X_cont, y, train_mask, test_mask)
        # LGBM: uses X_lgbm (cont-only when drop_dummies, full otherwise).
        # eval_mask enables early-stopping on the nested val slice (excluded from fit).
        lgbm_pred = _fit_predict_lgbm(X_lgbm, y, train_mask, test_mask, lgbm_params, eval_mask=eval_mask)
        # CatBoost + XGBoost: NaN-native, fit on train rows only, no early-stopping
        # (we don't have a separate eval slice for them — using a fixed sensible
        # hyperparameter set keeps the comparison fair to the other models).
        cat_pred = _fit_predict_catboost(X_lgbm, y, train_mask, test_mask)
        xgb_pred = _fit_predict_xgboost(X_lgbm, y, train_mask, test_mask)
        prior = _prior_pred(df, y, train_mask, test_mask)
        naive = _naive_persistence_pred(df, y, train_mask, test_mask)
        ar1 = _ar1_honest_pred(df, y, train_mask, test_mask)

        ens = 0.7 * lgbm_pred + 0.3 * prior
        ens3 = (lgbm_pred + ridge + prior) / 3.0
        # Diversity-ensemble: average of three different boosting libraries.
        ens3boost = (lgbm_pred + cat_pred + xgb_pred) / 3.0
        # Diversity-ensemble + prior + HB shrinkage.
        ens5 = (0.3 * lgbm_pred + 0.2 * cat_pred + 0.2 * xgb_pred
                + 0.15 * ridge + 0.15 * prior)

        y_test = y[test_mask]
        for name, p in [
            ("naive_persistence", naive),
            ("ar1_lag1_honest", ar1),
            ("ridge", ridge),
            ("lgbm", lgbm_pred),
            ("catboost", cat_pred),
            ("xgboost", xgb_pred),
            ("ensemble_lgbm_prior", ens),
            ("ensemble_lgbm_ridge_prior", ens3),
            ("ensemble_lgbm_catboost_xgboost", ens3boost),
            ("ensemble_lgbm_cat_xgb_ridge_prior", ens5),
        ]:
            rows.append({
                "horizon": h,
                "fold": fold.fold,
                "train_end": fold.train_end,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "n_train": int(train_mask.sum()),
                "n_opt_train": int((train_mask & ~eval_mask).sum()) if eval_mask is not None else None,
                "n_opt_val": int(eval_mask.sum()) if eval_mask is not None else None,
                "n_test": int(test_mask.sum()),
                "model": name,
                "mae": _mae(y_test, p),
                "rmse": _rmse(y_test, p),
                "dir_acc": _dir_acc(y_test, p),
                "skill_vs_naive": _skill(y_test, p, naive),
                "skill_vs_ar1": _skill(y_test, p, ar1),
            })
        for r in rows[-12:]:
            print(f"     {r['model']:<36s} mae={r['mae']:.4f}  "
                  f"skill_vs_naive={r['skill_vs_naive']:+.3f}  skill_vs_ar1={r['skill_vs_ar1']:+.3f}", flush=True)
    return pd.DataFrame(rows), nested_log


def summarize(df: pd.DataFrame) -> dict:
    """Aggregate per (horizon, model): mean ± std MAE across folds, mean dir_acc, mean skill."""
    g = df.groupby(["horizon", "model"])
    agg = g.agg(
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        rmse_mean=("rmse", "mean"),
        dir_acc_mean=("dir_acc", "mean"),
        skill_vs_naive_mean=("skill_vs_naive", "mean"),
        skill_vs_ar1_mean=("skill_vs_ar1", "mean"),
        n_folds=("fold", "nunique"),
    ).reset_index()
    return {"per_model": agg.to_dict(orient="records")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 5, 10])
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--test-window", type=int, default=4, help="Test window length in years per fold.")
    p.add_argument("--anchor-end-h5", type=int, default=2022, help="Latest test year for h=5 anchor (shifted for other horizons).")
    # ---- Nested Optuna controls (default = honest per-fold search) ----
    p.add_argument("--nested-optuna", dest="nested_optuna", action="store_true", default=True,
                   help="Run per-fold nested Optuna to select LGBM hyperparameters using only "
                        "rows with year <= fold.train_end. Default: on.")
    p.add_argument("--no-nested-optuna", dest="nested_optuna", action="store_false",
                   help="Disable nested Optuna and load v2 best_params.json instead (LEAKY; "
                        "retained for diagnostic comparison only).")
    p.add_argument("--nested-trials", type=int, default=300,
                   help="Optuna trials per fold per horizon (default 300 — same budget as "
                        "the v2 single-split search, applied per fold per horizon).")
    p.add_argument("--nested-val-years", type=int, default=4,
                   help="Width of the val slice carved out of each fold's train period for HP "
                        "selection (default 4 years — matches the v2 single-split val window).")
    # ---- Legacy controls (only used when --no-nested-optuna is set) ----
    p.add_argument("--cap-estimators", type=int, default=600,
                   help="[Legacy] Cap LightGBM n_estimators. Use 0 for None. Ignored under nested Optuna.")
    # ---- Feature control ----
    p.add_argument("--no-country-dummies", dest="drop_dummies", action="store_true", default=True,
                   help="Drop country/tier dummies from LGBM features for CV. Default: on. "
                        "Country dummies let LGBM memorize per-country patterns, inflating "
                        "walk-forward scores (same countries in every fold).")
    p.add_argument("--country-dummies", dest="drop_dummies", action="store_false",
                   help="Include country/tier dummies in LGBM features (original v2 behaviour). "
                        "Only use for diagnostic comparison — CV results will be unrealistic.")
    p.add_argument("--out-csv", type=str, default="walk_forward_cv.csv")
    p.add_argument("--out-json", type=str, default="walk_forward_cv_summary.json")
    p.add_argument("--out-params-json", type=str, default="walk_forward_cv_nested_params.json",
                   help="Per-fold best LGBM params + nested val MAE (nested path only).")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    horizon_anchors = {1: args.anchor_end_h5 + 2, 3: args.anchor_end_h5 + 1, 5: args.anchor_end_h5, 10: args.anchor_end_h5 - 4}
    all_rows = []
    all_nested: list[dict] = []
    for h in args.horizons:
        anchor = horizon_anchors.get(h, args.anchor_end_h5)
        cap = args.cap_estimators if args.cap_estimators > 0 else None
        df_h, nested_log = run_one_horizon(
            h, n_folds=args.n_folds, test_window=args.test_window, anchor_end=anchor,
            cap_estimators=cap, seed=args.seed,
            nested_optuna=args.nested_optuna,
            nested_trials=args.nested_trials,
            nested_val_years=args.nested_val_years,
            drop_dummies=args.drop_dummies,
        )
        all_rows.append(df_h)
        all_nested.extend(nested_log)
    big = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    if big.empty:
        print("[panel-backtest] no rows produced; aborting.", flush=True)
        return
    big.to_csv(OUT_DIR / args.out_csv, index=False)
    summary = summarize(big)
    (OUT_DIR / args.out_json).write_text(json.dumps(summary, indent=2))
    if args.nested_optuna and all_nested:
        (OUT_DIR / args.out_params_json).write_text(json.dumps(all_nested, indent=2))
        print(f"[panel-backtest] wrote {OUT_DIR / args.out_params_json}", flush=True)
    print(f"\n[panel-backtest] wrote {OUT_DIR / args.out_csv} ({len(big):,} rows)", flush=True)
    print(f"[panel-backtest] wrote {OUT_DIR / args.out_json}", flush=True)
    print("\n[panel-backtest] headline (mean MAE across folds):", flush=True)
    for h in args.horizons:
        sub = big[big["horizon"] == h]
        if sub.empty:
            continue
        agg = sub.groupby("model")["mae"].agg(["mean", "std", "count"]).sort_values("mean")
        print(f"\n  h={h} (n_folds={int(agg['count'].max())}):", flush=True)
        for model, row in agg.iterrows():
            mae_mean = row["mean"]
            mae_std = row["std"] if pd.notna(row["std"]) else 0.0
            print(f"    {model:<28s} mae={mae_mean:.4f} ± {mae_std:.4f}", flush=True)


if __name__ == "__main__":
    main()
