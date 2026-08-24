"""Quick diagnostic: does LGBM early-stop, and what's the effective tree count?"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.impute import SimpleImputer

from run_phase8_horizons_v2 import DROP_FEATURES, _add_country_and_tier_dummies
from _panel_backtest import _make_target, _features

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data" / "features" / "panel_wide.parquet"

df = pd.read_parquet(PANEL).sort_values(["iso3", "year"]).reset_index(drop=True)
df["target"] = _make_target(df, h=5)
df = df.dropna(subset=["target"]).reset_index(drop=True)

iso_levels = sorted(df["iso3"].unique().tolist())
X_cont, X_full, cont_cols, full_cols = _features(df, iso_levels, fit_years=df["year"].to_numpy())
y = df["target"].to_numpy(dtype=np.float32)
years = df["year"].to_numpy()

# Fold 1 setup
train_mask = years <= 2014
opt_train_mask = years <= 2010
opt_val_mask = train_mask & ~opt_train_mask
test_mask = (years >= 2015) & (years <= 2018)

Xa = X_cont.to_numpy()
fit_mask = opt_train_mask

imp = SimpleImputer(strategy="median")
Xt = imp.fit_transform(Xa[fit_mask])
Xv = imp.transform(Xa[opt_val_mask])
Xtest = imp.transform(Xa[test_mask])

print(f"Train (opt) rows: {Xt.shape[0]}, Val rows: {Xv.shape[0]}, Test rows: {Xtest.shape[0]}")
print(f"Features: {Xt.shape[1]} (cont-only, no dummies)")

# Overfit params from Optuna
params = {
    "objective": "regression_l1",
    "boosting_type": "gbdt",
    "n_estimators": 3866,
    "learning_rate": 0.0282,
    "num_leaves": 104,
    "min_child_samples": 55,
    "subsample": 0.784,
    "colsample_bytree": 0.963,
    "reg_alpha": 0.002,
    "reg_lambda": 0.002,
    "verbosity": -1,
    "n_jobs": -1,
}

print("\n--- Overfit params (Optuna-selected) ---")
model = lgb.LGBMRegressor(**params)
model.fit(Xt, y[fit_mask], eval_set=[(Xv, y[opt_val_mask])],
          callbacks=[lgb.early_stopping(stopping_rounds=80, verbose=False)])

print(f"n_estimators requested: {params['n_estimators']}")
print(f"best_iteration_: {model.best_iteration_}")
pred_train = model.predict(Xt)
pred_val = model.predict(Xv)
pred_test = model.predict(Xtest)
print(f"Train MAE: {np.mean(np.abs(pred_train - y[fit_mask])):.4f}")
print(f"Val MAE:   {np.mean(np.abs(pred_val - y[opt_val_mask])):.4f}")
print(f"Test MAE:  {np.mean(np.abs(pred_test - y[test_mask])):.4f}")

# Strongly regularized
print("\n--- Strongly regularized params ---")
params_reg = {
    "objective": "regression_l1",
    "boosting_type": "gbdt",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 100,
    "subsample": 0.7,
    "colsample_bytree": 0.5,
    "reg_alpha": 5.0,
    "reg_lambda": 5.0,
    "verbosity": -1,
    "n_jobs": -1,
}
model2 = lgb.LGBMRegressor(**params_reg)
model2.fit(Xt, y[fit_mask], eval_set=[(Xv, y[opt_val_mask])],
           callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
pred_train2 = model2.predict(Xt)
pred_val2 = model2.predict(Xv)
pred_test2 = model2.predict(Xtest)
print(f"best_iteration_: {model2.best_iteration_}")
print(f"Train MAE: {np.mean(np.abs(pred_train2 - y[fit_mask])):.4f}")
print(f"Val MAE:   {np.mean(np.abs(pred_val2 - y[opt_val_mask])):.4f}")
print(f"Test MAE:  {np.mean(np.abs(pred_test2 - y[test_mask])):.4f}")
print(f"(AR1 baseline test MAE for context: ~0.084)")
