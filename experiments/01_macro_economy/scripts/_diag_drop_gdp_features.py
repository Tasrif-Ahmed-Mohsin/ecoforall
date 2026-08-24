"""Quick diagnostic: does dropping all gdp_pc-derived features from cont_cols
collapse the implausibly low LGBM MAE to a realistic range?

If yes → the gdp_pc features are the root cause of the 0.002-0.005 MAE.
If no  → the leak is elsewhere (target autocorrelation, panel structure, etc.)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.impute import SimpleImputer

from run_phase8_horizons_v2 import DROP_FEATURES, _add_country_and_tier_dummies
from _panel_backtest import _make_target, _features, _mae, _dir_acc

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data" / "features" / "panel_wide.parquet"

df = pd.read_parquet(PANEL).sort_values(["iso3", "year"]).reset_index(drop=True)
df["target"] = _make_target(df, h=5)
df = df.dropna(subset=["target"]).reset_index(drop=True)

iso_levels = sorted(df["iso3"].unique().tolist())
X_cont, X_full, cont_cols, full_cols = _features(df, iso_levels, fit_years=df["year"].to_numpy())
y = df["target"].to_numpy(dtype=np.float32)
years = df["year"].to_numpy()

# Identify gdp_pc-derived columns
gdp_cols = [c for c in cont_cols if "gdp_pc" in c.lower()]
clean_cols = [c for c in cont_cols if c not in gdp_cols]

print(f"Original cont_cols: {len(cont_cols)}")
print(f"gdp_pc-derived cols: {len(gdp_cols)}: {gdp_cols}")
print(f"Clean cont_cols (no gdp_pc): {len(clean_cols)}")

# Simple fold: train <= 2014, val 2015-2018, test 2019-2022
train_mask = years <= 2010
val_mask = (years >= 2011) & (years <= 2014)
test_mask = (years >= 2015) & (years <= 2018)

print(f"\nTrain n={train_mask.sum()}, Val n={val_mask.sum()}, Test n={test_mask.sum()}")

# Regularized params (not Optuna-overfit)
params = {
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

for label, cols in [("WITH gdp_pc features", cont_cols), ("WITHOUT gdp_pc features", clean_cols)]:
    X = X_cont[cols].to_numpy()
    imp = SimpleImputer(strategy="median")
    Xt = imp.fit_transform(X[train_mask])
    Xv = imp.transform(X[val_mask])
    Xte = imp.transform(X[test_mask])

    model = lgb.LGBMRegressor(**params)
    model.fit(Xt, y[train_mask],
              eval_set=[(Xv, y[val_mask])],
              callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])

    pred_train = model.predict(Xt)
    pred_val = model.predict(Xv)
    pred_test = model.predict(Xte)

    print(f"\n--- {label} ({len(cols)} features) ---")
    print(f"  best_iteration: {model.best_iteration_}")
    print(f"  Train MAE: {_mae(y[train_mask], pred_train):.4f}")
    print(f"  Val MAE:   {_mae(y[val_mask], pred_val):.4f}")
    print(f"  Test MAE:  {_mae(y[test_mask], pred_test):.4f}")
    print(f"  Train dir_acc: {_dir_acc(y[train_mask], pred_train):.4f}")
    print(f"  Val dir_acc:   {_dir_acc(y[val_mask], pred_val):.4f}")
    print(f"  Test dir_acc:  {_dir_acc(y[test_mask], pred_test):.4f}")

    if label.startswith("WITH"):
        # Show top 10 feature importances
        imp_df = pd.DataFrame({
            "feature": cols,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False).head(15)
        print("\n  Top 15 feature importances:")
        for _, row in imp_df.iterrows():
            marker = " *** GDP_PC ***" if "gdp_pc" in row["feature"].lower() else ""
            print(f"    {row['feature']:40s} {int(row['importance']):6d}{marker}")
