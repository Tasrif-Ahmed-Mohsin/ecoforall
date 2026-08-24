"""Check the magnitude of the h=1 target distribution inside the nested val slice."""
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, r"E:\project_gmd\scripts")
from _panel_backtest import _make_target

panel = pd.read_parquet(r"E:\project_gmd\data\features\panel_wide.parquet")
panel = panel.sort_values(["iso3", "year"]).reset_index(drop=True)

# h=1 fold 0 nested val slice for anchor=2024: years 2018..2021 (train_end = 2024-4 = 2020? No.)
# _build_folds: anchor_end=2024, test_window=4. Fold 0 (latest): test_end=2024, test_start=2021,
# train_end = 2020. val_years=4 -> nested val = years <= train_end - 4 = 2016, i.e. years 2017..2020.
# Wait the code is years <= train_end - val_years = 2016. So opt train ends 2016, val = 2017..2020.

# Recreate the nested val slice for h=1 fold 0:
years = panel["year"].to_numpy()
mask_train = years <= 2020
mask_val = (years <= 2020) & (years > 2016)
print(f"opt_train rows (years<=2016): {mask_train.sum() - mask_val.sum()}")
print(f"nested_val rows (2017..2020): {mask_val.sum()}")

target = _make_target(panel, 1)
y_train = target[mask_train].dropna()
y_val = target[mask_val].dropna()
print(f"\ntarget_h1 distribution:")
print(f"  train y: n={len(y_train)}  mean={y_train.mean():.4f}  std={y_train.std():.4f}  |y|<0.01 frac={float((y_train.abs()<0.01).mean()):.4f}")
print(f"  val   y: n={len(y_val)}  mean={y_val.mean():.4f}  std={y_val.std():.4f}  |y|<0.01 frac={float((y_val.abs()<0.01).mean()):.4f}")

# A persistent-naive predictor (y_pred = 0) would achieve MAE = mean|y|.
print(f"\nNaive (y_pred=0) MAE on val: {y_val.abs().mean():.4f}")
print(f"Naive (y_pred=mean(train)) MAE on val: {(y_val - y_train.mean()).abs().mean():.4f}")

# What's the "honest" AR(1)-style prediction MAE we should expect on this val slice?
# y_{t+1} ~ y_t.  Compute it.
df = panel.loc[mask_val].copy()
df["y"] = target[mask_val].values
df = df.sort_values(["iso3", "year"])
df["y_prev"] = df.groupby("iso3")["y"].shift(1)
sub = df.dropna(subset=["y", "y_prev"])
print(f"\nAR(1) honest MAE on val: {(sub['y'] - sub['y_prev']).abs().mean():.4f}")

# Compare against the v2 single-split test MAE.
v2_test = target[(years > 2018) & (years <= 2022)].dropna()
print(f"v2 single-split test (2019..2022) MAE mean|y|: {v2_test.abs().mean():.4f}")