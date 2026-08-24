"""Phase 8, step 4 — feature-group ablation at horizon h.

Drops one feature group at a time, retrains Ridge + LightGBM at horizon h, and
reports skill change vs the full-feature baseline.

Groups:
  credit         bank_debt, total_loans
  wages_ineq     real_wage_jst, gini_income, gini_wealth
  asset_prices   equity_total_return, equity_capital_gain, equity_div_yield,
                 housing_capital_gain
  money_rates    short_rate, long_rate, real_interest_rate
  trade_fx       trade_gdp, current_account_gdp, fx_to_usd
  demographics   population
  macro          gdp_pc_real, inflation_cpi, gov_debt_gdp, gov_balance_dom,
                 social_spending

Run for one horizon (default h=5). Writes `data/features/ablation_table.csv`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.harmonize.common import FEATURES_DIR

PANEL = FEATURES_DIR / "panel_wide.parquet"

GROUPS: dict[str, list[str]] = {
    "credit": ["bank_debt", "total_loans"],
    "wages_ineq": ["real_wage_jst", "gini_income", "gini_wealth"],
    "asset_prices": [
        "equity_total_return", "equity_capital_gain", "equity_div_yield",
        "housing_capital_gain",
    ],
    "money_rates": ["short_rate", "long_rate", "real_interest_rate"],
    "trade_fx": ["trade_gdp", "current_account_gdp", "fx_to_usd"],
    "demographics": ["population"],
    "macro": [
        "gdp_pc_real", "inflation_cpi", "gov_debt_gdp", "gov_balance_dom",
        "social_spending",
    ],
}

# Anchors (same as run_phase8_horizons.py)
H5_TRAIN_END = 2014
H5_VAL_END = 2018
H5_TEST_END = 2022


def _build_target(panel: pd.DataFrame, h: int) -> pd.Series:
    panel = panel.sort_values(["iso3", "year"]).reset_index(drop=True)
    g_fwd = panel.groupby("iso3")["gdp_pc"].shift(-h)
    return np.log(g_fwd / panel["gdp_pc"]).rename(f"gdp_pc_growth_{h}y_fwd")


def _ablate(panel: pd.DataFrame, h: int, drop_cols: list[str]) -> dict:
    target = f"gdp_pc_growth_{h}y_fwd"
    df = panel.dropna(subset=[target]).reset_index(drop=True)

    shift = max(0, h - 5)
    train_end = H5_TRAIN_END - shift
    val_end = H5_VAL_END - shift
    test_end = H5_TEST_END - shift

    leak = {"iso3", "year", "gdp_pc"} | {c for c in df.columns if c.endswith("y_fwd")}
    feat_cols = [
        c for c in df.columns
        if c not in leak and c != "gdp_pc_growth_5y_fwd"
        and pd.api.types.is_numeric_dtype(df[c])
        and c not in drop_cols
    ]
    X = df[feat_cols].astype(np.float32).replace([np.inf, -np.inf], np.nan)
    y = df[target].astype(np.float32).to_numpy()
    years = df["year"].to_numpy()

    train = years <= train_end
    val = (years > train_end) & (years <= val_end)
    test = (years > val_end) & (years <= test_end)

    # Drop all-NaN training columns
    non_nan = X[train].notna().any(axis=0).to_numpy()
    kept_idx = [i for i, keep in enumerate(non_nan) if keep]
    if not kept_idx:
        return {"n_features": 0, "mae": float("nan"), "rmse": float("nan"),
                "dir_acc": float("nan"), "skill_vs_rw": float("nan")}
    X_kept = X.iloc[:, kept_idx]

    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline

    ridge = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0, random_state=0)),
    ])
    ridge.fit(X_kept[train].to_numpy(), y[train])

    import lightgbm as lgb
    lgbm = lgb.LGBMRegressor(
        n_estimators=2000, learning_rate=0.03, num_leaves=31,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        random_state=0, n_jobs=-1, verbosity=-1,
    )
    lgbm.fit(X_kept[train].to_numpy(), y[train],
             eval_set=[(X_kept[val].to_numpy(), y[val])],
             callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])

    ytest = y[test]
    pred_ridge = ridge.predict(X_kept[test].to_numpy())
    pred_lgbm = lgbm.predict(X_kept[test].to_numpy())
    pred_ens = 0.7 * pred_lgbm + 0.3 * pred_ridge

    err = pred_ens - ytest
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    dir_acc = float(np.mean(np.sign(pred_ens) == np.sign(ytest)))
    rw_mae = float(np.mean(np.abs(ytest)))
    return {
        "n_features": int(len(feat_cols)),
        "mae": mae,
        "rmse": rmse,
        "dir_acc": dir_acc,
        "skill_vs_rw": float(1 - mae / rw_mae) if rw_mae > 0 else float("nan"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--panel", type=Path, default=PANEL)
    p.add_argument("--out", type=Path, default=FEATURES_DIR / "ablation_table.csv")
    args = p.parse_args()

    panel = pd.read_parquet(args.panel)
    target = f"gdp_pc_growth_{args.horizon}y_fwd"
    if target not in panel.columns:
        panel[target] = _build_target(panel, args.horizon)

    # Full baseline (no drops)
    print(f"[ablation h={args.horizon}y] running full baseline...")
    full = _ablate(panel, args.horizon, drop_cols=[])
    full["group"] = "(none) full"
    full["dropped_features"] = ""

    rows = [full]
    for grp, cols in GROUPS.items():
        present = [c for c in cols if c in panel.columns]
        if not present:
            print(f"[ablation h={args.horizon}y] {grp}: no features present in panel, skip")
            continue
        print(f"[ablation h={args.horizon}y] dropping {grp} ({len(present)} cols)...")
        r = _ablate(panel, args.horizon, drop_cols=present)
        r["group"] = grp
        r["dropped_features"] = ",".join(present)
        rows.append(r)
        print(f"  -> n_feat={r['n_features']}  MAE={r['mae']:.4f}  RMSE={r['rmse']:.4f}  "
              f"dir_acc={r['dir_acc']:.3f}  skill_vs_rw={r['skill_vs_rw']:+.4f}")

    table = pd.DataFrame(rows)[["group", "n_features", "mae", "rmse", "dir_acc",
                                 "skill_vs_rw", "dropped_features"]]
    # Add delta vs full
    table["delta_mae_vs_full"] = table["mae"] - full["mae"]
    table["delta_skill_vs_full"] = table["skill_vs_rw"] - full["skill_vs_rw"]
    # Rank by skill loss (positive delta_mae = group was helping)
    table = table.sort_values("delta_mae_vs_full", ascending=False)

    table.to_csv(args.out, index=False)
    print(f"\n[ablation] wrote {args.out}")
    print(f"\n[ablation h={args.horizon}y] ranked by MAE change vs full:")
    print(table[["group", "n_features", "mae", "delta_mae_vs_full",
                 "skill_vs_rw", "delta_skill_vs_full"]].to_string(index=False))


if __name__ == "__main__":
    main()