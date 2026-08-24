"""Counterfactual: meta-ensemble WITH vs WITHOUT llm_pred feature.

Used to expose the LLM's actual contribution in MAE terms (instead of just
the linear weight decomposition, which is dominated by the integer `horizon`
column).
"""
import sys
sys.path.insert(0, ".")
import json
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from scripts._cross_horizon_ensemble import (
    _load_llm_snapshot, build_meta_dataset, _metrics
)


def fit_and_score(meta_df, label):
    feature_cols = []
    for h in [1, 3, 5, 10]:
        feature_cols += [f"ridge_h{h}", f"lgbm_h{h}", f"prior_h{h}"]
    if "ar1" in meta_df.columns:
        feature_cols.append("ar1")
    if "llm_pred" in meta_df.columns and meta_df["llm_pred"].notna().any():
        feature_cols.append("llm_pred")
    feature_cols += ["horizon"]

    train_mask = meta_df["split"] == "val"
    test_mask = meta_df["split"] == "test"

    X_tr = meta_df.loc[train_mask, feature_cols]
    X_te = meta_df.loc[test_mask, feature_cols]
    y_tr = meta_df.loc[train_mask, "y_true"].to_numpy()
    y_te = meta_df.loc[test_mask, "y_true"].to_numpy()

    try:
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    except TypeError:
        imputer = SimpleImputer(strategy="median")
    pipe = Pipeline([("imputer", imputer),
                      ("scaler", StandardScaler()),
                      ("ridge", RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], cv=5))])
    pipe.fit(X_tr, y_tr)
    pred_te = pipe.predict(X_te)

    per_h = {}
    for h in [1, 3, 5, 10]:
        m = (meta_df.loc[test_mask, "horizon"] == h).to_numpy()
        if not m.any():
            continue
        sub = meta_df.loc[test_mask].loc[m]
        per_h[f"h{h}"] = {
            "n": int(m.sum()),
            "meta_mae": float(mean_absolute_error(sub.y_true, pred_te[m])),
            "lgbm_mae": float(mean_absolute_error(sub.y_true, sub[f"lgbm_h{h}"])),
            "ridge_mae": float(mean_absolute_error(sub.y_true, sub[f"ridge_h{h}"])),
            "prior_mae": float(mean_absolute_error(sub.y_true, sub[f"prior_h{h}"])),
        }
        if "llm_pred" in meta_df.columns:
            mask = sub["llm_pred"].notna()
            if mask.any():
                per_h[f"h{h}"]["llm_mae"] = float(mean_absolute_error(
                    sub.loc[mask, "y_true"], sub.loc[mask, "llm_pred"]))

    print(f"\n=== {label} ===")
    print(f"features ({len(feature_cols)}): {feature_cols}")
    print(f"{'h':<6} {'n':>5} {'meta':>8} {'lgbm':>8} {'ridge':>8} {'prior':>8} {'llm':>8}")
    for k, v in per_h.items():
        llm = f"{v.get('llm_mae', float('nan')):8.4f}" if "llm_mae" in v else f"{'':>8}"
        print(f"{k:<6} {v['n']:>5} {v['meta_mae']:>8.4f} {v['lgbm_mae']:>8.4f} "
              f"{v['ridge_mae']:>8.4f} {v['prior_mae']:>8.4f} {llm}")
    return per_h


if __name__ == "__main__":
    llm_df, llm_audit = _load_llm_snapshot()
    meta, _, _ = build_meta_dataset(llm_snapshot=llm_df, llm_audit=llm_audit)
    print(f"meta rows={len(meta):,}, LLM attached={meta['llm_pred'].notna().sum()}")

    base = meta.copy()
    if "llm_pred" in base.columns:
        base = base.drop(columns=["llm_pred"])
    fit_and_score(base, "v1: WITHOUT llm_pred (counterfactual)")

    fit_and_score(meta.copy(), "v2: WITH llm_pred (current)")
