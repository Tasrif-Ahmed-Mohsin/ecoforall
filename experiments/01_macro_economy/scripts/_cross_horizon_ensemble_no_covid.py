"""Cross-horizon Ridge meta-ensemble — NO-COVID parallel experiment.

Sibling of `scripts/_cross_horizon_ensemble.py` that drops 2020 and 2021
(COVID-era years) from BOTH the meta-train and meta-test slices before
fitting the Ridge meta-learner.

Important nuances:
- For h=1 and h=3, the base forecasts' "test" split is dominated by 2020/2021
  rows (50 % and 66 % respectively). For h=5 and h=10, the test slice ends
  in 2019 / 2014 — COVID is already excluded by the forward-target window.
  So this experiment materially affects h=1 and h=3; for h=5/h=10 the
  meta numbers should be unchanged (we verify that).
- The base learners (LGBM, Ridge, prior) themselves are NOT retrained.
  Re-fitting them without COVID would require a full panel rebuild
  (~hours of compute) and a new `run_phase8_horizons_no_covid.py`. This
  script isolates the *meta-ensemble* effect, which is the cheap, honest
  first cut.
- Outputs go to `data/features/cross_horizon_meta_no_covid/` — never
  overwrites the canonical artifact. The original `_cross_horizon_ensemble.py`
  and its outputs are preserved for before/after comparison.

Outputs:
    data/features/cross_horizon_meta_no_covid/
        metrics.json
        predictions.parquet
        meta_ridge.joblib
    data/features/covid_compare.csv          (side-by-side vs canonical)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Make the sibling module importable so we reuse build_meta_dataset / HORIZONS.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from _cross_horizon_ensemble import (  # noqa: E402
    HORIZONS as _CANONICAL_HORIZONS,
    _add_ar1,
    _dir_acc,
    _load_one_horizon,
    _panel_path,
    build_meta_dataset,
)

FEATURES_DIR = ROOT / "data" / "features"
CANONICAL_DIR = FEATURES_DIR / "cross_horizon_meta"
OUT_DIR = FEATURES_DIR / "cross_horizon_meta_no_covid"

EXCLUDE_YEARS = (2020, 2021)


def _metrics(y, p) -> dict:
    return {
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "dir_acc": _dir_acc(y, p),
    }


def _drop_covid(meta: pd.DataFrame) -> pd.DataFrame:
    """Drop any row whose year is in EXCLUDE_YEARS, regardless of split."""
    n0 = len(meta)
    keep = ~meta["year"].isin(EXCLUDE_YEARS)
    out = meta.loc[keep].reset_index(drop=True)
    print(f"[no-covid] dropped {(~keep).sum():,} rows in years {EXCLUDE_YEARS} "
          f"(kept {len(out):,}/{n0:,})")
    print(f"[no-covid] per-horizon splits after drop:")
    for h in sorted(meta["horizon"].unique()):
        sub = out[out["horizon"] == h]
        print(f"           h={h}: {sub.split.value_counts().to_dict()}")
    return out


def train_meta_no_covid(meta: pd.DataFrame, horizons: list[int]) -> tuple[Pipeline, dict]:
    """Same as _cross_horizon_ensemble.train_meta, but on the COVID-excluded meta."""
    feature_cols = []
    for h in horizons:
        feature_cols += [f"ridge_h{h}", f"lgbm_h{h}", f"prior_h{h}"]
    if "ar1" in feature_cols:
        pass
    if "ar1" in meta.columns:
        feature_cols.append("ar1")
    feature_cols += ["horizon"]

    # Same convention: meta-train = val, meta-test = test.
    train_mask = meta["split"] == "val"
    test_mask = meta["split"] == "test"

    X_tr = meta.loc[train_mask, feature_cols].to_numpy()
    y_tr = meta.loc[train_mask, "y_true"].to_numpy()
    X_te = meta.loc[test_mask, feature_cols].to_numpy()
    y_te = meta.loc[test_mask, "y_true"].to_numpy()

    if "ar1" in feature_cols:
        for X in (X_tr, X_te):
            X[np.isnan(X)] = 0.0

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], cv=5)),
    ])
    pipe.fit(X_tr, y_tr)
    pred_te = pipe.predict(X_te)
    metrics_overall = _metrics(y_te, pred_te)

    # Per-horizon breakdown
    per_h = {}
    pred_full = np.full(len(meta), np.nan)
    pred_full[test_mask.to_numpy()] = pred_te
    meta["pred_meta"] = pred_full

    for h in horizons:
        m = (meta["horizon"] == h) & test_mask
        if m.sum() == 0:
            continue
        per_h[f"h{h}"] = {
            "n": int(m.sum()),
            **_metrics(
                meta.loc[m, "y_true"].to_numpy(),
                meta.loc[m, "pred_meta"].to_numpy(),
            ),
            "prior_mae": float(mean_absolute_error(
                meta.loc[m, "y_true"].to_numpy(),
                meta.loc[m, f"prior_h{h}"].to_numpy(),
            )),
            "lgbm_mae": float(mean_absolute_error(
                meta.loc[m, "y_true"].to_numpy(),
                meta.loc[m, f"lgbm_h{h}"].to_numpy(),
            )),
            "ridge_mae": float(mean_absolute_error(
                meta.loc[m, "y_true"].to_numpy(),
                meta.loc[m, f"ridge_h{h}"].to_numpy(),
            )),
        }

    chosen_alpha = float(pipe.named_steps["ridge"].alpha_)
    return pipe, {
        "overall_test": metrics_overall,
        "per_horizon_test": per_h,
        "feature_cols": feature_cols,
        "ridge_alpha": chosen_alpha,
        "meta_train_rows": int(train_mask.sum()),
        "meta_test_rows": int(test_mask.sum()),
        "excluded_years": list(EXCLUDE_YEARS),
        "note": (
            "Base learners (LGBM/Ridge/prior) are NOT retrained; this isolates "
            "the meta-ensemble effect of excluding COVID from train+test."
        ),
    }


def _compare_to_canonical(no_covid_metrics: dict) -> pd.DataFrame:
    """Build a side-by-side DataFrame: canonical vs no-COVID per horizon."""
    canonical_path = CANONICAL_DIR / "metrics.json"
    if not canonical_path.exists():
        print("[no-covid] no canonical metrics.json; skipping comparison")
        return pd.DataFrame()
    can = json.loads(canonical_path.read_text())
    rows = []
    for h_key in sorted(set(can.get("per_horizon_test", {}).keys()) |
                        set(no_covid_metrics.get("per_horizon_test", {}).keys())):
        c = can.get("per_horizon_test", {}).get(h_key, {})
        n = no_covid_metrics.get("per_horizon_test", {}).get(h_key, {})
        rows.append({
            "horizon": h_key,
            "n_canonical": c.get("n", 0),
            "n_no_covid": n.get("n", 0),
            "mae_canonical": round(c.get("mae", float("nan")), 4),
            "mae_no_covid": round(n.get("mae", float("nan")), 4),
            "mae_delta": (round((n.get("mae", float("nan")) - c.get("mae", float("nan"))), 4)
                          if c.get("mae") is not None and n.get("mae") is not None else None),
            "dir_acc_canonical": round(c.get("dir_acc", float("nan")), 3),
            "dir_acc_no_covid": round(n.get("dir_acc", float("nan")), 3),
            "prior_mae_canonical": round(c.get("prior_mae", float("nan")), 4),
            "prior_mae_no_covid": round(n.get("prior_mae", float("nan")), 4),
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    available = [h for h in _CANONICAL_HORIZONS
                 if (FEATURES_DIR / f"horizon_{h}y_v2" / "forecasts.parquet").exists()]
    if not available:
        raise SystemExit("No v2 horizons available. Run scripts/run_phase8_horizons_v2.py first.")
    print(f"[no-covid] available horizons: {available}")

    # Build the meta dataset (same as canonical) then drop COVID years.
    meta, horizons = build_meta_dataset()
    meta = _drop_covid(meta)

    # Fit the Ridge meta on the COVID-excluded slices.
    pipe, metrics = train_meta_no_covid(meta, horizons)
    print(f"[no-covid] ridge alpha={metrics['ridge_alpha']}  "
          f"overall_test MAE={metrics['overall_test']['mae']:.4f}  "
          f"dir_acc={metrics['overall_test']['dir_acc']:.3f}")
    print(f"[no-covid] per-horizon test:")
    print(f"{'h':<6} {'n':>5} {'meta':>8} {'lgbm':>8} {'ridge':>8} {'prior':>8} "
          f"{'meta_dir':>9} {'vs_prior':>9}")
    for k, v in metrics["per_horizon_test"].items():
        delta = v["prior_mae"] - v["mae"]
        print(f"{k:<6} {v['n']:>5} {v['mae']:>8.4f} {v['lgbm_mae']:>8.4f} "
              f"{v['ridge_mae']:>8.4f} {v['prior_mae']:>8.4f} "
              f"{v['dir_acc']:>9.3f} {delta:>+9.4f}")

    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    joblib.dump(pipe, OUT_DIR / "meta_ridge.joblib")
    meta.to_parquet(OUT_DIR / "predictions.parquet")
    print(f"[no-covid] wrote {OUT_DIR}")

    # Side-by-side comparison vs canonical.
    cmp_df = _compare_to_canonical(metrics)
    if not cmp_df.empty:
        cmp_path = FEATURES_DIR / "covid_compare.csv"
        cmp_df.to_csv(cmp_path, index=False)
        print(f"[no-covid] wrote comparison {cmp_path}")
        print(cmp_df.to_string(index=False))


if __name__ == "__main__":
    main()