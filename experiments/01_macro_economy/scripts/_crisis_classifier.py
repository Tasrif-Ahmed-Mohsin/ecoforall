"""Phase 9 (GMD-specific) — Crisis-probability side model.

GMD ships three explicit crisis dummies that the old 5-source panel didn't
carry:
    sov_debt_crisis, currency_crisis, banking_crisis

We build a binary classifier predicting
    p(any crisis in next H years | country-year)
per horizon H in {1, 3, 5, 10}. The label is forward-looking — for a row at
year t we mark y=1 if any of the three crisis flags equals 1 in the country-
year range [t+1, t+H] inclusive. We use the same Ridge/LGBM feature pipeline
as `run_phase8_horizons_v2.py` (country dummies, tier dummies, rank-transform
on continuous, Optuna-budget=0 for the default run, fallback to a strong fixed
config).

Validation: 5-fold walk-forward CV by anchor year. Reports ROC-AUC, PR-AUC,
Brier score, and calibration (mean predicted vs mean realised crisis rate).

Outputs:
    data/features/crisis_model/
        crisis_lgbm_h{h}.joblib
        crisis_metrics_h{h}.json     (per-fold + summary)
        crisis_cv_summary.json       (cross-horizon summary)
        crisis_cv.csv                (one row per fold × horizon × model)

This complements the existing point-forecast pipeline by adding a *probabilistic
tail-risk* signal — the kind of thing the existing v2 ensemble cannot surface
because it predicts the conditional mean only.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.harmonize.common import FEATURES_DIR  # noqa: E402
from run_phase8_horizons_v2 import (  # type: ignore  # noqa: E402
    DROP_FEATURES,
    _add_country_and_tier_dummies,
    _rank_transform,
)


PANEL = FEATURES_DIR / "panel_wide.parquet"
OUT_DIR = FEATURES_DIR / "crisis_model"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Three crisis dummies GMD ships. Forward-aggregation builds the label.
CRISIS_COLS = ["sov_debt_crisis", "currency_crisis", "banking_crisis"]
LEAK_COLS_BASE = {"iso3", "year", "gdp_pc"}


# ---------------------------------------------------------------------------
# Target construction
# ---------------------------------------------------------------------------
def _build_crisis_target(panel: pd.DataFrame, h: int) -> pd.Series:
    """y_t = 1 if any of {sov_debt_crisis, currency_crisis, banking_crisis}
    equals 1 in years [t+1, t+h] for the same iso3, else 0.
    """
    panel = panel.sort_values(["iso3", "year"]).reset_index(drop=True)
    any_crisis = panel[CRISIS_COLS].max(axis=1).astype(np.float32)
    # rolling sum over the next h rows (forward) per iso3, then take "any > 0"
    fwd_sum = (
        panel.assign(_y=any_crisis)
        .groupby("iso3")["_y"]
        .transform(lambda s: s[::-1].rolling(window=h, min_periods=1).sum()[::-1])
    )
    label = (fwd_sum > 0).astype(np.float32)
    # Important: shift by 1 so we don't leak the contemporaneous crisis as a
    # "future" crisis. We want strictly future.
    label = (
        panel.assign(_lbl=label)
        .groupby("iso3")["_lbl"]
        .transform(lambda s: s.shift(-1))
    )
    return label.rename(f"crisis_within_{h}y")


# ---------------------------------------------------------------------------
# Feature preparation (mirrors v2 trainer)
# ---------------------------------------------------------------------------
def _prepare_classifier(
    df: pd.DataFrame, target: str, iso_levels: list[str],
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    df_aug, _ = _add_country_and_tier_dummies(df, iso_levels)
    leak = LEAK_COLS_BASE | {c for c in df_aug.columns if c.endswith("y_fwd") or c.startswith("crisis_within_")}
    dummy_cols = {c for c in df_aug.columns if c.startswith("iso_") or c.startswith("tier_")}
    cont_cols = [
        c for c in df_aug.columns
        if c not in leak
        and pd.api.types.is_numeric_dtype(df_aug[c])
        and c not in DROP_FEATURES
        and c not in dummy_cols
        and c not in {"gdp_pc"}
    ]
    cont_cols = [c for c in cont_cols if not c.endswith("y_fwd")]
    X = df_aug[cont_cols + sorted(dummy_cols)].astype(np.float32).replace([np.inf, -np.inf], np.nan)
    keep = [c for c in X.columns if X[c].notna().any()]
    X = X[keep]
    y = df_aug[target].astype(np.float32)
    return X, y, keep


# ---------------------------------------------------------------------------
# Walk-forward CV
# ---------------------------------------------------------------------------
def _walk_forward_folds(panel: pd.DataFrame, n_folds: int = 5, test_window: int = 5) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    """Returns list of (anchor_end, h, train_mask, test_mask)."""
    max_year = int(panel["year"].max())
    folds = []
    # Anchor years evenly spaced back from the latest test slice.
    for f in range(n_folds):
        anchor_end = max_year - (n_folds - 1 - f) * test_window
        train_mask = (panel["year"] <= anchor_end - test_window).to_numpy()
        test_mask = ((panel["year"] > anchor_end - test_window) & (panel["year"] <= anchor_end)).to_numpy()
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        folds.append((anchor_end, f, train_mask, test_mask))
    return folds


def _fit_lgbm_binary(X_tr: pd.DataFrame, y_tr: pd.Series, seed: int = 0):
    """Fixed, strong LightGBM config for binary classification — no Optuna by
    default to keep the run quick. Cap estimators to 600 (matches the
    walk-forward CV convention from `_panel_backtest.py`).
    """
    import lightgbm as lgb
    # impute simple median per column
    med = X_tr.median(numeric_only=True)
    X_tr_imp = X_tr.fillna(med)

    pos_rate = float(y_tr.mean())
    # Class-weight rebalancing for the rare-class scenario (crises are <10%).
    spw = (1.0 - pos_rate) / max(pos_rate, 1e-3)
    model = lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
        scale_pos_weight=min(spw, 10.0),  # cap extreme rebalancing
    )
    model.fit(X_tr_imp, y_tr.to_numpy())
    return model, med


def _scoring(y_true: np.ndarray, p_hat: np.ndarray) -> dict:
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    return {
        "n": int(len(y_true)),
        "pos_rate": float(np.mean(y_true)),
        "pred_mean": float(np.mean(p_hat)),
        "roc_auc": float(roc_auc_score(y_true, p_hat)) if y_true.std() > 0 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, p_hat)) if y_true.std() > 0 else float("nan"),
        "brier": float(brier_score_loss(y_true, p_hat)),
    }


def _fit_evaluate_fold(
    panel: pd.DataFrame, h: int, train_mask: np.ndarray, test_mask: np.ndarray,
    iso_levels: list[str], seed: int,
) -> dict:
    target = f"crisis_within_{h}y"
    X, y, cols = _prepare_classifier(panel, target, iso_levels)
    # Drop rows where the target is NaN (insufficient forward coverage).
    valid = y.notna().to_numpy()
    train_mask = train_mask & valid
    test_mask = test_mask & valid

    X_tr, y_tr = X.loc[train_mask], y.loc[train_mask]
    X_te, y_te = X.loc[test_mask], y.loc[test_mask]

    # Rank-transform continuous features (consistent with v2 trainer).
    # For simplicity here we skip per-year rank and use column-wise rank across
    # all rows in X — this is the fallback the v2 trainer uses when no per-year
    # fit data is available.
    cont = [c for c in cols if not c.startswith("iso_") and not c.startswith("tier_")]
    X_rank, _ = _rank_transform(X_tr[cont], fit_idx=None)
    X_rank = pd.DataFrame(X_rank, columns=cont, index=X_tr.index)
    X_tr_full = pd.concat([X_rank, X_tr[[c for c in cols if c not in cont]]], axis=1)

    X_rank_te, _ = _rank_transform(X_te[cont], fit_idx=None)
    X_rank_te = pd.DataFrame(X_rank_te, columns=cont, index=X_te.index)
    X_te_full = pd.concat([X_rank_te, X_te[[c for c in cols if c not in cont]]], axis=1)

    model, med = _fit_lgbm_binary(X_tr_full, y_tr, seed=seed)
    p_hat = model.predict_proba(X_te_full.fillna(med))[:, 1]

    metrics = _scoring(y_te.to_numpy(), p_hat)
    metrics["anchor_end"] = int(panel.loc[test_mask, "year"].max())
    metrics["h"] = int(h)
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(horizons: list[int], n_folds: int, seed: int) -> None:
    print(f"[crisis] loading panel {PANEL} …")
    panel = pd.read_parquet(PANEL)
    print(f"[crisis] panel shape={panel.shape}  iso3s={panel['iso3'].nunique()}  "
          f"years={panel['year'].min()}-{panel['year'].max()}")

    # Build all 4 forward labels first.
    for h in horizons:
        target = f"crisis_within_{h}y"
        if target not in panel.columns:
            panel[target] = _build_crisis_target(panel, h)
            rate = panel[target].mean()
            print(f"[crisis] h={h}  base rate of any-crisis-within-{h}y: {rate:.4f}")

    iso_levels = sorted(panel["iso3"].unique().tolist())
    folds = _walk_forward_folds(panel, n_folds=n_folds)
    print(f"[crisis] {len(folds)} walk-forward folds: anchor_ends = "
          f"{[f[0] for f in folds]}")

    all_rows: list[dict] = []
    summary: dict = {"by_horizon": {}}

    for h in horizons:
        per_fold: list[dict] = []
        for anchor_end, f, train_mask, test_mask in folds:
            m = _fit_evaluate_fold(panel, h, train_mask, test_mask, iso_levels, seed=seed)
            m["fold"] = f
            per_fold.append(m)
            print(f"[crisis] h={h}  fold={f}  anchor_end={anchor_end}  "
                  f"n_test={m['n']}  pos_rate={m['pos_rate']:.3f}  "
                  f"pred_mean={m['pred_mean']:.3f}  roc_auc={m['roc_auc']:.3f}  "
                  f"pr_auc={m['pr_auc']:.3f}  brier={m['brier']:.4f}")
            all_rows.append({"horizon": h, "model": "lgbm_binary", **m})

        # aggregate
        df = pd.DataFrame(per_fold)
        agg = {
            "n_folds": int(len(df)),
            "roc_auc_mean": float(df["roc_auc"].mean()),
            "roc_auc_std": float(df["roc_auc"].std()),
            "pr_auc_mean": float(df["pr_auc"].mean()),
            "pr_auc_std": float(df["pr_auc"].std()),
            "brier_mean": float(df["brier"].mean()),
            "pos_rate_mean": float(df["pos_rate"].mean()),
            "pred_mean_avg": float(df["pred_mean"].mean()),
            "calibration_gap": float((df["pred_mean"] - df["pos_rate"]).abs().mean()),
        }
        summary["by_horizon"][str(h)] = agg
        print(f"[crisis] h={h}  AGG  roc_auc={agg['roc_auc_mean']:.3f}±{agg['roc_auc_std']:.3f}  "
              f"pr_auc={agg['pr_auc_mean']:.3f}±{agg['pr_auc_std']:.3f}  "
              f"calibration_gap={agg['calibration_gap']:.4f}")

    # Save outputs.
    cv_df = pd.DataFrame(all_rows)
    cv_path = OUT_DIR / "crisis_cv.csv"
    cv_df.to_csv(cv_path, index=False)
    print(f"[crisis] wrote {cv_path}")

    summary_path = OUT_DIR / "crisis_cv_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[crisis] wrote {summary_path}")

    print("[crisis] fitting final production models on all data...")
    for h in horizons:
        target = f"crisis_within_{h}y"
        X, y, cols = _prepare_classifier(panel, target, iso_levels)
        valid = y.notna().to_numpy()
        X_tr = X.loc[valid]
        y_tr = y.loc[valid]
        
        cont = [c for c in cols if not c.startswith("iso_") and not c.startswith("tier_")]
        X_rank, _ = _rank_transform(X_tr[cont], fit_idx=None)
        X_rank = pd.DataFrame(X_rank, columns=cont, index=X_tr.index)
        X_tr_full = pd.concat([X_rank, X_tr[[c for c in cols if c not in cont]]], axis=1)
        
        model, med = _fit_lgbm_binary(X_tr_full, y_tr, seed=seed)
        
        # Save the model, median imputer, and feature columns
        out_file = OUT_DIR / f"crisis_lgbm_h{h}.joblib"
        joblib.dump({"model": model, "median": med, "cols": cols, "cont_cols": cont}, out_file)
        print(f"[crisis] wrote {out_file}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 5, 10])
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(args.horizons, args.n_folds, args.seed)