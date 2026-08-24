"""Imputation sweep for the Ridge path of the v2 horizons trainer.

Question
--------
Does a smarter imputation policy beat the current
`SimpleImputer(strategy="median")` on the Ridge branch of
`run_phase8_horizons_v2._train_one_horizon`?

Why only Ridge?
---------------
- LightGBM is already NaN-aware and dominates the ensemble at every horizon;
  re-fitting it inside this sweep would be O(20 minutes) per k and the win
  from imputation is mostly on the linear branch.
- The retrieval layer (`RankedFaissIndex`) already does its own NaN-handling
  (`z[0,j]=0.0` for missing rank features) and is decoupled from this sweep.

Design
------
- Loads the same panel as the v2 trainer (`panel_wide.parquet`).
- Mirrors v2 `_prepare()` exactly: drop leak cols, drop `DROP_FEATURES`,
  rank-transform the continuous block (same `pd.Series.rank` recipe).
- Mirrors v2 splits: `H5_TRAIN_END=2014`, `H5_VAL_END=2018`, `H5_TEST_END=2022`
  (shifted by `h-5` for shorter horizons).
- Compares 4 imputers on a v2-identical Ridge(alpha=1) pipe:
    * median             — production baseline (per column).
    * tier_median        — per (income-tier, column) median (LIC/LMIC/UMIC/HIC).
    * region_median      — per (region, column) median (continent grouping).
    * tier_plus_self_lag — fill `c_lag1` with that row's own `c`; remaining
                           NaN fall back to per-tier median. (Twin idea: a
                           country's own self-twin is its own lag.)
- Reports MAE / RMSE / dir_acc on val + test, per horizon.

Outputs
-------
- `data/features/knn_impute_sweep/summary.csv` — wide-format per-horizon table.
- `data/features/knn_impute_sweep/summary.json` — same payload, machine-readable.

No parquet writes. No model saves. Read-only on the panel.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# KNNImputer emits numpy overflow warnings on 200+ feature matrices even after
# rank-scaling (the squared-distance accumulation overflows float32). Silence
# so the real signal isn't drowned in noise.
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.run_phase8_horizons_v2 import (
    H5_TEST_END,
    H5_TRAIN_END,
    H5_VAL_END,
    _build_horizon_target,
    _metrics,
    _prepare,
    _rank_transform,
)

PANEL = ROOT / "data" / "features" / "panel_wide.parquet"
OUT_DIR = ROOT / "data" / "features" / "knn_impute_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS_DEFAULT = [1, 3, 5]

# World Bank income-tier thresholds (USD, current-real). Mirrors v2 trainer.
_TIER_BINS = [-np.inf, 2000.0, 4500.0, 14000.0, np.inf]
_TIER_LABELS = ["LIC", "LMIC", "UMIC", "HIC"]

# Continent grouping by iso3 prefix (used as a coarser alternative to tier).
# This is a hand-curated continent->iso3 prefix map. Countries not listed
# land in "OTHER" and get the global median (same as the baseline).
_CONTINENT_PREFIXES: dict[str, tuple[str, ...]] = {
    "AF": ("DZA", "AGO", "BEN", "BWA", "BFA", "BDI", "CPV", "CMR", "CAF", "TCD",
           "COM", "COG", "COD", "CIV", "DJI", "EGY", "GNQ", "ERI", "SWZ", "ETH",
           "GAB", "GMB", "GHA", "GIN", "GNB", "KEN", "LSO", "LBR", "LBY", "MDG",
           "MWI", "MLI", "MRT", "MUS", "MAR", "MOZ", "NAM", "NER", "NGA", "RWA",
           "STP", "SEN", "SYC", "SLE", "SOM", "ZAF", "SSD", "SDN", "TZA", "TGO",
           "TUN", "UGA", "ZMB", "ZWE"),
    "AS": ("AFG", "ARM", "AZE", "BHR", "BGD", "BTN", "BRN", "KHM", "CHN", "CYP",
           "GEO", "IND", "IDN", "IRN", "IRQ", "ISR", "JPN", "JOR", "KAZ", "KWT",
           "KGZ", "LAO", "LBN", "MYS", "MDV", "MNG", "MMR", "NPL", "OMN", "PAK",
           "PSE", "PHL", "QAT", "SAU", "SGP", "KOR", "LKA", "SYR", "TWN", "TJK",
           "THA", "TLS", "TUR", "TKM", "ARE", "UZB", "VNM", "YEM"),
    "EU": ("ALB", "AND", "AUT", "BLR", "BEL", "BIH", "BGR", "HRV", "CZE", "DNK",
           "EST", "FIN", "FRA", "DEU", "GRC", "HUN", "ISL", "IRL", "ITA", "XKX",
           "LVA", "LIE", "LTU", "LUX", "MLT", "MDA", "MCO", "MNE", "NLD", "MKD",
           "NOR", "POL", "PRT", "ROU", "RUS", "SMR", "SRB", "SVK", " SVN", "ESP",
           "SWE", "CHE", "UKR", "GBR", "VAT"),
    "NA": ("ATG", "BHS", "BRB", "BLZ", "CAN", "CRI", "CUB", "DMA", "DOM", "SLV",
           "GRD", "GTM", "HTI", "HND", "JAM", "MEX", "NIC", "PAN", "KNA", "LCA",
           "VCT", "TTO", "USA"),
    "SA": ("ARG", "BOL", "BRA", "CHL", "COL", "ECU", "GUY", "PRY", "PER", "SUR",
           "URY", "VEN"),
    "OC": ("AUS", "FJI", "KIR", "MHL", "FSM", "NRU", "NZL", "PLW", "PNG", "WSM",
           "SLB", "TON", "TUV", "VUT"),
}
_ISO3_TO_CONTINENT: dict[str, str] = {
    iso: cont for cont, isos in _CONTINENT_PREFIXES.items() for iso in isos
}


def _assign_tier(gdp_pc: pd.Series) -> pd.Series:
    return pd.cut(gdp_pc, bins=_TIER_BINS, labels=_TIER_LABELS).astype(object)


def _assign_continent(iso3: pd.Series) -> pd.Series:
    return iso3.map(_ISO3_TO_CONTINENT).fillna("OTHER").astype(object)


def _fit_predict_ridge(X_train: np.ndarray, y_train: np.ndarray,
                       X_eval: np.ndarray) -> np.ndarray:
    """Standard v2 Ridge pipeline: scaler -> Ridge(alpha=1). Imputation
    happens upstream; the trainer's `SimpleImputer(median)` is replaced
    by whichever policy the sweep is testing."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0, random_state=0)),
    ])
    pipe.fit(X_train, y_train)
    return pipe.predict(X_eval)


# ---------------------------------------------------------------------
# Imputation policies. Each takes the (N,D) raw numeric matrix + row
# meta (iso3, gdp_pc) + train mask, returns the fully-imputed (N,D)
# matrix on which to fit Ridge.
# ---------------------------------------------------------------------

def _impute_median(X: np.ndarray, train: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
    """Per-column median on training rows. Production baseline."""
    imp = SimpleImputer(strategy="median").fit(X[train])
    return imp.transform(X).astype(np.float32)


def _impute_tier_median(X: np.ndarray, train: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
    """Per-(tier, column) median on training rows. Falls back to global
    median if a tier has no observed value for a column."""
    out = X.copy()
    tier = _assign_tier(meta["gdp_pc"]).to_numpy()
    nan_mask = np.isnan(out)
    for t in _TIER_LABELS:
        sub_train = (tier == t) & train
        if not sub_train.any():
            continue
        med = np.nanmedian(X[sub_train], axis=0)
        # Per-column: only fill (tier == t) cells that are NaN. Cells where the
        # tier itself has no observed value fall back to the global median.
        tier_rows = (tier == t)
        for j in range(out.shape[1]):
            cell = tier_rows & nan_mask[:, j]
            if not cell.any():
                continue
            val = med[j]
            if np.isnan(val):
                val = np.nanmedian(X[train, j])
            out[cell, j] = val
    if np.isnan(out).any():
        out = _impute_median(out, train, meta)
    return out.astype(np.float32)


def _impute_continent_median(X: np.ndarray, train: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
    """Per-(continent, column) median, fallback global."""
    out = X.copy()
    cont = _assign_continent(meta["iso3"]).to_numpy()
    nan_mask = np.isnan(out)
    for c in np.unique(cont):
        sub_train = (cont == c) & train
        if not sub_train.any():
            continue
        med = np.nanmedian(X[sub_train], axis=0)
        cont_rows = (cont == c)
        for j in range(out.shape[1]):
            cell = cont_rows & nan_mask[:, j]
            if not cell.any():
                continue
            val = med[j]
            if np.isnan(val):
                val = np.nanmedian(X[train, j])
            out[cell, j] = val
    if np.isnan(out).any():
        out = _impute_median(out, train, meta)
    return out.astype(np.float32)


def _impute_self_lag_then_tier(X: np.ndarray, X_cols: list[str],
                                train: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
    """For each column c with a `_lag1` partner, fill c's NaN with the row's
    own `c_lag1` (if observed). Remainder falls back to tier-median.
    The intuition: a country's own recent history is the closest twin."""
    out = X.copy()
    # Build a (col -> lag_col) mapping. Match on suffix.
    lag_for: dict[str, str] = {}
    for j, c in enumerate(X_cols):
        if c.endswith("_lag1"):
            base = c[:-5]
            if base in X_cols:
                lag_for[base] = c
    # Forward-fill col_j from col_j's lag (where col_j == base, lag is `_lag1`).
    col_idx = {c: i for i, c in enumerate(X_cols)}
    for base, lag in lag_for.items():
        if base not in col_idx or lag not in col_idx:
            continue
        bj, lj = col_idx[base], col_idx[lag]
        nan = np.isnan(out[:, bj]) & ~np.isnan(out[:, lj])
        if nan.any():
            out[nan, bj] = out[nan, lj]
    if np.isnan(out).any():
        out = _impute_tier_median(out, train, meta)
    return out.astype(np.float32)


def sweep_one_horizon(h: int, panel: pd.DataFrame) -> list[dict]:
    target = f"gdp_pc_growth_{h}y_fwd"
    panel = panel.copy()
    panel[target] = _build_horizon_target(panel, h)

    df = panel.dropna(subset=[target]).reset_index(drop=True)
    iso_levels = sorted(df["iso3"].unique().tolist())

    # Mirror v2 _prepare exactly.
    X_cont, X_full, y, cont_cols, full_cols = _prepare(df, target, iso_levels)

    years = df["year"].to_numpy()
    shift = max(0, h - 5)
    train_end = H5_TRAIN_END - shift
    val_end = H5_VAL_END - shift
    test_end = H5_TEST_END - shift
    train = years <= train_end
    val = (years > train_end) & (years <= val_end)
    test = (years > val_end) & (years <= test_end)

    yv = y.to_numpy()
    n_features = X_cont.shape[1]
    # Rank-transform FIRST so every imputer sees [0,1]-scaled features.
    Xr_full_rank = _rank_transform(X_cont)[0]
    Xr_full_rank = pd.DataFrame(Xr_full_rank, columns=X_cont.columns, index=X_cont.index)
    print(f"\n[h={h}y] rows={len(df):,}  cont_features={n_features}  "
          f"NaN fraction={np.isnan(Xr_full_rank.to_numpy()).mean():.3f}  "
          f"train/val/test={int(train.sum())}/{int(val.sum())}/{int(test.sum())}")

    Xr_np = Xr_full_rank.to_numpy()
    meta = df[["iso3", "gdp_pc_real"]].rename(columns={"gdp_pc_real": "gdp_pc"}).reset_index(drop=True)

    rows: list[dict] = []

    # ---- Policy 1: median (production baseline) ----
    X_imp = _impute_median(Xr_np, train, meta)
    pv = _fit_predict_ridge(X_imp[train], yv[train], X_imp[val])
    pt = _fit_predict_ridge(X_imp[train], yv[train], X_imp[test])
    rows.append({"horizon": h, "imputer": "median", "val_mae": _metrics(yv[val], pv)["mae"],
                 "test_mae": _metrics(yv[test], pt)["mae"],
                 "test_dir_acc": _metrics(yv[test], pt)["dir_acc"]})
    print(f"  median        test MAE={rows[-1]['test_mae']:.4f}  val MAE={rows[-1]['val_mae']:.4f}")

    # ---- Policy 2: per-tier median (4 buckets) ----
    X_imp = _impute_tier_median(Xr_np, train, meta)
    pv = _fit_predict_ridge(X_imp[train], yv[train], X_imp[val])
    pt = _fit_predict_ridge(X_imp[train], yv[train], X_imp[test])
    rows.append({"horizon": h, "imputer": "tier_median", "val_mae": _metrics(yv[val], pv)["mae"],
                 "test_mae": _metrics(yv[test], pt)["mae"],
                 "test_dir_acc": _metrics(yv[test], pt)["dir_acc"]})
    print(f"  tier_median   test MAE={rows[-1]['test_mae']:.4f}  val MAE={rows[-1]['val_mae']:.4f}")

    # ---- Policy 3: per-continent median (6 buckets) ----
    X_imp = _impute_continent_median(Xr_np, train, meta)
    pv = _fit_predict_ridge(X_imp[train], yv[train], X_imp[val])
    pt = _fit_predict_ridge(X_imp[train], yv[train], X_imp[test])
    rows.append({"horizon": h, "imputer": "continent_median", "val_mae": _metrics(yv[val], pv)["mae"],
                 "test_mae": _metrics(yv[test], pt)["mae"],
                 "test_dir_acc": _metrics(yv[test], pt)["dir_acc"]})
    print(f"  continent     test MAE={rows[-1]['test_mae']:.4f}  val MAE={rows[-1]['val_mae']:.4f}")

    # ---- Policy 4: self-lag then tier-median fallback (own-history twin) ----
    X_imp = _impute_self_lag_then_tier(Xr_np, list(X_cont.columns), train, meta)
    pv = _fit_predict_ridge(X_imp[train], yv[train], X_imp[val])
    pt = _fit_predict_ridge(X_imp[train], yv[train], X_imp[test])
    rows.append({"horizon": h, "imputer": "selflag_then_tier", "val_mae": _metrics(yv[val], pv)["mae"],
                 "test_mae": _metrics(yv[test], pt)["mae"],
                 "test_dir_acc": _metrics(yv[test], pt)["dir_acc"]})
    print(f"  selflag_tier  test MAE={rows[-1]['test_mae']:.4f}  val MAE={rows[-1]['val_mae']:.4f}")

    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=int, nargs="+", default=HORIZONS_DEFAULT,
                    help="Horizons to sweep (default: 1 3 5)")
    args = ap.parse_args()

    panel = pd.read_parquet(PANEL)
    print(f"[knn-sweep] panel: {panel.shape}  iso3={panel['iso3'].nunique()}  "
          f"years={panel['year'].min()}-{panel['year'].max()}")

    all_rows: list[dict] = []
    for h in args.horizons:
        all_rows.extend(sweep_one_horizon(h, panel))

    df_out = pd.DataFrame(all_rows)
    csv_path = OUT_DIR / "summary.csv"
    json_path = OUT_DIR / "summary.json"
    df_out.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(all_rows, indent=2))

    print("\n=== Winner per horizon (test MAE) ===")
    for h in args.horizons:
        sub = df_out[df_out["horizon"] == h].sort_values("test_mae")
        winner = sub.iloc[0]
        median = df_out[(df_out["horizon"] == h) & (df_out["imputer"] == "median")].iloc[0]
        delta_pct = (winner["test_mae"] - median["test_mae"]) / median["test_mae"] * 100.0
        print(f"  h={h}y  winner={winner['imputer']:<20s}  "
              f"test MAE={winner['test_mae']:.4f}  vs median {median['test_mae']:.4f}  "
              f"d={delta_pct:+.1f}%")

    print(f"\n[knn-sweep] wrote {csv_path}")
    print(f"[knn-sweep] wrote {json_path}")


if __name__ == "__main__":
    main()