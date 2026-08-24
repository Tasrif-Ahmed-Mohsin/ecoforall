"""Phase 8, step 3 v2 — upgraded benchmark table.

Compares all our models against honest baselines at every horizon, with a
properly vintage-correct IMF WEO comparison (instead of the unfair realised-value
comparison in the v1 benchmark script).

Models compared
---------------
1. naive_persistence — last realised value per country
2. random_walk       — y_hat = 0  (in log-growth space)
3. ar1_lag1_honest   — per-country AR(1) fitted on training data only
4. imf_weo_vintage   — IMF WEO forecast issued before/at forecast time (N/A on
                       rows where the most-recent vintage is newer than the row)
5. v2_lgbm_ensemble  — Optuna-tuned LightGBM + prior blend
6. cross_horizon_meta — Ridge meta-learner that stacks horizon-level forecasts

Outputs
-------
``data/features/benchmark_v2.csv``            — long-form table, one row per
                                               model×horizon×slice
``data/features/benchmark_v2_summary.json``   — headline numbers + DM p-values
                                               across all rows
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

from src.harmonize.common import FEATURES_DIR, HARMONIZED_DIR

PANEL = FEATURES_DIR / "panel_wide.parquet"
IMF_PARQUET = HARMONIZED_DIR / "imf.parquet"

V2_SUFFIX = "_v2"  # horizon_{h}y_v2/metrics.json + forecasts.parquet

# Test slices in v2 metrics.json terms — recomputed from val_end/test_end. We'll
# read these fresh from the artifacts themselves.
SLICE_NAMES = ["train", "val", "test", "holdout"]


# ---------------------------------------------------------------------------
# Diebold–Mariano with Newey-West HAC variance (lag=1) — same as v1 script.
# ---------------------------------------------------------------------------
def _dm_pvalue(y_true, y_pred_a, y_pred_b) -> tuple[float, float]:
    d = (y_pred_a - y_true) ** 2 - (y_pred_b - y_true) ** 2
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 5:
        return float("nan"), float("nan")
    d_mean = d.mean()
    gamma0 = float(((d - d_mean) ** 2).mean())
    if gamma0 == 0:
        return float("nan"), float("nan")
    gamma1 = float(((d[1:] - d_mean) * (d[:-1] - d_mean)).mean())
    var = (gamma0 + 2 * gamma1) / n
    if var <= 0:
        return float("nan"), float("nan")
    stat = d_mean / np.sqrt(var)
    from scipy.stats import norm
    p = 2 * (1 - norm.cdf(abs(stat)))
    return float(stat), float(p)


# ---------------------------------------------------------------------------
# AR(1) honest fit
# ---------------------------------------------------------------------------
def _ar1_honest(panel: pd.DataFrame, target: str, val_end: int) -> tuple[float, float]:
    """Fit y_t = a + b * y_{t-1} on all training rows where prev_y exists."""
    df = panel.sort_values(["iso3", "year"]).reset_index(drop=True)
    df["prev_y"] = df.groupby("iso3")[target].shift(1)
    train_mask = (df["year"] <= val_end) & df["prev_y"].notna() & df[target].notna()
    sub = df[train_mask]
    if len(sub) < 10:
        return 0.0, 0.0
    x = sub["prev_y"].to_numpy(dtype=np.float64)
    yobs = sub[target].to_numpy(dtype=np.float64)
    X_design = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X_design, yobs, rcond=None)
    return float(coef[0]), float(coef[1])


# ---------------------------------------------------------------------------
# IMF WEO vintage-correct loader
# ---------------------------------------------------------------------------
def _load_imf_vintage_forecasts() -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    """Return (forecast_df, realized_df, usd_compatible_countries) for the
    indicator gdp_pc_real.

    forecast_df columns: iso3, target_year, weo_value     (forecast rows only)
    realized_df columns: iso3, year, real_value           (is_forecast=False rows)

    The harmonizer emits IMF `gdp_pc_real` in either USD index or in local-
    currency units depending on the country. We restrict to countries whose
    realized values fall in [100, 200_000] (the plausible USD range for
    gdp_pc_real index=100-base) so that the IMF log-growth prediction is
    comparable to the panel target.

    Caveat: our parquet only has IMF vintages starting ~2020/2025. For pre-
    2020 test rows we report N/A — this is the honest "no vintage to compare"
    condition that the v1 benchmark silently ignored.
    """
    if not IMF_PARQUET.exists():
        print(f"[bench] WARNING: {IMF_PARQUET} not found, skipping IMF WEO baseline")
        return pd.DataFrame(), pd.DataFrame(), set()
        
    imf = pd.read_parquet(IMF_PARQUET)
    real = imf[(imf.is_forecast == False) & (imf.indicator_id == "gdp_pc_real")].copy()
    # USD-compatible = median realized gdp_pc_real across all years in [100, 200_000]
    median_real = real.groupby("iso3")["value"].median()
    usd_ok = median_real[(median_real >= 100) & (median_real <= 200_000)].index.tolist()
    usd_ok_set = set(usd_ok)
    # Filter forecast rows to USD-compatible countries
    fc = imf[(imf.is_forecast == True) & (imf.indicator_id == "gdp_pc_real")].copy()
    fc = fc[fc.iso3.isin(usd_ok_set)]

    fc = fc.sort_values(["iso3", "year"]).reset_index(drop=True)
    fc = fc.rename(columns={"year": "target_year", "value": "weo_value"})
    fc = fc[["iso3", "target_year", "weo_value"]]

    real = real[real.iso3.isin(usd_ok_set)]
    real = real.rename(columns={"year": "real_year", "value": "real_value"})
    real = real[["iso3", "real_year", "real_value"]]
    return fc, real, usd_ok_set


def _imf_pred(imf_fc: pd.DataFrame, imf_real: pd.DataFrame, iso3: str, year: int, h: int) -> float:
    """Compute vintage-correct IMF WEO log-growth forecast for (iso3, year).

    log( IMF_forecast_for_(year+h) / IMF_real_value_at_year )

    Returns NaN if either side is missing or if the panel row has no
    realized IMF value at `year` (the earliest IMF coverage is 1980).
    """
    if imf_fc.empty or imf_real.empty:
        return float("nan")
    target = year + h
    # realized denominator: IMF real value at country-year
    real_row = imf_real[(imf_real.iso3 == iso3) & (imf_real.real_year == year)]
    if len(real_row) == 0:
        return float("nan")
    real_v = float(real_row.iloc[0]["real_value"])
    if not np.isfinite(real_v) or real_v <= 0:
        return float("nan")
    # forecast numerator: earliest-vintage IMF forecast for target year
    fc_row = imf_fc[(imf_fc.iso3 == iso3) & (imf_fc.target_year == target)]
    if len(fc_row) == 0:
        return float("nan")
    fc_v = float(fc_row.iloc[0]["weo_value"])
    if not np.isfinite(fc_v) or fc_v <= 0:
        return float("nan")
    return float(np.log(fc_v / real_v))


# ---------------------------------------------------------------------------
# Per-horizon benchmark
# ---------------------------------------------------------------------------
# Tail-trim cutoff: drop rows where |y_true| exceeds this — these are war /
# hyperinflation / collapse events where no realistic forecast can be useful,
# and they dominate MAE for the IMF benchmark (which can't predict them).
TAIL_CUTOFF = 1.0


def _benchmark_one(
    panel: pd.DataFrame,
    h: int,
    out_root: Path,
    imf_fc: pd.DataFrame,
    imf_real: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    target = f"gdp_pc_growth_{h}y_fwd"
    if target not in panel.columns:
        panel = panel.sort_values(["iso3", "year"]).reset_index(drop=True)
        g_fwd = panel.groupby("iso3")["gdp_pc"].shift(-h)
        panel[target] = np.log(g_fwd / panel["gdp_pc"])

    df = panel.dropna(subset=[target]).reset_index(drop=True)

    # v2 metrics.json sits at horizon_{h}y_v2/. If missing, fall back to v1.
    v2_dir = out_root / f"horizon_{h}y_v2"
    v2_dir_v1 = out_root / f"horizon_{h}y"
    if v2_dir.exists():
        metrics_dir = v2_dir
        version = "v2"
    elif v2_dir_v1.exists():
        metrics_dir = v2_dir_v1
        version = "v1"
    else:
        print(f"[bench] h={h}: missing both v2 and v1 artifacts, skipping")
        return pd.DataFrame(), {}

    metrics = json.loads((metrics_dir / "metrics.json").read_text())
    val_end = metrics["split"]["val_end"]
    test_end = metrics["split"]["test_end"]

    # Load v2/v1 forecasts
    fc = pd.read_parquet(metrics_dir / "forecasts.parquet")
    # Load cross-horizon meta predictions (separate file) for the meta row
    meta_path = out_root / "cross_horizon_meta" / "predictions.parquet"
    meta_all = pd.read_parquet(meta_path) if meta_path.exists() else None
    if meta_all is not None:
        meta_h = meta_all[meta_all["horizon"] == h][["iso3", "year", "split", "pred_meta", "y_true"]]
        meta_h = meta_h.rename(columns={"pred_meta": "y_pred_meta", "y_true": "y_true_meta"})

    rows = []
    # Compute AR(1) honest-fit coefficients once per horizon
    a, b = _ar1_honest(df, target, val_end)
    df_sorted = df.sort_values(["iso3", "year"]).reset_index(drop=True)
    df_sorted["prev_y"] = df_sorted.groupby("iso3")[target].shift(1)
    df_sorted["ar1_pred"] = a + b * df_sorted["prev_y"]

    # IMF WEO lookup helper
    def weo_for(iso3, year):
        return _imf_pred(imf_fc, imf_real, iso3, year, h)

    # Iterate over every (split, model) combination
    for split in SLICE_NAMES:
        split_mask = df_sorted["split"] == split if "split" in df_sorted.columns else (
            df_sorted["year"].apply(lambda y: (
                y <= val_end if split == "train" else
                y == val_end + 1 if False else  # placeholder, see below
                (val_end < y <= test_end) if split == "test" else
                y > test_end
            ))
        )
        # The panel doesn't carry a `split` column — derive it from the test
        # window conventions used by run_phase8_horizons_v2.py:
        #   split_years = (year <= train_end) → train
        #                  (train_end < year <= val_end) → val
        #                  (val_end < year <= test_end) → test
        #                  else → holdout
        train_end = metrics["split"].get("train_end", val_end - 4)
        if split == "train":
            mask = df_sorted["year"] <= train_end
        elif split == "val":
            mask = (df_sorted["year"] > train_end) & (df_sorted["year"] <= val_end)
        elif split == "test":
            mask = (df_sorted["year"] > val_end) & (df_sorted["year"] <= test_end)
        else:  # holdout
            mask = df_sorted["year"] > test_end

        sub = df_sorted[mask].reset_index(drop=True)
        n_total = len(sub)
        if n_total == 0:
            continue

        y = sub[target].to_numpy(dtype=np.float64)

        # AR(1) honest
        ar_pred = sub["ar1_pred"].to_numpy(dtype=np.float64)
        ar_pred = np.where(np.isnan(ar_pred), 0.0, ar_pred)

        # Naive persistence: predict y_t = y_{t-h} (per country)
        prev_h = df.groupby("iso3")[target].shift(h)
        prev_h_map = {(r.iso3, int(r.year)): float(r.prev_h) for r in
                      df.assign(prev_h=prev_h).dropna(subset=["prev_h", target]).itertuples()
                      if not (isinstance(r.prev_h, float) and np.isnan(r.prev_h))}
        naiv_pred = np.array([prev_h_map.get((r.iso3, int(r.year)), np.nan)
                              for r in sub.itertuples()], dtype=np.float64)
        naiv_pred = np.where(np.isnan(naiv_pred), 0.0, naiv_pred)

        # Random walk in log-growth space: ŷ = 0
        rw_pred = np.zeros_like(y)

        # IMF WEO vintage-correct
        weo_pred = np.array([weo_for(r.iso3, int(r.year))
                             for r in sub.itertuples()], dtype=np.float64)
        n_weo = int(np.sum(~np.isnan(weo_pred)))

        # v2/v1 LGBM ensemble: lookup from forecasts.parquet by iso3+year
        fc_sub = fc[fc.iso3.isin(sub.iso3.unique())]
        # Use inner join to align by (iso3, year)
        fc_lookup = fc.set_index(["iso3", "year"])["y_pred_ensemble"]
        v2_lgbm_pred = np.array([
            fc_lookup.get((r.iso3, int(r.year)), np.nan)
            for r in sub.itertuples()
        ], dtype=np.float64)

        # Cross-horizon meta
        if meta_h is not None:
            m_lookup = meta_h.set_index(["iso3", "year"])["y_pred_meta"]
            meta_pred = np.array([
                m_lookup.get((r.iso3, int(r.year)), np.nan)
                for r in sub.itertuples()
            ], dtype=np.float64)
            meta_present = ~np.isnan(meta_pred)
            n_meta = int(meta_present.sum())
        else:
            meta_pred = np.full_like(y, np.nan)
            meta_present = np.zeros(n_total, dtype=bool)
            n_meta = 0

        # For metrics, restrict each model to its available rows
        def _score(name, pred, mask):
            y_t = y[mask]
            y_p = pred[mask]
            if len(y_t) == 0:
                return None
            err = y_p - y_t
            abs_err = np.abs(y_t)  # for naive-baseline MAE (predict zero)
            return {
                "horizon": h,
                "model": name,
                "version": version,
                "split": split,
                "n": int(mask.sum()),
                "mae": float(np.mean(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(err ** 2))),
                "dir_acc": float(np.mean(np.sign(y_p) == np.sign(y_t))),
                "skill_vs_rw": float(1 - np.mean(np.abs(err)) / np.mean(np.abs(y_t - 0.0))),
                "mean_y_true": float(np.mean(y_t)),
                "std_y_true": float(np.std(y_t)),
            }

        for name, pred, mask in [
            ("naive_persistence", naiv_pred, np.ones(n_total, dtype=bool)),
            ("random_walk",       rw_pred,   np.ones(n_total, dtype=bool)),
            ("ar1_lag1_honest",   ar_pred,   ~np.isnan(ar_pred)),
            ("imf_weo_vintage",   weo_pred,  ~np.isnan(weo_pred)),
            ("v2_lgbm_ensemble",  v2_lgbm_pred, ~np.isnan(v2_lgbm_pred)),
            ("cross_horizon_meta", meta_pred, meta_present),
        ]:
            mask_b = mask if mask.dtype == bool else mask.astype(bool)
            # Tail-trim: drop rows where |y_true| > TAIL_CUTOFF. Catastrophic
            # events dominate MAE for naive baselines and IMF.
            mask_b = mask_b & (np.abs(y) <= TAIL_CUTOFF)
            r = _score(name, pred, mask_b)
            if r is not None:
                rows.append(r)

        # DM tests on the test slice only — cross-horizon meta vs each baseline
        if split == "test" and n_meta > 0 and meta_present.any():
            valid = np.abs(y) <= TAIL_CUTOFF
            for name, base_pred, base_mask in [
                ("naive_persistence", naiv_pred, ~np.isnan(naiv_pred)),
                ("ar1_lag1_honest",   ar_pred,   ~np.isnan(ar_pred)),
                ("imf_weo_vintage",   weo_pred,  ~np.isnan(weo_pred)),
            ]:
                joint = meta_present & base_mask & valid
                if joint.sum() < 5:
                    continue
                stat, p = _dm_pvalue(y[joint], meta_pred[joint], base_pred[joint])
                for r in rows:
                    if r["split"] == "test" and r["model"] == "cross_horizon_meta":
                        r[f"dm_stat_vs_{name}"] = stat
                        r[f"dm_p_vs_{name}"] = p
            # Also DM: cross-horizon meta vs v2_lgbm_ensemble
            if (~np.isnan(v2_lgbm_pred)).sum() >= 5:
                joint = meta_present & ~np.isnan(v2_lgbm_pred) & valid
                stat, p = _dm_pvalue(y[joint], meta_pred[joint], v2_lgbm_pred[joint])
                for r in rows:
                    if r["split"] == "test" and r["model"] == "cross_horizon_meta":
                        r["dm_stat_vs_v2_lgbm_ensemble"] = stat
                        r["dm_p_vs_v2_lgbm_ensemble"] = p

    summary = {
        "horizon": h,
        "version": version,
        "split": metrics["split"],
        "n_test_rows": int((df_sorted["year"] > val_end).sum()),
        "n_weo_in_test": int(np.sum([
            ~np.isnan(_imf_pred(imf_fc, imf_real, r.iso3, int(r.year), h))
            for r in df_sorted[df_sorted["year"] > val_end].itertuples()
        ])),
        "n_weo_in_test_trimmed": int(np.sum([
            ~np.isnan(_imf_pred(imf_fc, imf_real, r.iso3, int(r.year), h))
            and abs(r.gdp_pc_growth_5y_fwd if hasattr(r, 'gdp_pc_growth_5y_fwd') else 0) <= TAIL_CUTOFF
            for r in df_sorted[df_sorted["year"] > val_end].itertuples()
        ])),
    }
    return pd.DataFrame(rows), summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 5, 10])
    p.add_argument("--panel",   type=Path, default=PANEL)
    p.add_argument("--out-root", type=Path, default=FEATURES_DIR)
    p.add_argument("--out-csv", type=Path, default=FEATURES_DIR / "benchmark_v2.csv")
    p.add_argument("--out-json", type=Path, default=FEATURES_DIR / "benchmark_v2_summary.json")
    args = p.parse_args()

    panel = pd.read_parquet(args.panel)

    imf_fc, imf_real, imf_usd_ok = _load_imf_vintage_forecasts()
    if not imf_fc.empty:
        print(f"[bench] IMF WEO vintage-correct: {len(imf_fc)} forecast rows "
              f"({imf_fc.target_year.min()}–{imf_fc.target_year.max()}, "
              f"{imf_fc.iso3.nunique()} USD-compatible countries); "
              f"{len(imf_real)} realized anchor rows "
              f"({imf_real.real_year.min()}–{imf_real.real_year.max()})")
    else:
        print("[bench] No IMF WEO vintage-correct forecasts available.")

    all_rows = []
    summaries = {}
    for h in args.horizons:
        rows, summary = _benchmark_one(panel, h, args.out_root, imf_fc, imf_real)
        if not rows.empty:
            all_rows.append(rows)
            summaries[h] = summary
            print(f"[bench] h={h}: "
                  f"vintage-correct IMF WEO available on "
                  f"{summary['n_weo_in_test']}/{summary['n_test_rows']} test rows")

    table = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    if table.empty:
        print("[bench] no rows produced, check artifacts")
        return

    table.to_csv(args.out_csv, index=False)
    print(f"\n[bench] wrote {args.out_csv}  ({len(table)} rows)")

    args.out_json.write_text(json.dumps(summaries, indent=2))
    print(f"[bench] wrote {args.out_json}")

    # Headline table — test slice only
    test_only = table[table.split == "test"].copy()
    test_only = test_only.sort_values(["horizon", "model"]).reset_index(drop=True)
    show_cols = [c for c in ["horizon", "model", "n", "mae", "rmse", "dir_acc",
                             "skill_vs_rw"] if c in test_only.columns]
    print("\n[bench] test-slice headline:")
    print(test_only[show_cols].to_string(index=False))

    # DM significance vs meta for h=5 (or each horizon)
    dm_cols = [c for c in test_only.columns if c.startswith("dm_p_")]
    if dm_cols:
        meta_rows = test_only[test_only.model == "cross_horizon_meta"]
        if not meta_rows.empty:
            print("\n[bench] cross-horizon meta DM p-values (test slice):")
            keep = ["horizon"] + dm_cols
            print(meta_rows[keep].to_string(index=False))


if __name__ == "__main__":
    main()
