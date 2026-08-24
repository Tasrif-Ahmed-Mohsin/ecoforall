"""Cross-horizon meta-ensemble.

Loads all v2 horizon forecasts (h=1, 3, 5, 10) and stacks them as features for
a Ridge meta-learner. Each row in the meta-dataset corresponds to one
(iso3, year) observation at one specific horizon. Because horizons have
different target windows, we cannot trivially predict one y from the others,
but we CAN build a per-horizon stacked ensemble: for each h, the meta-features
are the per-horizon candidate predictions (ridge_h, lgbm_h, prior_h, ar1_h)
plus context (horizon, country tier, recent realised growth).

Trained on val slice, evaluated on test slice. Reports MAE per horizon
and a weighted-average skill-vs-prior metric.

LLM-fusion (Option C)
---------------------
When `data/features/llm_baseline_holdout.csv` exists, the per-row `llm_pred`
is added as a 4th meta-feature for the matching (iso3, year, horizon). This
CSV is a frozen snapshot of the DeepSeek-LLM zero-shot forecast, captured
once and reused here. We do NOT call the LLM at inference time: at predict
time we look up the cached value (the same pattern used for AR(1)).

Snapshot policy:
  * Missing CSV  -> warning, skip the LLM feature (meta-learner falls back
                    to the v1 3-feature recipe).
  * Stale CSV    -> warning if older than 90 days, but proceed. The CSV
                    remains the ground-truth contract for reproducibility.
  * Per-row join -> left join on (iso3, year, horizon). Rows without an
                    LLM prediction get NaN (handled by median imputation
                    in the Ridge pipeline).

Outputs:
  data/features/cross_horizon_meta/metrics.json
  data/features/cross_horizon_meta/predictions.parquet
  data/features/cross_horizon_meta/llm_snapshot_used.json   (audit trail)
  data/features/cross_horizon_meta/weight_decomposition.json
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Resolve features root dynamically from this script's location so the same
# script works in both `e:\research\project` and `e:\project_gmd` clones.
ROOT = Path(__file__).resolve().parents[1] / "data" / "features"
HORIZONS = [1, 3, 5, 10]
META_OUT = ROOT / "cross_horizon_meta"
LLM_HOLDOUT_CSV = ROOT / "llm_baseline_holdout.csv"
LLM_STALE_DAYS = 90
LLM_REQUIRED_COLS = ["iso3", "year", "horizon", "llm_pred"]
# Glob for split-tagged LLM snapshots (e.g. llm_baseline_val_h1.csv,
# llm_baseline_test.csv). The canonical "holdout" snapshot is loaded by
# name (above) and is also glob-matched so the union loader is the
# single source of truth for what the meta-learner sees as the LLM
# feature.
LLM_SNAPSHOT_GLOB = "llm_baseline*.csv"


def _dir_acc(y, p):
    return float(np.mean(np.sign(y) == np.sign(p)))


def _metrics(y, p):
    return {
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "dir_acc": _dir_acc(y, p),
    }


def _load_one_horizon(h: int) -> pd.DataFrame:
    p = ROOT / f"horizon_{h}y_v2" / "forecasts.parquet"
    df = pd.read_parquet(p)
    df = df.rename(columns={
        "y_pred_ridge": f"ridge_h{h}",
        "y_pred_lgbm":  f"lgbm_h{h}",
        "y_pred_prior": f"prior_h{h}",
    })
    keep = ["iso3", "year", "split", "y_true",
            f"ridge_h{h}", f"lgbm_h{h}", f"prior_h{h}"]
    return df[keep].copy()


def _load_llm_snapshot() -> tuple[pd.DataFrame, dict]:
    """Load and union ALL frozen LLM zero-shot forecast snapshots.

    Reads every file matching ``LLM_SNAPSHOT_GLOB`` in the features root
    (e.g. ``llm_baseline_holdout.csv``, ``llm_baseline_val_h1.csv``,
    ``llm_baseline_val_h3.csv``). Each file is independently normalized
    then they are concatenated and de-duplicated on
    ``(iso3, year, horizon)`` (first non-null wins).

    Returns (df, audit_meta). df has columns
    ``[iso3, year, horizon, llm_pred]`` with one row per (iso3, year,
    horizon) the LLM was queried on across ANY snapshot.

    audit_meta is written to ``llm_snapshot_used.json`` and records:
      - per-file mtime, age, rowcount, status (ok/missing/stale/malformed)
      - combined rowcount, n_with_pred, attached_row_count, coverage
        (filled in by ``build_meta_dataset``)
      - aggregated model_versions, age_days (newest of all snapshots)

    Policy:
      * Missing ALL files -> df is empty, audit_meta status='missing'.
      * Any stale file (>LLM_STALE_DAYS days) -> flagged in that file's
        status but the union proceeds.
      * Any malformed file -> its rows are dropped, status='malformed'
        recorded per-file; the union still uses whatever loaded cleanly.
    """
    paths = sorted(ROOT.glob(LLM_SNAPSHOT_GLOB))
    audit: dict = {
        "glob": str(LLM_SNAPSHOT_GLOB),
        "files": [],
        "rowcount": 0,
        "n_with_pred": 0,
        "model_versions": [],
        "mtime_iso": None,
        "age_days": None,
        "stale_threshold_days": LLM_STALE_DAYS,
        "status": "missing",
    }
    if not paths:
        print(f"[meta][llm] no snapshots matched {LLM_SNAPSHOT_GLOB} "
              f"in {ROOT} -- meta-learner will fall back to v1 3-feature recipe")
        return pd.DataFrame(columns=LLM_REQUIRED_COLS), audit

    frames: list[pd.DataFrame] = []
    for p in paths:
        file_audit: dict = {"path": str(p), "status": "missing", "rowcount": 0}
        st = p.stat()
        file_audit["mtime_iso"] = time.strftime(
            "%Y-%m-%dT%H:%M:%S", time.localtime(st.st_mtime))
        file_audit["age_days"] = (time.time() - st.st_mtime) / 86400.0

        if file_audit["age_days"] > LLM_STALE_DAYS:
            file_audit["status"] = "stale"

        try:
            raw = pd.read_csv(p)
        except Exception as e:
            print(f"[meta][llm] failed to read {p}: {e}")
            file_audit["status"] = "malformed"
            audit["files"].append(file_audit)
            continue

        missing_cols = [c for c in LLM_REQUIRED_COLS if c not in raw.columns]
        if missing_cols:
            print(f"[meta][llm] {p.name} missing columns {missing_cols} -- skipping")
            file_audit["status"] = "malformed"
            audit["files"].append(file_audit)
            continue

        sub = raw.copy()
        sub["year"] = pd.to_numeric(sub["year"], errors="coerce")
        sub["horizon"] = pd.to_numeric(sub["horizon"], errors="coerce").astype("Int64")
        sub["llm_pred"] = pd.to_numeric(sub["llm_pred"], errors="coerce")
        sub = sub.dropna(subset=["iso3", "year", "horizon", "llm_pred"])
        # Reduce to one row per (iso3, year, horizon): keep the first non-null.
        sub = sub.sort_values(["iso3", "year", "horizon"]).drop_duplicates(
            subset=["iso3", "year", "horizon"], keep="first")

        file_audit["rowcount"] = int(len(sub))
        if file_audit["status"] == "missing":
            file_audit["status"] = "ok"
        if "model" in sub.columns:
            file_audit["model_versions"] = sorted(
                sub["model"].dropna().unique().tolist())
            audit["model_versions"] = sorted(set(
                audit["model_versions"] + file_audit["model_versions"]))
        audit["files"].append(file_audit)
        frames.append(sub[LLM_REQUIRED_COLS])

        print(f"[meta][llm] {p.name}: {file_audit['rowcount']:,} rows, "
              f"status={file_audit['status']}, age={file_audit['age_days']:.1f}d")

    if not frames:
        audit["status"] = "missing"
        print("[meta][llm] no snapshot files loaded successfully")
        return pd.DataFrame(columns=LLM_REQUIRED_COLS), audit

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["iso3", "year", "horizon"]).drop_duplicates(
        subset=["iso3", "year", "horizon"], keep="first")
    audit["rowcount"] = int(len(df))
    audit["n_with_pred"] = int(df["llm_pred"].notna().sum())
    # Aggregate mtime / age across files: newest mtime, oldest age for visibility.
    newest_mtime = max(
        time.mktime(time.strptime(f["mtime_iso"], "%Y-%m-%dT%H:%M:%S"))
        for f in audit["files"] if f.get("mtime_iso"))
    audit["mtime_iso"] = time.strftime(
        "%Y-%m-%dT%H:%M:%S", time.localtime(newest_mtime))
    audit["age_days"] = (time.time() - newest_mtime) / 86400.0
    audit["status"] = "ok"
    print(f"[meta][llm] union OK: {audit['rowcount']:,} rows from "
          f"{len(frames)} file(s), models={audit['model_versions']}, "
          f"newest_age={audit['age_days']:.1f}d")
    return df.reset_index(drop=True), audit


def _panel_path() -> Path | None:
    for cand in [
        ROOT / "panel_wide.parquet",
        ROOT.parent / "panel.parquet",
    ]:
        if cand.exists():
            return cand
    return None


def _add_ar1(panel: pd.DataFrame, h: int) -> pd.Series:
    """AR(1) honest-fit: predict the future h-year growth using the previous
    realised h-year growth per country. Computed from panel_wide.parquet.

    Returns a Series indexed by (iso3, year).
    """
    panel_p = _panel_path()
    if panel_p is None:
        return pd.Series(dtype=float)
    P = pd.read_parquet(panel_p)
    target = f"gdp_pc_growth_{h}y_fwd"
    if target not in P.columns:
        return pd.Series(dtype=float)
    P = P[["iso3", "year", target]].dropna().sort_values(["iso3", "year"])
    P["lag1"] = P.groupby("iso3")[target].shift(1)
    return P.set_index(["iso3", "year"])["lag1"]


def build_meta_dataset(panel_v2: pd.DataFrame | None = None,
                       llm_snapshot: pd.DataFrame | None = None,
                       llm_audit: dict | None = None,
                       ) -> tuple[pd.DataFrame, list[int], dict]:
    """Stack all horizons into a single meta-dataset with horizon as a feature.

    Each row = (iso3, year, h). Meta-features:
      ridge_h, lgbm_h, prior_h, (optional ar1_h), (optional llm_pred),
      horizon, year, iso3

    Returns (meta, horizons, llm_audit_dict). The llm_audit_dict is the same
    object passed in (or a new dict if None), mutated with attached_row_count.
    """
    frames = []
    for h in HORIZONS:
        d = _load_one_horizon(h)
        d["horizon"] = h
        frames.append(d)
    meta = pd.concat(frames, ignore_index=True)

    # Vectorized keep-filter (per-row, per-horizon): drop rows where ANY of the
    # 3 base predictions is NaN FOR THAT ROW'S HORIZON. After concat, each row
    # only fills in the 3 columns for its own horizon; the others are NaN by
    # design. A naive global dropna() would zero out the entire frame.
    n0 = len(meta)
    horizon_arr = meta["horizon"].to_numpy()
    keep_mask = np.zeros(len(meta), dtype=bool)
    for h in HORIZONS:
        m = horizon_arr == h
        if not m.any():
            continue
        sub = meta.loc[m]
        ok = (sub[f"ridge_h{h}"].notna()
              & sub[f"lgbm_h{h}"].notna()
              & sub[f"prior_h{h}"].notna()).to_numpy()
        keep_mask[m] = ok
    meta = meta.loc[keep_mask].reset_index(drop=True)
    print(f"[meta] dropped {(~keep_mask).sum():,} rows with NaN predictions "
          f"(kept {len(meta):,}/{n0:,})")

    # Try to attach AR(1) honest-fit per (iso3, year, horizon)
    panel_p = _panel_path()
    if panel_p is not None:
        P = pd.read_parquet(panel_p)
        ar1_cols = {}
        for h in HORIZONS:
            target = f"gdp_pc_growth_{h}y_fwd"
            if target not in P.columns:
                continue
            tmp = P[["iso3", "year", target]].dropna().sort_values(["iso3", "year"]).copy()
            tmp["ar1"] = tmp.groupby("iso3")[target].shift(1)
            ar1_cols[h] = tmp.set_index(["iso3", "year"])["ar1"]
        if ar1_cols:
            meta["ar1"] = [
                ar1_cols[h].get((iso, yr), np.nan) if h in ar1_cols else np.nan
                for iso, yr, h in zip(meta["iso3"], meta["year"], meta["horizon"])
            ]
            n_with_ar1 = meta["ar1"].notna().sum()
            print(f"[meta] AR(1) honest-fit attached for {n_with_ar1}/{len(meta)} rows")
        else:
            print("[meta] no AR(1) targets found in panel; skipping")
    else:
        print("[meta] no panel.parquet found; skipping AR(1)")

    # Attach LLM frozen-snapshot prediction per (iso3, year, horizon), if available.
    # No row drops: rows without llm_pred remain in the meta-dataset and will be
    # NaN-imputed in the meta-learner pipeline (same handling as AR(1)).
    if llm_audit is None:
        llm_audit = {"status": "not_loaded"}
    if llm_snapshot is None or llm_snapshot.empty:
        meta["llm_pred"] = np.nan
        print("[meta] no LLM snapshot provided; llm_pred column is all-NaN")
    else:
        meta = meta.merge(
            llm_snapshot, on=["iso3", "year", "horizon"], how="left")
        # merge reorders columns; re-attach llm_pred at known position
        n_with_llm = meta["llm_pred"].notna().sum()
        llm_audit["attached_row_count"] = int(n_with_llm)
        llm_audit["coverage"] = float(n_with_llm / max(len(meta), 1))
        print(f"[meta] llm_pred attached for {n_with_llm}/{len(meta)} rows "
              f"({100 * llm_audit['coverage']:.1f}%)")
    return meta, HORIZONS, llm_audit


def train_meta(meta: pd.DataFrame, horizons: list[int]) -> tuple[Pipeline, dict, pd.DataFrame]:
    """Train Ridge meta-learner on val, evaluate on test. Reports per-horizon.

    Returns (fitted_pipeline, metrics_dict, per_row_predictions_df).
    The third return is a small dataframe with per-row prediction decomposition:
    [iso3, year, horizon, y_true, prior, lgbm, ridge, ar1, llm_pred,
     pred_meta, delta_vs_lgbm_only, llm_attribution].
    """
    feature_cols = []
    for h in horizons:
        feature_cols += [f"ridge_h{h}", f"lgbm_h{h}", f"prior_h{h}"]
    if "ar1" in meta.columns:
        feature_cols.append("ar1")
    # LLM is optional: only present (and non-all-NaN) if the snapshot was loaded.
    if "llm_pred" in meta.columns and meta["llm_pred"].notna().any():
        feature_cols.append("llm_pred")
    # Context features
    feature_cols += ["horizon"]

    # Split convention:
    #   val  ->  meta-train (held out from base learners, the honest signal)
    #   test ->  meta-test
    train_mask = meta["split"] == "val"
    test_mask = meta["split"] == "test"

    X_tr_df = meta.loc[train_mask, feature_cols]
    X_te_df = meta.loc[test_mask, feature_cols]
    y_tr = meta.loc[train_mask, "y_true"].to_numpy()
    y_te = meta.loc[test_mask, "y_true"].to_numpy()

    # Pipeline now carries NaN handling upstream of scaling, so ar1 and llm_pred
    # NaNs are filled with median (=0 in practice since the inputs are
    # centred growth values with median ~0). This replaces the old
    # `X[np.isnan(X)] = 0.0` line which only handled ar1.
    # `keep_empty_features=True` (sklearn >= 1.4) lets the LLM column survive
    # even when training rows have all-NaN for it (since the snapshot only
    # covers test/holdout years).
    try:
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    except TypeError:
        # Older sklearn without the kwarg -- fall back to plain median.
        imputer = SimpleImputer(strategy="median")
    pipe = Pipeline([
        ("imputer", imputer),
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], cv=5)),
    ])
    pipe.fit(X_tr_df, y_tr)
    pred_te = pipe.predict(X_te_df)
    metrics_overall = _metrics(y_te, pred_te)

    # Per-horizon breakdown
    per_h = {}
    pred_full = np.full(len(meta), np.nan)
    pred_full[test_mask.to_numpy()] = pred_te
    meta["pred_meta"] = pred_full

    has_llm_col = "llm_pred" in meta.columns and meta["llm_pred"].notna().any()
    for h in horizons:
        m = (meta["horizon"] == h) & test_mask
        if m.sum() == 0:
            continue
        sub = meta.loc[m]
        per_h[f"h{h}"] = {
            "n": int(m.sum()),
            **{k: float(v) for k, v in _metrics(
                sub["y_true"].to_numpy(), sub["pred_meta"].to_numpy()).items()},
            "prior_mae": float(mean_absolute_error(
                sub["y_true"].to_numpy(), sub[f"prior_h{h}"].to_numpy())),
            "lgbm_mae": float(mean_absolute_error(
                sub["y_true"].to_numpy(), sub[f"lgbm_h{h}"].to_numpy())),
            "ridge_mae": float(mean_absolute_error(
                sub["y_true"].to_numpy(), sub[f"ridge_h{h}"].to_numpy())),
        }
        if has_llm_col:
            mask_llm_present = sub["llm_pred"].notna()
            if mask_llm_present.any():
                per_h[f"h{h}"]["llm_mae"] = float(mean_absolute_error(
                    sub.loc[mask_llm_present, "y_true"].to_numpy(),
                    sub.loc[mask_llm_present, "llm_pred"].to_numpy()))
                per_h[f"h{h}"]["llm_n"] = int(mask_llm_present.sum())

    # ---- per-source weight decomposition (coef * std-of-feature) ----------
    # Standardised coefficients scaled back by feature std are a fair
    # "contribution" measure for a linear stacker.
    ridge = pipe.named_steps["ridge"]
    scaler = pipe.named_steps["scaler"]
    coefs_std = ridge.coef_  # coefficients in standardised space
    stds = scaler.scale_     # std of each feature as seen at training time
    intercept = float(ridge.intercept_)
    contrib = coefs_std * stds  # contribution in original scale
    weight_decomposition = {
        "intercept": intercept,
        "features": [
            {"name": name, "coef_std": float(c),
             "std_at_train": float(s), "contribution": float(c * s)}
            for name, c, s in zip(feature_cols, coefs_std, stds)
        ],
        "ridge_alpha": float(ridge.alpha_),
    }
    total_abs = sum(abs(x["contribution"]) for x in weight_decomposition["features"]) or 1.0
    for f in weight_decomposition["features"]:
        f["share_of_total_abs"] = float(abs(f["contribution"]) / total_abs)

    # ---- per-row prediction breakdown (audit) -----------------------------
    # Compute the contribution of LLM to each prediction by zeroing out its
    # learned coefficient and re-predicting. delta_pred = pred_with_llm - pred_without_llm
    # is the "amount the LLM moved the prediction" for that row.
    decomp = pd.DataFrame({
        "iso3": meta.loc[test_mask, "iso3"].to_numpy(),
        "year": meta.loc[test_mask, "year"].to_numpy(),
        "horizon": meta.loc[test_mask, "horizon"].to_numpy(),
        "y_true": y_te,
    })
    # Save raw base predictions for the row's own horizon. `decomp` is test-only
    # so we operate on `meta.loc[test_mask]` (same row order as `decomp`).
    meta_te = meta.loc[test_mask].reset_index(drop=True)
    decomp["prior"] = np.nan
    decomp["lgbm"] = np.nan
    decomp["ridge"] = np.nan
    for h in horizons:
        mh = meta_te["horizon"] == h
        if not mh.any():
            continue
        decomp.loc[mh.to_numpy(), "prior"] = meta_te.loc[mh, f"prior_h{h}"].to_numpy()
        decomp.loc[mh.to_numpy(), "lgbm"] = meta_te.loc[mh, f"lgbm_h{h}"].to_numpy()
        decomp.loc[mh.to_numpy(), "ridge"] = meta_te.loc[mh, f"ridge_h{h}"].to_numpy()
    decomp["pred_meta"] = pred_te
    decomp["llm_attribution"] = 0.0
    if has_llm_col:
        decomp["llm_pred"] = meta_te["llm_pred"].to_numpy()
        # LLM attribution: how much did the LLM column move this row's
        # prediction vs the median-filled baseline? For a linear model:
        #   delta = coef_llm_in_orig_scale * (llm_pred - llm_median)
        # Rows where llm_pred is NaN get delta=0 by construction.
        llm_idx = feature_cols.index("llm_pred") if "llm_pred" in feature_cols else -1
        if llm_idx >= 0:
            imputer = pipe.named_steps["imputer"]
            llm_median = float(imputer.statistics_[llm_idx])
            if not np.isfinite(llm_median):
                # Fallback: imputer saw all-NaN during training (LLM snapshot
                # only covers test years). Use empirical median of the
                # observed test values instead.
                llm_median = float(decomp["llm_pred"].median())
            llm_coef_in_orig_scale = float(coefs_std[llm_idx] * stds[llm_idx])
            decomp["llm_attribution"] = llm_coef_in_orig_scale * (
                decomp["llm_pred"].fillna(llm_median).to_numpy() - llm_median)

    chosen_alpha = float(pipe.named_steps["ridge"].alpha_)
    metrics_out = {
        "overall_test": metrics_overall,
        "per_horizon_test": per_h,
        "feature_cols": feature_cols,
        "ridge_alpha": chosen_alpha,
        "meta_train_rows": int(train_mask.sum()),
        "meta_test_rows": int(test_mask.sum()),
        "n_test_rows_with_llm": int(
            meta.loc[test_mask, "llm_pred"].notna().sum()) if "llm_pred" in meta.columns else 0,
        "weight_decomposition": weight_decomposition,
    }
    return pipe, metrics_out, decomp


def main() -> None:
    global HORIZONS
    META_OUT.mkdir(parents=True, exist_ok=True)
    available = [h for h in HORIZONS if (ROOT / f"horizon_{h}y_v2" / "forecasts.parquet").exists()]
    missing = [h for h in HORIZONS if h not in available]
    print(f"[meta] available horizons: {available}")
    if missing:
        print(f"[meta] WARNING: missing forecasts for horizons {missing} -- skipping them")
    # Mutate the module-level HORIZONS for build_meta_dataset
    HORIZONS = available or HORIZONS
    if not available:
        raise SystemExit("No v2 horizons available. Run scripts/run_phase8_horizons_v2.py first.")

    # Load frozen LLM snapshot (Option C) BEFORE building the meta-dataset so
    # the LLM column is part of the meta-merge from the start. If the snapshot
    # is missing or stale we still proceed, just without that feature.
    llm_df, llm_audit = _load_llm_snapshot()

    meta, horizons, llm_audit = build_meta_dataset(
        llm_snapshot=llm_df, llm_audit=llm_audit)
    print(f"[meta] meta-dataset rows={len(meta):,}  iso3s={meta['iso3'].nunique()}  "
          f"splits={meta['split'].value_counts().to_dict()}")

    pipe, metrics, decomp = train_meta(meta, horizons)
    print(f"[meta] ridge alpha={metrics['ridge_alpha']}  "
          f"overall_test MAE={metrics['overall_test']['mae']:.4f}  "
          f"dir_acc={metrics['overall_test']['dir_acc']:.3f}")
    print(f"[meta] per-horizon test:")
    header = f"{'h':<6} {'n':>5} {'meta':>8} {'lgbm':>8} {'ridge':>8} {'prior':>8} {'llm':>8} {'meta_dir':>9} {'vs_prior':>9}"
    print(header)
    for k, v in metrics["per_horizon_test"].items():
        delta = v["prior_mae"] - v["mae"]
        llm_mae_str = f"{v.get('llm_mae', float('nan')):>8.4f}" if "llm_mae" in v else f"{'':>8}"
        print(f"{k:<6} {v['n']:>5} {v['mae']:>8.4f} {v['lgbm_mae']:>8.4f} "
              f"{v['ridge_mae']:>8.4f} {v['prior_mae']:>8.4f} {llm_mae_str} "
              f"{v['dir_acc']:>9.3f} {delta:>+9.4f}")

    # ---- write outputs -----------------------------------------------------
    (META_OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    joblib.dump(pipe, META_OUT / "meta_ridge.joblib")
    meta.to_parquet(META_OUT / "predictions.parquet")
    decomp.to_parquet(META_OUT / "holdout_decomposition.parquet")
    (META_OUT / "weight_decomposition.json").write_text(
        json.dumps(metrics["weight_decomposition"], indent=2))
    # Audit trail for the LLM snapshot used in this run
    (META_OUT / "llm_snapshot_used.json").write_text(
        json.dumps(llm_audit, indent=2))

    # Console summary of LLM contribution
    print(f"[meta] weight decomposition (top-8 by abs contribution):")
    feats = sorted(metrics["weight_decomposition"]["features"],
                   key=lambda f: abs(f["contribution"]), reverse=True)
    for f in feats[:8]:
        print(f"  {f['name']:<14}  coef_std={f['coef_std']:+.4f}  "
              f"contrib={f['contribution']:+.5f}  "
              f"share={f['share_of_total_abs']*100:5.1f}%")
    if "llm_pred" in metrics["feature_cols"]:
        llm_share = next((f["share_of_total_abs"] for f in feats
                          if f["name"] == "llm_pred"), 0.0)
        print(f"[meta] LLM column share of total |contribution|: {llm_share*100:.1f}%")
    print(f"[meta] wrote {META_OUT}")


if __name__ == "__main__":
    main()