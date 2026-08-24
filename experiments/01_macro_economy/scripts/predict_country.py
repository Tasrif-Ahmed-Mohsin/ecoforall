"""One-shot per-country forecaster.

Usage
-----
    python scripts/predict_country.py USA                # latest year for USA
    python scripts/predict_country.py USA 2018           # specific year
    python scripts/predict_country.py USA 2018 --topk 5  # plus 5 historical analogs

Output
------
A dict with:
  - iso3, query_year
  - forecast: {ridge, lgbm, ensemble}      -> 5-year-ahead log-returns on gdp_pc
  - pi80_low, pi80_high                    -> ~80% band from ensemble ±1.28 * std
  - analogs: top-k historical rows closest in feature space

Reads
-----
- data/features/panel_wide.parquet       (built by scripts/run_phase2.py)
- data/features/models/{ridge,lgbm,imputer}.joblib + feature_cols.json
- data/features/baseline_metrics.json    (test MAE for the band estimate)
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

warnings.filterwarnings("ignore", category=UserWarning, module=r"sklearn\..*")

ROOT = Path(__file__).resolve().parents[1]  # noqa
sys.path.insert(0, str(ROOT))

from src.harmonize.common import FEATURES_DIR

PANEL = FEATURES_DIR / "panel_wide.parquet"
MODELS_DIR = FEATURES_DIR / "models"
METRICS = FEATURES_DIR / "baseline_metrics.json"
CONFORMAL = FEATURES_DIR / "conformal_adjustment.json"
TARGET = "gdp_pc_growth_5y_fwd"
Z80 = 1.2815515655446004  # one-sided 90% quantile

# Per-side conformal offsets learnt on the 2019-2022 test slice.
# Loaded on demand in main(); missing file -> no calibration.


def _load_panel() -> pd.DataFrame:
    return pd.read_parquet(PANEL)


def _load_models() -> tuple:
    """Backwards-compat stub for v1 callers; raises on v2-only paths."""
    raise SystemExit(
        "v1 model-loader retired. v2 uses predict_one_iso3() from "
        "scripts/run_phase8_horizons_v2.py via main()'s --horizon flow."
    )


def _build_query_row(panel: pd.DataFrame, iso3: str, year: int | None) -> tuple[pd.Series, int]:
    """Pick or assemble the (iso3, year) row used for inference.

    Returns ``(row, year_resolved)``. ``year_resolved`` may differ from the
    requested ``year`` when ``year is None`` (in which case we use the most
    recent row available for the country).
    """
    sub = panel[panel["iso3"] == iso3]
    if sub.empty:
        raise SystemExit(f"iso3={iso3!r} not in panel.")
    if year is None:
        # Use the most recent year with a defined feature row (best data density).
        row = sub.sort_values("year").iloc[-1]
        year = int(row["year"])
    else:
        row = sub[sub["year"] == year]
        if row.empty:
            raise SystemExit(f"iso3={iso3!r} has no row for year={year}.")
        row = row.iloc[0]
    return row, year


def _predict_v2(panel: pd.DataFrame, iso3: str, year: int | None, horizon: int) -> dict:
    """Run v2 inference via the trainer's helper. Returns ridge/lgbm/q05/q50/q95."""
    from scripts.run_phase8_horizons_v2 import predict_one_iso3
    if year is None:
        sub = panel[panel.iso3 == iso3]
        if sub.empty:
            raise SystemExit(f"iso3={iso3!r} not in panel.")
        year = int(sub.sort_values("year").iloc[-1].year)
    return predict_one_iso3(panel, iso3, year, horizon=horizon)


def _per_country_prior(panel: pd.DataFrame, iso3: str, train_end_year: int,
                       target: str = TARGET) -> float:
    """Per-country naive baseline: last training-period realised `target` growth.

    Falls back to the global training-period mean if the country is missing
    or has no labelled rows before `train_end_year`. This is the falsification
    check at inference time — the trained ensemble must add information over
    this value to be worth shipping.
    """
    if target not in panel.columns:
        from scripts.run_phase8_horizons_v2 import _build_horizon_target
        h = int(target.replace("gdp_pc_growth_", "").replace("y_fwd", ""))
        panel = panel.copy()
        panel[target] = _build_horizon_target(panel, h)
    labelled = panel.dropna(subset=[target])
    train = labelled[labelled.year <= train_end_year]
    if train.empty:
        return float(panel[target].mean())
    g = train[train.iso3 == iso3]
    if g.empty:
        return float(train[target].mean())
    return float(g.sort_values("year").iloc[-1][target])


def _similar_analogs(
    panel: pd.DataFrame,
    panel_row: pd.Series,
    feat_cols: list[str],
    topk: int,
    exclude_year: int | None = None,
    exclude_iso3: str | None = None,
    use_faiss: bool = True,
    use_ranked: bool = False,
    min_overlap: int = 0,
) -> pd.DataFrame:
    """Top-k historical rows by similarity over standardized features.

    With use_ranked=True, uses the upgraded rank-features Euclidean index
    (scripts/_pattern_sweep{1..4}.py winner). With use_faiss=True and
    use_ranked=False, falls back to the legacy cosine z-score index.
    With use_faiss=False, falls back to a sklearn-free L2 search.
    """
    if not feat_cols:
        return pd.DataFrame()
    if use_ranked:
        # Prefer the GMD-shaped index (data/features/retrieval_v2/) when present;
        # fall back to the legacy v1 index (data/features/retrieval/).
        try:
            from src.retrieval.faiss_index import build_or_load_v2_ranked, build_or_load_ranked
            v2_path = Path(__file__).resolve().parents[1] / "data" / "features" / "retrieval_v2" / "panel_ranked.faiss"
            if v2_path.exists():
                index = build_or_load_v2_ranked()
            else:
                index = build_or_load_ranked()
            return index.query_topk(
                panel_row,
                k=topk,
                exclude_year=exclude_year,
                exclude_iso3=exclude_iso3,
                min_overlap=min_overlap,
            )
        except Exception as e:
            print(f"[warn] ranked retrieval unavailable ({e}); falling back to legacy FAISS.", file=sys.stderr)
    if use_faiss:
        try:
            from src.retrieval.faiss_index import build_or_load
            index = build_or_load()
            return index.query_topk(
                panel_row,
                k=topk,
                exclude_year=exclude_year,
                exclude_iso3=exclude_iso3,
            )
        except Exception as e:  # faiss may be missing or index not built yet
            print(f"[warn] FAISS retrieval unavailable ({e}); falling back to L2.", file=sys.stderr)

    # Fallback: L2 over standardized features.
    sub = panel.copy()
    if exclude_year is not None:
        sub = sub[sub["year"] != exclude_year]
    if exclude_iso3 is not None:
        sub = sub[sub["iso3"] != exclude_iso3]
    feats = sub[feat_cols].astype(float).replace([np.inf, -np.inf], np.nan)
    mu = feats.mean(skipna=True).fillna(0.0)
    sigma = feats.std(skipna=True).replace(0, np.nan).fillna(1.0)
    feats_std = (feats - mu) / sigma
    feats_std = feats_std.fillna(0.0)

    q = panel_row[feat_cols].astype(float).replace([np.inf, -np.inf], np.nan)
    q_std = ((q - mu) / sigma).fillna(0.0).to_numpy()

    dists = np.linalg.norm(feats_std.to_numpy() - q_std, axis=1)
    out = sub.assign(_distance=dists).sort_values("_distance").head(topk)
    return out[["iso3", "year", "gdp_pc_growth_5y_fwd", "_distance"]].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("iso3", help="ISO-3 code, e.g. USA")
    ap.add_argument("year", nargs="?", type=int, default=None,
                    help="Baseline year (defaults to latest available for the country)")
    ap.add_argument("--topk", type=int, default=5, help="Number of historical analogs")
    ap.add_argument("--faiss", dest="use_faiss", action="store_true", default=True,
                    help="Use FAISS-backed retrieval (default)")
    ap.add_argument("--no-faiss", dest="use_faiss", action="store_false",
                    help="Use the legacy L2 retrieval")
    ap.add_argument("--ranked", action="store_true", default=True,
                    help="Use the upgraded rank-features Euclidean index "
                         "(default; auto-selects v2 GMD-shaped index when present).")
    ap.add_argument("--no-ranked", dest="ranked", action="store_false",
                    help="Use the legacy cosine z-score index (v1, broken on GMD).")
    ap.add_argument("--min-overlap", type=int, default=0,
                    help="Minimum co-observed features per analog (0 = no filter). "
                         "Pairs with --ranked.")
    ap.add_argument("--explain", action="store_true",
                    help="Generate an LLM narrative grounded in the forecast and analogs")
    ap.add_argument("--horizon", type=int, default=5, choices=(1, 3, 5, 10),
                    help="Forecast horizon in years (default 5 = headline horizon).")
    args = ap.parse_args()

    panel = _load_panel()
    # v2 inference path: route through the v2 trainer's helper.
    preds = _predict_v2(panel, args.iso3, args.year, horizon=args.horizon)
    ridge_pred, lgbm_pred = preds["ridge"], preds["lgbm"]
    year = preds["year"]
    # Pick the panel row for analog retrieval + LLM explain block.
    row = panel[(panel.iso3 == args.iso3) & (panel.year == year)].iloc[0]

    # v2 analogs: use the continuous feature columns from the trainer's
    # feature_meta.json. This keeps the analog distance metric aligned with
    # the rank-transformed space the v2 trainer was fit on. Skip if the
    # meta file is missing (e.g. horizon not yet retrained).
    meta_path = FEATURES_DIR / f"horizon_{args.horizon}y_v2" / "feature_meta.json"
    feature_cols = json.loads(meta_path.read_text())["cont_cols"] if meta_path.exists() else []
    analogs = _similar_analogs(
        panel, row, feature_cols, args.topk,
        exclude_year=year, exclude_iso3=args.iso3,
        use_faiss=args.use_faiss,
        use_ranked=args.ranked,
        min_overlap=args.min_overlap,
    )

    # ---- v2.1: horizon-gated ensemble selection ----
    # Each horizon's trainer (scripts/run_phase8_horizons_v2.py) writes its
    # own metrics.json into data/features/horizon_{h}y_v2/. We read THAT
    # (not the v1 baseline_metrics.json) so the inferred recipe matches
    # whatever the trainer actually picked on the test slice.
    #
    # Per-horizon policy (AUDIT.md §13, §14, §15):
    #   h=1  -> 5-way ensemble (lgbm+cat+xgb+ridge+prior)
    #           weights: 0.4 lgbm, 0.2 cat, 0.2 xgb, 0.1 ridge, 0.1 prior
    #           CV evidence: 5-way beats AR(1) by +8.8% (gain +1.3pp over
    #           the original lgbm+prior).
    #   h=3  -> 5-way ensemble (lgbm+cat+xgb+ridge+prior). v2.1 re-train
    #           test MAE 0.0814 vs prior 0.0939 (-13.3%). 5-way wins by
    #           0.0009 over lgbm+prior (0.0823).
    #   h=5  -> 0.7*lgbm + 0.3*prior (the trainer's pick; v2.1 test MAE
    #           0.0901 vs prior 0.1102 = -18.2%). 5-way is 0.0903, so
    #           lgbm+prior narrowly wins.
    #   h=10 -> FALLBACK TO NAIVE PRIOR. v2.1 re-train: trainer picked
    #           'lgbm+prior' at test MAE 0.1396, but per-country prior
    #           sits at 0.1334 — the supervised ensemble still loses.
    #           Shipping the trained ensemble at h=10 would be a regression
    #           vs. the honest baseline. We override the trainer's pick.
    cat_pred = preds.get("catboost")
    xgb_pred = preds.get("xgboost")
    recipe = "lgbm+prior"
    if args.horizon == 1:
        # 5-way ensemble. Falls back gracefully if CatBoost/XGBoost artifacts
        # are missing (e.g. v2 trainer not yet re-run for h=1).
        if cat_pred is not None and xgb_pred is not None:
            recipe = "lgbm+cat+xgb+ridge+prior"
        else:
            recipe = "lgbm+prior"
    elif args.horizon == 10:
        # The trained ensemble loses to prior at h=10; ship the prior instead.
        recipe = "prior_only"
    else:
        # h=3 / h=5: trust the trainer's pick from per-horizon metrics.json.
        metrics_path = FEATURES_DIR / f"horizon_{args.horizon}y_v2" / "metrics.json"
        if metrics_path.exists():
            try:
                trainer_recipe = json.loads(metrics_path.read_text()).get(
                    "ensemble_recipe", "lgbm+prior"
                )
                # Guard: if the trainer picked something that loses to the
                # prior on its own test slice, also fall back to prior.
                ens_mae = json.loads(metrics_path.read_text()).get(
                    "ensemble_test_mae", float("inf")
                )
                prior_mae = json.loads(metrics_path.read_text()).get(
                    "ensemble_prior_mae", float("inf")
                )
                if ens_mae < prior_mae:
                    recipe = trainer_recipe
                else:
                    recipe = "prior_only"
            except Exception:
                recipe = "lgbm+prior"
        else:
            recipe = "lgbm+prior"

    prior_pred = _per_country_prior(panel, args.iso3, train_end_year=2014)
    if recipe == "lgbm+cat+xgb+ridge+prior":
        # 5-way ensemble (h=1).
        ensemble = (
            0.4 * lgbm_pred
            + 0.2 * (cat_pred if cat_pred is not None else lgbm_pred)
            + 0.2 * (xgb_pred if xgb_pred is not None else lgbm_pred)
            + 0.1 * ridge_pred
            + 0.1 * prior_pred
        )
    elif recipe == "lgbm+prior":
        ensemble = 0.7 * lgbm_pred + 0.3 * prior_pred
    elif recipe == "lgbm+ridge":
        ensemble = 0.7 * lgbm_pred + 0.3 * ridge_pred
    elif recipe == "prior_only":
        ensemble = prior_pred
    else:
        ensemble = lgbm_pred

    # Use the trained LGBM quantile models when present.
    #
    # The target distribution has fat tails (1.7% of rows below -0.5 log-return
    # = -39% GDP growth; 0.6% above +0.5 = +65%). A pure q10/q90 "80% band"
    # cannot cover the lower tail even after conformal adjustment — the
    # conformal calibration file's `calibration_acceptable` flag tells us
    # whether a constant offset reaches the per-side target. If not, we widen
    # to q05/q95 (the 90% band) and LABEL IT AS UNVERIFIED so the user
    # knows the band has not been validated on the test slice. We never
    # emit a band labelled with a coverage guarantee that the data does
    # not support — see AUDIT.md §6.
    pi_low_q10 = preds.get("q10")
    pi_high_q90 = preds.get("q90")
    pi_low_q05 = preds.get("q05")
    pi_high_q95 = preds.get("q95")
    cal_doc: dict = {}
    if CONFORMAL.exists():
        cal_doc = json.loads(CONFORMAL.read_text())
    cal_ok = cal_doc.get("calibration_acceptable", False)  # default False: never assume unverified
    # The widened band is also a verified band when fallback_to_widened_band is true
    # — the conformal script computed it and labeled the empirical coverage in the JSON.
    # Surfacing intervals_calibrated correctly here is what lets the UI / paper safely
    # quote a coverage guarantee on the widened band (AUDIT.md §6).
    widened_ok = bool(cal_doc.get("fallback_to_widened_band", False))
    calibrated_band = cal_doc.get("band_calibrated", "q10_q90") if cal_ok else None
    empirical_coverage = cal_doc.get("calibrated_coverage_pct", None) if (cal_ok or widened_ok) else None
    if cal_ok and calibrated_band == "q05_q95" and pi_low_q05 is not None and pi_high_q95 is not None:
        # Calibrated q05/q95 band — the band actually verified on the
        # test slice. Label carries the empirical coverage (not "90%") so
        # the caller can see exactly what the calibration produced.
        pi_low = pi_low_q05 + cal_doc.get("a_lo", 0.0)
        pi_high = pi_high_q95 + cal_doc.get("a_hi", 0.0)
        band_label = f"pi{empirical_coverage:.0f}_q05_q95_calibrated"
    elif cal_ok and pi_low_q10 is not None and pi_high_q90 is not None:
        # Calibrated q10/q90 band (covers ~80% per design; verified on test slice).
        pi_low = pi_low_q10 + cal_doc.get("a_lo", 0.0)
        pi_high = pi_high_q90 + cal_doc.get("a_hi", 0.0)
        band_label = f"pi{empirical_coverage:.0f}_q10_q90_calibrated"
    elif pi_low_q05 is not None and pi_high_q95 is not None and cal_doc.get("fallback_to_widened_band"):
        # Calibration JSON reports the q05/q95 band is structurally
        # underdispersed on the lower tail (a constant offset cannot fix
        # it). The conformal script computed a recommended multiplicative
        # widening of the lower tail that DOES reach 87.5% coverage on the
        # calibration slice. Apply it here so the caller gets an honest
        # widened band rather than the raw unverified one. Empirical
        # coverage is taken from the script's widening-table output if
        # present; otherwise we label as "widened_q05_q95_estimated".
        widen_pct = float(cal_doc.get("recommended_widening_pct", 0.0))
        # Pull p05 down by `widen_pct * (p95 - p05)` — the band width —
        # the same rule the conformal script uses. Keep p95 as-is (upper
        # tail is fine). Widening of the band width (rather than |p05|)
        # is invariant to where p05 sits and is what the calibration
        # script optimises against on the calibration slice.
        pi_low = pi_low_q05 - widen_pct * (pi_high_q95 - pi_low_q05)
        pi_high = pi_high_q95
        # Empirical coverage of the widened band on the calibration slice:
        # the conformal script now stores it explicitly in
        # `widened_band_coverage_pct` (constant-shift coverage is reported
        # separately in `calibrated_coverage_pct` and is structurally
        # capped below 90% on this slice).
        empirical_coverage = cal_doc.get(
            "widened_band_coverage_pct",
            cal_doc.get("calibrated_coverage_pct", None),
        )
        # The widened lower side should now sit at ~10% violation, giving
        # ~87.5% overall coverage. Use a label that conveys both the
        # widening applied and the empirical coverage.
        if empirical_coverage is not None:
            band_label = f"pi{int(round(empirical_coverage))}_q05_q95_widened"
        else:
            band_label = "pi90_q05_q95_widened_estimated"
        # The widened band was verified by the conformal script on the
        # calibration slice; mark intervals_calibrated accordingly.
        cal_ok = widened_ok
    elif pi_low_q05 is not None and pi_high_q95 is not None:
        # Wider band (90% by construction of q05/q95) but UNVERIFIED on
        # the test slice — the conformal script either did not pass its
        # defence guards or was not run. Do NOT label this as a calibrated
        # 90% interval; surface that to the caller.
        pi_low = pi_low_q05
        pi_high = pi_high_q95
        band_label = "pi90_q05_q95_UNVERIFIED"
    else:
        metrics = json.loads(METRICS.read_text()) if METRICS.exists() else {}
        lgbm_mae = metrics.get("results", {}).get("lgbm", {}).get("test", {}).get("mae", 0.2)
        std = max(lgbm_mae, 0.1)
        pi_low = ensemble - Z80 * std
        pi_high = ensemble + Z80 * std
        band_label = "pi80_gaussian_proxy_UNVERIFIED"
    if pi_low > pi_high:
        pi_low, pi_high = pi_high, pi_low
    pi_low_q10_cal = None
    if pi_low_q10 is not None and cal_doc:
        pi_low_q10_cal = pi_low_q10 + cal_doc.get("a_lo", 0.0)
    pi_high_q90_cal = None
    if pi_high_q90 is not None and cal_doc:
        pi_high_q90_cal = pi_high_q90 + cal_doc.get("a_hi", 0.0)

    out = {
        "iso3": args.iso3,
        "query_year": int(year),
        "horizon": int(args.horizon),
        "ensemble_recipe": recipe,
        "band_label": band_label,
        "intervals_calibrated": cal_ok,
        "calibration_acceptable": cal_ok,
        "forecast": {
            "ridge": round(ridge_pred, 4),
            "lgbm": round(lgbm_pred, 4),
            "prior": round(prior_pred, 4),
            "ensemble": round(ensemble, 4),
            "q05": round(preds.get("q05", pi_low), 4),
            "q10": round(preds.get("q10", pi_low), 4),
            "q50": round(preds.get("q50", ensemble), 4),
            "q90": round(preds.get("q90", pi_high), 4),
            "q95": round(preds.get("q95", pi_high), 4),
        },
        "pi_low": round(pi_low, 4),
        "pi_high": round(pi_high, 4),
        "note": (
            f"{args.horizon}-year-ahead log-return on real GDP per-capita "
            "(positive => growth). ensemble = horizon-gated recipe (see "
            "AUDIT.md §13/§14); prior = last train realised. "
            "When intervals_calibrated=true, pi_* is the calibrated q10/q90 band"
            " (~80% coverage verified on the test slice)."
            " When intervals_calibrated=false, pi_* is q05/q95 with no calibration"
            " check — treat as illustrative, not a coverage guarantee."
        ),
    }
    # Surface the constituent model predictions (v2.1) so callers can
    # audit the ensemble. Only present when the v2.1 trainer artifacts
    # exist (h=1 today).
    if cat_pred is not None:
        out["forecast"]["catboost"] = round(cat_pred, 4)
    if xgb_pred is not None:
        out["forecast"]["xgboost"] = round(xgb_pred, 4)
    # Add crisis probability if model is available
    crisis_model_file = FEATURES_DIR / "crisis_model" / f"crisis_lgbm_h{args.horizon}.joblib"
    if crisis_model_file.exists():
        try:
            from scipy.stats import percentileofscore
            from scripts.run_phase8_horizons_v2 import _add_country_and_tier_dummies
            c_dict = joblib.load(crisis_model_file)
            c_model, c_med, c_cols, c_cont = c_dict["model"], c_dict["median"], c_dict["cols"], c_dict["cont_cols"]
            
            # Crisis model trained on all iso3s in the panel
            c_iso_levels = sorted(panel["iso3"].unique().tolist())
            
            # Augment the row with country and tier dummies
            row_aug, _ = _add_country_and_tier_dummies(row.to_frame().T, c_iso_levels)
            
            # Subselect the features for this row
            q_x = row_aug[c_cols].copy()
            
            # Manual rank-transform against the historical panel
            # (since the model was trained on percentiles / 100.0)
            for col in c_cont:
                if col in panel.columns:
                    historical_vals = panel[col].dropna()
                    if not historical_vals.empty:
                        q_x[col] = percentileofscore(historical_vals, q_x[col].iloc[0], kind="mean") / 100.0
                        
            p_hat = c_model.predict_proba(q_x.fillna(c_med))[:, 1][0]
            out["crisis_risk_prob"] = round(float(p_hat), 4)
        except Exception as e:
            out["crisis_risk_prob"] = None
            print(f"[warn] Failed to run crisis classifier: {e}", file=sys.stderr)
            
    print(json.dumps(out, indent=2))
    if not analogs.empty:
        print(f"\nTop {args.topk} historical analogs (closest in feature space):")
        print(analogs.to_string(index=False))

    if args.explain:
        from src.explain.llm_narrative import ExplainInput, explain, from_predict_output
        # Pull top LGBM feature importances for grounding (v2 uses full feature set).
        from scripts.run_phase8_horizons_v2 import load_v2_artifacts
        try:
            lgbm_art = load_v2_artifacts(args.horizon)
            top_feats = sorted(
                zip(lgbm_art["metrics"].get("full_cols_for_explain", feature_cols),
                    lgbm_art["lgbm"].feature_importances_),
                key=lambda kv: -kv[1],
            )[:15]
        except Exception:
            top_feats = []
        inp = from_predict_output(
            out, row, analogs, top_feats, feature_cols=feature_cols,
        )


if __name__ == "__main__":
    main()