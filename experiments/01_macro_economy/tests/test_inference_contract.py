"""End-to-end inference: simulate `scripts/predict_country.py` for one country.

v2 contract: inference is via `scripts/run_phase8_horizons_v2.py:predict_one_iso3`,
which loads the v2 trainer artifacts at `data/features/horizon_{h}y_v2/`. This
test exercises that path for one country/year, plus the analog retrieval path
(which is dimension-agnostic — it only needs the panel and a list of feature
columns).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_predict():
    spec = importlib.util.spec_from_file_location(
        "predict_country",
        ROOT / "scripts" / "predict_country.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_v2_trainer():
    spec = importlib.util.spec_from_file_location(
        "run_phase8_horizons_v2",
        ROOT / "scripts" / "run_phase8_horizons_v2.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def pm():
    return _load_predict()


@pytest.fixture(scope="module")
def v2():
    return _load_v2_trainer()


def test_full_inference_pipeline(pm, v2) -> None:
    """End-to-end: load panel, run v2 predict, build per-country prior, retrieve analogs."""
    panel = pm._load_panel()
    # v2 inference (5y headline horizon).
    preds = v2.predict_one_iso3(panel, "USA", 2018, horizon=5)
    for k in ("ridge", "lgbm"):
        assert np.isfinite(preds[k]), f"{k} is non-finite: {preds[k]}"
    for q in ("q05", "q50", "q95"):
        assert np.isfinite(preds[q]), f"{q} is non-finite: {preds[q]}"

    # Per-country prior uses the v2 target name (5y forward growth).
    prior = pm._per_country_prior(panel, "USA", train_end_year=2014)
    assert np.isfinite(prior)

    # Ensemble recipe + scaled forecast.
    recipe = "lgbm+prior"
    if pm.METRICS.exists():
        recipe = json.loads(pm.METRICS.read_text()).get("ensemble_recipe", "lgbm+prior")
    if recipe == "lgbm+prior":
        ensemble = 0.7 * preds["lgbm"] + 0.3 * prior
    elif recipe == "lgbm+ridge":
        ensemble = 0.7 * preds["lgbm"] + 0.3 * preds["ridge"]
    else:
        ensemble = preds["lgbm"]
    assert np.isfinite(ensemble)

    # Analog retrieval — dimension-agnostic; just needs cont_cols from v2 meta.
    meta_path = pm.FEATURES_DIR / "horizon_5y_v2" / "feature_meta.json"
    feature_cols = json.loads(meta_path.read_text())["cont_cols"] if meta_path.exists() else []
    row = panel[(panel.iso3 == "USA") & (panel.year == preds["year"])].iloc[0]
    analogs = pm._similar_analogs(
        panel, row, feature_cols, topk=5,
        exclude_year=int(row.year), exclude_iso3="USA",
        use_faiss=True, use_ranked=False, min_overlap=0,
    )
    assert not analogs.empty
    assert {"iso3", "year", "gdp_pc_growth_5y_fwd"}.issubset(analogs.columns)


def test_ranked_index_path(pm, v2) -> None:
    """Rank-features Euclidean path with min_overlap=60 — should still return ≥1 analog.

    If the v1 FAISS index is missing on disk (the panel no longer carries v1
    columns like `bond_yield_lt`, `gini_*`, `trade_gdp`, ...), the function
    falls back to plain L2 — which does NOT produce an `n_overlap` column.
    We skip in that case rather than fail.
    """
    panel = pm._load_panel()
    meta_path = pm.FEATURES_DIR / "horizon_5y_v2" / "feature_meta.json"
    feature_cols = json.loads(meta_path.read_text())["cont_cols"] if meta_path.exists() else []
    row = panel[panel.iso3 == "USA"].sort_values("year").iloc[-1]
    out = pm._similar_analogs(
        panel, row, feature_cols, topk=5,
        exclude_year=int(row.year), exclude_iso3="USA",
        use_faiss=True, use_ranked=True, min_overlap=60,
    )
    if out.empty:
        pytest.skip("USA's latest year has no candidates passing min_overlap=60 (rare)")
    if "n_overlap" not in out.columns:
        pytest.skip("FAISS index missing v1 columns; ranked-FAISS fell back to L2 (no n_overlap column)")
    assert (out["n_overlap"] >= 60).all()


def test_query_year_fallback_to_latest(pm) -> None:
    panel = pm._load_panel()
    row, year = pm._build_query_row(panel, "USA", None)
    assert isinstance(year, int)
    assert year >= 2020, "USA should have a recent year in the panel"


def test_unknown_iso3_errors(pm) -> None:
    panel = pm._load_panel()
    with pytest.raises(SystemExit):
        pm._build_query_row(panel, "ZZZ", 2010)


def test_v2_predict_quantile_order(pm, v2) -> None:
    """q05 ≤ q50 ≤ q95 for the headline (USA, 2018, h=5) row — sanity check."""
    panel = pm._load_panel()
    p = v2.predict_one_iso3(panel, "USA", 2018, horizon=5)
    order = [p["q05"], p["q50"], p["q95"]]
    assert order == sorted(order), f"q05..q95 not monotone: {order}"


def test_misspelled_country_entity_extraction() -> None:
    """Verify that misspelled country prompts resolve correctly via fuzzy entity extraction."""
    from src.utils.country_lookup import extract_entities_from_prompt

    cases = {
        "afganistan 2011": ("AFG", 2011),
        "columbia 2015": ("COL", 2015),
        "philipines 2018": ("PHL", 2018),
        "bangladsh 2020": ("BGD", 2020),
        "ukrain 2022": ("UKR", 2022),
    }

    for prompt, expected in cases.items():
        iso3, year, _ = extract_entities_from_prompt(prompt)
        assert (iso3, year) == expected, f"Failed for '{prompt}': got ({iso3}, {year}), expected {expected}"

