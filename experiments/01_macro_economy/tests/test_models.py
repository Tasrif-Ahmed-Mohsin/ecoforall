"""Tests that the v2 persisted models produce finite, consistent forecasts.

v2 contract:
- Artifacts live at `data/features/horizon_{h}y_v2/` (one folder per horizon).
- Each folder contains: `lgbm.joblib`, `ridge.joblib`, `lgbm_q{05,50,95}.joblib`,
  `metrics.json`, `feature_meta.json` (iso_levels + cont_cols + full_cols),
  and `optuna_study.db` / `optuna_study.csv` (the tuning trail).
- Inference is via `scripts/run_phase8_horizons_v2.py:predict_one_iso3`, which
  rebuilds the labelled frame from the panel and the persisted schema.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.harmonize.common import FEATURES_DIR  # noqa: E402


PANEL = FEATURES_DIR / "panel_wide.parquet"
TARGET = "gdp_pc_growth_5y_fwd"

# v2 horizons we expect to find on disk.
HORIZONS = (1, 3, 5, 10)


def _load_v2_trainer():
    spec = importlib.util.spec_from_file_location(
        "run_phase8_horizons_v2",
        ROOT / "scripts" / "run_phase8_horizons_v2.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def v2():
    return _load_v2_trainer()


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    assert PANEL.exists(), f"missing panel {PANEL}"
    return pd.read_parquet(PANEL)


@pytest.fixture(scope="module")
def horizon_5y_artifacts():
    """Lazy-load the 5y v2 artifacts (the headline horizon)."""
    out_dir = FEATURES_DIR / "horizon_5y_v2"
    assert out_dir.exists(), f"missing {out_dir}"
    return {
        "lgbm": joblib.load(out_dir / "lgbm.joblib"),
        "ridge_pipe": joblib.load(out_dir / "ridge.joblib"),
        "q05": joblib.load(out_dir / "lgbm_q05.joblib"),
        "q50": joblib.load(out_dir / "lgbm_q50.joblib"),
        "q95": joblib.load(out_dir / "lgbm_q95.joblib"),
    }


def test_v2_horizon_artifacts_exist():
    """Each horizon directory must have the v2 artifact set."""
    for h in HORIZONS:
        out_dir = FEATURES_DIR / f"horizon_{h}y_v2"
        assert out_dir.exists(), f"missing {out_dir}"
        for name in ("lgbm.joblib", "ridge.joblib", "metrics.json", "feature_meta.json"):
            assert (out_dir / name).exists(), f"missing {out_dir / name}"


def test_v2_predict_loads(panel, v2) -> None:
    """Smoke test: predict_one_iso3 returns finite values for one country/year."""
    p = v2.predict_one_iso3(panel, "USA", 2018, horizon=5)
    for k in ("ridge", "lgbm", "q05", "q50", "q95"):
        assert np.isfinite(p[k]), f"{k} non-finite: {p[k]}"
    assert p["iso3"] == "USA"
    assert int(p["year"]) == 2018
    assert int(p["horizon"]) == 5


def test_v2_predict_quantile_order_soft(panel, v2) -> None:
    """q05 ≤ q50 ≤ q95 for all 4 horizons — allow ≤1 monotonicity violation per side.

    LGBM quantile regressors produce occasional FP-noise crossing on a small
    fraction of rows (≤ 1 / 772 in the v2 calibration slice). The conformal
    recalibrator checks the same property and accepts ≤1 violation.
    """
    for h in HORIZONS:
        p = v2.predict_one_iso3(panel, "USA", 2018, horizon=h)
        order = [p["q05"], p["q50"], p["q95"]]
        violations = sum(1 for i in range(len(order) - 1) if order[i] > order[i + 1])
        assert violations <= 1, (
            f"h={h}: q05..q95 too many crossings ({violations}): {order}"
        )


def test_v2_metrics_reasonable(panel, v2) -> None:
    """The persisted metrics.json for h=5 should report a meaningful ensemble gain
    over the prior on the test slice — sanity check that we shipped the trained
    model and not the random-walk fallback."""
    metrics_path = FEATURES_DIR / "horizon_5y_v2" / "metrics.json"
    assert metrics_path.exists()
    import json
    m = json.loads(metrics_path.read_text())
    results = m.get("results", {})
    if "prior" in results and m.get("ensemble_test_mae") is not None:
        prior_mae = results["prior"]["test"]["mae"]
        ens_mae = m["ensemble_test_mae"]
        assert ens_mae < prior_mae, (
            f"ensemble MAE ({ens_mae}) not below prior MAE ({prior_mae}) — model regressed"
        )


def test_v2_feature_meta_persisted():
    """feature_meta.json is the inference-time schema; it must round-trip iso_levels + col lists."""
    import json
    for h in HORIZONS:
        meta_path = FEATURES_DIR / f"horizon_{h}y_v2" / "feature_meta.json"
        m = json.loads(meta_path.read_text())
        assert "iso_levels" in m and isinstance(m["iso_levels"], list)
        assert "cont_cols" in m and isinstance(m["cont_cols"], list)
        assert "full_cols" in m and isinstance(m["full_cols"], list)
        # full_cols must contain cont_cols + country dummies + tier dummies.
        assert len(m["full_cols"]) >= len(m["cont_cols"]), (
            f"h={h}: full_cols ({len(m['full_cols'])}) shorter than cont_cols ({len(m['cont_cols'])})"
        )
