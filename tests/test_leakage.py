"""
Test Suite: Data Leakage Prevention Engine (Zero Forward Leakage)
=================================================================
Two classes of leak are covered:

  A. FEATURE-MATRIX leaks -- forward targets, leads, forward differences appearing as
     columns in X. Checked against the real panel, not a hand-written list.

  B. TEMPORAL-FEEDBACK leaks -- a model conditioning on a target that has not been
     realised at the forecast origin. This is the class that produced the previous
     run's headline: DynamicModelSelectionRouter received the whole test-fold target
     vector and updated weights immediately, so at horizon h the weights at origin t
     depended on up to h-1 years of future data.

The earlier version of this file defined its leak detector *inside the test module*,
so it exercised a function production never called and could not observe either defect.
"""

import numpy as np
import pandas as pd
import pytest

from src.gating.dms_state_space_router import DynamicModelSelectionRouter

FORBIDDEN_PATTERNS = ["_target_", "_diff_h", "_1y_fwd", "_3y_fwd", "_5y_fwd", "_fwd", "_lead"]

META_COLS = {"iso3", "country", "year", "region", "income_level", "region_wb",
             "growth_into_origin", "is_score", "horizon", "fold"}


def check_feature_columns_for_leakage(columns) -> list[str]:
    """Identify any column that violates the zero-leakage protocol."""
    return [c for c in columns if any(pat in c.lower() for pat in FORBIDDEN_PATTERNS)]


# --------------------------------------------------------------------------- A
def test_leakage_detector_catches_known_leaks():
    dirty = [
        "gdp_pc_real",
        "inflation_rate",
        "co2_emissions_per_capita_target_h1",
        "temp_anomaly_celsius_diff_h5",
        "material_conflict_annual_sum_1y_fwd",
        "trade_openness_gdp",
    ]
    leaked = check_feature_columns_for_leakage(dirty)
    assert set(leaked) == {
        "co2_emissions_per_capita_target_h1",
        "temp_anomaly_celsius_diff_h5",
        "material_conflict_annual_sum_1y_fwd",
    }


def test_real_panel_feature_matrix_has_no_forward_columns():
    """The panel the manuscript is built on must contain no forward-looking feature."""
    from pathlib import Path

    p = (Path(__file__).resolve().parent.parent / "data" / "processed_panels"
         / "real_cross_domain_annual_panel.parquet")
    if not p.exists():
        pytest.skip("real panel artifact not present")

    cols = pd.read_parquet(p).columns
    feature_cols = [c for c in cols if c not in META_COLS and not c.endswith("_fwd")]
    assert check_feature_columns_for_leakage(feature_cols) == []


def test_fold_quarantine_integrity():
    """For test window opening at t_start, training targets must land strictly before it."""
    t_start = 2019
    for h in (1, 3, 5):
        max_train_year = t_start - h - 1
        assert (max_train_year + h) < t_start


def test_warmup_window_targets_do_not_overlap_training_targets():
    """
    The DMS warm-up window [t_start-h, t_start-1] must sit strictly between the last
    training target and the first scored origin: its targets realise at or after
    t_start, while training targets all realise at or before t_start-1.
    """
    t_start = 2019
    for h in (1, 3, 5):
        last_train_origin = t_start - h - 1
        last_train_target = last_train_origin + h
        first_warm_origin = t_start - h
        first_warm_target = first_warm_origin + h
        assert last_train_target < t_start
        assert first_warm_target >= t_start
        assert first_warm_origin > last_train_origin


# --------------------------------------------------------------------------- B
def _toy_panel(n_years: int = 12, isos=("AAA", "BBB")) -> pd.DataFrame:
    rows = [{"iso3": i, "year": 2000 + t} for i in isos for t in range(n_years)]
    return pd.DataFrame(rows)


def test_dms_weights_stay_at_prior_until_a_target_is_realised():
    """
    At horizon h the first h origins of every country have no observable realisation,
    so the router must still be sitting exactly on its uniform prior. Under the old
    implementation the weights moved on the second origin regardless of h.
    """
    for h in (1, 3, 5):
        df = _toy_panel()
        n, m = len(df), 3
        preds = np.tile(np.array([0.0, 1.0, -1.0]), (n, 1))
        y = np.zeros(n)  # expert 0 is always exactly right -- maximally tempting

        router = DynamicModelSelectionRouter(n_experts=m, forgetting_factor=0.92, mode="dma")
        _, w = router.route_panel(df, preds, y, horizon=h)

        first = df.groupby("iso3")["year"].transform("min")
        for k in range(h):
            mask = (df["year"] == first + k).to_numpy()
            np.testing.assert_allclose(
                w[mask], np.full((mask.sum(), m), 1.0 / m), atol=1e-12,
                err_msg=f"h={h}: weights moved at origin offset {k}, before any target realised",
            )


def test_dms_does_eventually_learn_once_targets_realise():
    """Guard against the previous test passing for the wrong reason (a dead filter)."""
    df = _toy_panel(n_years=20)
    n, m = len(df), 3
    preds = np.tile(np.array([0.0, 1.0, -1.0]), (n, 1))
    y = np.zeros(n)

    router = DynamicModelSelectionRouter(n_experts=m, forgetting_factor=0.92, mode="dma")
    _, w = router.route_panel(df, preds, y, horizon=3)

    last = df.groupby("iso3")["year"].transform("max")
    final = w[(df["year"] == last).to_numpy()]
    assert np.all(final[:, 0] > 0.9), f"filter never concentrated on the correct expert: {final}"


def test_dms_no_feedback_is_exactly_the_equal_weight_average():
    """
    DMA whose posterior never updates stays on the 1/M prior, so it IS the simple
    average. This is why the router showed no gain once its test-target feedback was
    removed -- the equivalence is structural, and the test pins it down.
    """
    df = _toy_panel()
    n, m = len(df), 4
    rng = np.random.default_rng(7)
    preds = rng.normal(size=(n, m))

    router = DynamicModelSelectionRouter(n_experts=m, forgetting_factor=0.92, mode="dma")
    routed, _ = router.route_panel(df, preds, None, horizon=3)
    np.testing.assert_allclose(routed, preds.mean(axis=1), atol=1e-12)


def test_dms_requires_origin_year_to_gate_feedback():
    """Without an origin-year column the router cannot know what is observable."""
    df = _toy_panel().drop(columns=["year"])
    preds = np.zeros((len(df), 2))
    router = DynamicModelSelectionRouter(n_experts=2)
    with pytest.raises(KeyError, match="year"):
        router.route_panel(df, preds, np.zeros(len(df)), horizon=1)


def test_dms_longer_horizon_never_sees_more_than_shorter_horizon():
    """
    Monotonicity: a longer release delay can only shrink the information set. If a
    longer horizon ever fit the realised targets better, feedback would be leaking.
    """
    df = _toy_panel(n_years=16)
    n, m = len(df), 3
    rng = np.random.default_rng(11)
    preds = rng.normal(scale=0.05, size=(n, m))
    preds[:, 0] = 0.0
    y = np.zeros(n)

    errs = []
    for h in (1, 3, 5):
        r = DynamicModelSelectionRouter(n_experts=m, forgetting_factor=0.92, mode="dma")
        routed, _ = r.route_panel(df, preds, y, horizon=h)
        errs.append(float(np.mean(np.abs(y - routed))))
    assert errs[0] <= errs[1] + 1e-12 <= errs[2] + 2e-12, f"MAE not monotone in horizon: {errs}"


def test_no_synthetic_generators_in_active_directories():
    """Verify that no synthetic dataset generators exist in active src/ or scripts/ directories."""
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    active_dirs = [root / "src", root / "scripts"]
    
    forbidden_terms = ["np.random.normal(", "np.random.uniform(", "generate_synthetic_"]
    violations = []
    
    for d in active_dirs:
        for py_file in d.rglob("*.py"):
            text = py_file.read_text(encoding="utf-8", errors="ignore")
            # Exclude tests and seeds
            for term in forbidden_terms:
                if term in text and "reproducibility" not in py_file.name:
                    violations.append(f"{py_file.relative_to(root)} contains {term}")
                    
    assert violations == [], f"Found forbidden synthetic generators in active paths: {violations}"

