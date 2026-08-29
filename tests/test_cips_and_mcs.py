"""
Tests for Pesaran (2007) CIPS Unit Root Test, Model Confidence Set (MCS), and Regime Breakdown
"""

import numpy as np
import pandas as pd
import pytest
from src.econometrics.panel_granger import (
    pesaran_cadf_individual,
    pesaran_cips_test,
    pedroni_panel_cointegration_test,
)
from src.econometrics.model_confidence_set import model_confidence_set, block_bootstrap_indices
from src.evaluation.regime_breakdown import identify_regimes, evaluate_regime_performance


def test_pesaran_cadf_individual():
    np.random.seed(42)
    T = 40
    # Stationary AR(1) series
    y_i = np.zeros(T)
    y_bar = np.zeros(T)
    for t in range(1, T):
        y_i[t] = 0.3 * y_i[t - 1] + np.random.normal(0, 1)
        y_bar[t] = 0.2 * y_bar[t - 1] + np.random.normal(0, 0.5)

    stat = pesaran_cadf_individual(y_i, y_bar, lags=1)
    assert stat is not None
    assert np.isfinite(stat)
    # For stationary AR(1) with rho=0.3, CADF t-stat on (rho-1) should be negative
    assert stat < 0


def test_pesaran_cips_stationary_vs_unit_root():
    np.random.seed(42)
    N, T = 15, 35
    records = []

    for i in range(N):
        iso = f"C{i:02d}"
        # Generate stationary series
        y_stat = 0.0
        # Generate random walk (unit root)
        y_rw = 0.0
        for t in range(T):
            y_stat = 0.2 * y_stat + np.random.normal(0, 1)
            y_rw = y_rw + np.random.normal(0, 1)
            records.append({
                "iso3": iso,
                "year": 1980 + t,
                "stationary_var": y_stat,
                "unit_root_var": y_rw,
            })

    df = pd.DataFrame(records)

    res_stat = pesaran_cips_test(df, "stationary_var", lags=1, min_obs_per_country=20)
    assert res_stat["is_stationary_05pct"] is True
    assert res_stat["order_of_integration"] == "I(0) Stationary"

    res_rw = pesaran_cips_test(df, "unit_root_var", lags=1, min_obs_per_country=20)
    assert res_rw["is_stationary_05pct"] is False
    assert res_rw["order_of_integration"] == "I(1) Unit Root"


def test_model_confidence_set_elimination():
    np.random.seed(42)
    T = 100
    # Model 0: best (mean loss 0.02)
    # Model 1: equal to best (mean loss 0.02)
    # Model 2: slightly worse (mean loss 0.025)
    # Model 3: terrible (mean loss 0.10)
    l0 = np.random.normal(0.02, 0.005, size=T)
    l1 = np.random.normal(0.02, 0.005, size=T)
    l2 = np.random.normal(0.025, 0.005, size=T)
    l3 = np.random.normal(0.10, 0.01, size=T)

    losses = np.column_stack([l0, l1, l2, l3])
    names = ["Model_Best1", "Model_Best2", "Model_Mid", "Model_Terrible"]

    mcs_df = model_confidence_set(losses, names, alpha=0.10, n_boot=200, block_size=2, seed=42)

    assert len(mcs_df) == 4
    # The terrible model must NOT be in MCS 90%
    terrible_row = mcs_df[mcs_df["Model"] == "Model_Terrible"].iloc[0]
    assert not bool(terrible_row["In_MCS_90pct"])
    assert float(terrible_row["MCS_P_Value"]) < 0.10

    # Best model must be in MCS 90%
    best_row = mcs_df[mcs_df["Model"] == "Model_Best1"].iloc[0]
    assert bool(best_row["In_MCS_90pct"])


def test_regime_breakdown_logic():
    df = pd.DataFrame({
        "iso3": ["USA"] * 10 + ["GBR"] * 10,
        "year": list(range(2000, 2010)) * 2,
        "vdem_electoral_democracy": [0.8] * 7 + [0.6, 0.5, 0.4] + [0.85] * 10,
        "growth_into_origin": [0.03] * 8 + [-0.04, 0.02] + [0.02] * 10,
        "gdp_pc_growth_1y_fwd": [0.02] * 20,
        "pred_ar1": [0.02] * 20,
        "pred_eco": [0.02] * 20,
        "pred_all": [0.021] * 20,
    })

    regime_df = identify_regimes(df)
    assert "is_inst_transition" in regime_df.columns
    assert "is_macro_crisis" in regime_df.columns

    # Transition flag should be true for USA in late years due to democracy drop
    usa_late = regime_df[(regime_df["iso3"] == "USA") & (regime_df["year"] >= 2008)]
    assert any(usa_late["is_inst_transition"])

    # Evaluate performance function runs
    metrics = evaluate_regime_performance(
        regime_df,
        target_col="gdp_pc_growth_1y_fwd",
        model_pred_cols={"AR(1) Baseline": "pred_ar1", "Economy-Only Ridge": "pred_eco", "All-Domain Ridge (Concat)": "pred_all"},
        regime_mask=regime_df["is_macro_tranquil"],
        regime_name="Tranquil",
        horizon=1,
    )
    assert metrics["Regime"] == "Tranquil"
    assert "MAE_Economy-Only Ridge" in metrics


def test_pedroni_panel_cointegration():
    np.random.seed(42)
    N, T = 10, 30
    records = []
    for i in range(N):
        iso = f"C{i:02d}"
        x = 10.0
        e = 0.0
        for t in range(T):
            x = x + np.random.normal(0, 1)  # I(1)
            e = 0.3 * e + np.random.normal(0, 0.5)  # I(0) stationary residual (cointegrated)
            y = 2.0 + 1.5 * x + e
            records.append({"iso3": iso, "year": 1990 + t, "y": y, "x": x})
    df = pd.DataFrame(records)

    res = pedroni_panel_cointegration_test(df, y_col="y", x_col="x", lags=1, min_obs=15)
    assert res["n_countries"] == N
    assert "z_group_adf" in res
    assert res["cointegrated_5pct"] is True


def test_robustness_grid_tournament_parity():
    """Verify that robustness grid matches headline tournament exactly at baseline parameters."""
    from pathlib import Path
    bench_dir = Path(__file__).resolve().parent.parent / "data" / "benchmarks"
    
    alpha_path = bench_dir / "real_robustness_alpha_results.csv"
    tourn_path = bench_dir / "real_cross_domain_benchmark_results.csv"
    lambda_path = bench_dir / "real_robustness_lambda_results.csv"

    if not (alpha_path.exists() and tourn_path.exists() and lambda_path.exists()):
        pytest.skip("Benchmark CSV artifacts not yet generated.")

    alpha_df = pd.read_csv(alpha_path)
    tourn_df = pd.read_csv(tourn_path)
    lam_df = pd.read_csv(lambda_path)

    # 1. Verify alpha=100 exact parity with headline Economy Ridge & All-Domain Ridge
    a100 = alpha_df[alpha_df["Value"] == 100.0]
    assert len(a100) == 3

    for h in [1, 3, 5]:
        row_a = a100[a100["Horizon"] == h].iloc[0]
        row_eco = tourn_df[(tourn_df["Horizon"] == h) & (tourn_df["Model"] == "Economy-Only Ridge")].iloc[0]
        row_all = tourn_df[(tourn_df["Horizon"] == h) & (tourn_df["Model"] == "All-Domain Ridge (Concat)")].iloc[0]

        np.testing.assert_almost_equal(float(row_a["MAE_Eco_Ridge"]), float(row_eco["MAE"]), decimal=4)
        np.testing.assert_almost_equal(float(row_a["MAE_All_Domain_Concat"]), float(row_all["MAE"]), decimal=4)
        # Dilution penalty must be positive
        assert float(row_a["Dilution_Penalty_pct"]) > 0.0

    # 2. Verify all alpha values show positive dilution penalty
    for _, row in alpha_df.iterrows():
        assert float(row["Dilution_Penalty_pct"]) > 0.0

    # 3. Verify lambda=0.92 matches headline DMS router
    l92 = lam_df[lam_df["Value"] == 0.92]
    assert len(l92) == 3
    for h in [1, 3, 5]:
        row_l = l92[l92["Horizon"] == h].iloc[0]
        row_dms = tourn_df[(tourn_df["Horizon"] == h) & (tourn_df["Model"] == "DMS State-Space Router")].iloc[0]
        np.testing.assert_almost_equal(float(row_l["MAE"]), float(row_dms["MAE"]), decimal=4)


def test_all_src_modules_importable():
    """Verify that every python module in src/ imports cleanly without syntax or indentation errors."""
    import importlib
    import pkgutil
    import src
    from pathlib import Path

    src_path = Path(__file__).resolve().parent.parent / "src"
    for py_file in src_path.rglob("*.py"):
        if py_file.name.startswith("__"):
            continue
        rel_mod = "src." + ".".join(py_file.relative_to(src_path).with_suffix("").parts)
        mod = importlib.import_module(rel_mod)
        assert mod is not None


