"""Unit tests for the new Historical Scenario Engine & Crisis Pattern Matcher."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.explain.scenario_engine import (
    cluster_analog_outcomes,
    build_scenario_tree,
    detect_pattern_divergence,
    find_divergent_twins,
)
from src.explain.crisis_patterns import find_crisis_precedents, _check_vulnerability


def test_cluster_analog_outcomes_basic() -> None:
    data = pd.DataFrame({
        "iso3": ["KOR", "VNM", "THA", "PHL", "IDN", "PAK"],
        "year": [1975, 1998, 1985, 1980, 2000, 1998],
        "gdp_pc_growth_5y_fwd": [0.45, 0.35, 0.25, 0.05, 0.15, -0.15],
    })
    clusters = cluster_analog_outcomes(data, horizon=5, n_clusters=3)
    assert len(clusters) >= 2
    total_prob = sum(c["probability"] for c in clusters)
    assert pytest.approx(total_prob, 0.01) == 1.0


def test_build_scenario_tree() -> None:
    data = pd.DataFrame({
        "iso3": ["KOR", "VNM", "THA", "PHL", "IDN"],
        "year": [1975, 1998, 1985, 1980, 2000],
        "gdp_pc_growth_5y_fwd": [0.40, 0.30, 0.20, 0.10, 0.15],
    })
    tree = build_scenario_tree("BGD", 2005, horizon=5, analogs_df=data, ml_ensemble=0.22)
    assert "scenarios" in tree
    assert tree["n_analogs"] == 5
    assert np.isfinite(tree["weighted_forecast"])


def test_detect_pattern_divergence() -> None:
    q_row = pd.Series({"gdp_pc_real": 500.0, "inflation_rate": 5.0, "gov_debt_gdp": 30.0})
    a_row = pd.Series({"iso3": "KOR", "year": 1975, "gdp_pc_real": 620.0, "inflation_rate": 20.0, "gov_debt_gdp": 32.0})
    cols = ["gdp_pc_real", "inflation_rate", "gov_debt_gdp"]
    
    div = detect_pattern_divergence(q_row, a_row, cols, top_k=2)
    assert div["analog"]["iso3"] == "KOR"
    assert len(div["matches"]) >= 1
    assert len(div["divergences"]) >= 1


def test_find_divergent_twins() -> None:
    data = pd.DataFrame({
        "iso3": ["KOR", "PHL", "THA", "IDN"],
        "year": [1975, 1975, 1985, 1985],
        "gdp_pc_growth_5y_fwd": [0.50, 0.05, 0.30, 0.12],
    })
    twins = find_divergent_twins(data, horizon=5, min_outcome_gap=0.15)
    assert len(twins) >= 1
    assert twins[0]["outcome_gap"] >= 0.15


def test_check_vulnerability() -> None:
    row = pd.Series({
        "gov_debt_gdp": 95.0,        # Danger (>80)
        "current_account_gdp": -2.0,  # Safe (>-5)
        "inflation_rate": 12.0,       # Danger (>10)
    })
    vuln = _check_vulnerability(row)
    assert len(vuln) == 5
    statuses = {v["name"]: v["status"] for v in vuln}
    assert statuses["Government Debt / GDP"] == "danger"
    assert statuses["Current Account / GDP"] == "safe"
    assert statuses["Inflation Rate"] == "danger"


def test_entity_extraction_constituents() -> None:
    from src.utils.country_lookup import extract_entities_from_prompt
    iso3, year, horizon = extract_entities_from_prompt("england 2004")
    assert iso3 == "GBR"
    assert year == 2004

    iso3, year, _ = extract_entities_from_prompt("scotland 2010 5 year forecast")
    assert iso3 == "GBR"
    assert year == 2010

    iso3, _, _ = extract_entities_from_prompt("germany 2022")
    assert iso3 == "DEU"

