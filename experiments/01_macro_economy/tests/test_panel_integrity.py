"""Sanity tests against the harmonized panel + retrieval index artifacts."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.harmonize.common import FEATURES_DIR


PANEL = FEATURES_DIR / "panel_wide.parquet"
TARGET = "gdp_pc_growth_5y_fwd"


def test_panel_loads_and_shape() -> None:
    df = pd.read_parquet(PANEL)
    assert df.shape[0] > 1000, "panel should have many rows"
    assert {"iso3", "year", TARGET}.issubset(df.columns)
    assert df["iso3"].nunique() >= 100
    assert df["year"].min() <= 1960
    assert df["year"].max() >= 2020


def test_target_distribution_has_fat_tails() -> None:
    df = pd.read_parquet(PANEL)
    labelled = df[TARGET].dropna()
    assert labelled.std() > 0.3
    assert (labelled < -0.5).mean() > 0.005, "expect some deep-crisis observations"
    assert (labelled > 0.5).mean() > 0.003, "expect some hyper-growth observations"


def test_feature_columns_have_observations() -> None:
    """Allow up to 5 all-NaN columns (sparse Clio-Infra fields), but flag if more."""
    df = pd.read_parquet(PANEL)
    feature_cols = [c for c in df.columns
                    if c not in {"iso3", "year"}
                    and pd.api.types.is_numeric_dtype(df[c])]
    non_empty = sum(df[c].notna().any() for c in feature_cols)
    n_empty = len(feature_cols) - non_empty
    assert n_empty <= 5, f"too many all-NaN columns ({n_empty}): the panel should not have many"
    # Most columns should still be populated.
    assert non_empty / len(feature_cols) > 0.9


def test_ranked_index_loads_and_is_consistent() -> None:
    from src.retrieval.faiss_index import RankedV2Index
    p = FEATURES_DIR / "retrieval_v2" / "panel_ranked.faiss"
    if not p.exists():
        pytest.skip("retrieval_v2 panel_ranked.faiss not built; run scripts/_build_v2_faiss_index.py")
    idx = RankedV2Index.load()
    assert idx.index.ntotal == len(idx.rows)
    assert idx.index.ntotal > 1000
    assert len(idx.cols) > 50
    assert idx.mask.shape == (idx.index.ntotal, len(idx.cols))
    assert idx.mask.sum(axis=1).min() > 0, "every indexed row must observe ≥1 feature"


# Note: the legacy v1 FAISS loader test was removed during the 2026-07-20 GMD-only
# cleanup (data/features/retrieval/ deleted; the v1 24-column schema is not valid on
# GMD). The v2 path above is the canonical retrieval pin.
