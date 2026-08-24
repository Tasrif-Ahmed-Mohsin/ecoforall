"""Shared utilities and schema for harmonizing all ecodata sources.

Every source-specific module (`clio_infra.py`, `imf.py`, …) reads its raw file
and produces a DataFrame that conforms to `CANONICAL_SCHEMA`, then writes it
to `data/harmonized/<source>.parquet`.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pandas as pd

# ---------------------------------------------------------------------------
# Canonical schema
# ---------------------------------------------------------------------------
CANONICAL_SCHEMA = {
    "iso3": "string[pyarrow]",       # ISO-3166-1 alpha-3 (or NaN if unmapped)
    "year": "int64",                 # calendar year (rounded)
    "indicator_id": "string[pyarrow]",  # snake_case id, e.g. "gdp_pc_real"
    "value": "float64",
    "unit": "string[pyarrow]",       # free text: "USD", "%", "index", …
    "scale": "string[pyarrow]",      # "units" | "thousands" | "millions" | "billions"
    "source": "string[pyarrow]",     # "clio_infra" | "imf" | "jst" | "maddison" | "wb"
    "is_forecast": "boolean",        # True for IMF forecasts beyond latest_actual
}

CANONICAL_COLUMNS = list(CANONICAL_SCHEMA.keys())

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ECODATA_ROOT = PROJECT_ROOT.parent / "ecodata"
HARMONIZED_DIR = PROJECT_ROOT / "data" / "harmonized"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
COUNTRY_MASTER = PROJECT_ROOT / "data" / "country_master.csv"


def ensure_dirs() -> None:
    HARMONIZED_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)


def write_long(df: pd.DataFrame, source: str) -> Path:
    """Validate and write a long-format parquet for one source."""
    ensure_dirs()
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{source}: missing canonical columns {missing}")
    out = df[CANONICAL_COLUMNS].copy()
    out["source"] = source
    out_path = HARMONIZED_DIR / f"{source}.parquet"
    out.to_parquet(out_path, index=False)
    return out_path


def load_long(source: str) -> pd.DataFrame:
    return pd.read_parquet(HARMONIZED_DIR / f"{source}.parquet")


# ---------------------------------------------------------------------------
# ISO-3 helpers
# ---------------------------------------------------------------------------
def normalize_country_name(name: str) -> str:
    """Best-effort cleanup so pycountry / manual lookup have an easier job."""
    if not isinstance(name, str):
        return ""
    fixes = {
        "United States": "United States of America",
        "USA": "United States of America",
        "U.S.A.": "United States of America",
        "UK": "United Kingdom",
        "U.K.": "United Kingdom",
        "Britain": "United Kingdom",
        "Great Britain": "United Kingdom",
        "Russia": "Russian Federation",
        "Iran": "Iran, Islamic Republic of",
        "Iran, Islamic Rep.": "Iran, Islamic Republic of",
        "South Korea": "Korea, Republic of",
        "Korea, Rep.": "Korea, Republic of",
        "Korea": "Korea, Republic of",
        "North Korea": "Korea, Democratic People's Republic of",
        "Venezuela": "Venezuela, Bolivarian Republic of",
        "Venezuela, RB": "Venezuela, Bolivarian Republic of",
        "Syria": "Syrian Arab Republic",
        "Czechia": "Czech Republic",
        "Slovakia": "Slovakia",
        "Egypt": "Egypt, Arab Republic of",
        "Egypt, Arab Rep.": "Egypt, Arab Republic of",
        "Turkey": "Türkiye",
        "Türkiye": "Türkiye",
    }
    n = name.strip()
    return fixes.get(n, n)


def name_to_iso3(name: str, manual_map: dict[str, str] | None = None) -> str | None:
    """Convert a country name to ISO-3, with optional manual override."""
    if manual_map and name in manual_map:
        return manual_map[name]
    import pycountry  # local import; avoids hard dependency at import time

    fixed = normalize_country_name(name)
    try:
        c = pycountry.countries.lookup(fixed)
        return c.alpha_3
    except LookupError:
        # Try a fuzzy search
        try:
            c = pycountry.countries.search_fuzzy(fixed)[0]
            return c.alpha_3
        except LookupError:
            return None


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------
def round_year(series: pd.Series) -> pd.Series:
    """Round year-like values to int (handles Maddison's 1820.5 etc.)."""
    return series.round().astype("Int64").astype("int64")


# World Bank regional/aggregate ISO-3 codes. These are not real countries;
# we exclude them from training/eval so they don't dilute the country signal.
# Source: WB "Aggregates" classification.
WB_AGGREGATES = frozenset({
    "AFE", "AFW", "ARB", "CEB", "CSS", "EAP", "EAR", "EAS", "ECA", "ECS",
    "EMU", "EUU", "HIC", "HPC", "IBD", "IBT", "IDA", "IDB", "IDX", "LAC",
    "LCN", "LDC", "LIC", "LMC", "LMY", "LTE", "MEA", "MIC", "MNA", "NAC",
    "OED", "OSS", "PRE", "PSS", "PST", "SAS", "SSA", "SSF", "SST", "TEA",
    "TEC", "TLA", "TMN", "TSA", "TSS", "UMC", "WLD",
})


def coverage_report(df: pd.DataFrame, label: str) -> None:
    n = len(df)
    if n == 0:
        print(f"[{label}] EMPTY")
        return
    print(
        f"[{label}] rows={n:,}  countries={df['iso3'].nunique()}  "
        f"years={df['year'].min()}-{df['year'].max()}  "
        f"indicators={df['indicator_id'].nunique()}"
    )
    print(f"  indicator_ids: {sorted(df['indicator_id'].unique())[:8]}"
          f"{' …' if df['indicator_id'].nunique() > 8 else ''}")
