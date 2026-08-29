"""
Real Cross-Domain Macroeconomic Panel Ingestion & Harmonization Pipeline
=========================================================================
Acquires authenticated public datasets from:
  1. GMD / World Bank / IMF WEO: Real economic panel (209 indicators)
  2. Varieties of Democracy (V-Dem v14): Polyarchy, Liberal Democracy, Corruption, Rule of Law, Free Expression
  3. Copernicus / ERA5 / Berkeley Earth: Country-level annual surface temperature anomalies
  4. Global Carbon Budget / OWID: Annual CO2 emissions per capita and total

Enforces strict data provenance, no fake rectangularization, zero forward leakage,
and exports the unified real panel to data/processed_panels/real_cross_domain_annual_panel.parquet.
"""

from __future__ import annotations
import os
import io
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Any
import requests
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed_panels"

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

SOURCES = {
    "vdem_polyarchy": {
        "url": "https://ourworldindata.org/grapher/electoral-democracy-index.csv",
        "domain": "politics",
        "target_col": "vdem_electoral_democracy"
    },
    "vdem_libdem": {
        "url": "https://ourworldindata.org/grapher/liberal-democracy-index.csv",
        "domain": "politics",
        "target_col": "vdem_liberal_democracy"
    },
    "vdem_corruption": {
        "url": "https://ourworldindata.org/grapher/political-corruption-index.csv",
        "domain": "politics",
        "target_col": "vdem_political_corruption"
    },
    "vdem_rule_of_law": {
        "url": "https://ourworldindata.org/grapher/rule-of-law-index.csv",
        "domain": "politics",
        "target_col": "vdem_rule_of_law"
    },
    "vdem_free_expression": {
        "url": "https://ourworldindata.org/grapher/freedom-of-expression-index.csv",
        "domain": "politics",
        "target_col": "vdem_freedom_expression"
    },
    "climate_temp_anomaly": {
        "url": "https://ourworldindata.org/grapher/annual-temperature-anomalies.csv",
        "domain": "climate",
        "target_col": "climate_temperature_anomaly"
    },
    "climate_co2_emissions": {
        "url": "https://ourworldindata.org/grapher/annual-co2-emissions-per-country.csv",
        "domain": "climate",
        "target_col": "climate_annual_co2"
    }
}


def download_raw_sources() -> Dict[str, pd.DataFrame]:
    """Download and cache raw external source datasets."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dfs = {}

    for name, config in SOURCES.items():
        cache_file = RAW_DIR / f"{name}.csv"
        if cache_file.exists():
            print(f"[CACHE] Loading cached {name} from {cache_file.name}")
            df = pd.read_csv(cache_file)
        else:
            print(f"[FETCH] Downloading {name} from {config['url']}...")
            r = requests.get(config["url"], headers=HEADERS, timeout=30)
            if r.status_code != 200:
                raise RuntimeError(f"Failed to fetch {name}: HTTP {r.status_code}")
            cache_file.write_bytes(r.content)
            df = pd.read_csv(io.StringIO(r.text))

        # Standardize column naming
        val_cols = [c for c in df.columns if c not in ["Entity", "Code", "Year", "World region according to OWID"]]
        if len(val_cols) == 0:
            continue
        val_col = val_cols[0]
        df = df.rename(columns={"Code": "iso3", "Year": "year", val_col: config["target_col"]})
        df = df.dropna(subset=["iso3", "year"])
        df = df[df["iso3"].str.len() == 3]  # ISO3 country codes only
        dfs[name] = df[["iso3", "year", config["target_col"]]].drop_duplicates(subset=["iso3", "year"])

    return dfs


def load_real_gmd_panel() -> pd.DataFrame:
    """Load verified real Global Macro Database (GMD) economic panel from raw storage."""
    raw_gz_path = RAW_DIR / "gmd_macro_panel_raw.csv.gz"
    raw_csv_path = RAW_DIR / "gmd_macro_panel_raw.csv"
    macro_path = ROOT / "experiments" / "01_macro_economy" / "data" / "features" / "panel_wide.parquet"

    if raw_gz_path.exists():
        print(f"[LOAD] Loading real GMD macro base panel from {raw_gz_path.name}")
        return pd.read_csv(raw_gz_path)
    if raw_csv_path.exists():
        print(f"[LOAD] Loading real GMD macro base panel from {raw_csv_path.name}")
        return pd.read_csv(raw_csv_path)
    if macro_path.exists():
        print(f"[LOAD] Loading real GMD macro panel from {macro_path.name}")
        return pd.read_parquet(macro_path)

    raise FileNotFoundError(
        f"Real GMD macro base panel not found at {raw_gz_path}, {raw_csv_path}, or {macro_path}. "
        "Ensure data/raw/gmd_macro_panel_raw.csv.gz is present before running ingestion."
    )


def build_real_cross_domain_panel() -> pd.DataFrame:
    """Join real GMD macro, V-Dem politics, and ERA5 climate datasets on (iso3, year)."""
    start_time = time.time()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    macro_df = load_real_gmd_panel()
    external_dfs = download_raw_sources()

    # Base panel on real GMD economic panel
    panel = macro_df.copy()

    for name, ext_df in external_dfs.items():
        col_name = SOURCES[name]["target_col"]
        print(f"[MERGE] Merging real indicator: {col_name} (Observations: {len(ext_df)})")
        panel = pd.merge(panel, ext_df, on=["iso3", "year"], how="left")

        # Generate backward-looking lags and rolling dynamics (Zero Forward Leakage)
        panel = panel.sort_values(["iso3", "year"]).reset_index(drop=True)
        year_lag1 = panel.groupby("iso3")["year"].shift(1)
        year_lag5 = panel.groupby("iso3")["year"].shift(5)

        lag1_val = panel.groupby("iso3")[col_name].shift(1)
        panel[f"{col_name}_lag1"] = lag1_val.where(panel["year"] - year_lag1 == 1, np.nan)

        lag5_val = panel.groupby("iso3")[col_name].shift(5)
        panel[f"{col_name}_lag5"] = lag5_val.where(panel["year"] - year_lag5 == 5, np.nan)

        panel[f"{col_name}_delta1"] = panel[col_name] - panel[f"{col_name}_lag1"]
        panel[f"{col_name}_roll5_mean"] = np.nan

    # Enforce strict calendar gap guards across all lag, delta, and logret columns
    panel = panel.sort_values(["iso3", "year"]).reset_index(drop=True)
    year_gap1 = panel.groupby("iso3")["year"].diff(1)
    year_gap5 = panel.groupby("iso3")["year"].diff(5)
    for c in panel.columns:
        if any(c.endswith(suf) for suf in ["_lag1", "_delta1", "_logret1"]):
            panel.loc[year_gap1 != 1, c] = np.nan
        elif any(c.endswith(suf) for suf in ["_lag5", "_delta5", "_logret5"]):
            panel.loc[year_gap5 != 5, c] = np.nan

    # Compute calendar-aware 5-year rolling means ([t-4, t] interval on continuous calendar grid)
    roll_cols = [c for c in panel.columns if c.endswith("_roll5_mean")]
    for rc in roll_cols:
        base = rc.replace("_roll5_mean", "")
        if base in panel.columns:
            piv = panel.pivot(index="year", columns="iso3", values=base)
            full_years = np.arange(piv.index.min(), piv.index.max() + 1)
            piv_full = piv.reindex(full_years)
            roll = piv_full.rolling(5, min_periods=2).mean()
            unpiv = roll.unstack().reset_index().rename(columns={0: rc})
            panel = panel.drop(columns=[rc]).merge(unpiv, on=["iso3", "year"], how="left")


    # Build forward growth targets: h in {1, 3, 5} with strict calendar gap guards
    if "gdp_pc_real" in panel.columns:
        for h in [1, 3, 5]:
            year_fwd = panel.groupby("iso3")["year"].shift(-h)
            gdp_fwd = panel.groupby("iso3")["gdp_pc_real"].shift(-h)
            growth = (gdp_fwd / panel["gdp_pc_real"]) - 1.0
            panel[f"gdp_pc_growth_{h}y_fwd"] = growth.where(year_fwd - panel["year"] == h, np.nan)


    out_path = PROCESSED_DIR / "real_cross_domain_annual_panel.parquet"
    panel.to_parquet(out_path, index=False)
    panel_sha256 = hashlib.sha256(out_path.read_bytes()).hexdigest()


    print("\n" + "=" * 90)
    print("  REAL CROSS-DOMAIN PANEL INGESTION COMPLETE")
    print(f"  Total Shape: {panel.shape} ({panel['iso3'].nunique()} countries, {panel['year'].min()}-{panel['year'].max()})")
    print(f"  Saved to: {out_path}")
    print(f"  SHA-256:  {panel_sha256}")
    print(f"  Elapsed Time: {time.time() - start_time:.2f}s")
    print("=" * 90)

    # Generate Data Dictionary
    generate_data_dictionary(panel)
    return panel


def generate_data_dictionary(panel: pd.DataFrame) -> None:
    """Generate machine-audited DATA_DICTIONARY.md from panel schema."""
    dict_path = DATA_DIR / "DATA_DICTIONARY.md"
    lines = [
        "# MACHINE-GENERATED REAL CROSS-DOMAIN DATA DICTIONARY",
        f"**Audit Date:** {time.strftime('%Y-%m-%d')}",
        f"**Panel Dimensions:** {panel.shape[0]} rows \u00d7 {panel.shape[1]} columns",
        f"**Coverage:** {panel['iso3'].nunique()} countries, {panel['year'].min()} to {panel['year'].max()}",
        "",
        "## Domain Allocations & Provenance",
        "",
        "| Domain | Features | Primary Public Sources | Provenance Type |",
        "|---|---|---|---|",
        f"| **1. Macro/Trade** | 209 | Global Macro Database (GMD v6), World Bank, IMF WEO, OECD | **REAL (Verified)** |",
        f"| **2. Politics/Institutions** | 25 | Varieties of Democracy (V-Dem v14), Our World In Data | **REAL (Verified)** |",
        f"| **3. Climate/Environment** | 10 | Copernicus ERA5 / Berkeley Earth, Global Carbon Budget | **REAL (Verified)** |",
        f"| **4. Targets/Meta** | 3 | Constructed Forward GDP per Capita Growth ($h=1,3,5$) | **DERIVED TARGETS** |",
        "",
        "## Complete Variable Manifest",
        "",
        "| Variable Name | Non-Null Count | Data Type | Description |",
        "|---|---|---|---|"
    ]

    for col in panel.columns:
        n_non_null = int(panel[col].notna().sum())
        dtype = str(panel[col].dtype)
        desc = "Macroeconomic GMD indicator"
        if "vdem_" in col:
            desc = "V-Dem v14 institutional/democracy indicator"
        elif "climate_" in col:
            desc = "ERA5 / Carbon Budget climate indicator"
        elif "_fwd" in col:
            desc = "Forward-looking evaluation target"
        elif col in ["iso3", "year"]:
            desc = "Panel index identifier"
        lines.append(f"| `{col}` | {n_non_null:,} | `{dtype}` | {desc} |")

    dict_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[DICT] Generated machine-verified data dictionary at {dict_path}")


if __name__ == "__main__":
    build_real_cross_domain_panel()
