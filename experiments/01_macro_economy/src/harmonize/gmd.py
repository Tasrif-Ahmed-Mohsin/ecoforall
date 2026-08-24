"""Harmonize the Geo-Macroeconomic Dataset (Müller et al., 2026 v6).

Source: E:\\GMD_2026_06_csv\\GMD.csv

Wide format (~162 columns), one row per (country, year). This module:

  1. Reads the wide CSV
  2. Melts to long (iso3, year, indicator_id, value, ...)
  3. Strips the `forecast_*` columns (IMF-style — we never train on forecasts)
  4. Renames the ~162 GMD columns to canonical indicator_ids
  5. Writes `data/harmonized/gmd.parquet` in the long schema used by every
     other source (see `common.CANONICAL_SCHEMA`).

Run from e:\\research\\project_gmd:
    python -m src.harmonize.gmd
or:
    python scripts/run_phase1.py   (after wiring the SOURCES list)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common import (
    CANONICAL_COLUMNS,
    coverage_report,
    write_long,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GMD_CSV = Path(r"E:\GMD_2026_06_csv\GMD.csv")
SOURCE_NAME = "gmd"


# ---------------------------------------------------------------------------
# Column mapping: GMD's wide names  ->  our canonical indicator_ids
# ---------------------------------------------------------------------------
# This is the single biggest customisation point. Add/rename freely.
# Anything not in this map is DROPPED.
COLUMN_MAP: dict[str, str] = {
    # identifiers (handled separately)
    "ISO3": "__iso3__",
    "year": "__year__",
    # core target
    "rGDP_pc":       "gdp_pc_real",         # real GDP per capita, constant LCU
    "rGDP_pc_USD":   "gdp_pc_real_usd",     # real GDP per capita, constant USD
    "nGDP":          "gdp_nominal",         # nominal GDP, current LCU
    "nGDP_USD":      "gdp_nominal_usd",
    "rGDP":          "gdp_real",
    "rGDP_USD":      "gdp_real_usd",
    "pop":           "population",
    "deflator":      "gdp_deflator",
    # prices and rates
    "CPI":           "cpi",
    "infl":          "inflation_rate",
    "USDfx":         "fx_to_usd",
    "REER":          "reer",
    "strate":        "short_rate",
    "ltrate":        "long_rate",
    "cbrate":        "central_bank_rate",
    "unemp":         "unemployment_rate",
    "HPI":           "house_price_index",
    "rHPI":          "real_house_price_index",
    # fiscal and external
    "CA":            "current_account",
    "CA_GDP":        "current_account_gdp",
    "govdebt_GDP":   "gov_debt_gdp",
    "gen_govdebt_GDP":  "gen_gov_debt_gdp",
    "cgovdebt_GDP":     "cgov_debt_gdp",
    "govdef_GDP":       "gov_deficit_gdp",
    "gen_govdef_GDP":   "gen_gov_deficit_gdp",
    "govexp_GDP":       "gov_expenditure_gdp",
    "govrev_GDP":       "gov_revenue_gdp",
    "govtax_GDP":       "gov_tax_gdp",
    "gen_govtax_GDP":   "gen_gov_tax_gdp",
    "cons_GDP":         "consumption_gdp",
    "hcons_GDP":        "household_consumption_gdp",
    "gcons_GDP":        "gov_consumption_gdp",
    "inv_GDP":          "investment_gdp",
    "finv_GDP":         "fixed_investment_gdp",
    "exports_GDP":      "exports_gdp",
    "imports_GDP":      "imports_gdp",
    "CA_USD":           "current_account_usd",
    "govdebt":          "gov_debt",
    "govdef":           "gov_deficit",
    "govexp":           "gov_expenditure",
    "govrev":           "gov_revenue",
    "govtax":           "gov_tax",
    # monetary
    "M0": "money_supply_m0",
    "M1": "money_supply_m1",
    "M2": "money_supply_m2",
    "M3": "money_supply_m3",
    "M4": "money_supply_m4",
    # crisis dummies
    "SovDebtCrisis":  "sov_debt_crisis",
    "CurrencyCrisis": "currency_crisis",
    "BankingCrisis":  "banking_crisis",
    # income group (categorical)
    "income_group":   "__income_group__",
}


# Columns to drop explicitly (we never want these in features/target).
# `forecast_*` mirror IMF's forecast rows — strip them like we do in build_panel.
FORECAST_PREFIX = "forecast_"
ID_COLS = {"countryname", "ISO3", "id", "year", "income_group"}


def _melt_wide_to_long(wide: pd.DataFrame, min_year: int = 1900) -> pd.DataFrame:
    """Melt the GMD wide CSV to the canonical long schema.

    `min_year` (default 1900) drops the long-but-sparse historical tail in
    GMD (we observed years as old as 1086 for a handful of countries). The
    economic-modelling signal really starts ~1900 anyway.

    Returns DataFrame with columns:
        iso3, year, indicator_id, value, unit, scale, source, is_forecast
    """
    # 0. Drop junk historical years BEFORE melt — much cheaper than filtering
    # the long table afterwards.
    wide = wide[wide["year"] >= min_year].copy()

    # 1. Filter out forecast columns (anything starting with `forecast_`).
    keep_cols = [c for c in wide.columns if not c.startswith(FORECAST_PREFIX)]
    wide = wide[keep_cols].copy()

    # 2. Identify id columns and indicator columns.
    indicator_cols = [c for c in wide.columns
                      if c not in {"countryname", "id"}]  # ISO3, year, income_group kept

    # 3. Melt.
    long = wide.melt(
        id_vars=["ISO3", "year"],
        value_vars=[c for c in indicator_cols if c not in {"ISO3", "year"}],
        var_name="raw_name",
        value_name="value",
    )

    # 4. Rename GMD raw_name -> indicator_id via COLUMN_MAP. Anything not in
    #    the map (and not an identifier) gets dropped — return None rows.
    long["indicator_id"] = long["raw_name"].map(COLUMN_MAP)
    long = long.dropna(subset=["indicator_id"]).copy()
    # Strip the placeholder sentinels.
    long = long[~long["indicator_id"].str.startswith("__")].copy()

    # 5. ISO3 + year as canonical types.
    long["iso3"] = long["ISO3"].astype("string")
    long["year"] = long["year"].astype("int64")

    # 6. Fill canonical metadata columns.
    long["unit"] = "unknown"          # TODO: fill per-indicator
    long["scale"] = "units"
    long["source"] = SOURCE_NAME
    long["is_forecast"] = False

    return long[CANONICAL_COLUMNS]


def harmonize(min_year: int = 1900) -> Path:
    """Read GMD.csv, melt to long, write `data/harmonized/gmd.parquet`.

    Args:
        min_year: drop country-years strictly before this year. GMD has a long
            but sparse pre-1900 tail; default 1900.
    """
    print(f"[{SOURCE_NAME}] reading {GMD_CSV} …")
    wide = pd.read_csv(GMD_CSV)
    print(f"[{SOURCE_NAME}] wide shape: {wide.shape}")

    long = _melt_wide_to_long(wide, min_year=min_year)
    coverage_report(long, SOURCE_NAME)

    out_path = write_long(long, SOURCE_NAME)
    print(f"[{SOURCE_NAME}] wrote {out_path}")
    return out_path


if __name__ == "__main__":
    harmonize()
