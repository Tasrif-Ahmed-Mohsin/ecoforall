"""Build the ML-ready wide panel from the harmonized long parquets.

Steps
-----
1. Concatenate the 5 long parquets.
2. Drop IMF forecast rows (is_forecast=True) — we never train on forecasts.
3. Deduplicate overlapping `(iso3, year, indicator_id)` rows using a source
   priority (wb < jst < clio_infra < imf < maddison) — the *last* wins.
4. Pivot to wide: rows = (iso3, year), columns = indicator_id.
5. Compute the target: 5-year-ahead real GDP per-capita growth.
6. Compute feature lags and rolling stats for the core indicator list.
7. Write `data/features/panel_wide.parquet`.
"""
from __future__ import annotations

from pathlib import Path

import warnings

import numpy as np
import pandas as pd

# Silence pandas RuntimeWarning when log returns hit 0 (e.g. gdppc=0 pre-1820).
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"pandas\..*")

from ..harmonize.common import FEATURES_DIR, HARMONIZED_DIR, WB_AGGREGATES, ensure_dirs, load_long

# GMD-only clone. Single source = "gmd" (Geo-Macroeconomic Dataset v6).
# When you eventually fold the old 5-source panel back in, just restore the
# list below (see e:\research\project for the original multi-source config).
SOURCES = ["gmd"]

# When multiple sources report the same (iso3, year, indicator_id), this
# priority decides which value we keep. Higher index = preferred.
SOURCE_PRIORITY = {"gmd": 0}

# The core indicator set — matches what the GMD harmonizer actually emits.
# (Anything not in gmd.py:COLUMN_MAP will silently produce an all-NaN column
# and waste a feature slot — so this list is a strict subset of COLUMN_MAP.)
CORE_TARGETS = [
    # core target inputs
    "gdp_pc_real",           # real GDP per capita, constant LCU
    "gdp_pc_real_usd",       # real GDP per capita, constant USD (unit-stable!)
    "gdp_nominal",
    "gdp_deflator",
    "population",
    # prices and rates
    "cpi",
    "inflation_rate",
    "fx_to_usd",
    "reer",
    "short_rate",
    "long_rate",
    "central_bank_rate",
    "unemployment_rate",
    "real_house_price_index",
    # fiscal and external (% of GDP)
    "gov_debt_gdp",
    "gov_deficit_gdp",
    "gov_expenditure_gdp",
    "gov_revenue_gdp",
    "gov_tax_gdp",
    "current_account_gdp",
    "consumption_gdp",
    "investment_gdp",
    "fixed_investment_gdp",
    "exports_gdp",
    "imports_gdp",
    # monetary aggregates
    "money_supply_m1",
    "money_supply_m2",
    "money_supply_m3",
    # crisis dummies (these are 0/1 already — pure signal)
    "sov_debt_crisis",
    "currency_crisis",
    "banking_crisis",
]

# Country priority for the GDP-per-capita target when several sources carry it.
# (Sources are tried in this order; first non-null wins.)
# GMD ships BOTH a constant-LCU real GDP per capita AND a constant-USD one;
# prefer the USD one for cross-country comparability (this was a known pain
# point in the v2 IMF-vs-WB benchmark).
GDP_PC_CANDIDATES = ["gdp_pc_real_usd", "gdp_pc_real"]


def _load_all_long() -> pd.DataFrame:
    frames = [load_long(s) for s in SOURCES]
    df = pd.concat(frames, ignore_index=True)
    # Never train on IMF forecasts.
    df = df[~df["is_forecast"].fillna(False)].copy()
    return df


def _dedupe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_prio"] = df["source"].map(SOURCE_PRIORITY).fillna(-1)
    df = df.sort_values("_prio", kind="mergesort")
    df = df.drop_duplicates(subset=["iso3", "year", "indicator_id"], keep="last")
    return df.drop(columns="_prio")


def _to_wide(df: pd.DataFrame, value_col: str = "value") -> pd.DataFrame:
    wide = df.pivot_table(
        index=["iso3", "year"],
        columns="indicator_id",
        values=value_col,
        aggfunc="first",
    )
    wide.columns.name = None
    wide = wide.reset_index()
    return wide.sort_values(["iso3", "year"]).reset_index(drop=True)


def _add_gdp_pc_unified(wide: pd.DataFrame) -> pd.DataFrame:
    """Build a single `gdp_pc` column preferring real > real_ppp, by country-year."""
    wide = wide.copy()
    if "gdp_pc_real" in wide.columns or "gdp_pc_real_ppp" in wide.columns:
        candidates = [c for c in GDP_PC_CANDIDATES if c in wide.columns]
        wide["gdp_pc"] = wide[candidates].bfill(axis=1).iloc[:, 0]
    return wide


def _add_target(wide: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """5-year-ahead log-return on unified gdp_pc."""
    wide = wide.sort_values(["iso3", "year"]).copy()
    wide["gdp_pc_growth_5y_fwd"] = (
        wide.groupby("iso3")["gdp_pc"]
        .shift(-horizon)
        .pipe(lambda s: np.log(s / wide["gdp_pc"]))
    )
    return wide


def _add_lag_features(wide: pd.DataFrame, base_cols: list[str]) -> pd.DataFrame:
    """For each base column: t-1, t-5, 5y rolling mean, 5y delta, 5y log-return."""
    out = wide.sort_values(["iso3", "year"]).copy()
    g = out.groupby("iso3", group_keys=False)
    for col in base_cols:
        if col not in out.columns:
            continue
        s = out[col]
        out[f"{col}_lag1"] = g[col].shift(1)
        out[f"{col}_lag5"] = g[col].shift(5)
        out[f"{col}_roll5_mean"] = g[col].rolling(5, min_periods=3).mean().reset_index(level=0, drop=True)
        out[f"{col}_delta5"] = out[col] - g[col].shift(5)
        out[f"{col}_logret5"] = np.log(out[col] / g[col].shift(5))
    return out


def build(min_year: int = 1960, max_year: int = 2024, drop_aggregates: bool = True) -> Path:
    """Build the ML-ready wide panel.

    drop_aggregates=True removes the ~47 World Bank regional aggregate codes
    (ARB, HIC, EMU, etc.) from the panel before training. These are not real
    countries and dilute the per-country macro signal.
    """
    ensure_dirs()
    long = _load_all_long()
    print(f"[features] loaded long rows: {len(long):,}")

    deduped = _dedupe(long)
    print(f"[features] after dedupe (by source priority): {len(deduped):,}")

    wide = _to_wide(deduped)
    print(f"[features] wide shape: {wide.shape}  "
          f"iso3s={wide['iso3'].nunique()}  "
          f"years={wide['year'].min()}-{wide['year'].max()}")

    # Filter to a sensible modelling window BEFORE computing the forward target,
    # otherwise the shift(-horizon) lookup can resolve to a row that exists at
    # compute time but gets dropped by the truncation, leaving stale, non-null
    # labels in the written parquet. See scripts/_check_target_corruption.py.
    in_window = wide[(wide["year"] >= min_year) & (wide["year"] <= max_year)].copy()
    print(f"[features] window {min_year}-{max_year}: rows={len(in_window):,}")

    in_window = _add_gdp_pc_unified(in_window)
    in_window = _add_target(in_window, horizon=5)

    feature_cols = [c for c in CORE_TARGETS if c in in_window.columns and c != "gdp_pc_real_ppp"]
    # NOTE: gdp_pc_real_usd is the unit-stable target input — keep it in the
    # lag-feature pass even though it doubles as our primary target column.
    if "gdp_pc_real_usd" in in_window.columns:
        feature_cols = feature_cols + ["gdp_pc_real_usd"]
    feature_cols = sorted(set(feature_cols))
    in_window = _add_lag_features(in_window, feature_cols)

    if drop_aggregates:
        before = in_window["iso3"].nunique()
        in_window = in_window[~in_window["iso3"].isin(WB_AGGREGATES)].copy()
        print(f"[features] dropped WB aggregates: {before} -> {in_window['iso3'].nunique()} countries")

    print(f"[features] post-aggregate: rows={len(in_window):,}, "
          f"countries={in_window['iso3'].nunique()}, "
          f"target non-null={in_window['gdp_pc_growth_5y_fwd'].notna().sum():,}")

    out_path = FEATURES_DIR / "panel_wide.parquet"
    in_window.to_parquet(out_path, index=False)
    print(f"[features] wrote: {out_path}")
    return out_path


if __name__ == "__main__":
    print(build())