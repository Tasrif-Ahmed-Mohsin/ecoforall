"""Audit the panel's `gdp_pc_growth_5y_fwd` column for stale labels.

Background
----------
The GMD raw source covers years up to 2030 (forecast values). `build_panel.build()`
runs `_add_target(wide, horizon=5)` BEFORE the in-window truncation at `max_year=2024`.
That means forward-5y targets for rows in years 2020-2024 are computed using gdp_pc
values from years 2025-2029, which exist in `wide` at compute time but are dropped
before the parquet is written. Result: the stored `gdp_pc_growth_5y_fwd` column has
non-null values for years 2020-2024 that are STALE — they point to a future row that
no longer exists in the parquet.

How to fix
----------
In `src/features/build_panel.py::build()`, move the `max_year` truncation to BEFORE
the `_add_target(wide, horizon=5)` call. Then re-run `python -m src.features.build_panel`
and re-run training for any horizon whose metrics cite `gdp_pc_growth_5y_fwd` (h=5 in
particular; h=1/3/10 targets are built fresh by the trainer and unaffected).

Run this script any time the panel parquet is regenerated to confirm the corruption is gone.
"""
import pandas as pd, numpy as np
from pathlib import Path

p = pd.read_parquet(Path(__file__).resolve().parents[1] / "data/features/panel_wide.parquet")
p = p.sort_values(["iso3", "year"]).reset_index(drop=True)

p["_true_fwd5"] = np.log(
    p.groupby("iso3")["gdp_pc"].shift(-5) / p["gdp_pc"]
)

stored = p["gdp_pc_growth_5y_fwd"]
n_total = len(p)
n_stored = stored.notna().sum()
n_true = p["_true_fwd5"].notna().sum()

# Where stored is non-null but true forward is NaN -> stale
stale_mask = stored.notna() & p["_true_fwd5"].isna()
n_stale = int(stale_mask.sum())

# Where both non-null and disagree -> disagreement (none expected)
disagree = (
    stored.notna() & p["_true_fwd5"].notna()
    & (stored - p["_true_fwd5"]).abs() > 1e-6
).sum()

print(f"panel rows: {n_total:,}")
print(f"stored target non-null: {n_stored:,}")
print(f"true forward-5y non-null: {n_true:,}")
print(f"disagree (both non-null): {int(disagree):,}")
print(f"STALE rows (stored non-null, true NaN): {n_stale:,}")
if n_stale:
    print(f"  by year: {p[stale_mask].year.value_counts().sort_index().to_dict()}")
    print(f"  affected countries: {p[stale_mask].iso3.nunique()}")
print()
if n_stale == 0:
    print("PASS: panel target is clean (no stale labels)")
else:
    print("FAIL: stale labels present — fix build_panel.build() order, rebuild parquet")
    raise SystemExit(1)
