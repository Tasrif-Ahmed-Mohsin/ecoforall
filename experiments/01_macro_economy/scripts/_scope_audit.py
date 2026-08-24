"""One-shot audit script: print country/year coverage per source and modeled panel."""
import pandas as pd

print("=" * 70)
print("RAW HARMONIZED SOURCES (what was ingested)")
print("=" * 70)
total_iso = set()
for f in ["clio_infra", "imf", "jst", "maddison", "wb"]:
    df = pd.read_parquet(f"data/harmonized/{f}.parquet")
    iso = df["iso3"].unique().tolist()
    yr_lo = int(df.year.min())
    yr_hi = int(df.year.max())
    rows = len(df)
    print(f"  {f:12s}  rows={rows:>8,}  countries={len(iso):>4}  years={yr_lo}-{yr_hi}")
    total_iso.update(iso)
print(f"  {'UNION':12s}  unique countries = {len(total_iso)}")

print()
print("=" * 70)
print("MODELED PANEL (what is actually trained on)")
print("=" * 70)
panel = pd.read_parquet("data/features/panel_wide.parquet")
iso = sorted(panel["iso3"].unique().tolist())
yr_lo = int(panel.year.min())
yr_hi = int(panel.year.max())
rows = len(panel)
print(f"  rows={rows:>8,}  countries={len(iso):>4}  years={yr_lo}-{yr_hi}")
print(f"  target col: gdp_pc_growth_5y_fwd")
non_null = panel["gdp_pc_growth_5y_fwd"].notna().sum()
print(f"  rows with target: {non_null:,} (the rest are missing-end-of-series)")
print()
print("  country list:")
for i in range(0, len(iso), 8):
    print("    " + "  ".join(iso[i:i+8]))

print()
print("=" * 70)
print("TEST-SLICE COVERAGE (what we evaluated on)")
print("=" * 70)
for split, lo, hi in [("train", 0, 2014), ("val", 2015, 2018), ("test", 2019, 2022), ("hold", 2023, 2099)]:
    mask = (panel.year >= lo) & (panel.year <= hi) & panel["gdp_pc_growth_5y_fwd"].notna()
    n = mask.sum()
    n_iso = panel.loc[mask, "iso3"].nunique()
    print(f"  {split:5s}  {lo}-{hi}:  rows={n:>5,}  countries={n_iso:>3}")

print()
print("=" * 70)
print("RETRIEVAL INDEX")
print("=" * 70)
rows_idx = pd.read_parquet("data/features/retrieval/panel_ranked_rows.parquet")
print(f"  panel_ranked rows: {len(rows_idx):,}  countries: {rows_idx.iso3.nunique()}  years: {int(rows_idx.year.min())}-{int(rows_idx.year.max())}")
print(f"  min_overlap (production) = 60")
print(f"  expert weights file: {'weights.yml exists' if __import__('os').path.exists('weights.yml') else 'absent'}")

print()
print("=" * 70)
print("MODELS ON DISK")
print("=" * 70)
import os
for f in ["ridge.joblib", "lgbm.joblib", "imputer.joblib", "lgbm_q05.joblib"]:
    p = f"data/features/models/{f}"
    print(f"  {p:40s}  exists={os.path.exists(p)}")
for f in ["baseline_metrics.json", "conformal_adjustment.json"]:
    p = f"data/features/{f}"
    print(f"  {p:40s}  exists={os.path.exists(p)}")