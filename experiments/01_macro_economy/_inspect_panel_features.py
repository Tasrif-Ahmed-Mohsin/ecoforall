"""Inspect panel_wide.parquet for likely look-ahead leakage.

Hypothesis from the nested-CV run: even after dropping `*_y_fwd` columns the LGBM
inner-val MAE dropped to ~0.001, which means *some* feature in the panel still
carries future information for the target year. We'll print:
  - all column names (group by prefix)
  - for any column whose name hints at growth / change / shift / log / ratio,
    show value-vs-target-year alignment on a few sample rows.
"""
import pandas as pd
import numpy as np
from pathlib import Path

panel = pd.read_parquet(Path(r"E:\project_gmd\data\features\panel_wide.parquet"))
print(f"panel shape: {panel.shape}")
print(f"years: {panel['year'].min()} .. {panel['year'].max()}, unique={panel['year'].nunique()}")
print(f"isos: {panel['iso3'].nunique()}")

# Bucket columns by prefix to see feature families.
buckets: dict[str, list[str]] = {}
for c in panel.columns:
    head = c.split("_")[0] if "_" in c else c
    buckets.setdefault(head, []).append(c)

print("\ncolumn buckets (top 25):")
for k in sorted(buckets, key=lambda x: -len(buckets[x]))[:25]:
    print(f"  {k:20s} -> {len(buckets[k]):3d}  e.g. {buckets[k][:5]}")

# Suspicious-name features (anything suggesting growth, change, future, ratio, diff, lag).
suspicious_kw = (
    "growth", "change", "ratio", "diff", "future", "log_", "_log", "ln_",
    "gdp_pc_g", "forecast", "projected", "yoy", "y_on_y", "fwd", "lead",
    "shifted", "next_",
)
print("\nSuspicious column names:")
hits = [c for c in panel.columns if any(k in c.lower() for k in suspicious_kw)]
print(f"  count: {len(hits)}")
for c in hits:
    print(f"    {c}")

# Build a sample target for h=1 from gdp_pc and check if any feature at year T
# matches target at year T+1 (perfect-correlation test).
df = panel.sort_values(["iso3", "year"]).reset_index(drop=True)
g_fwd = df.groupby("iso3")["gdp_pc"].shift(-1)
target_h1 = np.log(g_fwd / df["gdp_pc"])
df["__target_h1__"] = target_h1

print("\nFeature-vs-target correlation check on h=1 (top 25 absolute correlations):")
# Restrict to rows where target is finite.
mask = df["__target_h1__"].notna()
sample = df.loc[mask, ["__target_h1__", "year"]].copy()
corrs = {}
for c in panel.columns:
    if c in {"iso3", "year", "gdp_pc", "__target_h1__"}:
        continue
    s = df.loc[mask, c]
    if not pd.api.types.is_numeric_dtype(s):
        continue
    try:
        corrs[c] = float(s.corr(df.loc[mask, "__target_h1__"]))
    except Exception:
        pass
top = sorted(corrs.items(), key=lambda kv: -abs(kv[1]) if not np.isnan(kv[1]) else 0)[:25]
for name, val in top:
    print(f"  {name:50s}  corr={val:+.4f}")
