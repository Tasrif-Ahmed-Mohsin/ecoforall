"""Inspect per-row llm_attribution from holdout_decomposition.parquet."""
import pandas as pd
d = pd.read_parquet(r"E:\project_gmd\data\features\cross_horizon_meta\holdout_decomposition.parquet")
print("rows:", len(d))
print("cols:", list(d.columns))
print(f"llm_attribution: min={d['llm_attribution'].min():.6f} "
      f"max={d['llm_attribution'].max():.6f} "
      f"mean={d['llm_attribution'].mean():.6f} "
      f"std={d['llm_attribution'].std():.6f}")
nz = (d['llm_attribution'].abs() > 1e-6).sum()
print(f"rows with non-zero llm_attribution: {nz} ({100*nz/len(d):.2f}%)")
n_llm = d['llm_pred'].notna().sum()
print(f"rows with non-null llm_pred: {n_llm} ({100*n_llm/len(d):.2f}%)")
print()
print("per-horizon attribution stats (only rows where llm_pred present):")
for h, sub in d.dropna(subset=['llm_pred']).groupby('horizon'):
    nz2 = (sub['llm_attribution'].abs() > 1e-6).sum()
    print(f"  h={h}: n_llm={len(sub)}  non_zero_attr={nz2}  "
          f"abs_max={sub['llm_attribution'].abs().max():.6f}  "
          f"mean={sub['llm_attribution'].mean():.6f}")
