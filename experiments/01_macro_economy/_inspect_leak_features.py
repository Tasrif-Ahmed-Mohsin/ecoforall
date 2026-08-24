"""Look for *implicit* look-ahead leakage: features that are NOT named *_fwd but whose
values at year T depend on year T+1..T+h. We test this by checking the year-aligned
correlation of every feature against the h=1 target and flagging anything whose
correlation is suspiciously high (>0.4) AND whose name does NOT contain "_fwd".

Then we cross-check with the panel: for the suspicious features, examine value at
year T vs value at year T+1 -- if they're identical (modulo small drift), the column
was built with .shift(-k) somewhere in the harmonization pipeline.
"""
import pandas as pd
import numpy as np

panel = pd.read_parquet(r"E:\project_gmd\data\features\panel_wide.parquet")
df = panel.sort_values(["iso3", "year"]).reset_index(drop=True)
g_fwd = df.groupby("iso3")["gdp_pc"].shift(-1)
target = np.log(g_fwd / df["gdp_pc"])
mask = target.notna()

print("Per-feature correlation with h=1 target (>0.3 abs, not *_fwd, not iso/year/gdp):")
hits = []
for c in panel.columns:
    if c in {"iso3", "year", "gdp_pc"} or "_fwd" in c.lower():
        continue
    s = df.loc[mask, c]
    if not pd.api.types.is_numeric_dtype(s):
        continue
    if s.nunique(dropna=True) < 3:
        continue
    try:
        cor = float(s.corr(target[mask]))
    except Exception:
        cor = np.nan
    if not np.isnan(cor) and abs(cor) > 0.3:
        hits.append((c, cor))

for name, cor in sorted(hits, key=lambda kv: -abs(kv[1])):
    print(f"  {name:50s}  {cor:+.4f}")

# Now check the SHAPE of look-ahead: for each suspicious feature f, see whether
# f(year T) equals f(year T+1) for many rows. If yes, it was shifted.
print("\nLag-1 self-similarity test (high value means feature is forward-shifted):")
for name, _ in sorted(hits, key=lambda kv: -abs(kv[1]))[:15]:
    s = df[["iso3", "year", name]].dropna()
    if s.empty:
        continue
    s = s.sort_values(["iso3", "year"])
    s["next"] = s.groupby("iso3")[name].shift(-1)
    diff = (s[name] - s["next"]).abs()
    med = float(diff.median()) if not diff.empty else float("nan")
    same = float((diff < 1e-6).mean()) if not diff.empty else float("nan")
    print(f"  {name:50s}  median|Δ|={med:.6g}  frac_identical(lag1)={same:.4f}")