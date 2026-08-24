"""Trace whether gdp_pc_growth_5y_fwd reaches the LGBM feature set in either
run_phase8_horizons_v2.py or _panel_backtest.py."""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, r"E:\project_gmd\scripts")
from run_phase8_horizons_v2 import (
    LEAK_COLS_BASE, DROP_FEATURES, _prepare,
    _add_country_and_tier_dummies, _horizon_target_name, _build_horizon_target,
)

panel = pd.read_parquet(r"E:\project_gmd\data\features\panel_wide.parquet")
print("panel columns containing '_fwd':")
for c in panel.columns:
    if "fwd" in c.lower():
        print(f"  {c}")

# Mimic the v2 _prepare path for h=1.
h = 1
target = _horizon_target_name(h)
panel2 = panel.copy()
panel2[target] = _build_horizon_target(panel2, h)
df = panel2.dropna(subset=[target]).reset_index(drop=True)
iso_levels = sorted(df["iso3"].unique().tolist())
X_cont, X_full, y, cont_cols, full_cols = _prepare(df, target, iso_levels)

print(f"\nh={h} v2 _prepare cont_cols: {len(cont_cols)}")
print(f"h={h} v2 _prepare full_cols: {len(full_cols)}")
print(f"\nAny '_fwd' col in cont_cols? {[c for c in cont_cols if 'fwd' in c.lower()]}")
print(f"Any 'growth' col in cont_cols? {[c for c in cont_cols if 'growth' in c.lower()]}")
print(f"Any 'gdp_pc_growth' col in cont_cols? {[c for c in cont_cols if 'gdp_pc_growth' in c.lower()]}")

# Same check for h=5
h = 5
target = _horizon_target_name(h)
panel2 = panel.copy()
panel2[target] = _build_horizon_target(panel2, h)
df = panel2.dropna(subset=[target]).reset_index(drop=True)
X_cont, X_full, y, cont_cols, full_cols = _prepare(df, target, iso_levels)
print(f"\nh={h} v2 _prepare cont_cols: {len(cont_cols)}")
print(f"Any 'gdp_pc_growth' col in cont_cols? {[c for c in cont_cols if 'gdp_pc_growth' in c.lower()]}")

# Per-iso quick look-ahead correlation for the suspicious feature.
df_sorted = panel.sort_values(["iso3", "year"]).reset_index(drop=True)
for h in [1, 3, 5]:
    target = np.log(df_sorted.groupby("iso3")["gdp_pc"].shift(-h) / df_sorted["gdp_pc"])
    mask = target.notna()
    cor = df_sorted.loc[mask, "gdp_pc_growth_5y_fwd"].corr(target[mask])
    print(f"corr(gdp_pc_growth_5y_fwd at year T, target_h{h} at year T) = {cor:+.4f}")