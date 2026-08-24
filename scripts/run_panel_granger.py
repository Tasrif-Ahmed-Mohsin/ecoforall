"""
Run Dumitrescu-Hurlin (2012) Panel Granger Causality Tests
===========================================================
Tests directional causality across the 4 domains:
- Economy: GDP pc growth, inflation
- Politics: Conflict, political stability, unrest
- Climate/Environment: Disasters, thermal anomalies, emissions
- Human/Society: Trust, fear, social cohesion
"""

from __future__ import annotations
import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from src.econometrics.panel_granger import dumitrescu_hurlin_test


def run_all_panel_granger_tests():
    panel_path = ROOT / "data" / "processed_panels" / "quad_domain_annual_panel.parquet"
    if not panel_path.exists():
        panel_path = ROOT / "data" / "quad_domain_annual_panel.parquet"

    print(f"Loading Panel: {panel_path}")
    df = pd.read_parquet(panel_path)

    test_pairs = [
        ("psychology_trust", "gdp_pc_growth_1y_fwd", "Social Trust -> GDP Growth"),
        ("psychology_fear", "material_conflict_annual_sum", "Societal Fear -> Material Conflict"),
        ("disaster_economic_damage_usd", "psychology_fear", "Climate Damage -> Societal Fear"),
        ("disaster_economic_damage_usd", "material_conflict_annual_sum", "Climate Disasters -> Conflict"),
        ("temperature_anomaly", "goldstein_stability_annual_mean", "Thermal Anomalies -> Stability"),
        ("political_stability_index", "renewable_energy_consumption_pct", "Political Stability -> Clean Energy"),
    ]

    print("=" * 80)
    print("DUMITRESCU-HURLIN (2012) PANEL GRANGER CAUSALITY TEST SUITE")
    print("=" * 80)

    results = []
    for cause, effect, label in test_pairs:
        if cause not in df.columns or effect not in df.columns:
            continue
        res = dumitrescu_hurlin_test(df, cause, effect, max_lag=2, min_obs_per_country=15)
        results.append({
            "Hypothesis / Channel": label,
            "W_bar": res["w_bar"],
            "Z_tilde": res["z_tilde"],
            "p_value": res["p_value"],
            "Countries (N)": res["n_countries"],
            "Causal?": "YES (p < 0.05)" if res["significant"] else "NO"
        })
        sig_str = "SIGNIFICANT (p < 0.05)" if res["significant"] else "Not Significant"
        print(f"[{label}] -> W_bar={res['w_bar']:.4f}, Z={res['z_tilde']:.4f}, p={res['p_value']:.4e} ({sig_str})")

    print("=" * 80)
    res_df = pd.DataFrame(results)
    out_csv = ROOT / "data" / "benchmarks" / "dumitrescu_hurlin_panel_granger_results.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"Saved results to: {out_csv}")


if __name__ == "__main__":
    run_all_panel_granger_tests()
