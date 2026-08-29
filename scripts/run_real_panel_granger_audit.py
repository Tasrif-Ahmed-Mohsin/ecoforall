"""
Real Panel Granger Causality & Cointegration Audit Suite (CIPS Pre-Tested & Exact Finite-T)
=============================================================================================
Tests whether real institutional and climate indicators Granger-cause real GDP growth
using the exact finite-T Dumitrescu-Hurlin (2012) test with:
  - Pesaran (2007) CIPS second-generation panel unit root pre-testing (accounting for CSD)
  - Pedroni (1999, 2004) panel cointegration diagnostics for I(1) series
  - CIPS-governed global integration differencing
  - Pesaran (2004) Cross-Sectional Dependence (CD) diagnostics
  - Holm-Bonferroni FWER stepdown control (m=7)

Exports verified Single-Source-of-Truth CSV artifacts to data/benchmarks/.
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.econometrics.panel_granger import (
    dumitrescu_hurlin_test,
    dumitrescu_hurlin_bootstrap_test,
    cross_sectionally_augmented_granger_test,
    holm_bonferroni_correction,
    pesaran_cd_test,
    pesaran_cips_test,
    pedroni_panel_cointegration_test,
)
from src.utils.reproducibility import seed_everything


def run_real_granger_audit():
    seed_everything(42)
    panel_path = ROOT / "data" / "processed_panels" / "real_cross_domain_annual_panel.parquet"
    if not panel_path.exists():
        raise FileNotFoundError(f"Real panel not found at {panel_path}")

    df = pd.read_parquet(panel_path)

    # Compute real GDP per capita growth if not present
    if "gdp_growth" not in df.columns:
        df["gdp_growth"] = df.groupby("iso3")["gdp_pc_real"].pct_change(fill_method=None)

    out_dir = ROOT / "data" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. SECOND-GENERATION PANEL UNIT ROOT AUDIT (PESARAN CIPS 2007) ---
    print("=" * 110)
    print("  1. PESARAN (2007) CIPS SECOND-GENERATION PANEL UNIT ROOT AUDIT (UNDER CSD)")
    print("  Null H0: Homogeneous non-stationary unit root across all sovereigns")
    print("=" * 110)

    vars_to_test = [
        ("vdem_electoral_democracy", "V-Dem Electoral Democracy"),
        ("vdem_liberal_democracy", "V-Dem Liberal Democracy"),
        ("vdem_political_corruption", "V-Dem Political Corruption"),
        ("vdem_rule_of_law", "V-Dem Rule of Law"),
        ("vdem_freedom_expression", "V-Dem Free Expression"),
        ("climate_temperature_anomaly", "ERA5 Temperature Anomaly"),
        ("climate_annual_co2", "Annual CO2 Emissions"),
        ("gdp_growth", "Real GDP per capita Growth"),
    ]

    cips_results = []
    cips_order_map = {}

    for col, var_name in vars_to_test:
        if col not in df.columns:
            continue
        # Level test
        res_level = pesaran_cips_test(df, variable_col=col, lags=1, min_obs_per_country=20)
        
        # First difference test
        df[f"{col}_diff"] = df.groupby("iso3")[col].diff()
        res_diff = pesaran_cips_test(df, variable_col=f"{col}_diff", lags=1, min_obs_per_country=20)

        order_code = "I(0)" if res_level["is_stationary_05pct"] else "I(1)"
        cips_order_map[col] = order_code

        cips_results.append({
            "Variable": var_name,
            "Column": col,
            "CIPS_Level": res_level["cips_stat"],
            "CIPS_Level_PVal": res_level["p_val_approx"],
            "Order_Level": res_level["order_of_integration"],
            "CIPS_Diff": res_diff["cips_stat"],
            "CIPS_Diff_PVal": res_diff["p_val_approx"],
            "Order_Diff": res_diff["order_of_integration"],
            "N_Countries": res_level["n_countries"],
            "Verdict_Under_CSD": "I(1) Unit Root in Levels -> Must First-Difference" if order_code == "I(1)" else "I(0) Stationary in Levels",
        })
        print(f"  {var_name:<30} | Level CIPS: {res_level['cips_stat']:6.3f} ({res_level['p_val_approx']}) -> {res_level['order_of_integration']:<15} | Diff CIPS: {res_diff['cips_stat']:6.3f} ({res_diff['p_val_approx']})")

    cips_df = pd.DataFrame(cips_results)
    cips_path = out_dir / "real_cips_unit_root_results.csv"
    cips_df.to_csv(cips_path, index=False)
    print(f"\n[SSoT] CIPS Unit Root results saved to: {cips_path}\n")

    # --- 2. PANEL COINTEGRATION AUDIT (PEDRONI 1999, 2004) FOR I(1) SERIES ---
    print("=" * 110)
    print("  2. PEDRONI (1999, 2004) RESIDUAL-BASED PANEL COINTEGRATION AUDIT")
    print("  Null H0: No cointegration between non-stationary series and sovereign GDP levels")
    print("=" * 110)

    # Compute log real GDP per capita level for cointegration tests
    df["log_gdp_pc_real"] = np.log(np.maximum(10.0, df["gdp_pc_real"]))
    coint_tests = [
        ("log_gdp_pc_real", "vdem_rule_of_law", "Log GDP pc ~ V-Dem Rule of Law"),
        ("log_gdp_pc_real", "climate_annual_co2", "Log GDP pc ~ Annual CO2 Emissions"),
    ]

    coint_results = []
    for y_col, x_col, label in coint_tests:
        if y_col in df.columns and x_col in df.columns:
            res_coint = pedroni_panel_cointegration_test(df, y_col=y_col, x_col=x_col, lags=2, min_obs=20)
            coint_results.append({
                "Relationship": label,
                "Y_Variable": y_col,
                "X_Variable": x_col,
                "T_Bar_ADF": res_coint["t_bar"],
                "Z_Group_ADF": res_coint["z_group_adf"],
                "P_Value": res_coint["p_value"],
                "N_Countries": res_coint["n_countries"],
                "Cointegrated_5pct": res_coint["cointegrated_5pct"],
                "Empirical_Verdict": res_coint["verdict"]
            })
            print(f"  [{label}] -> Z_Group_ADF={res_coint['z_group_adf']:6.3f}, p={res_coint['p_value']:.4e} -> {res_coint['verdict']}")

    coint_df = pd.DataFrame(coint_results)
    coint_path = out_dir / "real_cointegration_results.csv"
    coint_df.to_csv(coint_path, index=False)
    print(f"\n[SSoT] Cointegration results saved to: {coint_path}\n")

    # --- 3. DUMITRESCU-HURLIN (2012) PANEL GRANGER CAUSALITY AUDIT ---
    hypotheses = [
        {"cause": "vdem_electoral_democracy", "effect": "gdp_growth", "name": "V-Dem Electoral Democracy -> GDP Growth"},
        {"cause": "vdem_liberal_democracy", "effect": "gdp_growth", "name": "V-Dem Liberal Democracy -> GDP Growth"},
        {"cause": "vdem_political_corruption", "effect": "gdp_growth", "name": "V-Dem Corruption -> GDP Growth"},
        {"cause": "vdem_rule_of_law", "effect": "gdp_growth", "name": "V-Dem Rule of Law -> GDP Growth"},
        {"cause": "vdem_freedom_expression", "effect": "gdp_growth", "name": "V-Dem Free Expression -> GDP Growth"},
        {"cause": "climate_temperature_anomaly", "effect": "gdp_growth", "name": "ERA5 Temp Anomaly -> GDP Growth"},
        {"cause": "climate_annual_co2", "effect": "gdp_growth", "name": "CO2 Emissions -> GDP Growth"},
    ]

    print("=" * 110)
    print("  3. DUMITRESCU-HURLIN (2012) PANEL GRANGER AUDIT UNDER CROSS-SECTIONAL DEPENDENCE (CSD)")
    print("  Comparing: (A) Exact Finite-T DH | (B) CSD Panel Bootstrap (B=1000) | (C) CS-Augmented DH (Common Factor)")
    print("=" * 110)

    results = []
    p_values_dh = []
    p_values_boot = []
    p_values_cs = []

    for hyp in hypotheses:
        cause = hyp["cause"]
        effect = hyp["effect"]
        name = hyp["name"]

        c_order = cips_order_map.get(cause, "I(0)")
        e_order = cips_order_map.get(effect, "I(0)")

        # (A) Standard exact finite-T DH test
        dh_res = dumitrescu_hurlin_test(
            df,
            cause_col=cause,
            effect_col=effect,
            max_lag=2,
            min_obs_per_country=20,
            cause_order=c_order,
            effect_order=e_order,
        )

        # (B) CSD Panel Bootstrap test (B=1000)
        print(f"  [BOOTSTRAP] Running B=1000 CSD vector-resampling bootstrap for {name}...")
        boot_res = dumitrescu_hurlin_bootstrap_test(
            df,
            cause_col=cause,
            effect_col=effect,
            max_lag=2,
            min_obs_per_country=20,
            n_boot=1000,
            cause_order=c_order,
            effect_order=e_order,
            random_state=42,
        )

        # (C) Cross-Sectionally Augmented Granger test (Chudik & Pesaran 2016)
        cs_res = cross_sectionally_augmented_granger_test(
            df,
            cause_col=cause,
            effect_col=effect,
            max_lag=2,
            min_obs_per_country=25,
            cause_order=c_order,
            effect_order=e_order,
        )

        # Pesaran CD diagnostic: compute residuals from panel regression for this channel
        cd_stat, cd_pval = 0.0, 1.0
        try:
            resid_records = []
            for c in df["iso3"].unique():
                c_data = df[df["iso3"] == c].sort_values("year")[[cause, effect, "year"]].dropna()
                if len(c_data) < 20:
                    continue
                y = c_data[effect].values.astype(np.float64)
                x = c_data[cause].values.astype(np.float64)
                if c_order == "I(1)":
                    x = np.diff(x)
                    y = y[1:]
                X_mat = np.column_stack([np.ones(len(x)), x])
                try:
                    beta, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
                    resid = y - X_mat @ beta
                    years_arr = c_data["year"].values
                    if c_order == "I(1)":
                        years_arr = years_arr[1:]
                    for j, yr in enumerate(years_arr):
                        resid_records.append({"iso3": c, "year": int(yr), "residual": float(resid[j])})
                except Exception:
                    continue
            if len(resid_records) > 100:
                resid_df = pd.DataFrame(resid_records)
                cd_stat, cd_pval = pesaran_cd_test(resid_df, time_col="year", unit_col="iso3", res_col="residual")
        except Exception as e:
            print(f"    [CD Warning] {e}")

        results.append({
            "Hypothesis": name,
            "Cause_Variable": cause,
            "Effect_Variable": effect,
            "Integration_Order": f"{c_order} -> {e_order}",
            "N_Countries": dh_res["n_countries"],
            "W_bar_DH": dh_res["w_bar"],
            "Z_tilde_Fixed_T": dh_res["z_tilde"],
            "P_Value_DH_Raw": dh_res["p_value"],
            "Boot_Mean_Z": boot_res["boot_mean_z"],
            "Boot_SD_Z": boot_res["boot_sd_z"],
            "Boot_CV_95": boot_res["cv_05_boot"],
            "P_Value_Boot_CSD": boot_res["p_value_boot"],
            "W_bar_CS": cs_res["w_bar_cs"],
            "Z_tilde_CS": cs_res["z_tilde_cs"],
            "P_Value_CS": cs_res["p_value_cs"],
            "Pesaran_CD_Stat": round(cd_stat, 3),
            "Pesaran_CD_PVal": float(cd_pval),
        })
        p_values_dh.append(dh_res["p_value"])
        p_values_boot.append(boot_res["p_value_boot"])
        p_values_cs.append(cs_res["p_value_cs"])

    # 4. Holm-Bonferroni FWER Stepdown Adjustments across families (m=7)
    holm_dh = holm_bonferroni_correction(p_values_dh, alpha=0.05)
    holm_boot = holm_bonferroni_correction(p_values_boot, alpha=0.05)
    holm_cs = holm_bonferroni_correction(p_values_cs, alpha=0.05)

    for i in range(len(results)):
        results[i]["P_Value_DH_Holm"] = holm_dh[i]["p_value_adj"]
        results[i]["Significant_DH_05"] = holm_dh[i]["significant"]
        results[i]["P_Value_Boot_Holm"] = holm_boot[i]["p_value_adj"]
        results[i]["Significant_Boot_05"] = holm_boot[i]["significant"]
        results[i]["P_Value_CS_Holm"] = holm_cs[i]["p_value_adj"]
        results[i]["Significant_CS_05"] = holm_cs[i]["significant"]

    res_df = pd.DataFrame(results)
    out_csv = out_dir / "real_dumitrescu_hurlin_results.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"\n[SSoT] DH Panel Granger results saved to: {out_csv}")

    # Display clean summary
    print("\n" + "=" * 130)
    print(f"{'Hypothesis / Channel':<38} | {'Z_tilde (DH)':<12} | {'p (Boot CSD)':<12} | {'Z_CS (Common)':<14} | {'p_CS (Holm)':<12} | {'Pesaran CD':<10} | Verdict")
    print("-" * 130)
    for _, r in res_df.iterrows():
        p_boot_str = f"{r['P_Value_Boot_CSD']:.4f}" if r['P_Value_Boot_CSD'] >= 1e-4 else "< 10^-4"
        p_cs_str = f"{r['P_Value_CS_Holm']:.4e}" if r['P_Value_CS_Holm'] < 1e-3 else f"{r['P_Value_CS_Holm']:.4f}"
        verdict = "REJECT (Robust Precedence) ***" if (r["Significant_Boot_05"] and r["Significant_CS_05"]) else (
            "ATTENUATED (Common Factor Confounded)" if not r["Significant_CS_05"] else "REJECT (Bootstrap Only)"
        )
        print(f"{r['Hypothesis']:<38} | {r['Z_tilde_Fixed_T']:<12.3f} | {p_boot_str:<12} | {r['Z_tilde_CS']:<14.3f} | {p_cs_str:<12} | {r['Pesaran_CD_Stat']:<10.2f} | {verdict}")
    print("=" * 130)


if __name__ == "__main__":
    run_real_granger_audit()
