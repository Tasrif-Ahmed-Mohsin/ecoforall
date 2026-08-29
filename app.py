"""
Multi-Domain Macroeconomic Forecasting & Econometric Intelligence Console
========================================================================
Interactive Streamlit Dashboard & Forecasting Interface for:
  - 237 Sovereign Economies (1960–2024; N = 15,071 country-years)
  - Real GMD v6 Macro + V-Dem v14 Democracy + Copernicus ERA5 Climate Panels
  - Real-Time Bayesian State-Space Dynamic Model Selection (DMS)
  - Pesaran (2007) CIPS Panel Unit Root Diagnostics
  - Hansen, Lunde & Nason (2011) Model Confidence Set (MCS)
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.reproducibility import seed_everything
from src.gating.sovereign_segmentation_router import get_wb_region

st.set_page_config(
    page_title="Macroeconomic Panel Forecasting & Econometric Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_benchmark_data():
    bench_dir = ROOT / "data" / "benchmarks"
    
    tourn_path = bench_dir / "real_cross_domain_benchmark_results.csv"
    dh_path = bench_dir / "real_dumitrescu_hurlin_results.csv"
    cips_path = bench_dir / "real_cips_unit_root_results.csv"
    mcs_path = bench_dir / "real_model_confidence_set_results.csv"
    reg_path = bench_dir / "real_regime_breakdown_results.csv"

    tourn_df = pd.read_csv(tourn_path) if tourn_path.exists() else None
    dh_df = pd.read_csv(dh_path) if dh_path.exists() else None
    cips_df = pd.read_csv(cips_path) if cips_path.exists() else None
    mcs_df = pd.read_csv(mcs_path) if mcs_path.exists() else None
    reg_df = pd.read_csv(reg_path) if reg_path.exists() else None

    return tourn_df, dh_df, cips_df, mcs_df, reg_df


@st.cache_data
def load_country_metadata():
    panel_path = ROOT / "data" / "processed_panels" / "real_cross_domain_annual_panel.parquet"
    if panel_path.exists():
        df = pd.read_parquet(panel_path, columns=["iso3", "country", "year", "region", "income_level"])
        countries = df[["iso3", "country", "region", "income_level"]].drop_duplicates().sort_values("iso3")
        return countries
    return pd.DataFrame({"iso3": ["USA", "DEU", "GBR", "IND", "BRA", "ZAF", "JPN", "CHN"]})


def main():
    st.title("🌐 Multi-Domain Sovereign Growth Forecasting & Econometric Audit")
    st.markdown(
        "**Verified 65-Year Panel (1960–2024; $N=15,071$ country-years, 237 economies)**  \n"
        "*Integrating Global Macro Database (GMD v6), Varieties of Democracy (V-Dem v14), and Copernicus ERA5.*"
    )

    tourn_df, dh_df, cips_df, mcs_df, reg_df = load_benchmark_data()
    countries_df = load_country_metadata()

    tabs = st.tabs([
        "🏆 Forecasting Tournament",
        "🎯 Model Confidence Set",
        "⚖️ Regime Breakdown & Dilution",
        "🔬 CIPS, Cointegration & Causality",
        "📊 Hyperparameter Robustness",
        "🌍 Sovereign Explorer",
    ])

    with tabs[0]:
        st.header("Multi-Horizon Walk-Forward Tournament (5 Folds, 1960–2024)")
        st.markdown(
            r"Out-of-sample evaluation comparing single-domain specialists, static concatenation, "
            r"and dynamic model selection across $h \in \{1, 3, 5\}$ years under strict zero-leakage quarantine."
        )

        if tourn_df is not None:
            horizon_sel = st.selectbox("Select Forecast Horizon (h):", [1, 3, 5], format_func=lambda x: f"{x} Year{'s' if x > 1 else ''} Ahead")
            h_sub = tourn_df[tourn_df["Horizon"] == horizon_sel].sort_values("MAE")

            cols_to_show = ["Model", "MAE", "RMSE", "Lift_vs_AR1_pct", "Lift_vs_EcoRidge_pct", "DM_stat_yearclustered", "DM_pval_yearclustered"]
            renamed = {
                "Lift_vs_AR1_pct": "Lift vs AR(1) (%)",
                "Lift_vs_EcoRidge_pct": "Lift vs Eco-Ridge (%)",
                "DM_stat_yearclustered": "Year-Clustered DM",
                "DM_pval_yearclustered": "p-value",
            }
            display_df = h_sub[cols_to_show].rename(columns=renamed).reset_index(drop=True)
            st.dataframe(display_df, use_container_width=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                dms_row = h_sub[h_sub["Model"] == "DMS State-Space Router"].iloc[0]
                st.metric("DMS State-Space Router MAE", f"{dms_row['MAE']:.5f}", f"{dms_row['Lift_vs_AR1_pct']:+.2f}% vs AR(1)")
            with col2:
                eco_row = h_sub[h_sub["Model"] == "Economy-Only Ridge"].iloc[0]
                all_row = h_sub[h_sub["Model"] == "All-Domain Ridge (Concat)"].iloc[0]
                diff_pct = ((all_row['MAE'] - eco_row['MAE']) / eco_row['MAE']) * 100.0
                st.metric("Static Concat Dilution Penalty", f"{diff_pct:+.2f}%", "All-Domain vs Eco-Only Ridge", delta_color="inverse")
            with col3:
                eq_row = h_sub[h_sub["Model"] == "Equal-Weight Multi-Domain"].iloc[0]
                dms_vs_eq = ((eq_row['MAE'] - dms_row['MAE']) / eq_row['MAE']) * 100.0
                st.metric("Net Dynamic Filter Contribution", f"{dms_vs_eq:+.2f}%", "DMS vs Equal-Weight (1/M)")
        else:
            st.warning("Tournament results artifact not found.")

    with tabs[1]:
        st.header(r"Hansen, Lunde & Nason (2011) Model Confidence Set ($\widehat{\mathcal{M}}_{90\%}$)")
        st.markdown(
            r"Iterative Model Confidence Set procedure evaluated using moving-block bootstrap ($B=1,000$ replications) "
            r"on year-clustered loss differentials. Models with $p_{\text{MCS}} \ge 0.10$ belong to the superior set $\widehat{\mathcal{M}}_{90\%}$."
        )

        if mcs_df is not None:
            piv_mcs = mcs_df.pivot(index="Model", columns="Horizon", values=["MCS_P_Value", "In_MCS_90pct"])
            st.dataframe(mcs_df, use_container_width=True)
        else:
            st.warning("Model Confidence Set artifact not found.")

    with tabs[2]:
        st.header("Empirical Regime Breakdown: Information Dilution vs Crisis Utility")
        st.markdown(
            r"Tests **Proposition 1**: extraneous cross-domain features introduce parameter estimation variance "
            r"$\mathcal{O}(d_2/N)$ during tranquil regimes while providing crisis resilience during institutional and financial shocks."
        )

        if reg_df is not None:
            h_sel = st.selectbox("Select Horizon for Regime Breakdown:", [1, 3, 5], key="reg_h")
            r_sub = reg_df[reg_df["Horizon"] == h_sel].reset_index(drop=True)
            st.dataframe(r_sub, use_container_width=True)
        else:
            st.warning("Regime breakdown artifact not found.")

    with tabs[3]:
        st.header("Econometric Identification: CIPS Unit Root, Panel Cointegration & Granger Causality")

        st.subheader("1. Pesaran (2007) CIPS Panel Unit Root Diagnostics (Under Cross-Sectional Dependence)")
        if cips_df is not None:
            st.dataframe(cips_df, use_container_width=True)
        else:
            st.warning("CIPS artifact not found.")

        st.subheader("2. Pedroni (1999, 2004) Residual-Based Panel Cointegration Diagnostics")
        coint_path = ROOT / "data" / "benchmarks" / "real_cointegration_results.csv"
        if coint_path.exists():
            coint_df = pd.read_csv(coint_path)
            st.dataframe(coint_df, use_container_width=True)

        st.subheader("3. Dumitrescu-Hurlin (2012) Heterogeneous Panel Granger Non-Causality Tests")
        if dh_df is not None:
            st.dataframe(dh_df, use_container_width=True)
        else:
            st.warning("Dumitrescu-Hurlin artifact not found.")

    with tabs[4]:
        st.header("Hyperparameter Sensitivity & Robustness Grid")
        lam_path = ROOT / "data" / "benchmarks" / "real_robustness_lambda_results.csv"
        alp_path = ROOT / "data" / "benchmarks" / "real_robustness_alpha_results.csv"

        if lam_path.exists() and alp_path.exists():
            st.subheader("Panel A: Dynamic Model Selection Forgetting Factor (lambda) Sensitivity")
            st.dataframe(pd.read_csv(lam_path), use_container_width=True)

            st.subheader("Panel B: Ridge Regularization (alpha) & Information Dilution Penalty")
            st.dataframe(pd.read_csv(alp_path), use_container_width=True)
        else:
            st.warning("Robustness grid artifacts not found.")

    with tabs[5]:
        st.header("Sovereign Country Explorer")
        if not countries_df.empty:
            iso_list = countries_df["iso3"].tolist()
            selected_iso = st.selectbox("Select Sovereign Nation:", iso_list, format_func=lambda x: f"{x} - {countries_df[countries_df['iso3'] == x]['country'].values[0] if 'country' in countries_df.columns else x}")
            
            c_meta = countries_df[countries_df["iso3"] == selected_iso].iloc[0]
            st.json({
                "ISO3": selected_iso,
                "Country": str(c_meta.get("country", selected_iso)),
                "Region": str(c_meta.get("region", get_wb_region(selected_iso))),
                "Income Level": str(c_meta.get("income_level", "Unknown")),
            })
            st.info(f"To run full multi-horizon interactive point forecasts and conformal bounds for {selected_iso}, run: `python scripts/predict_country.py {selected_iso} 2023 --horizon 5`")


if __name__ == "__main__":
    main()
