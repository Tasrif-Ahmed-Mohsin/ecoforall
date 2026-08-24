"""
Streamlit Web Application: Universal Dynamic-Horizon Environment Engine (1960-2025).

Provides an interactive dashboard for:
 1. Multi-Target Quantile Scenario Forecaster & Conformal Prediction Fan Charts across ALL 8 indicators.
 2. FAISS Rank-Euclidean Historical Trajectory Twin Finder.
 3. Model Evaluation & DM Falsification Audit Metrics.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import yaml

from cross_domain_dataset_harmonizer import EnvironmentDatasetHarmonizer
from analog_engine import EnvironmentalAnalogEngine
from forecaster import EnvironmentMultiHorizonForecaster

st.set_page_config(
    page_title="65-Year Environmental Dynamic Horizon Engine",
    page_icon="🌍",
    layout="wide"
)

@st.cache_data
def load_data_and_models():
    harmonizer = EnvironmentDatasetHarmonizer()
    df_raw = harmonizer.load_raw_data()
    df_featured = harmonizer.build_features_and_targets(df_raw)
    
    df_train = df_featured[df_featured["year"] < 2018]
    df_calib = df_featured[(df_featured["year"] >= 2018) & (df_featured["year"] <= 2021)]
    
    forecaster = EnvironmentMultiHorizonForecaster()
    forecaster.fit(df_train, df_calib)
    
    analog_engine = EnvironmentalAnalogEngine(df_featured)
    
    return df_featured, forecaster, analog_engine, harmonizer.config

def main():
    st.title("🌍 Universal Dynamic-Horizon Environment & Climate Engine (1960–2025)")
    st.markdown("""
    This dynamic-horizon engine combines **Rank-Euclidean FAISS vector retrieval** with **multi-head quantile ML models**
    to discover historical climate trajectory twins and project future environmental risk across **ALL 8 Environmental Indicators**.
    """)
    
    df_featured, forecaster, analog_engine, config = load_data_and_models()
    
    st.sidebar.header("🕹️ Control Panel")
    all_iso3 = sorted(df_featured["iso3"].unique())
    selected_iso3 = st.sidebar.selectbox("Select Country (ISO3)", all_iso3, index=all_iso3.index("USA") if "USA" in all_iso3 else 0)
    
    all_years = sorted(df_featured[df_featured["iso3"] == selected_iso3]["year"].unique())
    selected_year = st.sidebar.slider("Select Historical / Current Year", int(min(all_years)), int(max(all_years)), int(max(all_years) - 5))
    
    targets = config["forecasting"]["target_indicators"]
    selected_target = st.sidebar.selectbox("Select Target Indicator", targets, index=0)
    
    tab1, tab2, tab3 = st.tabs([
        "🔮 Multi-Target Horizon Forecaster",
        "👯 Historical Trajectory Analog Finder",
        "📊 Model Falsification & System Audit"
    ])
    
    # TAB 1: Quantile Horizon Forecaster
    with tab1:
        st.subheader(f"Forecast Projections for {selected_iso3} — Indicator: {selected_target} (Base Year: {selected_year})")
        
        query_row = df_featured[(df_featured["iso3"] == selected_iso3) & (df_featured["year"] == selected_year)]
        
        if len(query_row) == 0:
            st.error("No valid feature record found for selected country and year.")
        else:
            preds = forecaster.predict(query_row, target_indicator=selected_target)
            
            horizons = config["forecasting"]["horizons"]
            forecast_rows = []
            
            base_val = query_row[selected_target].values[0] if selected_target in query_row.columns else 0.0
            
            for h in horizons:
                if h not in preds:
                    continue
                res = preds[h]
                forecast_rows.append({
                    "Horizon (Years)": f"+{h} Year(s) ({selected_year + h})",
                    "Point Ensemble": round(float(res["point_ensemble"][0]), 4),
                    "Ridge Baseline": round(float(res["ridge_baseline"][0]), 4),
                    "Conformal Lower (90%)": round(float(res["conformal_lower_90"][0]), 4),
                    "Conformal Upper (90%)": round(float(res["conformal_upper_90"][0]), 4),
                    "Quantile 5%": round(float(res["quantiles"]["q_0.05"][0]), 4),
                    "Quantile 95%": round(float(res["quantiles"]["q_0.95"][0]), 4),
                })
                
            forecast_df = pd.DataFrame(forecast_rows)
            st.dataframe(forecast_df, use_container_width=True)
            
            # Line chart visualization of fan chart
            st.subheader(f"Multi-Horizon Trajectory & Conformal Bounds for '{selected_target}'")
            
            timeline = [selected_year] + [selected_year + h for h in horizons]
            point_vals = [base_val] + [r["Point Ensemble"] for r in forecast_rows]
            lower_vals = [base_val] + [r["Conformal Lower (90%)"] for r in forecast_rows]
            upper_vals = [base_val] + [r["Conformal Upper (90%)"] for r in forecast_rows]
            
            chart_df = pd.DataFrame({
                "Year": timeline,
                "Point Forecast": point_vals,
                "Lower 90% Conformal": lower_vals,
                "Upper 90% Conformal": upper_vals
            }).set_index("Year")
            
            st.line_chart(chart_df)

    # TAB 2: Analog Finder
    with tab2:
        st.subheader(f"Top Historical Trajectory Twins for {selected_iso3} ({selected_year})")
        st.markdown("Retrieves countries in past decades that shared matching environmental state vectors and percentile ranks.")
        
        try:
            analogs = analog_engine.find_analogs(selected_iso3, selected_year, top_k=6)
            st.dataframe(analogs[["analog_iso3", "analog_year", "similarity_score", "current_co2", "current_temp_anomaly"]], use_container_width=True)
            
            st.markdown("### Historical Forward Realized Trajectories of Analogs")
            analog_chart_data = {}
            for idx, row in analogs.iterrows():
                label = f"{row['analog_iso3']} ({row['analog_year']})"
                trajs = row["forward_trajectories"]
                vals = [row["current_co2"]] + [trajs.get(f"h_{h}y", np.nan) for h in config["forecasting"]["horizons"]]
                analog_chart_data[label] = vals
                
            analog_chart_df = pd.DataFrame(analog_chart_data, index=[0, 1, 3, 5, 10])
            st.line_chart(analog_chart_df)
        except Exception as e:
            st.error(f"Error retrieving analogs: {str(e)}")

    # TAB 3: Audit & Falsification
    with tab3:
        st.subheader("Model Falsification & Statistical DM Test Audit Results across ALL Targets")
        audit_file = os.path.join("data", "audit_tournament_results.csv")
        
        if os.path.exists(audit_file):
            audit_df = pd.read_csv(audit_file)
            st.dataframe(audit_df, use_container_width=True)
            
            st.success("Diebold-Mariano tests evaluate out-of-sample forecast errors using HAC variance adjustments.")
        else:
            st.info("Run `python run_master_system_audit.py` to populate tournament audit metrics.")

if __name__ == "__main__":
    main()
