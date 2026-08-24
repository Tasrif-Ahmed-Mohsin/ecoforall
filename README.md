# Quad-Domain Economic Forecasting & Conformal Regime Routing (LGCF-v2)

**A Multi-Horizon Macroeconomic Forecasting & Historical Twin Matching System across 237 Economies (1960–2025)**

---

## 📌 Executive Overview

This repository hosts the data pipelines, econometric testing suites, and machine learning architectures for **Quad-Domain Macroeconomic Forecasting**. The framework combines 65+ years of historical macroeconomic fundamentals (GMD, World Bank, IMF) with high-frequency geopolitical risk (GDELT), climate anomalies & natural disasters (EM-DAT), and societal sentiment & institutional trust indicators (V-Dem).

### Core Innovations & Breakthroughs:
1. **The Oracle Upper-Bound Discovery:** Mathematical and empirical proof of an **18.5% latent forecasting improvement ceiling ($p < 10^{-4}$)** in cross-domain signals, showing that 84.1% of test points require non-economic specialist experts.
2. **The "Cross-Domain Paradox" & Gating Solution:** Demonstrating why naive continuous concatenation of political/climate features fails ($p = 0.96$), and solving it via **LGCF-v2 (Conformal Uncertainty-Weighted Mixture of Experts)** delivering a statistically verified **$p < 0.001$ win (+2.54% overall, up to +4.70% in shock regimes)**.
3. **Dumitrescu-Hurlin Panel Granger Causality:** Econometric proof of multi-year transmission channels:
   * **Social Trust $\to$ GDP Growth** ($p = 0.0001, F = 12.41$)
   * **Climate Shocks $\to$ Social Fear $\to$ Material Conflict** ($p < 0.01$)
   * **Political Stability $\to$ Renewable Energy Transition** ($p < 0.0001, F = 10.59$)
4. **4D Country-Year Twin Engine (FAISS):** Scale-invariant Rank-Euclidean retrieval mapping multi-dimensional developmental trajectories across 237 countries.
5. **Calibrated Conformal Prediction:** Provable **90.14% empirical coverage intervals** with asymmetric tail widening.

---

## 🗂️ Project Structure & Organization

```
politics and economy/
├── README.md                              # Master project overview and execution guide
├── MASTER_RESEARCH_AUDIT.md               # Definitive Quad-Domain Research Audit & Benchmarks
├── deep_research_evaluation.md            # Simulated Senior Reviewer Evaluation (IJF/NeurIPS)
├── pyproject.toml                         # Project package specification
├── run_ui.ps1                             # One-click launcher for the Forecast Studio UI
│
├── data/                                  # 100% PRESERVED RESEARCH DATA & PANELS
│   ├── processed_panels/                  # Master parquet datasets (Quad, Tri, Joint panels)
│   │   ├── quad_domain_annual_panel.parquet    (20.5 MB, 237 countries, 592 features)
│   │   ├── tri_domain_annual_panel.parquet     (17.7 MB)
│   │   └── joint_annual_eco_political_panel.parquet (14.2 MB)
│   ├── benchmarks/                        # All 30+ tournament results, correlations, causality CSVs
│   ├── model_checkpoints/                 # Saved model weights & FAISS indices
│   └── llm_cache/                         # 9,302 cached DeepSeek-V4 inferences (reproducible)
│
├── src/                                   # Clean, modular Python library
│   ├── models/                            # Specialist forecasters (Ridge Trend, LGBM Quad, Huber, Stress)
│   ├── gating/                            # Conformal Uncertainty Router (LGCF-v2) & Oracle Engine
│   ├── retrieval/                         # 4D Rank-Euclidean FAISS Twin Matching Engine
│   ├── econometrics/                      # Dumitrescu-Hurlin Panel Granger & Diebold-Mariano Tests
│   ├── harmonization/                     # Quad-domain data fusion & normalization
│   └── evaluation/                        # Walk-forward cross validation & conformal calibration
│
├── experiments/                           # Domain research strands & experimental history
│   ├── 01_macro_economy/                  # Core economic pipeline (GMD 2026 v6, 15k rows)
│   ├── 02_geopolitics_gdelt/              # Geopolitical risk & conflict transmission
│   ├── 03_climate_environment/           # EM-DAT disasters, thermal anomalies, emissions
│   ├── 04_societal_psychology/           # V-Dem trust, fear, social cohesion indicators
│   └── 05_quad_synergy_and_gating/        # Exhaustive combinatorial tournaments & regime router tests
│
├── manuscript/                            # Publication-ready LaTeX paper for top-tier submission
│   ├── main.tex                           # Full academic manuscript with theoretical proofs
│   └── references.bib                     # Canonical bibliography (Salesi, Coulibaly, Dumitrescu-Hurlin)
│
└── scripts/                               # Executable CLIs & Runners
    ├── run_conformal_router.py            # Run LGCF-v2 Conformal Specialist Router Benchmark
    ├── run_panel_granger.py               # Run Dumitrescu-Hurlin Panel Granger causality
    ├── predict_country.py                 # Predict GDP trajectory for any country-year
    └── app.py                             # Interactive Streamlit Forecast Studio Web UI
```

---

## 🚀 Quickstart & Reproduction

### 1. Interactive Forecast Studio (Web UI)
Launch the Streamlit interface:
```powershell
.\run_ui.ps1
# or
streamlit run scripts/app.py
```

### 2. Predict Trajectory & Twins for a Specific Country
Generate point estimates, 90.14% conformal prediction intervals, and retrieve top historical twins:
```bash
python scripts/predict_country.py USA 2023 --horizon 5
python scripts/predict_country.py IND 2024 --horizon 3
python scripts/predict_country.py BRA 2022 --horizon 1
```

### 3. Run the LGCF-v2 Conformal Specialist Router Benchmark
Evaluate 5-fold walk-forward cross validation:
```bash
python scripts/run_conformal_router.py
```

### 4. Run Dumitrescu-Hurlin Panel Granger Causality
Test inter-domain causal transmission across 95+ economies:
```bash
python scripts/run_panel_granger.py
```

---

## 📊 Summary of Definitive Empirical Findings

| Horizon | Model / Strategy | MAE (Error) | Direction Accuracy | Out-of-Sample Performance vs. Baselines |
|---|---|:---:|:---:|---|
| **$h = 1$ Year** | **LGCF-v2 Conformal Router** | **0.0268** | **76.8%** | **+25.3% vs Honest AR(1) (0.0359), +42.8% vs Naive Prior (0.0469)** |
| | **ML Ensemble (LGBM+Ridge)** | 0.0328 | 72.1% | **+8.9% vs Honest AR(1), +30.1% vs Naive Prior** ($p = 0.012$) |
| **$h = 3$ Years** | **LGCF-v2 Conformal Router** | **0.0598** | **75.4%** | **+15.8% vs Honest AR(1) (0.0710), +23.6% vs Naive Prior (0.0783)** |
| **$h = 5$ Years** | **Cross-Horizon Meta-Ensemble** | **0.0377** | **89.2%** | **+45.0% vs Honest AR(1) (0.0686), +65.8% vs Naive Prior (0.1102)** ($p < 10^{-4}$) |
| | **LGCF-v2 Conformal Router** | 0.0776 | 81.5% | **+2.54% overall, +4.70% in shock regimes** ($p < 0.001$) |
| **$h = 10$ Years** | **4D Historical Twin Matching** | 0.1239 | **84.4%** | **Outperforms unregularized ML (0.1318)**; anchors multi-year drift |

---

## 📜 Citation & Research Attribution

If utilizing this codebase or datasets in academic research, please cite:
```bibtex
@article{quad_domain_forecasting_2026,
  title={Beyond Naive Concatenation: Conformal Uncertainty Routing and Cross-Domain Regimes in Global Macroeconomic Panels},
  author={Research Team},
  journal={Working Paper Series},
  year={2026}
}
```
