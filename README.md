# Multi-Domain Macroeconomic Forecasting & State-Space Dynamic Model Selection

**A Rigorous Empirical Framework across 237 Economies (1960–2024; $N = 15,071$ country-years)**

---

## 📌 Executive Overview

This repository contains the verified data pipelines, econometric testing suites, and dynamic model selection architectures for **Multi-Horizon Sovereign Macroeconomic Panel Forecasting**. The framework combines 65 years of historical macroeconomic fundamentals (Global Macro Database v6, World Bank, IMF) with democratic institutional governance indices (Varieties of Democracy, V-Dem v14) and biophysical climate stress metrics (Copernicus ERA5 surface temperature anomalies, Global Carbon Budget CO2 emissions).

### Core Findings:

1. **The Cross-Domain Macroeconomic Forecasting Paradox**:
   - While V-Dem democratic institutions and rule of law Granger-cause real GDP growth ($\tilde{Z} = 14.23 \text{ to } 17.81, p < 10^{-4}$ under Holm-Bonferroni FWER control), **static uniform concatenation of all 241 features degrades out-of-sample forecast accuracy** relative to pure economy models across all horizons ($h \in \{1, 3, 5\}$).
   - Year-clustered Clark-West (2007) nested model tests confirm that the augmented model fails to significantly improve over the parsimonious economic baseline at any horizon.
   - This validates the finite-sample information dilution theory: non-economic features inflate parameter variance $\mathcal{O}(d_2/N)$ during tranquil regimes ($\approx 75\%$ of sovereign history).

2. **State-Space Dynamic Model Averaging (Koop-Korobilis DMA)**:
   - Online recursive Bayesian probability discounting ($\lambda = 0.92$) with a country-level residual variance across domain-quarantined functional specialists achieves the lowest error across all horizons, outperforming fitted per-country AR(1) baselines by **+9.71% at $h=1$, +13.75% at $h=3$, and +17.43% at $h=5$** ($p < 10^{-4}$ via year-clustered Diebold-Mariano testing).
   - **Methodological Scope**: Operates as an online convex combination ($\hat{y}_{t}^{\text{DMA}} = \sum_m \pi_{t|t-1, m} \hat{y}_t^{(m)}$) over fixed domain specialist models with shared variance $\sigma_{i,t}^2$, rather than estimating internal TVP Kalman filters within the base forecasters.
   - **Honest DMA-vs-EW & Specialist margins**: Gains over simple equal-weight averaging are modest: **+2.02% at $h=1$ ($p = 0.0020$), +1.62% at $h=3$ ($p = 0.0682$, insignificant at $\alpha=0.05$), and +2.28% at $h=5$ ($p = 0.0053$)**. At $h=5$, DMA does not statistically separate from the strongest single specialist (Economy LightGBM, $p = 0.2010$). Most cross-domain combination gains are captured by simple averaging.
   - With feedback disabled, DMA algebraically coincides with the 1/M average (`|DMA_no_feedback - Equal_Weight| = 0.00e+00`).


3. **Econometric Integrity & Stationarity Pre-Testing**:
   - Mandatory CIPS (Pesaran 2007) second-generation panel unit root pre-testing first-differences $I(1)$ series before Granger causality analysis.
   - Pervasive cross-sectional dependence ($\hat{CD} > 85$) is resolved via (1) vector-resampling CSD panel bootstrap ($B=1,000$) and (2) Chudik–Pesaran (2016) CS-DH common factor filtering: 5 political governance channels and $\text{CO}_2$ retain robust predictive precedence ($p < 10^{-5}$), while ERA5 temperature anomalies attenuate ($p = 0.0503$).


---

## 📊 Summary of Verified Empirical Benchmarks

### 1. Multi-Horizon Walk-Forward Tournament (5 Folds, 1960–2024)
*Source: `data/benchmarks/real_cross_domain_benchmark_results.csv`*

| Horizon ($h$) | Model Architecture | MAE | RMSE | Lift vs. AR(1) | Lift vs. Eco-Ridge | DM (Year-Clustered) | $p$-val |
|---|---|---|---|---|---|---|---|
| **$h = 1$ Year**<br>($N = 5,211$) | **DMS State-Space Router** | **0.03255** | **0.05753** | **+9.71%** | **+4.33%** | **5.786** | **$p < 10^{-4}$** |
| | Equal-Weight Multi-Domain (1/M) | 0.03322 | 0.05823 | +7.84% | +2.35% | 4.297 | $p = 0.0003$ |
| | Economy-Only LightGBM | 0.03386 | 0.05865 | +6.07% | +0.48% | 2.636 | $p = 0.0148$ |
| | All-Domain LightGBM (Concat) | 0.03379 | 0.05856 | +6.26% | +0.68% | 2.776 | $p = 0.0107$ |
| | **Economy-Only Ridge (206 feats)** | **0.03402** | **0.05866** | **+5.62%** | *Baseline* | 2.831 | **$p = 0.0095$** |
| | *All-Domain Ridge (Concat, 241 feats)* | *0.03461* | *0.05915* | *+4.00%* | *−1.71% (Degrades)* | 2.051 | *$p = 0.0519$* |
| | Stock-Watson DFM (5 Factors) | 0.03515 | 0.05944 | +2.50% | −3.31% | 0.937 | $p = 0.3583$ |
| | **AR(1) Baseline (growth_into_origin)** | **0.03605** | **0.06302** | *Baseline* | *−5.96%* | 0.000 | $p = 1.0000$ |
| **$h = 3$ Years**<br>($N = 4,773$) | **DMS State-Space Router** | **0.07715** | **0.12156** | **+13.75%** | **+6.66%** | **10.097** | **$p < 10^{-4}$** |
| | Equal-Weight Multi-Domain (1/M) | 0.07842 | 0.12282 | +12.33% | +5.11% | 8.886 | $p < 10^{-4}$ |
| | Economy-Only LightGBM | 0.07909 | 0.12481 | +11.58% | +4.31% | 7.601 | $p < 10^{-4}$ |
| | All-Domain LightGBM (Concat) | 0.07950 | 0.12510 | +11.13% | +3.81% | 6.978 | $p < 10^{-4}$ |
| | **Economy-Only Ridge** | **0.08265** | **0.12643** | **+7.60%** | *Baseline* | 3.853 | **$p = 0.0009$** |
| | *All-Domain Ridge (Concat)* | *0.08457* | *0.12826* | *+5.46%* | *−2.32% (Degrades)* | 2.973 | *$p = 0.0073$* |
| | **AR(1) Baseline (growth_into_origin)** | **0.08945** | **0.13808** | *Baseline* | *−8.23%* | 0.000 | $p = 1.0000$ |
| **$h = 5$ Years**<br>($N = 4,335$) | **DMS State-Space Router** | **0.11589** | **0.18141** | **+17.43%** | **+9.85%** | **8.970** | **$p < 10^{-4}$** |
| | Economy-Only LightGBM | 0.11723 | 0.18449 | +16.48% | +8.81% | 7.855 | $p < 10^{-4}$ |
| | Equal-Weight Multi-Domain (1/M) | 0.11860 | 0.18510 | +15.50% | +7.74% | 7.876 | $p < 10^{-4}$ |
| | All-Domain LightGBM (Concat) | 0.11900 | 0.18661 | +15.21% | +7.43% | 8.063 | $p < 10^{-4}$ |
| | **Economy-Only Ridge** | **0.12855** | **0.19426** | **+8.41%** | *Baseline* | 3.397 | **$p = 0.0030$** |
| | *All-Domain Ridge (Concat)* | *0.13109* | *0.19698* | *+6.60%* | *−1.97% (Degrades)* | 2.732 | *$p = 0.0132$* |
| | **AR(1) Baseline (growth_into_origin)** | **0.14035** | **0.21525** | *Baseline* | *−9.18%* | 0.000 | $p = 1.0000$ |

---

### 2. Panel Granger Non-Causality Tests under Cross-Sectional Dependence (CSD)
*Source: `data/benchmarks/real_dumitrescu_hurlin_results.csv` ($K=2$ lags, CIPS-governed differencing, CSD Bootstrap $B=1,000$, CS-DH Common Factor Filtering, Holm FWER $m=7$)*

| Transmission Channel | Countries ($N$) | Fixed-$T$ $\tilde{Z}$ | Boot 95% CV | Boot $p_{\text{CSD}}$ | CS-DH $Z_{\text{CS}}$ | CS-DH $p_{\text{Holm}}$ | Pesaran $\hat{CD}$ | Empirical Verdict |
|---|---|---|---|---|---|---|---|---|
| **V-Dem Free Expression $\to$ GDP Growth** | 173 | 16.322 | 4.73 | $< 10^{-4}$ | **10.041** | $< 10^{-4}$ *** | 107.81 | **Reject Null (Robust Precedence)** |
| **V-Dem Corruption $\to$ GDP Growth** | 173 | 14.233 | 4.10 | $< 10^{-4}$ | **8.959** | $< 10^{-4}$ *** | 107.15 | **Reject Null (Robust Precedence)** |
| **V-Dem Liberal Democracy $\to$ GDP Growth** | 173 | 17.809 | 4.74 | $< 10^{-4}$ | **7.563** | $< 10^{-4}$ *** | 109.69 | **Reject Null (Robust Precedence)** |
| **V-Dem Electoral Democracy $\to$ GDP Growth** | 173 | 15.232 | 4.78 | $< 10^{-4}$ | **5.829** | $< 10^{-4}$ *** | 108.36 | **Reject Null (Robust Precedence)** |
| **V-Dem Rule of Law $\to$ GDP Growth** | 173 | 14.530 | 3.75 | $< 10^{-4}$ | **5.378** | $< 10^{-4}$ *** | 108.66 | **Reject Null (Robust Precedence)** |
| **CO2 Emissions $\to$ GDP Growth** | 205 | 6.697 | 2.74 | $< 10^{-4}$ | **4.776** | $< 10^{-4}$ *** | 87.78 | **Reject Null (Robust Precedence)** |
| **ERA5 Temp Anomaly $\to$ GDP Growth** | 187 | 5.630 | 4.05 | $0.0130$ | 1.958 | $0.0503$ | 110.56 | **Attenuated (Common Factor Confounded)** |

> **CSD Methodological Audit**: Standard Dumitrescu–Hurlin assumes cross-sectional independence ($\text{Cov}(W_i, W_j)=0$). With Pesaran $\hat{CD} > 85$, standard $\tilde{Z}$ is inflated. Correcting for CSD via (1) vector-resampling bootstrap ($B=1,000$) preserving $\boldsymbol{\Sigma}_N$ and (2) Chudik–Pesaran (2016) CS-DH common factor filtering shows that all five institutional governance channels and CO$_2$ emissions retain robust predictive precedence ($p < 10^{-5}$), while surface temperature anomalies attenuate to $p=0.0503$, confirming that temperature causality was confounded by global common shocks.


---

### 3. Pedroni (1999, 2004) Panel Cointegration Diagnostics
*Source: `data/benchmarks/real_cointegration_results.csv`*

- **Log GDP pc $\sim$ V-Dem Rule of Law**: Group $Z_{\text{P-ADF}} = 4.212, p = 1.0000$ -> **Fail to Reject $H_0$ (No Cointegration)**.
- **Log GDP pc $\sim$ Annual CO2 Emissions**: Group $Z_{\text{P-ADF}} = 3.344, p = 0.9996$ -> **Fail to Reject $H_0$ (No Cointegration)**.
- *Econometric Implication*: Because no cointegration exists, first-differencing non-stationary indicators without an error-correction term is econometrically well-specified.

---

### 4. Hansen, Lunde & Nason (2011) Model Confidence Set ($\widehat{\mathcal{M}}_{90\%}$)
*Source: `data/benchmarks/real_model_confidence_set_results.csv` ($B=1,000$ Block Bootstrap, block size $= \max(2, h)$)*

- **$h=1$**: DMS State-Space Router is the sole model in $\widehat{\mathcal{M}}_{90\%}$ ($p_{\text{MCS}} = 1.000$; all other models $p \le 0.029$).
- **$h=3$**: Four architectures belong to $\widehat{\mathcal{M}}_{90\%}$: DMS ($p=1.000$), Equal-Weight Multi-Domain ($p=0.113$), Economy LightGBM ($p=0.113$), and All-Domain LightGBM ($p=0.113$).
- **$h=5$**: $\widehat{\mathcal{M}}_{90\%}$ narrows to DMS ($p=1.000$) and Economy LightGBM ($p=0.243$). Equal-Weight ($p=0.048$), All-Domain LightGBM ($p=0.023$), and all linear models ($p \le 0.002$) are eliminated.


---

### 5. Hyperparameter Robustness Grid ($\lambda \in [0.85, 1.00], \alpha \in [10, 200]$)
*Source: `data/benchmarks/real_robustness_*.csv` (Evaluated on exact 5-Fold 2000–2024 Walk-Forward CV)*

- Dynamic model adaptation ($\lambda \in [0.85, 0.92]$) consistently outperforms static recursive Bayesian Model Averaging ($\lambda = 1.00$) across medium and long horizons ($0.07704$ vs. $0.07732$ at $h=3$; $0.11572$ vs. $0.11613$ at $h=5$).
- The information dilution penalty of all-domain concatenation relative to domain-specialized Ridge persists across all tested regularizations $\alpha \in [10, 200]$ (+1.60% to +2.67%, with mean horizon penalty +1.81% to +2.28%).
- At headline $\alpha = 100.0$, the grid replicates the exact tournament MAEs (Economy Ridge: 0.03402, 0.08265, 0.12855; Concat Ridge: 0.03461, 0.08457, 0.13109).


---

## 🗂️ Reproduction & Verification Guide

### 0. Build Harmonized Cross-Domain Panel from Raw Data:
Reconstructs the complete 15,071 row $\times$ 246 column panel from genuine raw files in `data/raw/` (`gmd_macro_panel_raw.csv.gz`, V-Dem, ERA5, Carbon Budget):
```bash
python src/harmonization/ingest_real_cross_domain_panel.py
```
*Expected Output Parquet:* `data/processed_panels/real_cross_domain_annual_panel.parquet`  
*Verified SHA-256:* `4d6706d03724469b8fd5a8ca171057617e74643a9bab4b6d6795ae9aaf95135a`

### 1. Forensic Data Provenance & Cryptographic Verification Audit:
Verifies genuine external provenance, timestamps, line counts, and SHA-256 hashes across all raw inputs, processed parquet, and benchmark CSVs:
```bash
python scripts/verify_raw_data_provenance.py
```

### 2. Execute Full Forecasting Tournament (Pseudo-Real-Time Walk-Forward):
```bash
python scripts/run_real_multidomain_benchmark.py
```


### 3. Execute CIPS Panel Unit Root, Cointegration & Granger Causality Audit:
```bash
python scripts/run_real_panel_granger_audit.py
```

### 4. Execute Model Confidence Set Audit:
```bash
python scripts/run_model_confidence_set_audit.py
```

### 5. Execute Regime Breakdown Audit:
```bash
python scripts/run_regime_breakdown_audit.py
```

### 6. Execute Hyperparameter Robustness Grid Audit:
```bash
python scripts/run_robustness_grid_audit.py
```

### 7. Auto-Generate LaTeX Manuscript Tables (SSoT):
Reads benchmark CSV artifacts directly from `data/benchmarks/` and programmatically updates LaTeX tables in `manuscript/tables/`:
```bash
python scripts/generate_manuscript_tables.py
```

### 8. Run Verification & Test Suite:
```bash
python scripts/_audit_verify_baseline_and_leakage.py
python -m pytest tests/ -v
```
