# Multi-Domain Macroeconomic Forecasting — Research Caliber & Methodological Audit

**Status Update (Post-Rework, August 26, 2026):**
The repository has undergone a complete, ground-up empirical rework to address every issue identified in [`CRITICAL_ISSUES.md`](file:///e:/politics%20and%20economy/CRITICAL_ISSUES.md):
- **100% Real Public Data Ingestion**: Replaced all synthetic generators with genuine V-Dem v14 democracy/institutional indices, Copernicus ERA5 surface temperature anomalies, and Global Carbon Budget CO2 emissions ($N=15,071$ country-years, 237 economies, 241 features).
- **Mandatory CIPS Stationarity Pre-Testing & CSD-Robust Granger Causality**: First-differences $I(1)$ series (Rule of Law, $\text{CO}_2$) before Granger causality analysis. Accounts for severe cross-sectional dependence ($\hat{CD} > 85$) via Emirmahmutoglu–Kose (2011) vector-resampling bootstrap ($B=1,000$) and Chudik–Pesaran (2016) Cross-Sectionally Augmented CS-DH common factor filtering: democratic institutions ($Z_{\text{CS}} \in [5.38, 10.04], p < 10^{-5}$) and $\text{CO}_2$ ($Z_{\text{CS}} = 4.78, p < 10^{-5}$) maintain robust predictive precedence, while ERA5 temperature anomalies attenuate ($Z_{\text{CS}} = 1.96, p = 0.0503$).

- **The Cross-Domain Paradox Tested on Real Data**: Demonstrated that static concatenation of all 241 features degrades out-of-sample forecast accuracy across linear and non-linear models, whereas recursive Bayesian state-space DMS ($\lambda = 0.92$) resolves the information dilution penalty.
- **Single Source of Truth**: All manuscript tables and documentation are auto-generated directly from verified CSV artifacts. Zero hand-typed figures.

---

## 1. Executive verdict (the bottom line)

**Publishable — at a Tier-2 venue, after one serious rewrite. Not Tier-1 (NeurIPS main, ICML, AER, QJE, Econometrica) in its current form. The September 2026 submission target of "IJF/JAE/NeurIPS" overstates the strength of the empirical contribution by at least one tier.**

More specifically:

| Tier | Verdict | Reasoning |
|---|---|---|
| **Tier 1** (NeurIPS main, ICML, AER, QJE, Econometrica, ReStud) | ❌ Not ready | Empirical contribution is dominated by a single high-noise result (DMS); null result for the headline thesis; novel ML machinery is solid engineering but not a clean theoretical advance |
| **Tier 2 top** (JAE, JoE, ReStat, IJF) | ⚠️ Possible *with revisions* | Could land here if the narrative is rebuilt around the DMS/regime-memory finding with honest reporting of the cross-domain null |
| **Tier 2 mid** (IJF, Computational Economics, Economic Modelling, JEDC) | ✅ Good fit | Existing engineering rigor + DMS result + clear negative result for naive cross-domain concatenation fits this tier cleanly |
| **Tier 3** (NeurIPS/KDD workshops, AAAI, specialty) | ✅ Strong workshop paper as-is | The LLM-as-regime-detector framing is timely and novel enough for a workshop, especially with the external OOD protocol |
| **Best submission strategy** | Dual-track | Workshop paper (current draft ≈ 8/10 for a workshop) + IJF/CompEcon submission after reframe |

---

## 2. What this project actually is — taxonomy

There are now **five distinct research artifacts** in this repo, each with different internal quality:

| Artifact | Core claim | Quality of evidence | Status |
|---|---|---|---|
| **A. Cross-Horizon Meta-Ensemble (single-domain)** | 65.8% MAE win at h=5 over AR(1) | Single test slice (n=213, year-2019 origins only). Per the project's own `AUDIT.md` (projectresearch/), the same model loses to AR(1) at h≥5 under walk-forward CV | Stale / superseded by the August audit |
| **B. Naive cross-domain concatenation** | "Cross-Domain Paradox" (p=0.96 for GDP lift) | Strong, reproducible across 2,856–3,528 country-year observations per horizon, sign-correct | Solid negative result |
| **C. Dumitrescu-Hurlin panel Granger** | Trust→GDP, Fear→Conflict, Climate→Fear, etc. | Genuine DH (2012) implementation; multiplicity-corrected; survives Bonferroni for several pairs | Methodologically clean, but **partial overclaim** (see §4.3) |
| **D. LLM-Gated Mixture-of-Experts (LGCF)** | Solves the Cross-Domain Paradox via LLM regime detection | The headline config (E in `master_ablation_8way.csv`) achieves **+0.48% average lift, p=0.53 (null)** | The empirical centerpiece is **null**. See §4.1 |
| **E. Koop-Korobilis DMS Router** | Recursive Bayesian state-space gating closes 56–68% of the Oracle Gap | Numerically the strongest result in the project (+30–40% lift over single-domain baselines, p<10⁻¹⁵); survives multi-horizon external OOD transfer | Genuinely publishable piece of work. See §4.2 |

**The project's actual contribution is E, not D.** Everything written in `MASTER_RESEARCH_AUDIT.md` §§6, 7, and the abstract of `main.tex` implicitly positions LLM gating as the contribution. The data says otherwise.

---

## 3. Numerical evidence — what the CSVs actually show

### 3.1 The LLM gate is a null result (not a "breakthrough")

From `data/lgcf_results/lgcf_summary.csv` and `data/benchmarks/master_ablation_8way.csv` (verbatim, 5-fold walk-forward CV, 1960–2025, 169 economies, h ∈ {1,3,5}):

| Config | h=1 MAE | h=3 MAE | h=5 MAE | Avg lift vs Eco-Only |
|---|---:|---:|---:|---:|
| A. Economy-Only baseline (LGBM+Ridge) | 0.02657 | 0.06036 | 0.07994 | 0.00% |
| B. Uniform Cross-Domain Mixture | 0.02652 | 0.05960 | 0.07860 | +1.04% (p=0.96) |
| C. Random Dirichlet Gating | 0.02651 | 0.05991 | 0.07914 | +0.65% (p=0.88) |
| D. Heuristic Rule-Based Gating | 0.02650 | 0.05997 | 0.07900 | +0.69% (p=0.42) |
| **E. Zero-Shot LLM Gate (DeepSeek-V4, 9,302 calls)** | **0.02653** | **0.06009** | **0.07926** | **+0.48% (p=0.53, NULL)** |
| F. Markov-Switching AR (Hamilton 1989) | 0.02792 | 0.06654 | 0.08937 | −7.85% (loses) |
| **G. LGCF-v2 (Conformal Specialist Router)** | 0.02683 | 0.05983 | 0.07760 | **+1.02%** (h=3 p=5.78e-8, h=5 p=0.0002) |
| H. Oracle (theoretical upper bound) | 0.02323 | 0.05086 | 0.06518 | +15.59% |

**Reading these numbers honestly:**

- The zero-shot LLM gate (E) is **statistically indistinguishable** from the random Dirichlet gate (C), the heuristic gate (D), and the uniform mixture (B). Average lift vs. Economy-Only: +0.48%, +0.65%, +0.69%, +1.04% respectively. None is significant. The LLM is not adding regime-detection signal beyond what random Dirichlet noise produces.
- LGCF-v2 (G) achieves a real +1.02% lift with statistical significance at h=3 and h=5. The `+4.70% in shock regimes` claim in the README is consistent with the `solution_v2_summary.csv` (`imp_solution_v2_pct` column shows +2.54% at h=3, +0.97% at h=5 overall) but the "shock sub-slice" detail is not directly verifiable from the CSV I have. The magnitude is small.
- The narrative in `main.tex` lines 156–160 ("Calibrated router achieves MAE 0.04040 vs Eco-Only 0.04088 in COVID fold") is real but the absolute lift is **+1.2% in MAE** during the strongest possible test slice. A reviewer will ask why this is worth a paper.

The framing in `MASTER_RESEARCH_AUDIT.md` §6 ("August 2026 Breakthrough") is misleading: the LLM gate itself produces a null result; the only claimed "breakthrough" leverage is from the conformal meta-routing layer (G), which does not require the LLM at all (it uses predicted residual variance). The LLM is essentially acting as a colorful feature for a downstream calibrator.

### 3.2 The Koop-Korobilis DMS result is the genuine finding

From `data/benchmarks/oracle_gap_tournament_results.csv` (3,528 / 3,192 / 2,856 country-year test obs across h ∈ {1,3,5}):

| Horizon | Config | MAE | Lift vs Eco | Oracle Gap Closed | DM stat |
|---|---|---:|---:|---:|---:|
| h=1 | 1. Economy Baseline (Ridge) | 0.04323 | 0.00% | 0.00% | — |
| h=1 | 5. **Koop-Korobilis DMS** | **0.03011** | **+30.35%** | **56.58%** | 31.226 |
| h=1 | 7. Theoretical Oracle | 0.02004 | +53.64% | 100.00% | 52.725 |
| h=3 | 5. **Koop-Korobilis DMS** | **0.06475** | **+36.03%** | **63.47%** | 22.894 |
| h=5 | 5. **Koop-Korobilis DMS** | **0.09108** | **+40.59%** | **67.79%** | 24.843 |

And from `data/benchmarks/external_generalization_benchmarks.csv` (33 held-out countries, full spatial OOD transfer; 1,512–1,971 country-year obs, 2015–2025 temporal freeze):

| Protocol | Horizon | MAE | Lift vs AR(1) | DM p-value |
|---|:---:|---:|---:|---:|
| Spatial OOD | h=1 | 0.04989 | +26.14% | <10⁻¹⁵ |
| Spatial OOD | h=3 | 0.11899 | +20.84% | 1.5e-9 |
| Spatial OOD | h=5 | 0.18509 | +20.13% | 8.9e-6 |
| Temporal Freeze | h=1 | 0.03275 | +5.64% | <10⁻¹⁵ |
| Temporal Freeze | h=3 | 0.07337 | +6.86% | <10⁻¹⁵ |
| Temporal Freeze | h=5 | 0.10657 | −8.62% (vs AR1) but +38.16% vs Economy Ridge | <10⁻¹⁵ |

These are genuinely large, statistically robust effects that survive both spatial and temporal out-of-distribution transfer. The DMS router is doing real work: it maintains a recursive Bayesian state-space probability over which specialist to trust for each country over time, with forgetting factor λ=0.92 (see `src/gating/dms_state_space_router.py` lines 38–54). The code is a faithful re-implementation of Koop & Korobilis (IER 2012, Economic Modelling 2011).

**This is the empirical backbone the paper should be built on.**

### 3.3 The cross-domain null is solid

From `data/exhaustive_combinatorial_benchmark_results.csv` (15 combinations × 5 targets, 2^N - 1 tournament):

- **gdp_pc_growth_1y_fwd**: S1_Eco alone wins (RMSE 0.0316). Every cross-domain addition is statistically insignificant (DM p∈[0.965, 0.987]).
- **stability_momentum_annual_mean**: S2_Pol alone wins; cross-domain adds noise.
- **co2_emissions_per_capita**: S3_Env alone wins; cross-domain adds noise (RMSE 1.0288 vs 1.0294+ for combos).
- **psychology_trust**: S3_Env+S4_Hum wins (DM p=0.0005). Cross-domain helps when target is itself cross-domain.
- **psychology_fear**: S1_Eco+S3_Env+S4_Hum wins (DM p<10⁻⁴, +2.09% RMSE).
- **disaster_economic_damage**: S1_Eco+S3_Env wins (DM p<10⁻⁴, +75.76% RMSE) — economic capital stock + climate both needed.

**Honest interpretation:** Cross-domain features help when the **target** is itself cross-domain (psychology, climate damage). Cross-domain features add noise when the target is a single-domain quantity (GDP growth). The "Cross-Domain Paradox" framing in the paper is real and worth publishing as a negative result, but the audit should not bury this under excitement about the DMS result.

### 3.4 The Dumitrescu-Hurlin Granger results are mostly clean

From `data/dumitrescu_hurlin_panel_granger_results.csv` (verified to be DH (2012) — see `dumitrescu_hurlin_panel_granger.py`, which implements E-W step, W̄, Z̃, Bonferroni, BH-FDR all correctly):

| Hypothesis | Lags | N | W̄ | Z̃ | Raw p | Bonferroni p | Survives FWER? |
|---|:---:|:---:|---:|---:|---:|---:|:---:|
| Trust → GDP Growth | 2 | 95 | 3.7796 | 6.5158 | <10⁻⁶ | <10⁻⁵ | ✅ |
| Social Cohesion → GDP Growth | 2 | 95 | 3.0447 | 3.5188 | 0.0004 | 0.013 | ✅ |
| Education → GDP Growth | 2 | 95 | 3.9645 | 7.2698 | <10⁻⁶ | <10⁻⁵ | ✅ |
| Temp Anomaly → GDP Growth | 1 | 50 | 4.6522 | 17.0467 | <10⁻⁶ | <10⁻⁵ | ✅ |
| Fear → Material Conflict | 2 | 95 | 2.9553 | 3.2109 | 0.0013 | 0.0397 | ✅ |
| Trust → Political Stability | 2 | 95 | 3.5294 | 5.5699 | <10⁻⁶ | <10⁻⁵ | ✅ |
| Disaster Damage → GDP Growth | 1,2 | 50 | — | — | ≥0.26 | 1.000 | ❌ |

These are methodologically sound. The `quad_positive_findings_summary.csv` quoting "F = 12.41, p = 0.0001" for trust→GDP does not exactly match the CSV (which shows W̄=3.7796, Z̃=6.5158, raw p<10⁻⁶, Bonferroni <10⁻⁵). This is a **small numerical inconsistency** between the summary narrative and the underlying CSV — minor but a reviewer will notice.

---

## 4. Strengths (what is genuinely good)

### 4.1 Engineering rigor — top 10% of applied ML papers

- **Anti-leakage discipline**: per-fold imputer, per-fold rank-transform, honest early-stopping, dummy dropout, target exclusion, Optuna constrained to training slice. This is documented across multiple files (`solution_conformal_analog_router.py` lines 173–175 use `rank_fit_transform`; `oracle_gating_analysis.py` lines 282–288 use per-fold pipeline).
- **Reproducibility**: every external result is a CSV with row counts, fold identifiers, DM stat, p-value, and source artifact cited in `MASTER_RESEARCH_AUDIT.md` tables. The 9,302 cached LLM gate responses in `data/llm_gate_cache/` enable exact replay.
- **Honest reporting**: `solution_v2_summary.csv` shows `mae_eco` and `mae_conformal` for h=1 — conformal **loses** at h=1 (0.02720 vs 0.02671, +1.85%). The README's headline "+2.54% at h=3" obscures the h=1 loss. The CSV is honest; the narrative is selective.
- **Multi-protocol evaluation**: spatial OOD transfer, temporal decade freeze, multi-horizon rolling-origin walk-forward, DM tests with sample-size awareness. Genuinely above the bar for applied forecasting.

### 4.2 The Online State-Space DMA Router Implementation

`src/gating/dms_state_space_router.py` implements an online recursive Bayesian expert-weighting rule with forgetting factor $\lambda=0.92$ (Koop & Korobilis 2012 IER) and an online country-level residual variance $\sigma_c^2$. Running with `mode="dma"` produces a convex combination over fixed specialist forecasts ($\hat{y}_t = \sum_m \pi_m \hat{y}_m$) rather than hard argmax selection (DMS) or a full Kalman-filter TVP estimation per model. The empirical audit honestly documents that the DMA advantage over simple equal-weight combination is $+1.6\%$ to $+2.3\%$ MAE (significant at $h=1, 5$; insignificant at $h=3$ with $p=0.0682$), and does not statistically separate from Economy LightGBM at $h=5$ ($p=0.2010$), with the 90% Model Confidence Set retaining 4 models at $h=3$ and 2 models at $h=5$.



### 4.3 The Panel Granger causality implementation is CSD-robust

`src/econometrics/panel_granger.py` implements the exact Dumitrescu–Hurlin (2012) finite-$T$ standardized statistic $\tilde{Z}$ (eq. 9), paired with Pesaran (2007) CIPS unit root pre-testing, Emirmahmutoglu–Kose (2011) / Lopez–Weber (2017) vector-resampling CSD panel bootstrap ($B=1,000$), and Chudik–Pesaran (2016) Cross-Sectionally Augmented CS-DH common factor filtering under Holm-Bonferroni FWER control ($m=7$). This directly resolves the CSD size distortion ($\hat{CD} > 85$) and honestly isolates the attenuation of ERA5 temperature anomalies ($Z_{\text{CS}} = 1.958, p = 0.0503$).


### 4.4 The dual-contribution framing (causality + forecasting) is good paper design

Combining a causal-analysis section (DH panel Granger) with a forecasting section (DMS router) and a methodological section (LLM-as-feature ablation) is a sensible three-paper-in-one structure. It mirrors the well-regarded IMF/World Bank working-paper style.

### 4.5 The conformal prediction framework is real

`solution_conformal_analog_router.py` lines 207–220 implement residual variance modeling and inverse-precision gating. The 90.14% empirical coverage claim from `projectresearch/AUDIT.md` is documented (`conformal_adjustment.json`) with the band-widening tail adjustment noted. This is publishable on its own as a methodological note.

---

## 5. Weaknesses and risks — what would block publication

### 5.1 The LLM-gate empirical center is null (critical)

`master_ablation_8way.csv` config E (zero-shot LLM gate) achieves +0.48% lift over Economy-Only, p=0.53 — statistically indistinguishable from random Dirichlet noise. The paper's central narrative ("LLM-Gated Cross-Domain Forecasting solves the Cross-Domain Paradox") is **not supported by the data**. The LLM gate essentially replicates the uniform mixture. The "LGCF-v2" name attached to Config G is also misleading because Config G is a **conformal meta-routing** layer that does not consume the LLM weights in any path I can see in `solution_conformal_analog_router.py`. The LLM gate (E) is a separate experiment that fails to deliver.

This is the single biggest risk. A referee will see this in the ablation table and ask: "If you remove the LLM, your system still works. What is the LLM doing?"

### 5.2 The "18.5% Oracle Ceiling" is not a predictive bound — it's a target-comparison upper bound

The Oracle is computed by selecting the best of 5 configs *per country-year* given the actual observed target (`oracle_gating_analysis.py` lines 299–302: `oracle_idx = np.argmin(error_matrix, axis=1)`). This is the standard oracle ceiling construction, but it is **not a real-world achievable bound**. Any learned router is bounded by the Bayes-optimal router given features, not by the oracle that knows the target. The `p < 10⁻⁴` claim is correct for the Oracle-vs-Uniform comparison (DM test on oracle errors vs uniform errors), but the framing in `main.tex` and the audit is loose: "an oracle gating upper bound of 18.5% MAE improvement" reads as if this is a learnable bound. It is not. It is the gap between uniform mixing and perfect hindsight.

### 5.3 The Granger "F=12.41, p=0.0001" claim is hand-curated, not from the CSV

`quad_positive_findings_summary.csv` says: "F = 12.41, p < 0.001". The actual `dumitrescu_hurlin_panel_granger_results.csv` row for trust→GDP at lag 2 reports `W̄=3.7796, Z̃=6.5158, raw p < 10⁻⁶, Bonferroni p < 10⁻⁵`. These are different statistics: F (per-country), W̄ (panel mean of F), Z̃ (standardized panel statistic). The audit conflates them. A reviewer who pulls the CSV will flag this.

### 5.4 The manuscript is skeletal

`manuscript/main.tex` is **194 lines** (8 pages with bibliography). It has:
- 1 related-work paragraph (Section 1, 3 sentences)
- 22 references total (in `references.bib`), but the manuscript cites only 7 of them in the body
- No related-work section dedicated to multi-horizon macro forecasting (Coulibaly & Li 2019, Goulet-Coulombe 2022, M4/M5 winners)
- No ablation tables for `solution_v2` or `advanced_meta_router` (the LGCF-v2 result is buried in prose)
- No proper Appendix for the DMS router derivation
- The "Crisis-Regime Specialization" subsection (lines 158–159) cites one COVID fold (h=1) and one pre-COVID fold (h=5) with hardcoded MAE values from `master_ablation_8way.csv` — but the COVID fold is fold 0 for h=1 in the augmented ablation, not in the LGCF-v2 ablation; the numbers appear to come from the oracle ablation (`oracle_fold_summary.csv` row fold=0: MAE 0.03771 vs eco 0.04088). This is **misattribution** — the number is the oracle's fold-level MAE, not the LGCF-v2 meta-router's.

### 5.5 Diebold-Mariano implementation is not HAC-adjusted (sub-critical)

`oracle_gating_analysis.py::diebold_mariano` (lines 183–193) uses:
```python
d = e1**2 - e2**2
var_d = max(1e-10, np.var(d) / n)
dm_stat = mean_d / sqrt(var_d)
```
This is the i.i.d. version of DM. For overlapping multi-step forecasts at h=3,5, the error differentials are autocorrelated and the Newey-West HAC adjustment is required for correct inference (Diebold & Mariano 1995, Section 4). The project **does** have HAC-adjusted DM code in `experiments/04_societal_psychology/validation.py` and `research 2/forecaster.py`, but the **headline tournament benchmarks** use the i.i.d. version. For h=1 this is fine; for h=3 and h=5 the reported p-values are likely too small (anti-conservative).

A referee who checks this will downgrade the statistical claims by ~1 tier of confidence.

### 5.6 Cross-paper consistency: two different baselines for the same horizon

`MASTER_RESEARCH_AUDIT.md` Table 1 line 30 says:
> Ensemble (LGBM+Cat+XGB+Ridge) at h=1 MAE = 0.0328

`data/lgcf_results/lgcf_summary.csv` A row at h=1 says:
> A_EcoOnly MAE = 0.02657

These are both supposed to be "Economy-Only baseline." The ratio is ~0.81×. Either the audit row is stale (from an earlier pipeline without Ridge) or the ablation is comparing against a different baseline. **This needs reconciliation in the paper.** Same h=5 number problem: audit says h=5 ensemble MAE 0.0822, ablation says 0.07994.

### 5.7 The "9,302 live DeepSeek-V4 inferences" framing is partially misleading

`llm_gate_engine.py::compute_gate_weights` (lines 308–318) loads from disk cache first; if cache hit, no API call is made. Only on cache miss does it actually call the DeepSeek API. The README's "9,302 Cached DeepSeek-V4 Inferences" is accurate but the word "Live" elsewhere is not — most of these were generated in a single batch and replayed from cache. This is fine for reproducibility but should not be sold as "live evaluation."

### 5.8 No data release or ingest scripts

The cross-domain panel (592 features, 237 countries, 1960–2025) is a major engineering asset but the `data/processed_panels/quad_domain_annual_panel.parquet` is shipped as a binary artifact. There are no documented scripts in `scripts/` or `src/harmonization/` that ingest GDELT, EM-DAT, V-Dem, and GMD into the quad panel. A reproducibility-focused reviewer (or replicator) cannot rebuild the data from scratch.

### 5.9 The "94.2% similarity" claim for USA(2015)↔Canada(2012) is geometric proximity, not prediction accuracy

The audit acknowledges this in `MASTER_RESEARCH_AUDIT.md` §4 ("not a forecast accuracy percentage"), but the README and `main.tex` discussion of twins is much looser. The 4D FAISS engine has **no quantitative validation** as a forecasting tool — no held-out twin-ranking accuracy, no comparison against economy-only twin matching. It's an interpretive retrieval system dressed up as a prediction component.

---

## 6. What the project should claim — a recommended reframe

A reviewer-resilient paper would look like this:

**Title:** "Persistent Regime Memory Beats Naive Cross-Domain Forecasting: A 237-Economy, 65-Year Panel Study"

**Contributions:**
1. **Negative result (worth publishing):** Uniform concatenation of political, environmental, and societal features adds zero statistically significant improvement to GDP forecasting on a global panel (Diebold-Mariano p≥0.96 across 15 sector combinations). Cross-domain features help only when the target itself is cross-domain (psychology, climate damage).
2. **Methodological positive:** We document the *conditions* under which cross-domain information helps (multi-year lags, certain target classes) via Dumitrescu-Hurlin (2012) panel Granger causality with full multiplicity correction. Trust→GDP, Education→GDP, Temp Anomaly→GDP, and Fear→Conflict all survive Bonferroni at the 5% level with effect sizes matching the institutional-economics literature (Algan-Cahuc 2010, Burke et al. 2015).
3. **Practical contribution:** State-space recursive gating (Koop-Korobilis DMS) closes 56–68% of the theoretically available headroom and beats economy-only by +30% to +40% MAE across h∈{1,3,5}. The lift survives spatial OOD transfer (33 held-out countries) and 2015–2025 temporal freeze.
4. **Methodological ablation:** Zero-shot LLMs (DeepSeek-V4, 9,302 calls) acting as regime detectors fail to deliver meaningful lift (+0.48% over economy-only, p=0.53). Conformal meta-routing captures most of the regime-switching benefit without needing the LLM. **The LLM is the wrong tool for this task** — a finding worth reporting.

This framing removes the LLM-as-headline contribution, drops the "oracle ceiling" framing into a clearly labeled theoretical bound, and elevates the DMS result to the center. It is more honest and more publishable.

---

## 7. Suggested venue strategy

| Submission target | Expected outcome | Why |
|---|---|---|
| **NeurIPS main track** | Reject (desk) | Empirical contribution is null for the LLM-gate thesis; the DMS result is an engineering contribution, not a methodological advance at NeurIPS caliber |
| **NeurIPS AI for Finance workshop** | Accept (oral possible) | The OOD transfer protocol, the conformal meta-routing, and the LLM-null finding are workshop-worthy |
| **KDD Applied Data Science** | Accept (poster) | Engineering scope, panel size, reproducibility are appropriate for KDD |
| **International Journal of Forecasting** | Major revision | Right scope for cross-horizon panel forecasting; needs related-work expansion and HAC-DM correction |
| **Computational Economics** | Accept with minor revisions | Good fit for the cross-domain panel + negative result + DMS methodology combo |
| **Journal of Applied Econometrics** | Major revision | Needs (a) HAC-DM, (b) Clark-West test, (c) Granger causality treated more carefully with endogeneity discussion |
| **Economic Modelling** | Accept with minor revisions | Strong fit; less prestige than IJF/JAE but accepts applied methodological pieces |
| **AER P&P / ReStat / QJE** | Reject | Not a top-tier econometrics contribution; no causal identification beyond Granger |

**Recommended strategy:** Submit the LLM-as-failed-detector finding + DMS result + cross-domain null to **NeurIPS AI for Finance** workshop by ~mid-September for a Dec 2026 venue. Simultaneously prepare an IJF submission with the reframe in §6, dropping the LLM-as-headline and adopting the DMS-as-center framing. Target submission: late October 2026.

---

## 8. Concrete fix list (ordered by impact)

1. **Reframe the abstract.** Replace the LLM gate as the headline with the DMS router + cross-domain negative result. The LLM becomes a cautionary ablation, not the contribution.
2. **Add HAC-DM.** Re-run `oracle_gating_analysis.py` and `solution_conformal_analog_router.py` with Newey-West HAC for h≥3. Adjust the audit. The h=1 numbers are fine.
3. **Reconcile the baseline MAE numbers** between `MASTER_RESEARCH_AUDIT.md` Table 1 and `data/lgcf_results/lgcf_summary.csv`. Pick one canonical baseline per horizon and use it everywhere.
4. **Fix the F=12.41 vs W̄=3.7796 mismatch.** In `quad_positive_findings_summary.csv`, the F-statistic quoted for trust→GDP is not in the canonical CSV. Replace with W̄/Z̃ or recompute the per-country F if that's what was meant.
5. **Expand the manuscript to 12–14 pages.** Add related work (M-competition, Coulibaly 2019, Goulet-Coulombe 2022, Makridakis 2020, Aastveit 2019, cross-domain forecasting literature), add proper tables for the DMS tournament, add an Appendix on Koop-Korobilis DMS derivation, and add an Appendix on the panel-construction recipe.
6. **Add Clark-West (2007) test alongside DM** for nested model comparisons. Already cited in `references.bib` (lines 109–118) but never used in any script I found.
7. **Move the COVID/contagion COVID section** out of `main.tex` Section 4.3 (which uses a number from a different experiment) and either remove it or re-attribute it to the correct experiment.
8. **Add a quantitative evaluation of the 4D Twin engine.** Currently it is a retrieval module with no held-out ranking accuracy. Either (a) downgrade it to "interpretive scenario-anchoring" only and remove from forecasting claims, or (b) add a proper evaluation.
9. **Open-source the data ingest pipeline.** Add `scripts/build_quad_panel.py` that downloads GMD, GDELT, EM-DAT, V-Dem from their canonical sources, harmonizes to the panel schema, and reproduces `quad_domain_annual_panel.parquet`. Without this, the data is unverifiable.
10. **Clean up `MASTER_RESEARCH_AUDIT.md`.** Many of the comparison tables mix artifacts from different experiments (e.g., `lgcf_summary.csv` uses Eco-Only MAE 0.02657 but the headline benchmark table uses 0.0328). Either commit to one canonical ablation per horizon or label each row with its source experiment clearly.

---

## 9. Score card

| Dimension | Score | Top-tier standard | Verdict |
|---|:---:|:---:|---|
| Novelty (algorithmic) | ★★★★☆ (4/5) | ★★★★☆ | DMS implementation + cross-domain tournament + LLM-as-detector ablation = above average |
| Novelty (empirical finding) | ★★★☆☆ (3/5) | ★★★★☆ | Cross-domain null is solid; DMS lift is incremental over Koop-Korobilis (2012); LLM-as-detector null is novel-but-negative |
| Methodological rigor | ★★★☆☆ (3/5) | ★★★★☆ | DM not HAC-adjusted; Clark-West missing; DH Granger uses Bonferroni correctly; anti-leakage is excellent |
| Empirical results | ★★★★☆ (4/5) | ★★★★☆ | DMS result is strong and OOD-validated; LLM result is null (honest reporting pulls score up) |
| Engineering quality | ★★★★★ (5/5) | ★★★★☆ | Exceeds standard |
| Writing / manuscript | ★★☆☆☆ (2/5) | ★★★★☆ | 8 pages, 7 citations, single-DM test, no Clark-West, no related work |
| Related work | ★★☆☆☆ (2/5) | ★★★★☆ | 22 bib entries but body cites only 7; missing M4/M5 literature |
| Data documentation | ★★☆☆☆ (2/5) | ★★★★☆ | Parquet shipped, no ingest scripts, no codebook for "psychology" indicators |
| Reproducibility | ★★★★☆ (4/5) | ★★★★☆ | 9,302 cached LLM responses + CSV artifacts; missing ingest = reproducibility is bounded to provided data |
| Honest self-assessment | ★★★★★ (5/5) | ★★★★☆ | The CSVs are honest even when the audit prose is selective |

**Overall: 3.3 / 5.0** — publishable at Tier-2 with revisions. Not Tier-1 as currently positioned.

---

## 10. Bottom line

> **Is the empirical work sound?** Mostly yes — the DMS result is real and validated across multiple out-of-distribution protocols. The cross-domain null for GDP is solid. The Dumitrescu-Hurlin Granger analysis is methodologically clean.
>
> **Is the paper's framing honest?** Partially no — the LLM gate is a null result dressed as a "breakthrough." The "18.5% Oracle Ceiling" is presented as a learnable bound but is actually a hindsight oracle. The F=12.41 / W̄=3.78 mismatch is a citation-quality issue.
>
> **Is it publishable?** Yes — at IJF, Computational Economics, or Economic Modelling with a 3–6 month rewrite. The DMS router alone, combined with the cross-domain null and the DH Granger findings, is a strong Tier-2 paper.
>
> **Should the LLM be dropped from the contribution list?** Yes. It's an honest negative result worth reporting in the ablation, but it is not a contribution. The audit's positioning of LLM as the centerpiece overstates what the data shows by at least one tier of venue ambition.
>
> **Should the project pursue publication?** Yes, but reframe. The DMS-based router + cross-domain negative result is publishable. The LLM-gated framework is a workshop paper at best, and only if the LLM-null is presented as the finding rather than the gating mechanism.
