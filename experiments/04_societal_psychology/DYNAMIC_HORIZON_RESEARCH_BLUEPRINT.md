# DYNAMIC HORIZON RESEARCH BLUEPRINT & MASTER PROMPT

## Overview

This document provides a complete, domain-agnostic specification and master prompt for building a **Universal Dynamic-Horizon Pattern Recognition & Forecasting Engine**.

Unlike traditional models tied to a single domain (e.g., country-level GDP) or fixed yearly steps ($h \in \{1, 3, 5, 10\}$ years), this architecture dynamically adapts to:
* **Any Time Series Domain**: Political Science & Geopolitical Risk (Conflict Events, Polling Trends, Regime Stability), Macroeconomics, Financial Volatility/Returns, Energy Grid Load, Supply Chain Congestion, Climate Anomalies, or Epidemiology.
* **Any Time Frequency**: Minutes (`1T`), Hours (`1H`), Days (`1D`), Weeks (`1W`), Months (`1M`), or Years (`1Y`).
* **Any Forecast Horizon Set**: Fully parameterized target steps $h \in \{h_1, h_2, h_3, h_4\}$ configured via a central metadata specification.

---

## System Architecture

```
                     ┌─────────────────────────────────────────┐
                     │          YAML CONFIGURATION             │
                     │  Domain: "crypto_volatility"            │
                     │  Frequency: "1H" (Hourly)               │
                     │  Horizons: [1, 4, 24, 168] (hours)     │
                     └────────────────────┬────────────────────┘
                                          │
    ┌─────────────────────────────────────┴─────────────────────────────────────┐
    │                                                                           │
    ▼                                                                           ▼
┌──────────────────────────────────────┐                   ┌──────────────────────────────────────┐
│       FEATURE PIPELINE (DYNAMIC)     │                   │       RETRIEVAL ENGINE (GENERIC)     │
│ - Config-driven lags [t-1, t-4, t-24]│                   │ - Rank-Euclidean FAISS Index         │
│ - Windowed rolling stats             │                   │ - Sequence Length Matching           │
│ - Time-frequency embedding (Cyclic)  │                   │ - Overlap Mask Filtering             │
└──────────────────┬───────────────────┘                   └──────────────────┬───────────────────┘
                   │                                                          │
                   └──────────────────────────┬───────────────────────────────┘
                                              │
                                              ▼
                        ┌──────────────────────────────────────────┐
                        │      MULTI-HEAD QUANTILE FORECASTER      │
                        │  (LightGBM / Ridge Stack per horizon h)  │
                        └─────────────────────┬────────────────────┘
                                              │
                                              ▼
                        ┌──────────────────────────────────────────┐
                        │   LLM DOMAIN-ADAPTIVE NARRATOR FUSION    │
                        │ (Prompt injected with domain metadata)   │
                        └──────────────────────────────────────────┘
```

---

## 1. Central Configuration Contract (`config.yaml`)

```yaml
# Example 1: Geopolitical Conflict & Risk Intensity (ACLED / GDELT / GPR)
domain:
  name: "geopolitical_conflict_risk" # Options: "geopolitical_conflict_risk", "election_polling", "regime_stability", "gdp_macro"
  entity_label: "iso3"               # Entity column name ('iso3', 'country_code', 'ticker')
  time_col: "timestamp"              # Datetime index column name
  frequency: "1M"                    # Pandas offset alias ('1D', '1W', '1M', '1Y')
  time_unit_label: "months"          # Human-readable unit for display & LLM

forecasting:
  target_indicator: "violent_event_count" # Options: 'fatalities', 'gpr_index', 'conflict_events'
  target_transform: "log_return"         # Options: 'log_return', 'absolute_change', 'raw_level'
  horizons: [1, 3, 6, 12]                # Dynamic forecast steps h in units of months

retrieval:
  lookback_window: 12                    # 12-month lookback window in state vector
  distance_metric: "rank_euclidean"        # Options: 'rank_euclidean', 'faiss_cosine'
  min_overlap_ratio: 0.70                # Require 70% non-null indicator overlap

models:
  quantiles: [0.05, 0.10, 0.50, 0.90, 0.95]
  cv_folds: 5
  cv_window_type: "expanding"
```

```yaml
# Example 2: Election Polling & Voter Sentiment Dynamics
domain:
  name: "election_polling_dynamics"
  entity_label: "candidate_id"       # Entity column name ('candidate_id', 'party_id', 'district_id')
  time_col: "timestamp"
  frequency: "1W"                    # Weekly aggregation
  time_unit_label: "weeks"

forecasting:
  target_indicator: "poll_share_pct"
  target_transform: "absolute_change"
  horizons: [1, 4, 12, 26]           # Forecast horizons in weeks

retrieval:
  lookback_window: 12
  distance_metric: "rank_euclidean"
  min_overlap_ratio: 0.75

models:
  quantiles: [0.05, 0.10, 0.50, 0.90, 0.95]
  cv_folds: 5
  cv_window_type: "expanding"
```

---

## 2. Master System Prompt for AI Coding Agents

Copy and paste the block below into your AI coding assistant to build the codebase end-to-end:

```markdown
# SYSTEM PROMPT: Universal Dynamic-Horizon Pattern Engine

You are building a domain-agnostic, variable-frequency time-series forecasting system that combines vector similarity retrieval with multi-horizon quantile ML forecasting.

The engine MUST NOT hardcode indicator names, yearly frequency, or fixed forecast steps. All domain logic, time frequencies, and target horizons are driven dynamically by a single `config.yaml` file.

---

### PHASE 1: Canonical Ingestion & Dynamic Data Schema
1. Unified Data Harmonizer: Convert raw multi-variate time series into a canonical schema:
   `[entity_id, timestamp, indicator_id, value]`
2. Matrix Assembly: Construct a wide multi-index panel DataFrame indexed by `(entity_id, timestamp)`.
3. Validation Pipeline: Build a generic data audit tool that reports missingness, temporal gaps, and panel balance for any entity type.

---

### PHASE 2: Dynamic Feature & Metric Transformation
1. Frequency-Aware Feature Generation: Read `config.yaml` and build features automatically:
   - Lags: Generate $t - h$ for every $h \in \text{horizons}$.
   - Rolling Aggregations: Compute rolling mean, std, min, max over lookback windows $[1 \times \text{window}, 2 \times \text{window}]$.
   - Cyclical Embeddings: Sine/Cosine encoding for time units (hour-of-day, day-of-week, month-of-year) matching `domain.frequency`.
   - Event Shocks & Calendar Anchors: Generate distance-to-event features (e.g. days/months to next scheduled election, legislative session openings, sanctions implementation).
   - Sentiment & Exogenous Proxies: Incorporate normalized text/media sentiment scores (e.g. GDELT tone, policy speech embeddings) if configured.
2. Percentile Rank Mapping: Normalize continuous feature vectors per timestamp slice to $[0, 1]$ uniform rank percentiles across entities to enforce scale-invariant similarity matching.

---

### PHASE 3: Vector Retrieval & Analog Trajectory Engine
1. State Vector Representation: Assemble state vectors $V_{i, t}$ capturing current features + lookback sequence.
2. Rank-Euclidean FAISS Index: Construct an L2 FAISS index over rank-transformed vectors.
3. Overlap Filter: Enforce `min_overlap_ratio` filtering to disqualify entity-time slices lacking minimum co-observed indicators.
4. Relative Match Scoring: Rescale pairwise Euclidean distances into relative match confidence scores $[0.0, 1.0]$.
5. API Contract: `find_analogs(entity_id, timestamp, k=8)` returns the top-K historical analogs with entity ID, historical timestamp, similarity score, and realised forward outcome paths.

---

### PHASE 4: Horizon-Agnostic Multi-Head Estimators
1. Dynamic Head Building: Iterate through `horizons` in `config.yaml`:
   - Target Creation: Build $y_{t+h}$ via `target_transform`.
   - Estimators: Train Ridge Regression + LightGBM Point Regressor + Quantile Regressors ($q_{05}, q_{50}, q_{95}$).
   - Baselines: Fit AR(1) and Naive Persistence baselines ($y_{t+h} = y_t$).
2. Stacking Meta-Learner: Train a Ridge meta-stacker over candidate prediction outputs across all horizons to enforce multi-step trajectory coherence.

---

### PHASE 5: Anti-Leakage Walk-Forward CV & DM Verification
1. Rolling Origin Cross-Validation: Implement a 5-fold expanding-window CV.
2. Anti-Leakage Rules:
   - Strictly fit imputers and rank transformations inside training fold boundaries.
   - Run hyperparameter selection (Optuna) inside nested validation loops.
   - Do not memorize static entity fixed-effect dummies across time shifts.
3. Statistical Falsification: Run Diebold-Mariano (DM) tests with Newey-West HAC variance adjustment comparing ML Ensemble against AR(1) and Naive baselines.

---

### PHASE 6: Split-Conformal Prediction Bands
1. Quantile Calibration: Compute split-conformal coverage on an out-of-time evaluation slice.
2. Conformity Scores:
   $$s_i^{lower} = q_{0.05}(x_i) - y_i, \quad s_i^{upper} = y_i - q_{0.95}(x_i)$$
3. Fallback Mechanism: If empirical coverage $< 85\%$, surface `calibration_acceptable: false` and widen interval bounds using empirical quantile residuals.

---

### PHASE 7: LLM Narrative Synthesizer & Web Interface
1. Domain-Adaptive Explainer: Inject prompt template with `domain.name`, `time_unit_label`, `horizons`, numeric quantile forecasts, and top-5 historical analogs.
2. Streamlit Application (`web_app.py`):
   - Model Evaluation Dashboard (Walk-forward metrics & DM test results).
   - Scenario Forecaster (Quantile fan charts & conformal bounds).
   - Pattern Matching Explorer (Ranked analogs with realised trajectory overlays).
```

---

## 3. Domain Adaptation Examples

| Domain | Entity (`entity_label`) | Frequency (`frequency`) | Horizons (`horizons`) | Target Indicator | Lookback |
|---|---|---|---|---|---|
| **Geopolitical Risk & Violence** | `country_iso3` | `1M` (Monthly) | `[1, 3, 6, 12]` (Months) | Violent Incidents / Fatalities (ACLED) | 12 Months |
| **Election Polling & Sentiment** | `candidate_id` / `party_id` | `1W` (Weekly) | `[1, 4, 12, 26]` (Weeks) | Poll Share / Net Approval | 12 Weeks |
| **Legislative Polarization** | `country_iso3` | `1M` (Monthly) | `[1, 3, 6, 12]` (Months) | Polarization / Voting Distance | 24 Months |
| **Regime Fragility & Governance** | `iso3` | `1Y` (Yearly) | `[1, 3, 5, 10]` (Years) | V-Dem Liberal Democracy Score | 5 Years |
| **High-Frequency Stock Volatility** | `ticker` | `1H` (Hourly) | `[1, 4, 24, 168]` (Hours) | Realized Volatility | 24 Hours |
| **Energy Grid Peak Load** | `region_id` | `1H` (Hourly) | `[1, 6, 24, 48]` (Hours) | Peak Megawatt Demand | 48 Hours |
| **Commodity Cycles (Oil/Gas)** | `commodity` | `1W` (Weekly) | `[1, 4, 12, 26]` (Weeks) | Spot Price Return | 12 Weeks |
| **Supply Chain Freight** | `route_id` | `1W` (Weekly) | `[1, 4, 12, 52]` (Weeks) | Freight Congestion Index | 12 Weeks |
| **Climate & Weather Anomalies** | `grid_cell` | `1D` (Daily) | `[1, 7, 30, 90]` (Days) | Sea Surface Temp Delta | 30 Days |

---

## 4. Special Considerations for Political Data & Geopolitical Time-Series

When adapting this engine for political science, election polling, and geopolitical risk modeling, adhere to the following domain-specific guidelines:

1. **Handling Structural Breaks & Regime Shocks**:
   * Political systems frequently experience non-linear structural shifts (e.g., coups, constitutional changes, major elections). Enforce **Rank-Euclidean FAISS matching** over recent moving windows to find historical analogs that share post-break dynamics.
2. **Sparse Event Aggregation & Irregular Polling**:
   * Event data (ACLED, GDELT) and polling series often exhibit irregular sampling and zero-inflated distributions.
   * Apply log transformations $\log(1 + y)$ or Tweedie/Poisson quantile loss objectives when predicting raw event counts or fatalities.
3. **Exogenous Calendar & Text Features**:
   * Explicitly include distance-to-election counters (days/weeks until voting day) as dynamic state features.
   * Fuse media sentiment (e.g., GDELT Goldstein scale or tone) with traditional macro-political indicators.
4. **Tail Risk & Conformal Interval Widening**:
   * Political shocks carry heavy tail risks. Ensure split-conformal coverage calibration (Phase 6) monitors interval validity during high-volatility pre-election or active conflict periods.

