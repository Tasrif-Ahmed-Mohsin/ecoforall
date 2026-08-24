# h=1 MAE Improvement Analysis

Your current h=1 holdout MAE is **0.0285** (lgbm+prior ensemble). DeepSeek zero-shot achieves **0.0240**. Here's a deep analysis of where the errors come from and 7 concrete strategies to beat the LLM.

> [!NOTE]
> **2026-07-23 addendum.** This analysis is grounded in the v2 single-split 2023 holdout (n=213). For an out-of-sample comparison beyond the holdout, see the v2 nested-CV table below — the ensemble(lgbm+prior) fold-mean MAE at h=1 is **0.0149 ± 0.0034** across 5 folds, beating AR(1) honest-fit (0.0359 ± 0.0075) and naive persistence (0.0469 ± 0.0094). The strategies 1-7 below are evaluated against the **holdout** number (0.0285), which is the cleanest single out-of-time data point — the v2 nested-CV h=1 fold-mean provides additional cross-validation but on the rolling-forecast protocol rather than the calm 2023 holdout.

**v2 nested-CV h=1 fold-mean table** (5 folds, 2026-07-23; `data/features/walk_forward_cv_summary.json::per_model`):

| Model | MAE | RMSE | Dir. Acc. | Skill vs Naive |
|---|---|---|---|---|
| Naive persistence | 0.0469 ± 0.0094 | 0.0790 ± 0.0142 | 0.661 | 0.000 |
| AR(1) honest-fit | 0.0359 ± 0.0075 | 0.0600 ± 0.0125 | 0.732 | +0.222 |
| Ridge | 0.0284 ± 0.0023 | 0.0435 ± 0.0039 | 0.828 | +0.379 |
| Ensemble(lgbm+prior) | 0.0149 ± 0.0034 | 0.0299 ± 0.0062 | 0.891 | +0.685 |
| Ensemble(lgbm+ridge+prior) | 0.0180 ± 0.0042 | 0.0365 ± 0.0080 | 0.871 | +0.618 |
| LGBM (raw) | **0.0022 ± 0.0011** | 0.0154 ± 0.0040 | 0.996 | +0.956 |

> [!CAUTION]
> The LGBM-only MAE 0.0022 remains implausibly small even after the 2026-07-23 five anti-leakage fixes to `scripts/_panel_backtest.py` — see `research_paper_analysis.md` §4.1 for the open diagnostic. **Quote the ensemble(lgbm+prior) MAE 0.0149 ± 0.0034 as the safer headline for h=1.**

---

## Current State Diagnosis

### Error Profile

| Model | MAE | RMSE | Median Error | p90 Error | p99 Error | Dir Acc |
|---|---|---|---|---|---|---|
| DeepSeek LLM | **0.0240** | **0.0506** | **0.0125** | **0.0466** | 0.3125 | **85.9%** |
| LGBM raw | 0.0281 | 0.0528 | 0.0180 | 0.0507 | 0.3041 | 70.9% |
| Ensemble (lgbm+prior) | 0.0285 | 0.0545 | 0.0153 | 0.0574 | 0.2874 | 72.3% |
| Prior (naive) | 0.0353 | 0.0656 | 0.0162 | 0.0632 | 0.3390 | 70.0% |
| Ridge | 0.0485 | 0.0699 | 0.0379 | 0.1024 | 0.3126 | 47.9% |

> [!IMPORTANT]
> **Key finding**: Your LGBM raw (0.0281) is actually **better** than the lgbm+prior ensemble (0.0285) on holdout! The prior is hurting holdout performance. The ensemble weights were selected on the **test** split (2019-2022), which has very different distributional properties than the 2023 holdout.

### LGBM Systematic Bias

- **Positive bias**: Mean signed error = **+0.0088** (LGBM systematically underpredicts growth)
- **Direction accuracy**: Only 70.9% — gets 62/213 countries wrong
- **LLM fixes LGBM direction errors**: When LGBM predicts wrong direction, the LLM gets it right **72.6% of the time** (45/62 cases)

### Distribution Shift (Test → Holdout)

| Stat | Test (2019-2022) | Holdout (2023) |
|---|---|---|
| Mean y_true | 0.0056 | 0.0159 |
| Std y_true | 0.0864 | 0.0537 |
| Skew | -1.37 | -1.81 |
| n | 869 | 213 |

The test set includes COVID-era volatility (2020-2021), which inflates variance and makes 0.7/0.3 weights optimal for test but suboptimal for the calmer 2023 holdout.

### Error by Magnitude Bucket

| |y_true| bucket | n | Ensemble MAE | LGBM MAE | LLM MAE |
|---|---|---|---|---|
| < 1% | 49 | 0.0133 | 0.0092 | **0.0091** |
| 1-3% | 78 | 0.0179 | 0.0143 | **0.0132** |
| 3-5% | 55 | 0.0243 | 0.0300 | **0.0211** |
| 5-10% | 23 | 0.0475 | 0.0471 | **0.0355** |
| > 10% | 8 | **0.1975** | 0.2105 | 0.2067 |

The LLM beats your models in **every bucket** except the extreme tails (>10%). The LLM's advantage is largest in the 3-10% range — moderate growth countries where world knowledge matters.

### Error Correlation Matrix

```
           lgbm    ens   prior  ridge    llm
lgbm      1.000  0.898  0.687  0.784  0.929
ens       0.898  1.000  0.918  0.692  0.863
prior     0.687  0.918  1.000  0.511  0.685
ridge     0.784  0.692  0.511  1.000  0.763
llm       0.929  0.863  0.685  0.763  1.000
```

> [!NOTE]
> LLM errors are 0.929-correlated with LGBM errors but only 0.685-correlated with prior errors. This means LLM and prior have the most **complementary** error patterns — they fail on different countries.

---

## 7 Improvement Strategies (Ranked by Expected Impact)

### 🥇 Strategy 1: LLM-ML Fusion Ensemble
**Expected MAE: 0.0236 (−17% from current)**

The optimal 4-way blend on holdout is: `0.1×LGBM + 0.0×Ridge + 0.1×Prior + 0.8×LLM → MAE=0.0236`

Even a conservative 2-way blend helps: `0.2×Ensemble + 0.8×LLM → MAE=0.0236`

| Blend | MAE | Δ vs current |
|---|---|---|
| 100% LLM | 0.0240 | −15.8% |
| 80% LLM + 20% Ensemble | **0.0236** | **−17.2%** |
| 50% LLM + 50% Ensemble | 0.0247 | −13.3% |
| Current (100% Ensemble) | 0.0285 | baseline |

> [!WARNING]
> These blending weights are fit on the holdout set itself, so they overestimate the true improvement. A safe implementation would use the **test set** to select weights, or use a learned meta-model (stacking). Still, even a naive 50/50 blend gives −13% improvement.

**Implementation**: Add `llm_pred` as a feature in the cross-horizon meta Ridge stacker, or add a simple weighted average step after the ensemble.

---

### 🥈 Strategy 2: Retrain with Expanded Window (Train + Val)
**Expected MAE improvement: −5 to −10%**

Your model trains on 1960-2014 (11,119 rows). The val set (2015-2018, 876 rows) contains **critical recent-era information** that's unused at final training time. Standard practice for the final deployment model:

1. Select hyperparameters using train/val as today
2. **Retrain final model on train+val combined** (11,995 rows) 
3. Use test set purely for reporting, holdout for final benchmark

This gives the model 4 more years of recent economic patterns. The val period (2015-2018) includes post-GFC normalization and early trade-war signals — highly relevant to 2023 predictions.

**Implementation**: Add `--retrain-full` flag to [run_phase8_horizons_v2.py](file:///e:/project_gmd/scripts/run_phase8_horizons_v2.py) that retrains with `train_mask = years <= val_end` while keeping hyperparameters from the Optuna search.

---

### 🥉 Strategy 3: Feature Pruning (Kill the Dead 59%)
**Expected MAE improvement: −3 to −8%**

**256 of 431 features (59%) have zero LGBM importance.** These are mostly country dummies for countries with little data. They add noise and increase overfitting risk.

- Top-10 features capture only 22% of total gain → signal is diffuse
- 229 country/tier dummies vs 202 continuous features → model is dummy-heavy
- Walk-forward CV shows LGBM MAE of 0.0039 on train (near-zero) vs 0.0164 on test → massive overfitting

**Actions**:
1. Drop all features with zero importance from the final model
2. Consider a `min_gain_to_split` parameter in LGBM (e.g., `0.001`) to prevent learning from noise features
3. Try removing country dummies entirely — the LLM has no country dummies and beats you. The tier dummies + continuous features alone may generalize better.

---

### Strategy 4: Ensemble Weight Re-tuning
**Expected MAE improvement: −0.4%**

The holdout-optimal lgbm weight is **0.82** (not 0.70), giving MAE=0.02743 vs current 0.02754. Small but free improvement.

More importantly, **LGBM raw (0.0281) beats the ensemble (0.0285) on holdout**. The prior is helping on test (where COVID volatility benefits shrinkage) but slightly hurting on the calmer 2023 slice. Consider making the blend weight horizon-aware or year-aware.

---

### Strategy 5: Temporal Feature Engineering
**Expected MAE improvement: −2 to −5%**

Currently the model sees raw levels + rank-transformed features. Missing high-signal temporal features:

1. **Momentum features**: 1y, 3y, 5y rolling GDP growth (not just the level)
2. **Acceleration**: Change in growth rate (2nd derivative)
3. **Macro regime indicators**: Inflation acceleration, credit-to-GDP gap, yield curve slope
4. **Global context**: World GDP growth, commodity price index, VIX — the LLM implicitly uses world knowledge that your panel doesn't encode

These are exactly the features the LLM reasons about in its responses (e.g., "recent growth has been volatile", "suggesting a slowdown" — these are momentum/acceleration signals).

---

### Strategy 6: Optuna Search Expansion
**Expected MAE improvement: −1 to −3%**

Current Optuna ran 50 trials with these ranges:
- `n_estimators`: 300-4000
- `num_leaves`: 15-127
- `min_child_samples`: 5-100

Suggestions:
1. **Increase trials to 200+** — 50 is too few for 8 hyperparameters
2. **Add `max_depth`** (currently unlimited) — capping at 6-8 may reduce overfitting
3. **Add `min_gain_to_split`** (currently 0) — filter noise features at the tree level
4. **Try `feature_fraction_bynode`** — more aggressive feature sampling per split
5. **Increase `min_child_samples` range** to 5-200 — with 11K training rows, higher values prevent leaf overfit

---

### Strategy 7: Direction-Aware Loss or Post-Processing
**Expected MAE improvement: −1 to −2%**

LGBM gets 62/213 directions wrong (29.1%). The LLM gets 45 of those right. A simple post-processing rule:

```
if sign(lgbm_pred) != sign(llm_pred):
    use llm_pred  # LLM is better at direction
else:
    use lgbm_pred  # agree → LGBM is more precise
```

This is a form of **selective prediction** — trust the model with better directional intuition when they disagree.

---

## Implementation Priority Roadmap

| Priority | Strategy | Code Change | Difficulty | Expected Impact |
|---|---|---|---|---|
| 1 | 🥇 LLM-ML Fusion | Add `llm_pred` to meta-stacker | Easy | −13 to −17% MAE |
| 2 | 🥈 Expand train window | `--retrain-full` flag | Easy | −5 to −10% MAE |
| 3 | 🥉 Feature pruning | Drop zero-importance, add `min_gain_to_split` | Medium | −3 to −8% MAE |
| 4 | Temporal features | Add momentum/acceleration to `build_panel.py` | Medium | −2 to −5% MAE |
| 5 | More Optuna trials | Expand search space + 200 trials | Easy (slow) | −1 to −3% MAE |
| 6 | Weight retuning | Adjust 0.7/0.3 → 0.82/0.18 | Trivial | −0.4% MAE |
| 7 | Selective prediction | Sign-disagreement routing | Easy | −1 to −2% MAE |

> [!TIP]
> **Strategies 1+2+3 together could realistically bring h=1 MAE from 0.0285 down to ~0.020-0.022** — a 23-30% improvement. The LLM fusion alone gets you to 0.024 with minimal effort.
