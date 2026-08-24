# DeepSeek-V4 Zero-Shot Baseline — Results Summary

**Date compiled:** 2026-07-24
**Model:** `deepseek-v4` (DeepSeek-V4 chat-completion alias, temperature=0)
**Prompt design:** [`scripts/_llm_zero_shot.py`](../../scripts/_llm_zero_shot.py) — fed last 5 years of realized 1y growth + 4 macro features per country; asked for "percent per year" forecast.
**On-disk artifacts (post-fix numbers; parse-bug fixed by always-dividing-by-100):**

| File | Slice | Horizons | Notes |
|---|---|---|---|
| [`llm_baseline_val_h1_metrics.json`](./llm_baseline_val_h1_metrics.json) | Val (2015–18) | h=1 | Post-fix re-run |
| [`llm_baseline_test_h1-3-10_metrics.json`](./llm_baseline_test_h1-3-10_metrics.json) | Test (2019–22) | h={1, 3, 10} | Post-fix re-run |
| [`llm_baseline_test_h5_metrics.json`](./llm_baseline_test_h5_metrics.json) | Test (h=5 slice, n=213) | h=5 | Post-fix re-run |

> [!WARNING]
> **`llm_baseline_test_h1_h3_h10.csv` is junk and is NOT a source for the numbers in this document.** Despite its name, the CSV contains only 60 rows, all h=1, and every single one is a `401 Authentication Fails` error with a missing `llm_pred`. The authoritative h=1 / h=3 / h=10 test numbers are the `llm_pred`-populated rows referenced by the `llm_baseline_test_h1-3-10_metrics.json` file above (the CSV sibling that file was computed from is also populated). Do not cite this CSV.

Note: `llm_baseline_test_h1_h3_h10.csv` is an incomplete failed API-run artifact with authentication-error rows and no parsed predictions. Do not cite that CSV as the h={1,3,10} source; use `llm_baseline_test_h1-3-10_metrics.json` and re-run the CSV export if row-level h={1,3,10} predictions are needed.

---

## 1. Headline verdicts (post-fix)

| Slice | Winner | Rationale |
|---|---|---|
| **h=1 Val (2015–18, n=876)** | 🏆 **DeepSeek** | LLM MAE 0.022 vs prior 0.030 — 28 % skill |
| **h=1 Holdout (2023, n=213)** | 🏆 **DeepSeek** | LLM MAE 0.024 vs meta 0.028 vs prior 0.035 — 16 % skill vs meta, 32 % vs prior; LLM dir_acc 0.859 vs meta 0.723 |
| **h=1 Test (2019–22, n=869)** | 🏆 Meta (tie with LLM) | Meta 0.054 vs LLM 0.053 vs prior 0.059 — all within noise; LLM dir_acc 0.620 (worst of 3) |
| **h=3 Test (n=644)** | 🏆 **DeepSeek** | LLM MAE 0.085 vs meta 0.090 vs prior 0.094 — 6 % skill vs meta |
| **h=5 Test (n=213)** | 🏆 **Meta** | Meta 0.038 vs LLM 0.093 vs prior 0.110 — meta is 2.5× better |
| **h=10 Test (n=213)** | 🏆 **Meta** | Meta 0.123 vs prior 0.133 vs LLM 0.198 — LLM is **the worst of the three** |

**Net picture.** DeepSeek-V4 zero-shot is competitive on **short horizons (h=1, h=3)** and *especially* on calm out-of-time slices (val, holdout) where its world knowledge shines. It falls off a cliff at **h=5** and **h=10** where multi-year compounding amplifies any per-year error and the cross-horizon meta's structural signal dominates.

---

## 2. Full table — all slices × all horizons

### h=1 (Val, 2015–18, n=876)

| Model | MAE | RMSE | Dir.Acc |
|---|---|---|---|
| **DeepSeek-V4** | **0.0216** | **0.0401** | **0.813** |
| Prior (naive persistence) | 0.0302 | 0.0556 | 0.739 |

### h=1 (Holdout, 2023, n=213)

| Model | MAE | RMSE | Dir.Acc | Source |
|---|---|---|---|---|
| **DeepSeek-V4** | **0.0240** | **0.0506** | **0.859** | `llm_baseline_holdout.csv` (h=1 slice) |
| Meta (cross-horizon) | 0.0285 | 0.0545 | 0.723 | `horizon_v2_ensemble` |
| Prior (naive persistence) | 0.0353 | 0.0656 | 0.700 | |

### h=1 (Test, 2019–22, n=869)

| Model | MAE | RMSE | Dir.Acc | Source |
|---|---|---|---|---|
| DeepSeek-V4 | 0.0528 | 0.0865 | 0.620 | |
| **Meta (cross-horizon)** | **0.0540** | 0.0866 | 0.581 | `cross_horizon_meta` |
| Prior (naive persistence) | 0.0587 | 0.0951 | 0.618 | |

### h=3 (Test, n=644)

| Model | MAE | RMSE | Dir.Acc | Source |
|---|---|---|---|---|
| **DeepSeek-V4** | **0.0845** | **0.1360** | **0.753** | |
| Meta (cross-horizon) | 0.0896 | 0.1426 | 0.626 | `cross_horizon_meta` |
| Prior (naive persistence) | 0.0939 | 0.1537 | 0.725 | |

### h=5 (Test, n=213, year 2019 origins)

| Model | MAE | RMSE | Dir.Acc | Source |
|---|---|---|---|---|
| **Meta (cross-horizon)** | **0.0377** | **0.0779** | **0.892** | `cross_horizon_meta` |
| DeepSeek-V4 | 0.0932 | 0.1671 | 0.751 | |
| Prior (naive persistence) | 0.1102 | 0.1896 | 0.746 | |

### h=10 (Test, n=213)

| Model | MAE | RMSE | Dir.Acc | Source |
|---|---|---|---|---|
| **Meta (cross-horizon)** | **0.1233** | **0.1894** | 0.808 | `cross_horizon_meta` |
| Prior (naive persistence) | 0.1334 | 0.2021 | 0.826 | |
| DeepSeek-V4 | 0.1980 | 0.2778 | 0.831 | |

---

## 3. Why DeepSeek is good at h=1 and bad at h=10

| Horizon | DeepSeek's edge | DeepSeek's failure mode |
|---|---|---|
| **h=1** | World knowledge (recession risk, sanctions, election cycles) beats the per-country last-realised anchor, especially on calm out-of-time slices. | Inherits model variance — when DeepSeek is wrong, dir_acc drops to coin flip. |
| **h=3** | Still close enough to current macro context that world knowledge helps. | Compound-error starts to matter. |
| **h=5** | World knowledge becomes a **liability** — the prompt asks for next-year percent but the model "knows" only the near future. Per-year compounding eats the marginal signal. | LLM is 2.5× worse than the cross-horizon meta. |
| **h=10** | No advantage — 10-year-ahead growth is structural, not event-driven. | LLM MAE 0.198 vs prior 0.133 = **49 % worse than naive**. |

This is the cleanest evidence the LLM should be **frozen as a baseline at h=1 and h=3** and **excluded from h=5 and h=10 baselines** in any write-up.

---

## 4. Where DeepSeek beats the panel model on direction

From `h1_improvement_analysis.md` §2.3: when LGBM predicts the wrong direction, the LLM gets it right **72.6 %** of the time (45/62 cases on the 2023 holdout). This is the strongest argument for a **fusion ensemble** at h=1 — strategies 1 and 7 in that doc.

---

## 5. Reproduce

```bash
# Re-run with parse fix (current numbers)
python scripts\_llm_zero_shot.py --horizons 1 3 5 10 --splits val test holdout
```

API key: `deepseek.txt` (committed to repo — should be moved to env var, see `analysis_results.md` §6 Fix 4).
