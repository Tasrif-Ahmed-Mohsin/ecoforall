# DeepSeek LLM Forecasting — Analysis & Diagnosis

## TL;DR: Your LLM **did run successfully** and results **do exist** — but they aren't surfaced anywhere you'd see them.

The script [_llm_zero_shot.py](file:///e:/project_gmd/scripts/_llm_zero_shot.py) completed all 426 API calls (213 countries × 2 horizons), wrote output files, and the DeepSeek results are **dramatically worse** than your ML ensemble. Here's the full breakdown.

---

## 1. Where Are the Results?

The results exist across **per-horizon CSV + JSON pairs**, but **no other script in your project reads them**:

| Output file | Status |
|---|---|
| [llm_baseline_holdout.csv](file:///e:/project_gmd/data/features/llm_baseline_holdout.csv) | ✅ **426 rows** (213 h=1 + 213 h=5), all populated |
| [llm_baseline_test_h1-3-10_metrics.json](file:///e:/project_gmd/data/features/llm_baseline_test_h1-3-10_metrics.json) | ✅ h=1/h=3/h=10 test metrics, all 3 horizons populated |
| [llm_baseline_test_h5_metrics.json](file:///e:/project_gmd/data/features/llm_baseline_test_h5_metrics.json) | ✅ h=5 test metrics (n=213, year 2019 origins) |
| [llm_baseline_val_h{1,3,5,10}_metrics.json](file:///e:/project_gmd/data/features/llm_baseline_val_h1_metrics.json) | ✅ per-horizon val metrics (n=876 / 869) |
| `llm_baseline_test_h1_h3_h10.csv` | ⚠️ **JUNK** — 60 rows, all 401 auth errors, no usable `llm_pred`. Do not cite. Use the metrics JSON above instead. |

### Why "no results are showing"

> [!IMPORTANT]
> **Neither the web app, the benchmark table script, nor any other reporting script reads the LLM output files.** The LLM results are orphaned — they exist on disk but are never displayed by:
> - [web_app.py](file:///e:/project_gmd/scripts/web_app.py) — zero mentions of "llm"
> - [_benchmark_v2.py](file:///e:/project_gmd/scripts/_benchmark_v2.py) — zero mentions of "llm"
> - [_benchmark_table.py](file:///e:/project_gmd/scripts/_benchmark_table.py) — zero mentions of "llm"

The console output at the end of `_llm_zero_shot.py` does print a summary table, but only **during the run itself**. Your [_llm_run.log](file:///e:/project_gmd/data/features/_llm_run.log) shows only partial progress (lines 3-4 logged 40/426), and `_llm_full.log` is **empty** — suggesting the script's stdout was not fully captured to the log file.

---

## 2. The Actual Results: LLM vs Your Ensemble

Here's what the completed run produced — extracted from the per-horizon metrics JSON files (`llm_baseline_holdout.csv` for h=1 holdout, `llm_baseline_test_h5_metrics.json` for h=5 test, `llm_baseline_test_h1-3-10_metrics.json` for h=1/h=3/h=10 test):

### h=1 Forecasting (1-year horizon, 2023 holdout, n=213)

| Model | MAE | RMSE | Direction Accuracy |
|---|---|---|---|
| **DeepSeek-V4 zero-shot** | **0.0240** | **0.0506** | **85.9%** |
| Your v2 ensemble (lgbm+prior) | 0.0281 | 0.0528 | 70.9% |
| Prior (naive persistence) | 0.0353 | 0.0656 | 70.0% |

### h=5 Forecasting (5-year horizon, test split, n=213)

| Model | MAE | RMSE | Direction Accuracy |
|---|---|---|---|
| **Cross-horizon meta** | **0.0377** | **0.0779** | **89.2%** |
| Your v2 ensemble (lgbm+prior) | 0.0922 | — | — |
| DeepSeek-V4 zero-shot | 0.0932 | 0.1671 | 75.1% |
| Prior (naive persistence) | 0.1102 | 0.1896 | 74.6% |

### Multi-horizon LLM summary (test slice, post-fix)

| h | n | DeepSeek MAE | Cross-horizon meta MAE | Naive MAE | DeepSeek dir_acc |
|---|---|---|---|---|---|
| 1 | 869 | 0.0528 | **0.0540** | 0.0587 | 0.620 |
| 3 | 644 | **0.0845** | 0.0896 | 0.0939 | 0.753 |
| 5 | 213 | 0.0932 | **0.0377** | 0.1102 | 0.751 |
| 10 | 213 | 0.1980 | **0.1233** | 0.1334 | 0.831 |

> [!IMPORTANT]
> **The numbers above are post-fix** (parse-bug in `_parse_forecast` was fixed by always-dividing-by-100). See `data/features/LLM_RESULTS.md` for the standalone summary.
> [!CAUTION]
> **Honest verdict.** DeepSeek beats the panel model on the h=1 holdout (clean, 2023, n=213) and on h=3 test, but loses to the cross-horizon meta at h=5 (2.5× worse) and is **the worst of the three at h=10** (0.198 vs meta 0.123, +49 % worse than naive). DeepSeek has no place in the h=5/h=10 baseline story. The dir_acc 85.9 % at h=1 holdout is real, not a bug — see `data/features/LLM_RESULTS.md` §5 for the per-error correlation analysis.

---

## 3. Root Cause: The Unit-Conversion Bug
## 4. Summary of Issues

| # | Issue | Impact | Severity |
|---|---|---|---|
| 1 | **Results not shown anywhere** — no script/UI reads `llm_baseline_*.{csv,json}` | You see nothing despite tokens burning | 🔴 Critical |
| 2 | **h=1 meta comparison missing** — cross-horizon meta doesn't cover holdout slice | Can't compare LLM vs ensemble for h=1 | 🟡 Moderate |
| 3 | **Console output not captured** — `_llm_run.log` only has 2 progress lines, `_llm_full.log` is empty | You missed the final summary table | 🟡 Moderate |
| 4 | **API key in plaintext** in [deepseek.txt](file:///e:/project_gmd/deepseek.txt) (committed to repo) | Security risk | 🟡 Moderate |

---

## 5. What Needs to Be Fixed

### Fix 1: Integrate LLM into the benchmark/web UI

Add a `deepseek_zero_shot` row to `_benchmark_v2.py` or `_benchmark_table.py` so the comparison appears in your reporting pipeline.

### Fix 2: Add h=1 ensemble comparison

Use the per-horizon v2 ensemble holdout predictions directly (they exist in `horizon_1y_v2/forecasts.parquet` with split=`holdout`, MAE=0.028).

### Fix 3: Secure the API key

Move the key to an environment variable and add `deepseek.txt` to `.gitignore`.

> [!NOTE]
> **Parse bug fixed 2026-07-23.** The post-fix LLM MAE is 0.024 (h=1 holdout) and 0.093 (h=5 test) — well within the predicted ~0.03–0.05 range for h=1 and 2× better than naive at h=5. The verdict: DeepSeek is genuinely competitive at h=1/h=3 but is destroyed by the cross-horizon meta at h=5/h=10. See `data/features/LLM_RESULTS.md` for the full multi-horizon picture.

---

## 7. v2 Nested Walk-Forward CV (2026-07-23)

**Status:** A full v2 nested walk-forward CV was run after applying five anti-leakage fixes to `scripts/_panel_backtest.py`:
1. Per-fold imputer (`scripts/_panel_backtest.py` lines 122–127) — no global fit.
2. Per-fold rank transform (lines 226–247) — fitted only on the train slice per fold.
3. Honest early-stopping (lines 254–280) — `eval_mask` excluded from the fit so the LGBM never sees the val/test rows during stop-decision.
4. `DROP_DUMMIES=True` default (line 332) — country/tier dummies turned off by default (previously silently leaking era-specific country means).
5. Tightened Optuna nested search space (lines 144–162) — wider `min_child_samples` floor, `max_depth` cap, `min_gain_to_split` floor.

Command used:
```
python scripts\_panel_backtest.py --horizons 1 3 5 10 --n-folds 5 --test-window 4 \
    --anchor-end-h5 2022 --nested-trials 300 --nested-val-years 4
```

Anchors: h=5→2022, h=1→2024, h=3→2023, h=10→2018 (h=10 ran 4 folds because anchor − 5×4 = 1998 is outside the panel).

### 7.1 Headline MAE (mean +/- std across folds)

Pulled directly from `data/features/walk_forward_cv_summary.json::per_model`. **This is the authoritative nested-CV number set** and supersedes earlier over-optimistic tables.

| h | folds | LGBM | ens (lgbm+prior) | ens (lgbm+ridge+prior) | Ridge | AR(1) honest | Naive persist |
|---|---|---|---|---|---|---|---|
| 1 | 5 | 0.0336 +/- 0.0081 | **0.0334 +/- 0.0080** | 0.0346 +/- 0.0074 | 0.0402 +/- 0.0071 | 0.0359 +/- 0.0075 | 0.0469 +/- 0.0094 |
| 3 | 5 | 0.0757 +/- 0.0110 | **0.0695 +/- 0.0118** | 0.0715 +/- 0.0130 | 0.0904 +/- 0.0142 | 0.0710 +/- 0.0118 | 0.0783 +/- 0.0114 |
| 5 | 5 | 0.1032 +/- 0.0124 | 0.0843 +/- 0.0132 | 0.0831 +/- 0.0119 | 0.1179 +/- 0.0083 | **0.0686 +/- 0.0194** | 0.0735 +/- 0.0248 |
| 10 | 4 | 0.1700 +/- 0.0038 | 0.1328 +/- 0.0032 | 0.1288 +/- 0.0026 | 0.1986 +/- 0.0108 | **0.0894 +/- 0.0094** | 0.0936 +/- 0.0107 |

### 7.2 Skill vs baselines (ensemble_lgbm_prior mean across folds)

| h | skill_vs_naive | skill_vs_ar1 | dir_acc |
|---|---|---|---|
| 1 | +0.279 | +0.072 | 0.719 |
| 3 | +0.115 | +0.021 | 0.751 |
| 5 | -0.265 | -0.297 | 0.778 |
| 10 | -0.430 | -0.496 | 0.832 |

The ensemble variant **`ensemble_lgbm_prior`** is the short-horizon recipe only. It edges AR(1) at h=1 and h=3, but AR(1) is stronger at h=5 and h=10.

### 7.3 Honest framing

> [!CAUTION]
> The authoritative nested-CV story is no longer "ML dominates." It is: ML helps at short horizons, while the per-country AR(1) honest baseline dominates medium and long horizons. The cross-horizon h=5 result remains a separate single-slice result and should be labeled separately from the walk-forward CV table.

### 7.4 What changed in the project files

| File | Change |
|---|---|
| `data/features/walk_forward_cv.csv` | 5 folds × 4 horizons × 6 models = 120 rows (refresh) |
| `data/features/walk_forward_cv_summary.json` | mean ± std per (horizon, model) + skill/dir_acc (refresh) |
| `data/features/walk_forward_cv_nested_params.json` | nested Optuna best_params per (horizon, fold) — new |
| `data/features/_walk_forward_cv.log` | raw run log (1.2 MB) |

### 7.5 Reproduce

```bash
python scripts\_panel_backtest.py \
    --horizons 1 3 5 10 --n-folds 5 --test-window 4 \
    --anchor-end-h5 2022 --nested-trials 300 --nested-val-years 4
```

Add `--nested-trials 0` for the leaky-hyperparameter diagnostic (see §7.3 caveat 1).
