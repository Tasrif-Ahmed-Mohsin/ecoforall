# project_gmd — GMD 2026 v6 clone

This is a **sibling clone** of `e:\research\project`, prepared to run the
existing 5-phase pipeline on the **Geo-Macroeconomic Dataset (Müller et al.,
2026 v6)** at `E:\GMD_2026_06_csv\GMD.csv`.

**The original `e:\research\project` is untouched.** This folder is for you
to hack on without breaking the working pipeline.

---

## What is already identical to project/

- All 25+ scripts under `scripts/`
- All 4 source modules under `src/`
- All 12 unit tests under `tests/`
- Same target column name: `gdp_pc_growth_5y_fwd`
- Same horizon set: `{1, 3, 5, 10}`
- Same train/val/test boundaries: `2014 / 2018 / 2022`
- Same conformal calibration slice: `2019–2022`
- Same walk-forward CV: `n_folds = 5`

## What is new (GMD-specific)

| File | What it does |
|---|---|
| `src/harmonize/gmd.py` | NEW. Reads `E:\GMD_2026_06_csv\GMD.csv`, melts the 162-column wide format to the canonical long schema, strips `forecast_*` columns, writes `data/harmonized/gmd.parquet`. |
| `data/harmonized/gmd.parquet` | NOT YET GENERATED. Run step 1 below to create it. |
| Old `data/harmonized/*.parquet` | COPIED but irrelevant — they hold the old IMF/WB/JST/Maddison/Clio-Infra source. Phase 2 will need editing to point at `gmd` instead. |
| Old `data/features/*.parquet,*.json,*.joblib` | COPIED but irrelevant — every per-horizon model + cross-horizon meta + walk-forward CV summary was trained on the old panel. **You must retrain.** |

## 5 commands to run the whole thing

```powershell
cd E:\research\project_gmd

# 1. Harmonize the raw GMD CSV -> long parquet (~10 s)
python -m src.harmonize.gmd

# 2. Edit src/features/build_panel.py SOURCES list to ["gmd"] (and adjust
#    CORE_TARGETS / SOURCE_PRIORITY for GMD's indicator_ids). Then:
python -m src.features.build_panel

# 3. Train all four horizon-level models (Ridge + LGBM q05/q10/q50/q90/q95 +
#    Optuna search). This will take a few minutes per horizon on a laptop.
python scripts/run_phase8_horizons_v2.py

# 4. Cross-horizon Ridge meta-ensemble over the four horizons.
python scripts/_cross_horizon_ensemble.py

# 5. Verify everything.
python -m pytest -q                                      # 12 tests should pass
python scripts/_panel_backtest.py                        # 5-fold walk-forward CV
python scripts/_conformal_calibrate.py                   # recalibrate PI band
python scripts/predict_country.py USA                    # end-to-end inference
python scripts/_benchmark_v2.py                          # consolidated benchmark
```

## Key file-edits you will need (placeholder list)

| Script | Change |
|---|---|
| `src/features/build_panel.py:28` | `SOURCES = ["gmd"]` |
| `src/features/build_panel.py:32` | `SOURCE_PRIORITY = {"gmd": 0}` |
| `src/features/build_panel.py:35-55` | Replace `CORE_TARGETS` with GMD indicator_ids |
| `src/features/build_panel.py:59` | `GDP_PC_CANDIDATES = ["gdp_pc_real"]` (the rGDP_pc column) |
| `src/harmonize/__init__.py` | `from . import gmd` |
| `scripts/run_phase1.py` | If it iterates over source names, leave as-is — `gmd` will be picked up automatically. |

The 25 scripts under `scripts/` need **no edits** — they all read
`TARGET = "gdp_pc_growth_5y_fwd"`, which is computed in `build_panel.py`
from `gdp_pc_real`, and the horizons stay `{1, 3, 5, 10}`.

## Quick schema sanity check (run any time)

```powershell
python -c "import pandas as pd; df=pd.read_parquet('data/harmonized/gmd.parquet'); print(df.shape); print(df.head()); print(df.indicator_id.value_counts().head(20))"
```

## What this clone is NOT

- It is **not a config-driven refactor**. The original `e:\research\project`
  is still hardcoded; if you want one pipeline that runs on either dataset,
  that's a separate (multi-hour) refactor.
- It is **not a paper-ready report**. The v1 baseline metrics, v2 metrics,
  cross-horizon meta, walk-forward CV, and benchmark JSONs are all copied
  from the old dataset and **lie about the new dataset** if you look at them.
  Regenerate everything before quoting any number.

## Why this design

The user said *"keep the style, clone it, so my current things remain, then
I will do things my own"*. This clone:

1. Zero risk to the working pipeline.
2. One new file (`src/harmonize/gmd.py`) is the only true new code.
3. The 5 commands above are the entire "do the same thing" recipe.
4. Everything else is mechanical edits you'll do as you go.
