"""Comprehensive data + model integrity audit.

Answers five questions:

 1. Is there data leakage? Could any column theoretically encode the future
    target (year t+5)?
 2. Is the target column constructed correctly? Compare to a hand-derived
    value for a few known countries.
 3. Does the model beat sensible naive baselines on the test slice?
 4. Does the conformal calibration actually calibrate out-of-sample on the
    2023-2024 forward holdout?
 5. Per-country diagnostic: spot-check known macro events (Russia 2020,
    GBR 2020, COVID) to see if the model reacts at all.
"""
import sys, json, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from src.harmonize.common import FEATURES_DIR

panel = pd.read_parquet(FEATURES_DIR / "panel_wide.parquet")
all_cols = json.loads((FEATURES_DIR / "models" / "feature_cols.json").read_text())
kept_idx = json.loads((FEATURES_DIR / "models" / "imputer_idx.json").read_text())
imputer = joblib.load(FEATURES_DIR / "models" / "imputer.joblib")
ridge = joblib.load(FEATURES_DIR / "models" / "ridge.joblib")
lgbm = joblib.load(FEATURES_DIR / "models" / "lgbm.joblib")
q10 = joblib.load(FEATURES_DIR / "models" / "lgbm_q10.joblib")
q50 = joblib.load(FEATURES_DIR / "models" / "lgbm_q50.joblib")
q90 = joblib.load(FEATURES_DIR / "models" / "lgbm_q90.joblib")
cal = json.loads((FEATURES_DIR / "conformal_adjustment.json").read_text())
metrics_doc = json.loads((FEATURES_DIR / "baseline_metrics.json").read_text())
RECIPE = metrics_doc.get("ensemble_recipe", "lgbm+prior")
TRAIN_END = metrics_doc.get("split", {}).get("train_end", 2014)
CAL_OK = cal.get("calibration_acceptable", True)

# Load wider q05/q95 if present so the audit matches predict_country.py.
try:
    from joblib import load as _jload
    q05 = _jload(FEATURES_DIR / "models" / "lgbm_q05.joblib")
    q95 = _jload(FEATURES_DIR / "models" / "lgbm_q95.joblib")
except Exception:
    q05 = q95 = None

TARGET = "gdp_pc_growth_5y_fwd"


def _prior_pred(df: pd.DataFrame) -> pd.Series:
    """Per-country last train realised target — same formula predict_country.py uses."""
    train = df[df.year <= TRAIN_END].dropna(subset=[TARGET])
    if train.empty:
        return pd.Series(np.full(len(df), float(df[TARGET].mean())), index=df.index)
    gmean = float(train[TARGET].mean())
    last = train.sort_values("year").groupby("iso3")[TARGET].last().to_dict()
    return df["iso3"].map(lambda iso: float(last.get(iso, gmean)))


def predict_block(df: pd.DataFrame) -> pd.DataFrame:
    X = df[all_cols].astype(float).replace([np.inf, -np.inf], np.nan)
    kept_names = [all_cols[i] for i in kept_idx]
    Xi = imputer.transform(X[kept_names].to_numpy())
    full = np.zeros((len(df), len(all_cols)), dtype=np.float32)
    full[:, kept_idx] = Xi
    out = df[["iso3", "year"]].copy()
    out["y"] = df[TARGET].to_numpy()
    out["ridge"] = ridge.predict(full)
    out["lgbm"] = lgbm.predict(full)
    out["prior"] = _prior_pred(df).to_numpy()
    if RECIPE == "lgbm+prior":
        out["ensemble"] = 0.7 * out["lgbm"] + 0.3 * out["prior"]
    elif RECIPE == "lgbm+ridge":
        out["ensemble"] = 0.7 * out["lgbm"] + 0.3 * out["ridge"]
    else:
        out["ensemble"] = out["lgbm"]
    out["q10"] = q10.predict(full) + cal.get("a_lo", 0.0)
    out["q90"] = q90.predict(full) + cal.get("a_hi", 0.0)
    if q05 is not None and q95 is not None:
        out["q05"] = q05.predict(full)
        out["q95"] = q95.predict(full)
    # Mirror predict_country.py band selection.
    if not CAL_OK and q05 is not None and q95 is not None:
        out["pi_low"] = np.minimum(out["q05"], out["q95"])
        out["pi_high"] = np.maximum(out["q05"], out["q95"])
        out["band_label"] = "pi90_q05_q95"
    else:
        out["pi_low"] = np.minimum(out["q10"], out["q90"])
        out["pi_high"] = np.maximum(out["q10"], out["q90"])
        out["band_label"] = "pi80_q10_q90"
    return out


print("=" * 70)
print("(1) LEAKAGE AUDIT")
print("=" * 70)
# Question: does anything in the feature column names imply t+something?
suspicious = [c for c in all_cols if any(t in c for t in ("_fwd", "_future", "_t+1", "_t+5", "_next"))]
print(f"Columns with forward-looking tokens: {suspicious or '(none)'}")
# Also examine feature construction: all features are *_lag1, *_lag5, *_roll5_mean, *_delta5, *_logret5.
# delta5 = col[t] - col[t-5]  -- uses values up to t only  (correct).
# logret5 = log(col[t] / col[t-5])  -- uses values up to t only (correct).
# lag5 = col[t-5] (correct).
# So no feature should encode future. Confirm by reading build_panel.py.
print("\nAll engineered suffixes in the panel:")
suffixes = []
for c in all_cols:
    for suf in ("_lag1", "_lag5", "_roll5_mean", "_delta5", "_logret5"):
        if c.endswith(suf):
            suffixes.append(suf)
            break
    else:
        suffixes.append("(raw)")
from collections import Counter
print(Counter(suffixes).most_common())

print("\n" + "=" * 70)
print("(2) TARGET REALISM")
print("=" * 70)
# Spot-check: hand-compute target for USA 2010 from gdp_pc_real.
sample = (
    panel[(panel.iso3 == "USA") & (panel.year.isin([2008, 2009, 2010, 2013, 2014, 2015]))]
    [["iso3", "year", "gdp_pc", "gdp_pc_real", TARGET]]
)
print(sample.to_string(index=False))
# Hand compute for USA 2010:  log(gdp_pc[2015]) - log(gdp_pc[2010])
df_usa = panel[(panel.iso3 == "USA") & (panel.year.isin([2010, 2015]))]
g10 = float(df_usa[df_usa.year == 2010]["gdp_pc"].iloc[0])
g15 = float(df_usa[df_usa.year == 2015]["gdp_pc"].iloc[0])
print(f"\nHand-derived for USA 2010: log({g15:.0f}/{g10:.0f}) = {np.log(g15/g10):+.4f}")
print("Stored value should match.")
print()
print("Target distribution stats:")
y = panel.dropna(subset=[TARGET])[TARGET]
print(f"  n      = {len(y)}")
print(f"  mean   = {y.mean():+.4f}  (raw: ~+28% per 5yr)")
print(f"  std    = {y.std():.4f}")
print(f"  q01    = {y.quantile(0.01):+.4f}")
print(f"  q50    = {y.quantile(0.50):+.4f}")
print(f"  q99    = {y.quantile(0.99):+.4f}")
print(f"  min    = {y.min():+.4f}  max = {y.max():+.4f}")
print(f"  below -0.5 (negative shock): {(y < -0.5).sum()} rows ({(y < -0.5).mean()*100:.2f}%)")

print("\n" + "=" * 70)
print("(3) vs. NAIVE BASELINES (test slice 2019-2022)")
print("=" * 70)
test = predict_block(panel[(panel.year >= 2019) & (panel.year <= 2022)].dropna(subset=[TARGET]))
# Baseline 1: predict 0 for everyone
mae_zero = (test["y"] - 0).abs().mean()
dir_zero = ((test["y"] > 0) & (0 > 0) | (test["y"] < 0) & (0 < 0)).sum() / len(test)
# Baseline 2: predict each country's own lagged target (mean-of-prior-realized)
def country_baseline(df, target_col):
    pred = df.groupby("iso3")[target_col].shift(1)  # t-1 over the test slice: actually needs train mean.
    return pred
# Train-prior baseline: predict each country's mean training target.
train = panel[panel.year <= 2014].dropna(subset=[TARGET])
country_train_mean = train.groupby("iso3")[TARGET].mean().to_dict()
global_mean = train[TARGET].mean()
def country_prior(t):
    out = []
    for iso, yval in zip(t["iso3"], t["year"]):
        # Use the most recent train-period realized value (year 2014 if present, else mean).
        train_iso = train[(train.iso3 == iso)]
        if not train_iso.empty:
            out.append(train_iso[train_iso.year == train_iso.year.max()][TARGET].iloc[0])
        else:
            out.append(global_mean)
    return pd.Series(out, index=t.index)
priors = country_prior(test)
mae_prior = (test["y"] - priors).abs().mean()
dir_prior = ((np.sign(test["y"]) == np.sign(priors)) & (priors != 0)).mean()

def report(name, p, y):
    p = np.asarray(p); y = np.asarray(y)
    mask = ~np.isnan(p) & ~np.isnan(y)
    p = p[mask]; y = y[mask]
    mae = np.mean(np.abs(p - y))
    rmse = np.sqrt(np.mean((p - y)**2))
    dir_acc = np.mean((np.sign(p) == np.sign(y)) & (y != 0))
    in80 = ((y >= p) & (y <= 0)).mean()  # crude: how often y > 0 when p > 0
    return mae, rmse, dir_acc

m_m, m_r, m_d = report("zero",   np.zeros(len(test)), test["y"])
p_m, p_r, p_d = report("prior",  priors, test["y"])
e_m, e_r, e_d = report("ens",    test["ensemble"], test["y"])
r_m, r_r, r_d = report("ridge",  test["ridge"], test["y"])
l_m, l_r, l_d = report("lgbm",   test["lgbm"], test["y"])
print(f"  baseline-zero   : MAE={m_m:.4f}  RMSE={m_r:.4f}  dir-acc={m_d*100:.1f}%")
print(f"  baseline-prior  : MAE={p_m:.4f}  RMSE={p_r:.4f}  dir-acc={p_d*100:.1f}%")
print(f"  ridge           : MAE={r_m:.4f}  RMSE={r_r:.4f}  dir-acc={r_d*100:.1f}%")
print(f"  lgbm            : MAE={l_m:.4f}  RMSE={l_r:.4f}  dir-acc={l_d*100:.1f}%")
print(f"  ensemble        : MAE={e_m:.4f}  RMSE={e_r:.4f}  dir-acc={e_d*100:.1f}%")
# Empirical coverage of the conformal band on the TEST slice
in_band = ((test["y"] >= test["pi_low"]) & (test["y"] <= test["pi_high"])).mean()
band_label = test["band_label"].iloc[0]
band_target = 90.0 if "pi90" in band_label else 80.0
print(f"\n  Band in-sample on test (where cal was fit): label={band_label}, "
      f"coverage={in_band*100:.2f}% (target {band_target:.0f}%)")

print("\n" + "=" * 70)
print("(4) CONFORMAL HONESTY: holdout 2023-2024 (NEVER SEEN BY QUANTILE OR CAL)")
print("=" * 70)
hold = predict_block(panel[panel.year > 2022].dropna(subset=[TARGET]))
print(f"  rows: {len(hold)}, countries: {hold.iso3.nunique()}")
in_band_h = ((hold["y"] >= hold["pi_low"]) & (hold["y"] <= hold["pi_high"])).mean()
band_label_h = hold["band_label"].iloc[0]
band_target_h = 90.0 if "pi90" in band_label_h else 80.0
print(f"  Band empirical coverage on holdout: {in_band_h*100:.2f}% (target {band_target_h:.0f}%, label={band_label_h})")
mae_h = (hold["y"] - hold["ensemble"]).abs().mean()
dir_h = ((np.sign(hold["y"]) == np.sign(hold["ensemble"])) & (hold["y"] != 0)).mean()
print(f"  holdout MAE  = {mae_h:.4f}")
print(f"  holdout dir-acc = {dir_h*100:.1f}%")
print("\n  holdout rows:")
print(hold[["iso3","year","y","ensemble","pi_low","pi_high"]].to_string(index=False))

print("\n" + "=" * 70)
print("(5) PER-COUNTRY SPOT CHECKS")
print("=" * 70)
for iso, yr, expected in [
    ("RUS", 2020, "model should predict negative (sanctions shock ahead)"),
    ("GBR", 2020, "model should be uncertain (COVID + Brexit regime change)"),
    ("CHN", 2018, "model should predict positive (high growth period)"),
    ("USA", 2008, "model in-sample, sanity check"),
    ("DEU", 1990, "reunification shock -> should see big shock"),
    ("JPN", 1990, "lost-decade entry -> negative growth forecast"),
]:
    rows = panel[(panel.iso3 == iso) & (panel.year == yr)]
    if rows.empty:
        print(f"  {iso} {yr}: not in panel")
        continue
    pred = predict_block(rows).iloc[0]
    flag = ""
    if pred["y"] < -0.1 and expected.startswith("model in-sample"):
        flag = "  (in-sample)"
    print(f"  {iso} {yr}: y_actual={pred['y']:+.4f}  ensemble={pred['ensemble']:+.4f}  "
          f"pi=[{pred['pi_low']:+.4f}, {pred['pi_high']:+.4f}]  [{expected}]{flag}")