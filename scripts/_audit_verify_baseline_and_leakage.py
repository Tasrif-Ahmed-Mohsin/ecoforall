"""
INDEPENDENT AUDIT VERIFICATION (read-only diagnostic; writes nothing to data/benchmarks)
=======================================================================================
Tests two suspected defects in scripts/run_real_multidomain_benchmark.py:

  D1. AR(1) baseline train/test regressor mismatch
      fit()  builds growth_lag = pct_change(gdp_pc_real)      [mean 0.020, sd 0.067]
      predict_panel() is passed lag_growth_col="gdp_pc_real_logret5" [mean 0.085, sd 0.194]
      -> the fitted rho_i is applied to a different variable at test time.

  D2. DMS router consumes test-fold targets
      route_panel(te_df, specialist_matrix, y_te) updates weights with y_te[i].
      At origin t the target y_{t+h} is unrealised for h-1 further years, so weights
      at origin t use information from up to h-1 years in the future.

Reproduces the same folds/features as the original script.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.reproducibility import seed_everything
from src.gating.sovereign_segmentation_router import get_wb_region
from src.gating.dms_state_space_router import DynamicModelSelectionRouter
from src.models.macro_baselines import PerCountryARForecaster, EqualWeightCombinationForecaster

FOLDS = [
    {"test_start": 2019, "test_end": 2024},
    {"test_start": 2015, "test_end": 2018},
    {"test_start": 2011, "test_end": 2014},
    {"test_start": 2007, "test_end": 2010},
    {"test_start": 2000, "test_end": 2006},
]


def horizon_matched_ar(tr_df, te_df, target_col, h):
    """Correct per-country AR: regress y_{t+h} on the SAME lag variable at train and test.

    Uses growth into the origin year (realised at t, no leakage), identical column
    for fit and predict, with empirical-Bayes shrinkage to the pooled estimate.
    """
    lagcol = "growth_into_t"
    tr = tr_df.dropna(subset=[target_col, lagcol])
    if len(tr) < 20:
        return np.full(len(te_df), 0.02 * h)
    Xg = np.column_stack([np.ones(len(tr)), tr[lagcol].values])
    bg, *_ = np.linalg.lstsq(Xg, tr[target_col].values, rcond=None)
    g_a, g_r = float(bg[0]), float(bg[1])

    params = {}
    for iso, grp in tr.groupby("iso3"):
        if len(grp) >= 5:
            Xc = np.column_stack([np.ones(len(grp)), grp[lagcol].values])
            try:
                bc, *_ = np.linalg.lstsq(Xc, grp[target_col].values, rcond=None)
                n = len(grp)
                w = n / (n + 2.0)
                params[iso] = (w * float(bc[0]) + (1 - w) * g_a,
                               w * float(bc[1]) + (1 - w) * g_r)
            except Exception:
                params[iso] = (g_a, g_r)
        else:
            params[iso] = (g_a, g_r)

    out = []
    for iso, lag in zip(te_df["iso3"].values, te_df[lagcol].values):
        a, r = params.get(iso, (g_a, g_r))
        x = lag if np.isfinite(lag) else 0.02
        out.append(a + r * x)
    return np.clip(np.array(out, dtype=np.float64), -0.5, 0.5)


def main():
    seed_everything(42)
    df = pd.read_parquet(ROOT / "data" / "processed_panels" / "real_cross_domain_annual_panel.parquet")
    df["region_wb"] = df["iso3"].apply(get_wb_region)
    df = df.sort_values(["iso3", "year"]).copy()
    # growth realised INTO the origin year t: available at t, no leakage
    df["growth_into_t"] = df.groupby("iso3")["gdp_pc_real"].pct_change(fill_method=None)

    meta = {"iso3", "country", "year", "region", "income_level", "region_wb", "growth_into_t"}
    allf = [c for c in df.columns if c not in meta and not c.endswith("_fwd")]
    eco = [c for c in allf if not c.startswith("vdem_") and not c.startswith("climate_")]
    pol = [c for c in allf if c.startswith("vdem_")]
    cli = [c for c in allf if c.startswith("climate_")]

    print(f"features: total={len(allf)} eco={len(eco)} pol={len(pol)} cli={len(cli)}\n")
    rows = []

    for h in (1, 3, 5):
        tcol = f"gdp_pc_growth_{h}y_fwd"
        clean = df.dropna(subset=[tcol]).sort_values(["iso3", "year"]).copy()
        parts = []

        for fi, f in enumerate(FOLDS):
            ts, te_ = f["test_start"], f["test_end"]
            tr_df = clean[clean["year"] <= ts - h - 1].copy()
            te_df = clean[(clean["year"] >= ts) & (clean["year"] <= te_)].copy()
            if len(tr_df) == 0 or len(te_df) == 0:
                continue
            y_tr = tr_df[tcol].values.astype(np.float64)
            y_te = te_df[tcol].values.astype(np.float64)

            def prep(colset):
                im, sc = SimpleImputer(strategy="median"), StandardScaler()
                A = np.array(tr_df[colset].values, dtype=np.float64)
                B = np.array(te_df[colset].values, dtype=np.float64)
                A[~np.isfinite(A)] = np.nan
                B[~np.isfinite(B)] = np.nan
                return (np.clip(sc.fit_transform(im.fit_transform(A)), -5, 5),
                        np.clip(sc.transform(im.transform(B)), -5, 5),
                        im)

            Xtr_e, Xte_e, im_e = prep(eco)
            Xtr_p, Xte_p, _ = prep(pol)
            Xtr_c, Xte_c, _ = prep(cli)

            # --- AR baselines: as-shipped vs horizon-matched (correct) ---
            arm = PerCountryARForecaster(horizon=h)
            arm.fit(tr_df, target_col=tcol, lag_growth_col="growth_into_t")
            te_df["ar_shipped"] = np.clip(
                arm.predict_panel(te_df, lag_growth_col="growth_into_t"), -0.5, 0.5)
            te_df["ar_fixed"] = horizon_matched_ar(tr_df, te_df, tcol, h)

            # random-walk / no-change reference
            te_df["rw"] = np.clip(np.nan_to_num(te_df["growth_into_t"].values, nan=0.02) * h, -0.5, 0.5)

            for name, Xtr, Xte in (("eco_ridge", Xtr_e, Xte_e),
                                   ("pol_ridge", Xtr_p, Xte_p),
                                   ("cli_ridge", Xtr_c, Xte_c)):
                m = Ridge(alpha=100.0, random_state=42).fit(Xtr, y_tr)
                te_df[name] = np.clip(m.predict(Xte), -0.5, 0.5)

            gl = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.03, max_depth=5,
                                   random_state=42, verbose=-1, n_jobs=-1)
            gl.fit(im_e.transform(np.where(np.isfinite(tr_df[eco].values.astype(float)),
                                           tr_df[eco].values.astype(float), np.nan)), y_tr)
            te_df["eco_lgbm"] = np.clip(gl.predict(im_e.transform(
                np.where(np.isfinite(te_df[eco].values.astype(float)),
                         te_df[eco].values.astype(float), np.nan))), -0.5, 0.5)

            # --- DMS: with test targets (as shipped) vs without (honest) ---
            M_shipped = np.column_stack([te_df["ar_shipped"], te_df["eco_ridge"],
                                         te_df["pol_ridge"], te_df["cli_ridge"], te_df["eco_lgbm"]])
            r1 = DynamicModelSelectionRouter(n_experts=5, forgetting_factor=0.92, mode="dma")
            te_df["dms_leaky"], _ = r1.route_panel(te_df, M_shipped, y_te)
            r2 = DynamicModelSelectionRouter(n_experts=5, forgetting_factor=0.92, mode="dma")
            te_df["dms_nofeedback"], _ = r2.route_panel(te_df, M_shipped, None)

            M_fixed = np.column_stack([te_df["ar_fixed"], te_df["eco_ridge"],
                                       te_df["pol_ridge"], te_df["cli_ridge"], te_df["eco_lgbm"]])
            r3 = DynamicModelSelectionRouter(n_experts=5, forgetting_factor=0.92, mode="dma")
            te_df["dms_leaky_fixedexp"], _ = r3.route_panel(te_df, M_fixed, y_te)
            r4 = DynamicModelSelectionRouter(n_experts=5, forgetting_factor=0.92, mode="dma")
            te_df["dms_honest_fixedexp"], _ = r4.route_panel(te_df, M_fixed, None)

            te_df["eqw_shipped"] = EqualWeightCombinationForecaster.combine(
                [M_shipped[:, j] for j in range(5)])
            te_df["eqw_fixed"] = EqualWeightCombinationForecaster.combine(
                [M_fixed[:, j] for j in range(5)])
            parts.append(te_df)

        comb = pd.concat(parts, ignore_index=True)
        yt = comb[tcol].values
        mae = lambda c: float(np.mean(np.abs(yt - comb[c].values)))
        base_shipped, base_fixed = mae("ar_shipped"), mae("ar_fixed")

        print(f"===== h = {h}   (N = {len(yt)} test country-years) =====")
        print(f"  target sd = {np.std(yt):.5f}   mean|y| = {np.mean(np.abs(yt)):.5f}")
        print(f"  {'model':<26} {'MAE':>9} {'vs AR-shipped':>15} {'vs AR-FIXED':>13}")
        for c in ["ar_shipped", "ar_fixed", "rw", "eco_ridge", "pol_ridge", "cli_ridge",
                  "eco_lgbm", "eqw_shipped", "eqw_fixed", "dms_leaky", "dms_nofeedback",
                  "dms_leaky_fixedexp", "dms_honest_fixedexp"]:
            m = mae(c)
            print(f"  {c:<26} {m:9.5f} {100*(base_shipped-m)/base_shipped:14.2f}% "
                  f"{100*(base_fixed-m)/base_fixed:12.2f}%")
            rows.append({"h": h, "model": c, "MAE": round(m, 5),
                         "lift_vs_AR_shipped": round(100*(base_shipped-m)/base_shipped, 2),
                         "lift_vs_AR_fixed": round(100*(base_fixed-m)/base_fixed, 2)})
        print()

    pd.DataFrame(rows).to_csv(ROOT / "data" / "benchmarks" / "_audit_verification.csv", index=False)
    print("diagnostic written to data/benchmarks/_audit_verification.csv")


if __name__ == "__main__":
    main()
