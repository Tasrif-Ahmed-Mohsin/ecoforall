"""Year-clustered vs pooled forecast inference (diagnostic only).

The tournament reports Diebold-Mariano stats of 11-35 and Clark-West p-values of
0.0 computed on POOLED country-years. With ~170 sovereigns sharing each year's
global shocks, the pooled standard error ignores cross-sectional dependence.
This recomputes the same comparisons with year-clustered SEs (Driscoll-Kraay
style: average the loss differential within each year, then test across years).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm, t as tdist
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.utils.reproducibility import seed_everything
from src.models.macro_baselines import PerCountryARForecaster

FOLDS = [(2019, 2024), (2015, 2018), (2011, 2014), (2007, 2010), (2000, 2006)]


def dm_pooled(y, p1, p2):
    d = np.abs(y - p1) - np.abs(y - p2)
    s = np.std(d, ddof=1) / np.sqrt(len(d))
    return float(np.mean(d) / s)


def dm_year_clustered(y, p1, p2, years):
    d = np.abs(y - p1) - np.abs(y - p2)
    g = pd.DataFrame({"d": d, "yr": years}).groupby("yr")["d"].mean().values
    G = len(g)
    stat = float(np.mean(g) / (np.std(g, ddof=1) / np.sqrt(G)))
    p = float(2 * (1 - tdist.cdf(abs(stat), df=G - 1)))
    return stat, p, G


def main():
    seed_everything(42)
    df = pd.read_parquet(ROOT / "data" / "processed_panels" / "real_cross_domain_annual_panel.parquet")
    df = df.sort_values(["iso3", "year"]).copy()
    df["growth_into_t"] = df.groupby("iso3")["gdp_pc_real"].pct_change(fill_method=None)
    meta = {"iso3", "country", "year", "region", "income_level", "growth_into_t"}
    allf = [c for c in df.columns if c not in meta and not c.endswith("_fwd")]
    eco = [c for c in allf if not c.startswith("vdem_") and not c.startswith("climate_")]

    print(f"{'h':>2} {'comparison':<34} {'POOLED DM':>10} {'YEAR-CLUSTERED DM':>19} {'p':>8} {'#yrs':>5}")
    print("-" * 84)
    for h in (1, 3, 5):
        tcol = f"gdp_pc_growth_{h}y_fwd"
        clean = df.dropna(subset=[tcol]).copy()
        parts = []
        for ts, te_ in FOLDS:
            tr = clean[clean["year"] <= ts - h - 1]
            te = clean[(clean["year"] >= ts) & (clean["year"] <= te_)].copy()
            if not len(tr) or not len(te):
                continue
            ytr = tr[tcol].values.astype(float)
            im, sc = SimpleImputer(strategy="median"), StandardScaler()
            A = np.array(tr[eco].values, float); A[~np.isfinite(A)] = np.nan
            B = np.array(te[eco].values, float); B[~np.isfinite(B)] = np.nan
            Xtr = np.clip(sc.fit_transform(im.fit_transform(A)), -5, 5)
            Xte = np.clip(sc.transform(im.transform(B)), -5, 5)

            am = PerCountryARForecaster(horizon=h)
            am.fit(tr, target_col=tcol, lag_growth_col="growth_into_t")
            te["ar_shipped"] = np.clip(am.predict_panel(te, lag_growth_col="growth_into_t"), -.5, .5)


            # horizon-matched AR on the same lag column at train and test
            trv = tr.dropna(subset=[tcol, "growth_into_t"])
            Xg = np.column_stack([np.ones(len(trv)), trv["growth_into_t"].values])
            bg, *_ = np.linalg.lstsq(Xg, trv[tcol].values, rcond=None)
            pr = {}
            for iso, gp in trv.groupby("iso3"):
                if len(gp) >= 5:
                    Xc = np.column_stack([np.ones(len(gp)), gp["growth_into_t"].values])
                    bc, *_ = np.linalg.lstsq(Xc, gp[tcol].values, rcond=None)
                    w = len(gp) / (len(gp) + 2.0)
                    pr[iso] = (w * bc[0] + (1 - w) * bg[0], w * bc[1] + (1 - w) * bg[1])
            te["ar_fixed"] = np.clip([
                (pr.get(i, (bg[0], bg[1]))[0] + pr.get(i, (bg[0], bg[1]))[1] * (x if np.isfinite(x) else .02))
                for i, x in zip(te["iso3"], te["growth_into_t"])], -.5, .5)

            te["eco_ridge"] = np.clip(Ridge(alpha=100., random_state=42).fit(Xtr, ytr).predict(Xte), -.5, .5)
            g = lgb.LGBMRegressor(n_estimators=150, learning_rate=.03, max_depth=5,
                                  random_state=42, verbose=-1, n_jobs=-1)
            g.fit(im.transform(A), ytr)
            te["eco_lgbm"] = np.clip(g.predict(im.transform(B)), -.5, .5)
            parts.append(te)

        c = pd.concat(parts, ignore_index=True)
        y, yr = c[tcol].values, c["year"].values
        for lbl, base, cand in [
            ("eco_ridge vs AR-shipped (as published)", "ar_shipped", "eco_ridge"),
            ("eco_ridge vs AR-FIXED", "ar_fixed", "eco_ridge"),
            ("eco_lgbm  vs AR-FIXED", "ar_fixed", "eco_lgbm"),
            ("eco_lgbm  vs eco_ridge", "eco_ridge", "eco_lgbm"),
        ]:
            dp = dm_pooled(y, c[base].values, c[cand].values)
            dc, pc, G = dm_year_clustered(y, c[base].values, c[cand].values, yr)
            star = "***" if pc < .01 else ("**" if pc < .05 else ("*" if pc < .1 else " ns"))
            print(f"{h:>2} {lbl:<34} {dp:>10.2f} {dc:>19.2f} {pc:>8.4f}{star} {G:>5}")
        print()


if __name__ == "__main__":
    main()
