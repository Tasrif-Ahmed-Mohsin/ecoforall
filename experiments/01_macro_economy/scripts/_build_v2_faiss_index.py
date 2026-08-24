"""Build a GMD-shaped ranked FAISS retrieval index.

Why this script exists
----------------------
The v1 `panel_ranked.faiss` was built with a feature list that includes
v1-only columns (gini_income, real_wage_jst, trade_gdp, …). GMD doesn't
emit 17 of those 24 columns, so the v1 index is effectively useless on the
GMD panel (most rows have low n_overlap → no useful neighbors).

This script builds a NEW index (`retrieval_v2/`) whose feature columns are
the **v2 trainer's cont_cols** as persisted in
`data/features/horizon_{h}y_v2/metrics.json::feature_meta.cont_cols`
(or, if absent, the ones freshly backfilled by `_backfill_v2_cont_cols.py`).
Uses the rank-features + Euclidean recipe that won the v1 sweep.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import faiss
from scipy.stats import rankdata

from src.harmonize.common import FEATURES_DIR

PANEL = FEATURES_DIR / "panel_wide.parquet"
OUT_DIR = FEATURES_DIR / "retrieval_v2"
TARGET = "gdp_pc_growth_5y_fwd"
K = 10


def _load_v2_cont_cols() -> tuple[list[str], str]:
    """Return (cont_cols, horizon_source). cont_cols is from any v2 metrics
    that has feature_meta persisted (they're identical across horizons)."""
    for h in (5, 3, 1, 10):
        m_path = FEATURES_DIR / f"horizon_{h}y_v2" / "metrics.json"
        if not m_path.exists():
            continue
        m = json.loads(m_path.read_text())
        cont = m.get("feature_meta", {}).get("cont_cols", [])
        if cont:
            return cont, f"horizon_{h}y_v2"
    raise SystemExit(
        "No v2 feature_meta.cont_cols found. Run `python scripts\\_backfill_v2_cont_cols.py` first."
    )


def _fit_transform(df: pd.DataFrame, cols: list[str]):
    """Rank-features -> z-score. Returns (mat, mask, mu, sigma, sorted_vals_per_col, transform_fn)."""
    sub = df[cols].astype(float).replace([np.inf, -np.inf], np.nan)
    n_rows, n_cols = sub.shape
    ranks = np.zeros((n_rows, n_cols), dtype=np.float32)
    sorted_vals = {}
    for j, c in enumerate(cols):
        v = sub[c].to_numpy(dtype=np.float64)
        obs = ~np.isnan(v)
        if obs.any():
            ranks[obs, j] = (rankdata(v[obs], method="average") - 1.0) / obs.sum()
        sorted_vals[c] = np.sort(v[obs]) if obs.any() else np.array([], dtype=np.float64)
    mu = ranks.mean(axis=0).astype(np.float64)
    sigma = ranks.std(axis=0).astype(np.float64)
    sigma = np.where(sigma < 1e-9, 1.0, sigma)
    mat = ((ranks - mu) / sigma).astype(np.float32)
    mask = sub.notna().astype(np.uint8).to_numpy()

    def transform_fn(q: pd.Series) -> np.ndarray:
        z = np.zeros((1, n_cols), dtype=np.float32)
        for j, c in enumerate(cols):
            v = q.get(c, np.nan)
            if pd.isna(v):
                z[0, j] = 0.0
                continue
            arr = sorted_vals[c]
            if len(arr) == 0:
                z[0, j] = 0.0
                continue
            frac = float((arr <= float(v)).mean())
            z[0, j] = float((frac - mu[j]) / sigma[j])
        return np.ascontiguousarray(z, dtype=np.float32)

    return mat, mask, mu.astype(np.float32), sigma.astype(np.float32), sorted_vals, transform_fn


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cont_cols, src = _load_v2_cont_cols()
    print(f"[v2-faiss] cont_cols source: {src}")
    print(f"[v2-faiss] n cont_cols: {len(cont_cols)}")

    panel = pd.read_parquet(PANEL)
    print(f"[v2-faiss] panel shape: {panel.shape}")

    # Columns-membership sanity check
    missing = [c for c in cont_cols if c not in panel.columns]
    if missing:
        print(f"[v2-faiss] WARN: {len(missing)} cont_cols missing from panel (will be NaN-only): "
              f"{missing[:10]}{'...' if len(missing) > 10 else ''}")

    indexed = panel.dropna(subset=[TARGET]).copy()
    rows = indexed[["iso3", "year"]].reset_index(drop=True)
    rows[TARGET] = indexed[TARGET].to_numpy()
    print(f"[v2-faiss] indexed rows: {len(rows):,}")

    # Use ONLY cont_cols that actually exist in the panel
    usable_cols = [c for c in cont_cols if c in panel.columns]
    print(f"[v2-faiss] using {len(usable_cols)} / {len(cont_cols)} cont_cols (rest are NaN-only)")

    mat, mask, mu, sigma, sorted_vals, transform_fn = _fit_transform(indexed, usable_cols)
    mat = np.ascontiguousarray(mat, dtype=np.float32)
    index = faiss.IndexFlatL2(mat.shape[1])  # Euclidean
    index.add(mat)

    # Persist
    faiss.write_index(index, str(OUT_DIR / "panel_ranked.faiss"))
    np.save(OUT_DIR / "mu.npy", mu)
    np.save(OUT_DIR / "sigma.npy", sigma)
    np.save(OUT_DIR / "mask.npy", mask)
    rows.to_parquet(OUT_DIR / "rows.parquet", index=False)
    (OUT_DIR / "cols.json").write_text(json.dumps(usable_cols))
    np.savez(OUT_DIR / "sorted_vals.npz", **{c: v for c, v in sorted_vals.items()})

    # Quick smoke test on USA 2023 (or nearest country-year present)
    try:
        q = panel[(panel.iso3 == "USA") & (panel.year == 2023)].iloc[0]
    except IndexError:
        q = indexed.iloc[0]
    z = transform_fn(q)
    sims, ids = index.search(z, K)
    print(f"\n[v2-faiss] SMOKE TEST (query iso3={q.get('iso3')!r} year={q.get('year')!r}):")
    print(f"[v2-faiss] top-{K} n_overlap values: "
          f"{[(rows.iloc[i]['iso3'], int(rows.iloc[i]['year']), int((mask[ids[0][i]] & z[0].astype(bool)).sum())) for i in range(K)]}")

    print(f"\n[v2-faiss] wrote index to {OUT_DIR / 'panel_ranked.faiss'}")
    print(f"[v2-faiss] ({index.ntotal:,} rows, {mat.shape[1]} features)")


if __name__ == "__main__":
    main()
