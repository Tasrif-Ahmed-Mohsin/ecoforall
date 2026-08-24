"""Evaluate the pattern finder end-to-end.

The validation question is: when we retrieve the top-k most-similar historical
country-years to a query country-year, does the *realized 5y growth* of those
neighbors actually cluster around the realized growth of the query?

Two tests:

  (A) Self-prediction test
      For every labelled (iso3, year) in the test slice (>= 2015), look up its
      top-20 analogs from TRAIN years only (excluding the country itself).
      Compute the median neighbor growth vs the realized growth and report
      median and mean absolute error.

  (B) Direction test
      For every labelled row, ask: do >50% of neighbors agree with the query
      on the sign of growth?

This is the right evaluation for a retrieval system that is meant to surface
historical situations similar to a current one — not a forecast accuracy
metric, but a "do the patterns look like the patterns" metric.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json
import numpy as np
import pandas as pd

from src.harmonize.common import FEATURES_DIR
from src.retrieval.faiss_index import build_or_load

PANEL = FEATURES_DIR / "panel_wide.parquet"
TARGET = "gdp_pc_growth_5y_fwd"
TRAIN_END = 2014

K_NEIGHBORS = 20


def _load_panel() -> pd.DataFrame:
    return pd.read_parquet(PANEL)


def main() -> None:
    panel = _load_panel()
    labelled = panel.dropna(subset=[TARGET]).copy()
    labelled["year"] = labelled["year"].astype(int)

    # Use only test/holdout years as queries (>= 2015) so the index doesn't include the query itself.
    queries = labelled[labelled["year"] >= 2015].reset_index(drop=True)
    print(f"[eval] queries: {len(queries)} rows, "
          f"{queries.iso3.nunique()} countries, "
          f"years {queries.year.min()}-{queries.year.max()}")
    print(f"[eval] each query retrieves top-{K_NEIGHBORS} analogs via FAISS")

    index = build_or_load()
    # Filter the underlying index rows to TRAIN-only so neighbors are honest.
    train_rows_mask = panel.iloc[index.rows.index] if False else None  # not used; we use FAISS rows df
    indexed = index.rows.copy()
    train_ids = set(indexed[indexed.year <= TRAIN_END].index)
    print(f"[eval] FAISS index has {len(indexed)} rows; "
          f"{len(train_ids)} are train-period (<= {TRAIN_END})")

    # Per-query neighborhood stats.
    rows = []
    abs_errs = []
    dir_matches = []
    queries_with_neighbors = 0
    n_rows = len(queries)
    print(f"[eval] running {n_rows} queries …")
    for i, q in queries.iterrows():
        # Over-fetch, then filter to TRAIN-only and exclude the query country.
        nb = index.query_topk(q, k=K_NEIGHBORS * 4, exclude_iso3=q.iso3, exclude_year=q.year)
        nb = nb[nb.year <= TRAIN_END]
        # Drop any row with a NaN target.
        nb = nb.dropna(subset=[TARGET]).head(K_NEIGHBORS)
        if len(nb) < 3:
            continue
        queries_with_neighbors += 1
        nb_growth = nb[TARGET].to_numpy()
        q_growth = float(q[TARGET])
        median_growth = float(np.median(nb_growth))
        mean_growth = float(np.mean(nb_growth))
        ae = abs(median_growth - q_growth)
        abs_errs.append(ae)
        sign_q = np.sign(q_growth)
        nb_signs = np.sign(nb_growth)
        # Sign agreement: fraction of neighbors whose sign matches the query.
        if sign_q == 0:
            dir_match = float((nb_signs == 0).mean())
        else:
            dir_match = float((nb_signs == sign_q).mean())
        dir_matches.append(dir_match)
        rows.append({
            "iso3": q.iso3, "year": int(q.year),
            "q_growth": q_growth,
            "n_neighbors": len(nb),
            "neighbor_median": median_growth,
            "neighbor_mean": mean_growth,
            "neighbor_min": float(np.min(nb_growth)),
            "neighbor_max": float(np.max(nb_growth)),
            "dir_match_pct": dir_match * 100,
            "abs_err": ae,
        })
    out = pd.DataFrame(rows)
    print(f"\n[eval] queries with >=3 train neighbors: {queries_with_neighbors}")
    if len(out) == 0:
        return
    print(f"[eval] cross-row abs error stats:")
    print(f"  median absolute error (median neighbor vs realized): {np.median(abs_errs):.4f}")
    print(f"  mean   absolute error                              : {np.mean(abs_errs):.4f}")
    print(f"  RMSE                                            : {np.sqrt(np.mean(np.array(abs_errs)**2)):.4f}")
    print(f"[eval] sign agreement (fraction of neighbors matching query's sign of growth):")
    print(f"  median dir-match    = {np.median(dir_matches)*100:.1f}%")
    print(f"  mean   dir-match    = {np.mean(dir_matches)*100:.1f}%")
    print(f"  share of queries with >=50% sign match: {(np.array(dir_matches) >= 0.5).mean()*100:.1f}%")
    print(f"  share of queries with >=75% sign match: {(np.array(dir_matches) >= 0.75).mean()*100:.1f}%")
    # Distribution buckets.
    print(f"\n[eval] distribution of absolute errors:")
    bins = [0, 0.05, 0.10, 0.25, 0.50, 1.0, np.inf]
    counts, _ = np.histogram(abs_errs, bins=bins)
    for b, c, bn in zip(bins[:-1], counts, bins[1:]):
        print(f"  |err| in [{b:.2f}, {bn:.2f}): {c} queries ({c/len(abs_errs)*100:.1f}%)")

    # Save the per-query table for inspection.
    out_path = FEATURES_DIR / "pattern_eval.parquet"
    out.to_parquet(out_path, index=False)
    print(f"\n[eval] wrote {out_path}")

    # 10 hardest (largest |err|) and 10 easiest (smallest |err|).
    print("\n[eval] 10 EASIEST queries (median neighbor growth closest to query's):")
    easiest = out.nsmallest(10, "abs_err")
    for _, r in easiest.iterrows():
        print(f"  {r.iso3} {r.year}: realized={r.q_growth:+.4f}  "
              f"median(nb)={r.neighbor_median:+.4f}  "
              f"|err|={r.abs_err:.4f}  sign-match={r.dir_match_pct:.0f}%")
    print("\n[eval] 10 HARDEST queries:")
    hardest = out.nlargest(10, "abs_err")
    for _, r in hardest.iterrows():
        print(f"  {r.iso3} {r.year}: realized={r.q_growth:+.4f}  "
              f"median(nb)={r.neighbor_median:+.4f}  "
              f"|err|={r.abs_err:.4f}  sign-match={r.dir_match_pct:.0f}%")


if __name__ == "__main__":
    main()