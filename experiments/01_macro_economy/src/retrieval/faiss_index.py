"""FAISS-backed retrieval of similar historical country-year situations.

Design
------
- Vectors: standardized feature columns from panel_wide.parquet.
- Missing values are imputed with column means during indexing; the *same*
  mean is reused at query time so retrieval is comparable across rows.
- Distances use L2 (smaller = closer). The ranked outputs include:
    iso3, year, y_true (the realized 5y growth), distance, n_overlap
  `n_overlap` is the number of non-missing feature columns the candidate
  shares with the query — useful for dismissing self-neighbors caused by
  macro-finance features that only exist for ~18 countries.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import faiss
import numpy as np
import pandas as pd

try:
    import yaml  # PyYAML; only required when --weights is used.
except ImportError:  # pragma: no cover - allow retrieval to run without PyYAML.
    yaml = None


def _load_weights(path: str | Path | None) -> dict[str, float]:
    """Load expert-nudged feature weights from a YAML file.

    Returns an empty dict (= equal weights) when path is None or the
    file does not exist. Warns on unknown keys so typos don't silently
    drop features to weight=1.0.
    """
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        print(f"[faiss_index] weights file not found: {p} — using equal weights",
              file=sys.stderr)
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML is required for --weights; install with `pip install pyyaml`")
    raw = yaml.safe_load(p.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"weights file {p} must be a YAML mapping of {{feature: weight}}")
    return {str(k): float(v) for k, v in raw.items()}

from src.harmonize.common import FEATURES_DIR
from src.features.build_panel import build  # noqa: F401  (re-exported if you want to rebuild)

PANEL = FEATURES_DIR / "panel_wide.parquet"
INDEX_DIR = FEATURES_DIR / "retrieval"
DEFAULT_K = 10


def _load_panel() -> pd.DataFrame:
    return pd.read_parquet(PANEL)


def _standardize(df: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean-impute NaN, then z-score per column. Return (matrix, mu, sigma)."""
    sub = df[cols].astype(float).replace([np.inf, -np.inf], np.nan)
    mu_s = sub.mean(skipna=True).fillna(0.0)
    sigma_s = sub.std(skipna=True).replace(0, np.nan).fillna(1.0)
    sub_filled = sub.fillna(mu_s)
    mu = mu_s.to_numpy()
    sigma = sigma_s.to_numpy()
    z = (sub_filled.to_numpy() - mu) / sigma
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return z, mu.astype(np.float32), sigma.astype(np.float32)


class FaissIndex:
    """Wraps a FAISS Flat-L2 index over a panel of (iso3, year) rows."""

    def __init__(self):
        self.index: faiss.IndexFlatL2 | None = None
        self.rows: pd.DataFrame | None = None        # aligned rows (iso3, year, y_true, n_overlap)
        self.mu: np.ndarray | None = None
        self.sigma: np.ndarray | None = None
        self.cols: list[str] = []

    @property
    def path(self) -> Path:
        return INDEX_DIR / "panel.faiss"

    def build(self, panel: pd.DataFrame | None = None,
              target_col: str = "gdp_pc_growth_5y_fwd",
              exclude_cols: tuple[str, ...] = ("iso3", "year", "gdp_pc", "gdp_pc_real")) -> "FaissIndex":
        """Build the index from the panel (or from disk if panel is None)."""
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        if panel is None:
            panel = _load_panel()
        # Feature columns: numeric, non-leakage.
        feat_cols = [
            c for c in panel.columns
            if c not in set(exclude_cols) | {"gdp_pc_growth_5y_fwd"}
            and pd.api.types.is_numeric_dtype(panel[c])
        ]
        # Only index rows that have a defined target, so retrieval reports are useful.
        indexed = panel.dropna(subset=[target_col]).copy()
        rows = indexed[["iso3", "year"]].reset_index(drop=True)
        rows[target_col] = indexed[target_col].to_numpy()

        mat, mu, sigma = _standardize(indexed, feat_cols)

        # Build using inner-product on the L2-normalized vectors (== cosine) which is
        # more robust when countries have sparser feature coverage than USA.
        mat = np.ascontiguousarray(mat, dtype=np.float32)
        faiss.normalize_L2(mat)
        index = faiss.IndexFlatIP(mat.shape[1])
        index.add(mat)

        # Persist a numpy copy of every row's per-feature mask (1 if observed, 0 if NaN).
        mask = indexed[feat_cols].notna().astype(np.uint8).to_numpy()

        self.index = index
        self.rows = rows
        self.mu = mu
        self.sigma = sigma
        self.cols = feat_cols
        self.mask = mask
        self.target_col = target_col
        self._save()
        return self

    def _save(self) -> None:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.path))
        np.save(INDEX_DIR / "mu.npy", self.mu)
        np.save(INDEX_DIR / "sigma.npy", self.sigma)
        np.save(INDEX_DIR / "mask.npy", self.mask)
        self.rows.to_parquet(INDEX_DIR / "rows.parquet", index=False)
        (INDEX_DIR / "cols.json").write_text(json.dumps(self.cols))

    @classmethod
    def load(cls) -> "FaissIndex":
        obj = cls()
        obj.index = faiss.read_index(str(obj.path))
        obj.mu = np.load(INDEX_DIR / "mu.npy")
        obj.sigma = np.load(INDEX_DIR / "sigma.npy")
        obj.mask = np.load(INDEX_DIR / "mask.npy")
        obj.rows = pd.read_parquet(INDEX_DIR / "rows.parquet")
        obj.cols = json.loads((INDEX_DIR / "cols.json").read_text())
        obj.target_col = "gdp_pc_growth_5y_fwd"
        return obj

    def _vectorize_query(self, panel_row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """Standardize a single panel_row into a (1, D) array using stored mu/sigma."""
        cols = self.cols
        x = panel_row[cols].astype(float).to_frame().T.replace([np.inf, -np.inf], np.nan)
        mu_s = pd.Series(self.mu, index=cols)
        x_imp = x.fillna(mu_s).to_numpy()
        z = (x_imp - self.mu) / self.sigma
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        q_mask = x.notna().astype(np.uint8).to_numpy()[0]  # (D,)
        z = np.ascontiguousarray(z, dtype=np.float32)
        faiss.normalize_L2(z)
        return z, q_mask

    def query_topk(
        self,
        panel_row: pd.Series,
        k: int = DEFAULT_K,
        exclude_year: int | None = None,
        exclude_iso3: str | None = None,
    ) -> pd.DataFrame:
        """Return top-k most-similar historical rows by cosine similarity.

        n_overlap counts how many features the candidate also actually observed
        (not NaN) — useful because countries with sparse features cluster to
        noisy neighbors.
        """
        q, q_mask = self._vectorize_query(panel_row)
        # Overfetch so we have headroom after the optional filters.
        k_fetch = min(max(k * 10, k + 50), len(self.rows))
        sims, ids = self.index.search(q, k_fetch)
        sims, ids = sims[0], ids[0]
        rows = self.rows.iloc[ids].reset_index(drop=True)
        # n_overlap: candidate mask AND query mask (intersection of features observed by both).
        inter = (self.mask[ids] & q_mask).sum(axis=1)
        # Convert IP similarity to a "distance-like" score: 1 - sim. Larger = worse.
        rows["similarity"] = sims
        rows["n_overlap"] = inter
        rows["distance"] = 1.0 - sims
        if exclude_year is not None:
            rows = rows[rows["year"] != exclude_year]
        if exclude_iso3 is not None:
            rows = rows[rows["iso3"] != exclude_iso3]
        return rows.head(k).reset_index(drop=True)


def build_or_load() -> FaissIndex:
    p = INDEX_DIR / "panel.faiss"
    if p.exists():
        return FaissIndex.load()
    return FaissIndex().build()


# ----------------------------------------------------------------------
# Rank-features + Euclidean index (the "best pattern detection" setup).
# Selected after sweeping 6 feature sets x 3 transforms x 2 metrics
# (see scripts/_pattern_sweep.py / _pattern_sweep2.py). Rank-features
# are outlier-immune (LBN, VEN no longer pull the centroid), and
# Euclidean is consistently stronger than cosine when the matrix is
# already on a comparable scale (rank CDF -> z-score).
# ----------------------------------------------------------------------

WELL_COVERED_RAW = {
    "gdp_growth", "inflation_rate", "inflation_cpi", "cpi", "real_wage",
    "real_wage_jst", "short_rate", "long_rate", "real_interest_rate",
    "gov_debt_gdp", "gov_balance_dom", "current_account_gdp", "trade_gdp",
    "bank_debt", "total_loans", "social_spending", "unemployment_rate",
    "gini_income", "gini_income_wb", "gini_wealth",
    "equity_total_return", "equity_capital_gain", "equity_div_yield",
    "housing_capital_gain",
}


def _ranked_feat_cols(panel: pd.DataFrame) -> list[str]:
    engineered = [c for c in panel.columns
                  if c.endswith(("_lag1", "_lag5", "_roll5_mean", "_delta5", "_logret5"))]
    panel_cols = set(engineered)
    return sorted((panel_cols | WELL_COVERED_RAW) & set(panel.columns))


class RankedFaissIndex:
    """Cosine/Euclidean index over rank-transformed, z-scored features.

    Optional expert-nudged weights (a dict {feature: multiplier}) are
    applied multiplicatively to the z-scored matrix before L2. Weights
    are baked in at build time so the persisted FAISS index already
    encodes them; the same weight vector is reused at query time via
    `_vectorize_query`.
    """

    def __init__(self):
        self.index: faiss.Index | None = None
        self.rows: pd.DataFrame | None = None
        self.cols: list[str] = []
        self.mu: np.ndarray | None = None      # per-column rank-CDF mean
        self.sigma: np.ndarray | None = None   # per-column rank-CDF std
        self.w: np.ndarray | None = None       # per-column expert multiplier
        self.sorted_vals: dict[str, np.ndarray] = {}  # per-col sorted values for query-time CDF
        self.mask: np.ndarray | None = None    # (N, D) uint8, 1 if observed

    @property
    def path(self) -> Path:
        return INDEX_DIR / "panel_ranked.faiss"

    def build(self, panel: pd.DataFrame | None = None,
              target_col: str = "gdp_pc_growth_5y_fwd",
              metric: str = "euclidean",
              weights: dict[str, float] | None = None,
              weights_path: str | Path | None = None) -> "RankedFaissIndex":
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        if panel is None:
            panel = pd.read_parquet(PANEL)
        if weights is None:
            weights = _load_weights(weights_path)
        self.cols = _ranked_feat_cols(panel)
        indexed = panel.dropna(subset=[target_col]).copy()
        rows = indexed[["iso3", "year"]].reset_index(drop=True)
        rows[target_col] = indexed[target_col].to_numpy()

        sub = indexed[self.cols].astype(float).replace([np.inf, -np.inf], np.nan)
        self.mask = sub.notna().astype(np.uint8).to_numpy()

        # Rank each column to its empirical CDF, then z-score across rows.
        ranks = np.empty(sub.shape, dtype=np.float32)
        self.sorted_vals = {}
        for j, c in enumerate(self.cols):
            v = sub[c].to_numpy(dtype=np.float64)
            obs = ~np.isnan(v)
            if obs.any():
                ranks[obs, j] = (pd.Series(v[obs]).rank(method="average").to_numpy() - 1) / obs.sum()
            self.sorted_vals[c] = np.sort(v[obs]) if obs.any() else np.array([], dtype=np.float64)
        self.mu = ranks.mean(axis=0).astype(np.float64)
        self.sigma = ranks.std(axis=0).astype(np.float64)
        self.sigma[self.sigma < 1e-9] = 1.0
        mat = ((ranks - self.mu) / self.sigma).astype(np.float32)

        # Apply expert-nudged weights: w[c] multiplies column c of the
        # z-scored matrix. Unknown keys default to 1.0 (no change).
        if weights:
            w = np.array([weights.get(c, 1.0) for c in self.cols], dtype=np.float32)
            unknown = sorted(set(weights) - set(self.cols))
            if unknown:
                print(f"[faiss_index] ignoring {len(unknown)} unknown weight keys: "
                      f"{unknown[:5]}{'...' if len(unknown) > 5 else ''}", file=sys.stderr)
            mat = (mat * w).astype(np.float32)
            self.w = w
        else:
            self.w = np.ones(len(self.cols), dtype=np.float32)

        mat = np.ascontiguousarray(mat)
        if metric == "cosine":
            faiss.normalize_L2(mat)
            self.index = faiss.IndexFlatIP(mat.shape[1])
        elif metric == "euclidean":
            self.index = faiss.IndexFlatL2(mat.shape[1])
        else:
            raise ValueError(metric)
        self.index.add(mat)
        self.rows = rows
        self.metric = metric
        self.target_col = target_col
        self._save()
        return self

    def _save(self) -> None:
        faiss.write_index(self.index, str(self.path))
        np.save(INDEX_DIR / "panel_ranked_mu.npy", self.mu)
        np.save(INDEX_DIR / "panel_ranked_sigma.npy", self.sigma)
        np.save(INDEX_DIR / "panel_ranked_mask.npy", self.mask)
        if self.w is not None:
            np.save(INDEX_DIR / "panel_ranked_w.npy", self.w)
        self.rows.to_parquet(INDEX_DIR / "panel_ranked_rows.parquet", index=False)
        (INDEX_DIR / "panel_ranked_cols.json").write_text(json.dumps(self.cols))
        # sorted_vals: per-column sorted observed values, used for query-time CDF.
        # Save as a single npz of object arrays.
        np.savez(INDEX_DIR / "panel_ranked_sorted.npz",
                 **{c: v for c, v in self.sorted_vals.items()})

    @classmethod
    def load(cls) -> "RankedFaissIndex":
        obj = cls()
        obj.index = faiss.read_index(str(obj.path))
        obj.mu = np.load(INDEX_DIR / "panel_ranked_mu.npy")
        obj.sigma = np.load(INDEX_DIR / "panel_ranked_sigma.npy")
        obj.mask = np.load(INDEX_DIR / "panel_ranked_mask.npy")
        w_path = INDEX_DIR / "panel_ranked_w.npy"
        obj.w = np.load(w_path) if w_path.exists() else None
        obj.rows = pd.read_parquet(INDEX_DIR / "panel_ranked_rows.parquet")
        obj.cols = json.loads((INDEX_DIR / "panel_ranked_cols.json").read_text())
        obj.target_col = "gdp_pc_growth_5y_fwd"
        obj.metric = "euclidean"
        # Load sorted values lazily; on demand.
        obj._sorted_loaded = False
        return obj

    def _ensure_sorted(self) -> None:
        if getattr(self, "_sorted_loaded", False):
            return
        npz = np.load(INDEX_DIR / "panel_ranked_sorted.npz", allow_pickle=False)
        self.sorted_vals = {c: npz[c] for c in self.cols}
        self._sorted_loaded = True

    def _vectorize_query(self, panel_row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        self._ensure_sorted()
        D = len(self.cols)
        z = np.zeros((1, D), dtype=np.float32)
        for j, c in enumerate(self.cols):
            v = panel_row.get(c, np.nan)
            if pd.isna(v):
                z[0, j] = 0.0
                continue
            arr = self.sorted_vals.get(c, np.array([], dtype=np.float64))
            if len(arr) == 0:
                z[0, j] = 0.0
                continue
            frac = float((arr <= float(v)).mean())
            z[0, j] = float((frac - self.mu[j]) / self.sigma[j])
        # Match the build-time weight transform so query and indexed vectors
        # live in the same weighted space.
        if self.w is not None:
            z = (z * self.w).astype(np.float32)
        z = np.ascontiguousarray(z, dtype=np.float32)
        if self.metric == "cosine":
            faiss.normalize_L2(z)
        q_mask = panel_row[self.cols].astype(float).notna().astype(np.uint8).to_numpy()
        return z, q_mask

    def query_topk(
        self,
        panel_row: pd.Series,
        k: int = 10,
        exclude_year: int | None = None,
        exclude_iso3: str | None = None,
        min_overlap: int = 0,
    ) -> pd.DataFrame:
        q, q_mask = self._vectorize_query(panel_row)
        over_fetch = max(k * 10, 200) if min_overlap > 0 else max(k * 10, k + 50)
        if self.metric == "cosine":
            sims, ids = self.index.search(q, over_fetch)
        else:
            d, ids = self.index.search(q, over_fetch)
            sims = -d
        sims, ids = sims[0], ids[0]
        rows = self.rows.iloc[ids].reset_index(drop=True)
        inter = (self.mask[ids] & q_mask).sum(axis=1)
        rows["similarity"] = sims
        rows["n_overlap"] = inter
        rows["distance"] = 1.0 - sims if self.metric == "cosine" else -sims
        if exclude_year is not None:
            rows = rows[rows["year"] != exclude_year]
        if exclude_iso3 is not None:
            rows = rows[rows["iso3"] != exclude_iso3]
        if min_overlap > 0:
            rows = rows[rows["n_overlap"] >= min_overlap]
        return rows.head(k).reset_index(drop=True)


def build_or_load_ranked(weights_path: str | Path | None = None,
                          force_rebuild: bool = False) -> RankedFaissIndex:
    """Load the persisted rank-Euclidean index, or rebuild with the given
    expert weights. If `weights_path` differs from the one baked at
    build-time, set `force_rebuild=True` to rebuild and persist a new
    weighted index (saved under a `panel_ranked.faiss` lock is bypassed
    only when --rebuild is passed at the CLI)."""
    p = INDEX_DIR / "panel_ranked.faiss"
    if p.exists() and not force_rebuild:
        idx = RankedFaissIndex.load()
        # If caller asked for a different weights file, force a rebuild.
        if weights_path is not None:
            print(f"[faiss_index] --weights requires rebuild; pass --rebuild",
                  file=sys.stderr)
        return idx
    return RankedFaissIndex().build(weights_path=weights_path)


# ----------------------------------------------------------------------
# v2 (GMD-shaped) retrieval: same rank-features + Euclidean recipe, but the
# feature column list comes from a v2 trainer (horizon_{h}y_v2) so the
# index stays compatible with the GMD panel, which lacks 17 of the 24 v1
# well-covered raw columns. Persisted under data/features/retrieval_v2/.
# ----------------------------------------------------------------------

V2_INDEX_DIR = FEATURES_DIR / "retrieval_v2"


class RankedV2Index:
    """Loads the v2-shaped ranked FAISS index from disk; does not rebuild
    (use scripts/_build_v2_faiss_index.py for that)."""

    def __init__(self):
        self.index = None
        self.rows = None
        self.cols: list[str] = []
        self.mu = None
        self.sigma = None
        self.mask = None
        self.sorted_vals: dict = {}
        self._sorted_loaded = False
        self.metric = "euclidean"
        self.target_col = "gdp_pc_growth_5y_fwd"

    @property
    def path(self) -> Path:
        return V2_INDEX_DIR / "panel_ranked.faiss"

    @classmethod
    def load(cls) -> "RankedV2Index":
        obj = cls()
        obj.index = faiss.read_index(str(V2_INDEX_DIR / "panel_ranked.faiss"))
        obj.mu = np.load(V2_INDEX_DIR / "mu.npy")
        obj.sigma = np.load(V2_INDEX_DIR / "sigma.npy")
        obj.mask = np.load(V2_INDEX_DIR / "mask.npy")
        obj.rows = pd.read_parquet(V2_INDEX_DIR / "rows.parquet")
        obj.cols = json.loads((V2_INDEX_DIR / "cols.json").read_text())
        return obj

    def _ensure_sorted(self) -> None:
        if self._sorted_loaded:
            return
        npz = np.load(V2_INDEX_DIR / "sorted_vals.npz", allow_pickle=False)
        self.sorted_vals = {c: npz[c] for c in self.cols}
        self._sorted_loaded = True

    def _vectorize_query(self, panel_row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        self._ensure_sorted()
        D = len(self.cols)
        z = np.zeros((1, D), dtype=np.float32)
        for j, c in enumerate(self.cols):
            v = panel_row.get(c, np.nan)
            if pd.isna(v):
                z[0, j] = 0.0
                continue
            arr = self.sorted_vals.get(c, np.array([], dtype=np.float64))
            if len(arr) == 0:
                z[0, j] = 0.0
                continue
            frac = float((arr <= float(v)).mean())
            z[0, j] = float((frac - self.mu[j]) / self.sigma[j])
        z = np.ascontiguousarray(z, dtype=np.float32)
        q_mask = panel_row[self.cols].astype(float).notna().astype(np.uint8).to_numpy()
        return z, q_mask

    def query_topk(
        self,
        panel_row: pd.Series,
        k: int = 10,
        exclude_year: int | None = None,
        exclude_iso3: str | None = None,
        min_overlap: int = 0,
    ) -> pd.DataFrame:
        q, q_mask = self._vectorize_query(panel_row)
        over_fetch = max(k * 10, 200) if min_overlap > 0 else max(k * 10, k + 50)
        d, ids = self.index.search(q, over_fetch)
        sims, ids = (-d[0]).astype(np.float32), ids[0]
        rows = self.rows.iloc[ids].reset_index(drop=True)
        inter = (self.mask[ids] & q_mask).sum(axis=1)
        rows["similarity"] = sims
        rows["n_overlap"] = inter
        rows["distance"] = -sims
        if exclude_year is not None:
            rows = rows[rows["year"] != exclude_year]
        if exclude_iso3 is not None:
            rows = rows[rows["iso3"] != exclude_iso3]
        if min_overlap > 0:
            rows = rows[rows["n_overlap"] >= min_overlap]
        return rows.head(k).reset_index(drop=True)


def build_or_load_v2_ranked() -> RankedV2Index:
    """Load the GMD-shaped ranked retrieval index (built by
    scripts/_build_v2_faiss_index.py)."""
    p = V2_INDEX_DIR / "panel_ranked.faiss"
    if not p.exists():
        raise FileNotFoundError(
            f"v2 retrieval index not built: {p}. Run `python scripts\\_build_v2_faiss_index.py` first."
        )
    return RankedV2Index.load()


def query_topk(iso3: str, year: int, k: int = 10) -> pd.DataFrame:
    panel = _load_panel()
    row = panel[(panel["iso3"] == iso3) & (panel["year"] == year)]
    if row.empty:
        raise SystemExit(f"iso3={iso3!r} year={year}: not in panel.")
    return build_or_load().query_topk(
        row.iloc[0], k=k, exclude_year=year, exclude_iso3=iso3
    )


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("iso3")
    ap.add_argument("year", type=int)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--weights", type=str, default=None,
                    help="YAML file of {feature: weight} for expert-nudged retrieval")
    ap.add_argument("--rebuild", action="store_true",
                    help="Force rebuild of the persisted FAISS index")
    args = ap.parse_args()
    idx = build_or_load_ranked(weights_path=args.weights, force_rebuild=args.rebuild)
    panel = pd.read_parquet(PANEL)
    row = panel[(panel["iso3"] == args.iso3) & (panel["year"] == args.year)]
    if row.empty:
        raise SystemExit(f"iso3={args.iso3!r} year={args.year}: not in panel.")
    df = idx.query_topk(row.iloc[0], k=args.k,
                        exclude_year=args.year, exclude_iso3=args.iso3)
    print(df.to_string(index=False))