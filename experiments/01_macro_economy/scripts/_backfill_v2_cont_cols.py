"""Backfill v2 cont_cols into feature_meta without retraining.

Why this exists
---------------
`scripts/run_phase8_horizons_v2.py::_train_one_horizon` writes
`feature_meta.json` with the cont_cols list. The original retarget run
populated these into the on-disk metrics files in some sessions but not
others. This script replays `_prepare` against the **current** GMD panel
and writes `feature_meta` into every `data/features/horizon_{h}y_v2/`
metrics.json. No retraining, no Optuna, no model reload.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from scripts.run_phase8_horizons_v2 import (
    _add_country_and_tier_dummies,
    _horizon_target_name,
    _build_horizon_target,
)


def _cont_cols_for(panel: pd.DataFrame, target: str, iso_levels: list[str]) -> list[str]:
    """Re-derive cont_cols using the trainer's exact `_prepare` heuristic.

    We re-implement it inline (not import) because the trainer's `_prepare`
    also casts NaN-handled matrices; we only need the column list.
    """
    if target not in panel.columns:
        panel = panel.copy()
        panel[target] = _build_horizon_target(panel, _horizon_to_h(target))

    df = panel.dropna(subset=[target]).reset_index(drop=True)
    from scripts.run_phase8_horizons_v2 import LEAK_COLS_BASE, DROP_FEATURES  # type: ignore
    leak = LEAK_COLS_BASE | {c for c in df.columns if c.endswith("y_fwd")}
    df_aug, _ = _add_country_and_tier_dummies(df, iso_levels)
    dummy_cols = {c for c in df_aug.columns if c.startswith("iso_") or c.startswith("tier_")}
    cont_cols = [
        c for c in df_aug.columns
        if c not in leak
        and pd.api.types.is_numeric_dtype(df_aug[c])
        and c not in DROP_FEATURES
        and c not in dummy_cols
        and c not in {"gdp_pc", "gdp_pc_growth_5y_fwd"}
    ]
    cont_cols = [c for c in cont_cols if not c.endswith("y_fwd")]
    # Drop entirely-NaN columns (matches trainer's `cont_keep` filter).
    X = df_aug[cont_cols].astype(float)
    cont_keep = [c for c in cont_cols if X[c].notna().any()]
    return cont_keep


def _horizon_to_h(target: str) -> int:
    # "gdp_pc_growth_5y_fwd" -> 5
    s = target.replace("gdp_pc_growth_", "").replace("_fwd", "").rstrip("y")
    return int(s)


def main() -> None:
    panel = pd.read_parquet(ROOT / "data" / "features" / "panel_wide.parquet")
    iso_levels = sorted(panel["iso3"].unique().tolist())
    print(f"[backfill] loaded panel: {len(panel):,} rows × {panel.shape[1]} cols")
    print(f"[backfill] iso_levels: {len(iso_levels)}")

    for h in (1, 3, 5, 10):
        out_dir = ROOT / "data" / "features" / f"horizon_{h}y_v2"
        m_path = out_dir / "metrics.json"
        if not m_path.exists():
            print(f"[backfill] h={h}: no metrics.json, skip")
            continue
        target = _horizon_target_name(h)
        cont_cols = _cont_cols_for(panel, target, iso_levels)
        m = json.loads(m_path.read_text())
        m.setdefault("feature_meta", {})
        m["feature_meta"]["cont_cols"] = cont_cols
        m["feature_meta"]["iso_levels"] = iso_levels[:50] + (["..."] if len(iso_levels) > 50 else [])
        m["feature_meta"]["target"] = target
        m["feature_meta"]["n_cont"] = len(cont_cols)
        m_path.write_text(json.dumps(m, indent=2))
        print(f"[backfill] h={h}: wrote {len(cont_cols)} cont_cols to feature_meta")


if __name__ == "__main__":
    main()
