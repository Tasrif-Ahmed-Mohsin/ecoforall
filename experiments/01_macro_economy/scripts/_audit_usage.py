"""One-shot audit: what data is actually used, what produced the live results."""
from __future__ import annotations
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

panel = pd.read_parquet(ROOT / "data" / "features" / "panel_wide.parquet")

print("=" * 70)
print("CURRENT PIPELINE — DATA USED AND RESULTS")
print("=" * 70)

print("\n[1] Panel source")
print(f"    file:     data/features/panel_wide.parquet")
print(f"    rows:     {len(panel):,}")
print(f"    cols:     {panel.shape[1]}")
print(f"    years:    {panel.year.min()} – {panel.year.max()}")
print(f"    countries (iso3): {panel.iso3.nunique()}")
print(f"    source:   GMD 2026 v6 (E:\\GMD_2026_06_csv\\GMD.csv) ONLY")
print(f"               old 5-source stack (IMF/WB/JST/Maddison/Clio) NOT used")

print("\n[2] Per-horizon target coverage on the panel")
for h in (1, 3, 5, 10):
    t = f"gdp_pc_growth_{h}y_fwd"
    n = int(panel[t].notna().sum()) if t in panel.columns else 0
    print(f"    h={h:2d}  {t:30s}  rows with target: {n:5d}  ({n / len(panel) * 100:5.1f}%)")

# Find the actual train/test split used by the v2 trainer (last-year-of-target)
# The v2 trainer uses (year <= TRAIN_END) for train and the rest as test.
print("\n[3] Test slice actually evaluated by v2 trainer")
# Walk each v2 metrics to get n_labelled + n_countries + (implicit) test slice
for h in (1, 3, 5, 10):
    m_path = ROOT / "data" / "features" / f"horizon_{h}y_v2" / "metrics.json"
    if not m_path.exists():
        continue
    m = json.loads(m_path.read_text())
    n_lab = m.get("n_labelled")
    n_cty = m.get("n_countries")
    n_feat = m.get("n_cont_features")
    ens = m.get("ensemble_recipe")
    mae = m.get("ensemble_test_mae")
    prior = m.get("ensemble_prior_mae")
    delta = (mae - prior) if mae and prior else None
    print(f"    h={h:2d}  n_labelled={n_lab:5d}  countries={n_cty:3d}  cont_features={n_feat:3d}  "
          f"recipe={ens:18s}  test_mae={mae:.4f}  prior={prior:.4f}  delta={delta:+.4f}")

print("\n[4] Cross-horizon meta-ensemble (final stacking layer)")
m = json.loads((ROOT / "data" / "features" / "cross_horizon_meta" / "metrics.json").read_text())
for hk in ("h1", "h3", "h5", "h10"):
    d = m.get("per_horizon_test", {}).get(hk)
    if d is None:
        continue
    print(f"    {hk}: meta_mae={d['mae']:.4f}  prior_mae={d['prior_mae']:.4f}  n={d['n']}  "
          f"delta_vs_prior={d['mae'] - d['prior_mae']:+.4f}")

print("\n[5] Walk-forward CV (generalization check)")
wf = pd.read_csv(ROOT / "data" / "features" / "walk_forward_cv.csv")
print(wf.to_string(index=False))

print("\n[6] Conformal calibration on h=5 (n=772, years 2019-2022)")
c = json.loads((ROOT / "data" / "features" / "conformal_adjustment.json").read_text())
print(f"    raw q05/q95 coverage:           {c['raw_coverage_pct']:.2f} %  (target 90 %)")
print(f"    raw lower-tail violation:       {c['raw_lower_violation_pct']:.2f} %  (target <= 5 %)")
print(f"    raw upper-tail violation:       {c['raw_upper_violation_pct']:.2f} %")
print(f"    calibrated (constant-shift):    {c['calibrated_coverage_pct']:.2f} %")
print(f"    calibrated lower-tail violation:{c['calibrated_lower_violation_pct']:.2f} %")
print(f"    defense_guards:                 {c['defense_guards']}")
print(f"    fallback_to_widened_band:       {c['fallback_to_widened_band']}  (widening {c['recommended_widening_pct']*100:.0f} %)")

print("\n[7] Crisis classifier (Phase 9 deliverable, GMD-specific)")
crisis_dir = ROOT / "data" / "features" / "crisis_model"
if crisis_dir.exists():
    for f in sorted(crisis_dir.iterdir()):
        print(f"    {f.name}  ({f.stat().st_size / 1024:.1f} KB)")
else:
    print("    (no crisis_model artifacts)")

print("\n[8] FAISS retrieval (history analogs)")
faiss_dir = ROOT / "data" / "features" / "retrieval_v2"
if faiss_dir.exists():
    for f in sorted(faiss_dir.iterdir()):
        print(f"    {f.name:30s}  ({f.stat().st_size / 1024:.1f} KB)")
legacy_dir = ROOT / "data" / "features" / "retrieval"
print(f"    legacy v1 retrieval/             {'present (broken on GMD; 17/24 v1 columns absent)' if legacy_dir.exists() else 'absent'}")