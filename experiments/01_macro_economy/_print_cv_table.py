"""Print per-model walk-forward CV rows for the three v2.1 horizons."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

for h, relpath in [
    (1, "data/features/walk_forward_cv_h1_v21_summary.json"),
    (3, "data/features/walk_forward_cv_h3_v21_summary.json"),
    (5, "data/features/walk_forward_cv_v21_summary.json"),
]:
    p = ROOT / relpath
    if not p.exists():
        print(f"[h={h}] MISSING: {p}")
        continue
    d = json.loads(p.read_text(encoding="utf-8"))
    rows = d["per_model"]
    print(f"\n[h={h}] {relpath}  ({len(rows)} rows, n_folds={rows[0]['n_folds']})")
    print(f"  {'MODEL':30s}  {'MAE':>9s}  {'STD':>7s}  {'SKILL_AR1':>10s}  {'SKILL_PRIOR':>12s}")
    for r in sorted(rows, key=lambda x: x["mae_mean"]):
        print(
            f"  {r['model']:30s}  {r['mae_mean']:.4f}  {r['mae_std']:.4f}  "
            f"{r['skill_vs_ar1_mean']:+.4f}    {r['skill_vs_naive_mean']:+.4f}"
        )
