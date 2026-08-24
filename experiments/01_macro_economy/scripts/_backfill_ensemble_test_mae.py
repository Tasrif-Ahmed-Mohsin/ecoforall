"""Backfill `ensemble_test_mae` into existing v2 metrics.json files (no retrain)."""
import json
from pathlib import Path

for h in (1, 3, 5, 10):
    p = Path(f"data/features/horizon_{h}y_v2/metrics.json")
    m = json.loads(p.read_text())
    if "ensemble_test_mae" not in m:
        cands = m.get("ensemble_candidates", {})
        best = m.get("ensemble_recipe")
        if best in cands:
            m["ensemble_test_mae"] = float(cands[best])
            p.write_text(json.dumps(m, indent=2))
            print(f"h={h}: backfilled ensemble_test_mae={m['ensemble_test_mae']:.4f} (best={best})")
        else:
            print(f"h={h}: no best recipe in cands; skipping")
    else:
        print(f"h={h}: already has ensemble_test_mae={m['ensemble_test_mae']:.4f}")
