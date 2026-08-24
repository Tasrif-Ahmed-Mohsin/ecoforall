"""Quick standalone reporter for the cross-horizon meta headline."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
m = json.loads((ROOT / "data/features/cross_horizon_meta/metrics.json").read_text())
print("CROSS-HORIZON META  (GMD, test slice 2019-2022 unless noted)")
rows = m["per_horizon_test"]
print("type:", type(rows).__name__, "keys:", list(rows.keys()) if hasattr(rows, "keys") else "n/a")
if isinstance(rows, dict):
    for h, r in rows.items():
        mae = r.get("meta_mae", r.get("mae", float("nan")))
        prior = r.get("prior_mae", float("nan"))
        delta = r.get("delta_vs_prior", float("nan"))
        n = r.get("n", r.get("n_test", "?"))
        print(f"  h={h:>2}: meta_mae={mae:.4f}  prior_mae={prior:.4f}  delta={delta:+.4f}  n={n}")
elif isinstance(rows, list):
    print("first row keys:", list(rows[0].keys()))