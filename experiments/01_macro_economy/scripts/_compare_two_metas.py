"""Print both 'meta' concepts side by side so the naming doesn't get mixed up."""
import json
from pathlib import Path

print("=== Per-horizon trainer ensemble (single horizon, picks best of lgbm / lgbm+ridge / lgbm+prior / lgbm+ridge+prior) ===")
for h in (1, 3, 5, 10):
    m = json.loads(Path(f"data/features/horizon_{h}y_v2/metrics.json").read_text())
    cands = m["ensemble_candidates"]
    best = m["ensemble_recipe"]
    prior = m["results"]["prior"]["test"]["mae"]
    ens = m["ensemble_test_mae"]
    print(f"  h={h:2d}  prior={prior:.4f}  best_ens={ens:.4f}  recipe={best}  candidates={cands}")

print()
print("=== Cross-horizon meta-ensemble (stacks all 4 horizons + AR(1) via Ridge meta-learner) ===")
meta = json.loads(Path("data/features/cross_horizon_meta/metrics.json").read_text())
per_h = meta["per_horizon_test"]
for h in sorted(per_h.keys(), key=lambda k: int(k[1:])):
    row = per_h[h]
    n = row.get("n", "?")
    meta_mae = row["mae"]
    prior_mae = row["prior_mae"]
    lgbm_mae = row["lgbm_mae"]
    ridge_mae = row["ridge_mae"]
    delta = meta_mae - prior_mae
    win = "WIN" if meta_mae < prior_mae else "lose"
    print(
        f"  h={h:3s}  n={n:4}  meta={meta_mae:.4f}  lgbm={lgbm_mae:.4f}  "
        f"ridge={ridge_mae:.4f}  prior={prior_mae:.4f}  "
        f"delta_vs_prior={delta:+.4f}  ({win})"
    )