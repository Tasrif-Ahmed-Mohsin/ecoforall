"""Print h=10 v2 test-slice metrics + ensemble recipe."""
import json, sys
m = json.load(open(r"E:/research/project/data/features/horizon_10y_v2/metrics.json"))
print("h=10 v2 split:", m["split"])
print("h=10 v2 test MAEs:")
for k, v in m["results"].items():
    t = v["test"]
    print(f"  {k:>10s}: mae={t['mae']:.4f}  rmse={t['rmse']:.4f}  dir_acc={t['dir_acc']:.3f}  n={t['n']}")
print("ensemble candidates:", {k: round(v, 4) for k, v in m["ensemble_candidates"].items()})
print("ensemble_prior_mae:", round(m["ensemble_prior_mae"], 4))
print("ensemble_recipe:", m["ensemble_recipe"])
print("optuna best params:")
for k, v in m["optuna_best_params"].items():
    print(f"  {k}: {v}")
