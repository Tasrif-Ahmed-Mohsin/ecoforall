"""Diff per-horizon MAE before/after wiring llm_pred into the Ridge meta.

Reads metrics.json (current) and metrics.before_llm.json (saved pre-run).
"""
import json, pathlib
ROOT = pathlib.Path(r"E:\project_gmd\data\features\cross_horizon_meta")
M = json.loads((ROOT / "metrics.json").read_text())
B = json.loads((ROOT / "metrics.before_llm.json").read_text())

print(f"{'horizon':<8} {'n':>5} {'mae_before':>11} {'mae_after':>11} {'Δ MAE':>10} {'% vs base':>11} {'Δ dir_acc':>11}")
print("-" * 72)
for h in ["h1", "h3", "h5", "h10"]:
    a, z = M["per_horizon_test"][h], B["per_horizon_test"][h]
    mae_a, mae_b = a["mae"], z["mae"]
    delta = mae_b - mae_a  # positive = improvement (lower is better)
    pct = 100 * delta / mae_b
    da = a["dir_acc"] - z["dir_acc"]
    print(f"{h:<8} {a['n']:>5} {mae_b:>11.6f} {mae_a:>11.6f} {delta:>+10.6f} {pct:>+10.3f}% {da:>+11.4f}")

oa, ob = M["overall_test"], B["overall_test"]
print()
print(f"{'overall':<8} {M['meta_test_rows']:>5} {ob['mae']:>11.6f} {oa['mae']:>11.6f} "
      f"{ob['mae'] - oa['mae']:>+10.6f}")
print()
print("feature_cols (n):", len(M["feature_cols"]))
print("LLM in cols:", "llm_pred" in M["feature_cols"])
print("n_test_rows_with_llm:", M["n_test_rows_with_llm"])
print("ridge alpha:", M["ridge_alpha"])
for f in M["weight_decomposition"]["features"]:
    if f["name"] == "llm_pred":
        print(f"llm_pred  coef_std={f['coef_std']:+.6f}  std={f['std_at_train']:.6f}  "
              f"contrib={f['contribution']:+.6f}  share={f['share_of_total_abs']*100:.2f}%")
        break
