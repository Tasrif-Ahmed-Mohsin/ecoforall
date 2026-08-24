"""Print v2.1 walk-forward CV summaries cleanly. Run with: python _print_cv_summary.py [h1|h3|h5|h10|all]"""
import json
import sys
from pathlib import Path

FEAT = Path(__file__).resolve().parent / "data" / "features"


def show(label: str, path: Path) -> None:
    if not path.exists():
        print(f"[{label}] NOT FOUND: {path.name}")
        return
    d = json.loads(path.read_text())
    pm = d.get("per_model", [])
    if not pm:
        print(f"[{label}] empty per_model in {path.name}")
        return
    print("=" * 80)
    print(f"[{label}]  {path.name}  ({len(pm)} models)")
    print("=" * 80)
    if isinstance(pm, list):
        print(f"  (per-row fields: {sorted(pm[0].keys())})")
    hdr = f"{'MODEL':40s} {'mae_mean':>10s} {'skill_ar1':>11s} {'skill_prior':>12s}"
    print(hdr)
    print("-" * 80)
    if isinstance(pm, list):
        rows = sorted(pm, key=lambda r: r.get("mae_mean", r.get("mae", float("inf"))))
        for m in rows:
            name = m.get("model", "?")
            mae = m.get("mae_mean", m.get("mae", float("nan")))
            ar1 = m.get("skill_vs_ar1_mean", float("nan"))
            prior = m.get("skill_vs_naive_mean", float("nan"))
            print(f"{name:40s} {mae:>10.4f} {ar1:>11.4f} {prior:>12.4f}")
    else:
        rows = sorted(pm.items(), key=lambda kv: kv[1].get("mae_mean", float("inf")))
        for name, m in rows:
            mae = m.get("mae_mean", float("nan"))
            ar1 = m.get("skill_vs_ar1_mean", float("nan"))
            prior = m.get("skill_vs_naive_mean", float("nan"))
            print(f"{name:40s} {mae:>10.4f} {ar1:>11.4f} {prior:>12.4f}")
    print()


def main() -> None:
    args = sys.argv[1:] or ["all"]
    if "h1" in args or "all" in args:
        show("h=1", FEAT / "walk_forward_cv_h1_v21_summary.json")
    if "h3" in args or "all" in args:
        show("h=3", FEAT / "walk_forward_cv_h3_v21_summary.json")
    if "h5" in args or "all" in args:
        show("h=5", FEAT / "walk_forward_cv_v21_summary.json")  # h=5 is the un-suffixed file
    if "h10" in args or "all" in args:
        # We never ran v2.1 CV at h=10; the v1 trainer's per-horizon h=10 result is on disk
        # in horizon_10y_v2/metrics.json. Show what is there.
        p = FEAT / "horizon_10y_v2" / "metrics.json"
        if p.exists():
            print("=" * 80)
            print("[h=10]  horizon_10y_v2/metrics.json (v1 trainer only, NO v2.1 ensemble candidates)")
            print("=" * 80)
            m = json.loads(p.read_text())
            print(f"  ensemble_recipe:   {m.get('ensemble_recipe')}")
            print(f"  ensemble_test_mae: {m.get('ensemble_test_mae'):.4f}")
            print(f"  ensemble_prior_mae:{m.get('ensemble_prior_mae'):.4f}")
            print(f"  delta vs prior:    {m.get('ensemble_test_mae', 0) - m.get('ensemble_prior_mae', 0):+.4f}")
            print()
            print("  Per-base-model test results:")
            for k, v in m.get("results", {}).items():
                if isinstance(v, dict) and "test" in v:
                    print(f"    {k:20s} MAE={v['test']['mae']:.4f}  dir_acc={v['test']['dir_acc']:.3f}")
        else:
            print("[h=10] no metrics.json found")


if __name__ == "__main__":
    main()