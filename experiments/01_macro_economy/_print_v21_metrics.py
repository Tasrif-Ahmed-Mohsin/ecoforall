"""Print v2.1 trainer per-horizon metrics cleanly. Run with: python _print_v21_metrics.py [h1|h3|h5|h10|all]"""
import json
import sys
from pathlib import Path

FEAT = Path(__file__).resolve().parent / "data" / "features"


def show(h: int) -> None:
    p = FEAT / f"horizon_{h}y_v2" / "metrics.json"
    if not p.exists():
        print(f"[h={h}] no metrics.json")
        return
    m = json.loads(p.read_text())
    print("=" * 78)
    print(f"[h={h}]  horizon_{h}y_v2/metrics.json  (v2.1 trainer)")
    print("=" * 78)
    print(f"  recipe:           {m['ensemble_recipe']}")
    print(f"  test_mae:         {m['ensemble_test_mae']:.4f}")
    print(f"  prior_mae:        {m['ensemble_prior_mae']:.4f}")
    delta = m['ensemble_test_mae'] - m['ensemble_prior_mae']
    win = "WIN " if delta < 0 else "LOSE"
    print(f"  delta vs prior:   {delta:+.4f}   [{win} vs prior]")
    print()
    print("  Ensemble candidates (sorted by test MAE):")
    cands = m['ensemble_candidates']
    prior = m['ensemble_prior_mae']
    for k, v in sorted(cands.items(), key=lambda kv: kv[1]):
        d = v - prior
        flag = " <- prior" if v < prior else "  (loses)" if d > 0 else ""
        print(f"    {k:40s} {v:.4f}  delta={d:+.4f}{flag}")
    print()
    print("  Base-model test slice:")
    for k, v in m.get("results", {}).items():
        if isinstance(v, dict) and "test" in v:
            t = v['test']
            d = t['mae'] - prior
            print(f"    {k:20s} MAE={t['mae']:.4f}  dir_acc={t['dir_acc']:.3f}  delta={d:+.4f}")
    print()


def main() -> None:
    args = sys.argv[1:] or ["all"]
    for h in (1, 3, 5, 10):
        if f"h{h}" in args or "all" in args:
            show(h)


if __name__ == "__main__":
    main()