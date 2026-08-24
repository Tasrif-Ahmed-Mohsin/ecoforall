"""Print the existing hold-out metrics for all four horizons."""
import json

print("HOLD-OUT METRICS (forecast origins > test_end, currently years 2023-2024 for h<=5)")
print()
for h in (1, 3, 5, 10):
    path = f"data/features/horizon_{h}y_v2/metrics.json"
    m = json.load(open(path))
    split = m["split"]
    print(f"h={h:>2}: train<={split['train_end']}, val<={split['val_end']}, "
          f"test<={split['test_end']}, n_labelled={m['n_labelled']}, "
          f"n_countries={m['n_countries']}")
    for model in ("prior", "lgbm", "ridge"):
        r = m["results"].get(model, {})
        hk = r.get("hold")
        if hk:
            print(f"      {model:>6} hold: MAE={hk['mae']:.4f}  dir_acc={hk['dir_acc']:.3f}  "
                  f"rmse={hk['rmse']:.4f}  n={hk['n']}")
    em = m.get("ensemble_recipe", "?")
    # try to find ensemble-named key
    ens_hold = None
    for ens_name in ("lgbm+prior", "lgbm+ridge+prior", "lgbm+ridge"):
        block = m["results"].get(ens_name)
        if isinstance(block, dict) and isinstance(block.get("hold"), dict):
            ens_hold = (ens_name, block["hold"])
            break
    if ens_hold:
        print(f"      ensemble ({em}) hold[{ens_hold[0]}]: MAE={ens_hold[1]['mae']:.4f}  "
              f"dir_acc={ens_hold[1]['dir_acc']:.3f}  n={ens_hold[1]['n']}")
    print()
