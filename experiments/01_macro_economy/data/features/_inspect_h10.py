import pandas as pd
import json

df = pd.read_csv(r'data\features\llm_baseline_val_h10.csv')
print('total rows:', len(df))
print('horizons:', df['horizon'].value_counts().to_dict())
print('llm_pred non-null:', int(df['llm_pred'].notna().sum()), '/', len(df))
print()
print('--- final metrics JSON ---')
m = json.load(open(r'data\features\llm_baseline_val_h10_metrics.json'))
for k, v in m.get('horizons', {}).items():
    print(k)
    for kk in ('llm', 'prior', 'meta'):
        if kk in v:
            d = v[kk]
            print('  {:6s} n={:4d}  mae={:.4f}  dir={:.3f}'.format(
                kk, d['n'], d['mae'], d['dir_acc']))
