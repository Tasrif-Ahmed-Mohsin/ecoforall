import pandas as pd, numpy as np
from sklearn.metrics import mean_absolute_error

p = pd.read_parquet('data/features/cross_horizon_meta/predictions.parquet')

def metrics(s, h):
    cols = [f'lgbm_h{h}', f'ridge_h{h}', f'prior_h{h}']
    s = s.dropna(subset=cols)
    if len(s) == 0:
        return None
    rec = {
        'h': h,
        'n': len(s),
        'years': f'{int(s.year.min())}-{int(s.year.max())}',
        'lgbm_mae': round(float(mean_absolute_error(s.y_true, s[f'lgbm_h{h}'])), 4),
        'ridge_mae': round(float(mean_absolute_error(s.y_true, s[f'ridge_h{h}'])), 4),
        'prior_mae': round(float(mean_absolute_error(s.y_true, s[f'prior_h{h}'])), 4),
        'lgbm_dir': round(float((np.sign(s.y_true) == np.sign(s[f'lgbm_h{h}'])).mean()), 3),
        'prior_dir': round(float((np.sign(s.y_true) == np.sign(s[f'prior_h{h}'])).mean()), 3),
    }
    if s.pred_meta.notna().any():
        sm = s.dropna(subset=['pred_meta'])
        rec['meta_mae'] = round(float(mean_absolute_error(sm.y_true, sm.pred_meta)), 4)
        rec['meta_dir'] = round(float((np.sign(sm.y_true) == np.sign(sm.pred_meta)).mean()), 3)
    return rec

# Clean slice: drop 2020, 2021 (COVID) from the test+holdout combined slice
clean = p[~p.year.isin([2020, 2021])].dropna(subset=['y_true'])
print('=== clean (no 2020/2021) — test+holdout combined ===')
rows = [r for r in (metrics(clean[clean.horizon == h], h) for h in [1, 3, 5, 10]) if r]
print(pd.DataFrame(rows).to_string(index=False))

print()
print('=== delta vs full slice (incl. COVID) ===')
for h in [1, 3, 5, 10]:
    full = p[(p.horizon == h) & (p.split.isin(['test', 'holdout']))].dropna(subset=['y_true', f'lgbm_h{h}'])
    cln = full[~full.year.isin([2020, 2021])]
    if len(full) == 0 or len(cln) == 0:
        continue
    f_lgbm = round(float(mean_absolute_error(full.y_true, full[f'lgbm_h{h}'])), 4)
    c_lgbm = round(float(mean_absolute_error(cln.y_true, cln[f'lgbm_h{h}'])), 4)
    f_dir = round(float((np.sign(full.y_true) == np.sign(full[f'lgbm_h{h}'])).mean()), 3)
    c_dir = round(float((np.sign(cln.y_true) == np.sign(cln[f'lgbm_h{h}'])).mean()), 3)
    print(f'h={h}: lgbm_mae full={f_lgbm} (n={len(full)}, dir={f_dir}) -> clean={c_lgbm} (n={len(cln)}, dir={c_dir})')

# Holdout-only (year 2023) on h=1 is also a clean reference
print()
print('=== h=1 only — COVID-excluded alternatives ===')
for label, mask in [
    ('all test+holdout (incl COVID)', p[(p.horizon == 1) & (p.split.isin(['test', 'holdout']))].dropna(subset=['y_true', 'lgbm_h1'])),
    ('test only (2019-2022, incl COVID)', p[(p.horizon == 1) & (p.split == 'test')].dropna(subset=['y_true', 'lgbm_h1'])),
    ('test NO 2020/2021', p[(p.horizon == 1) & (p.split == 'test') & (~p.year.isin([2020, 2021]))].dropna(subset=['y_true', 'lgbm_h1'])),
    ('holdout only (2023, no COVID by construction)', p[(p.horizon == 1) & (p.split == 'holdout')].dropna(subset=['y_true', 'lgbm_h1'])),
    ('test 2019 only', p[(p.horizon == 1) & (p.split == 'test') & (p.year == 2019)].dropna(subset=['y_true', 'lgbm_h1'])),
    ('test 2022 only', p[(p.horizon == 1) & (p.split == 'test') & (p.year == 2022)].dropna(subset=['y_true', 'lgbm_h1'])),
]:
    if len(mask) == 0:
        print(f'  {label}: n=0')
        continue
    yrs = sorted(mask.year.unique())
    rec = {
        'n': len(mask),
        'years': f'{yrs[0]}-{yrs[-1]}',
        'lgbm_mae': round(float(mean_absolute_error(mask.y_true, mask['lgbm_h1'])), 4),
        'lgbm_dir': round(float((np.sign(mask.y_true) == np.sign(mask['lgbm_h1'])).mean()), 3),
        'prior_mae': round(float(mean_absolute_error(mask.y_true, mask['prior_h1'])), 4),
        'prior_dir': round(float((np.sign(mask.y_true) == np.sign(mask['prior_h1'])).mean()), 3),
    }
    print(f'  {label}: {rec}')
