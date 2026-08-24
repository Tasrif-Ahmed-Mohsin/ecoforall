import pandas as pd
import shutil

paths = [
    r'data\features\llm_baseline_val_h1.csv',
    r'data\features\llm_baseline_val_h3.csv',
    r'data\features\llm_baseline_val_h5.csv',
    r'data\features\llm_baseline_val_h10.csv',
    r'data\features\llm_baseline_holdout.csv',
]

for p in paths:
    df = pd.read_csv(p)
    n_before = len(df)
    cols = [c for c in ('iso3', 'year', 'horizon') if c in df.columns]
    df = df.drop_duplicates(subset=cols, keep='first').reset_index(drop=True)
    n_after = len(df)
    if n_after != n_before:
        df.to_csv(p, index=False)
    print('{:55s}  {} -> {} rows'.format(p, n_before, n_after))
