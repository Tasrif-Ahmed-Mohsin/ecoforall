"""Inspect v2 forecast parquet schemas."""
import pandas as pd
from pathlib import Path
root = Path(r"E:\research\project\data\features")
for h in [1, 3, 5, 10]:
    p = root / f"horizon_{h}y_v2" / "forecasts.parquet"
    if not p.exists():
        print(f"h={h}: missing"); continue
    df = pd.read_parquet(p)
    print(f"h={h}: cols={df.columns.tolist()[:15]} rows={len(df)} years={df['year'].min()}-{df['year'].max()} iso3s={df['iso3'].nunique()}")
    print(df.head(2).to_string()); print("---")