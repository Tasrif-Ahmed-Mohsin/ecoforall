import json
for h in (1,3,5,10):
    m = json.load(open(f'data/features/horizon_{h}y_v2/metrics.json'))
    print(f'\n=== h={h} ===')
    print('split:', m['split'])
    print('n_labelled:', m['n_labelled'])
    print('n_countries:', m['n_countries'])
    print('results keys:', list(m['results'].keys()))
    for k,v in m['results'].items():
        for stage in ('train','val','test','hold'):
            if stage in v:
                sub = v[stage]
                print(f'  {k}.{stage}: n={sub["n"]} mae={sub["mae"]:.4f} dir_acc={sub["dir_acc"]:.3f}')
