import json

m = json.load(open(r'data\features\cross_horizon_meta\metrics.json'))
print('=== LLM Contribution Summary (after full Option C union) ===')
print()
print('overall test MAE: {:.4f}  dir_acc: {:.3f}'.format(
    m['overall_test']['mae'], m['overall_test']['dir_acc']))
print()

hdr = ('horiz', 'n', 'meta', 'prior', 'lgbm', 'ridge', 'llm')
print('{:<6} {:>5} {:>9} {:>9} {:>9} {:>9} {:>9}'.format(*hdr))
for h, v in m['per_horizon_test'].items():
    prior = v.get('prior_mae')
    lgbm = v.get('lgbm_mae')
    ridge = v.get('ridge_mae')
    llm = v.get('llm_mae')
    fmt = lambda x: '{:.4f}'.format(x) if x is not None else '-'
    print('{:<6} {:>5} {:>9} {:>9} {:>9} {:>9} {:>9}'.format(
        h, v['n'], fmt(v['mae']), fmt(prior), fmt(lgbm), fmt(ridge), fmt(llm)))
print()
print('n_test_rows_with_llm:', m.get('n_test_rows_with_llm'))
print()

wd = m['weight_decomposition']
for f in wd['features']:
    if f['name'] == 'llm_pred':
        s = f['share_of_total_abs']
        print('LLM feature in meta-learner:')
        print('  coef_std    = {:+.4f}   (non-zero -> LLM column is contributing)'.format(f['coef_std']))
        print('  std_at_train= {:.4f}   (low because only 8% of meta-train rows have LLM data)'.format(f['std_at_train']))
        print('  share       = {:.4%}    of total |contribution|'.format(s))
print()
print('Top 3 contributors:')
for f in sorted(wd['features'], key=lambda x: -abs(x['contribution']))[:3]:
    print('  {:<11s} coef_std={:+.4f}  contrib={:+.5f}  share={:.1%}'.format(
        f['name'], f['coef_std'], f['contribution'], f['share_of_total_abs']))
