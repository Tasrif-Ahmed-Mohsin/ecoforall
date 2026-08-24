import optuna
s = optuna.create_study(
    storage='sqlite:///E:/research/project/data/features/horizon_10y_v2/optuna_study.db',
    study_name='lgbm_h10', load_if_exists=True)
with open('E:/research/project/data/features/_h10_progress.txt', 'w') as f:
    f.write(f"trials={len(s.trials)} completed={sum(1 for t in s.trials if t.state.name=='COMPLETE')} best={s.best_value if s.best_trial else None}\n")
