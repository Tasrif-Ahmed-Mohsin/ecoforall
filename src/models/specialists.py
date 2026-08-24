"""
Specialist Forecaster Models (LGCF-v2 Architecture)
===================================================
Provides 4 decorrelated, orthogonal specialist experts:
1. Macro Linear Trend Expert (Ridge)
2. Deep Non-linear Quad Tree Expert (LightGBM)
3. Robust Heavy-Tailed Outlier Expert (Huber)
4. Crisis/Stress Quantile Specialist (GBDT Quantile on Downturns)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb


@dataclass
class SpecialistTrainedSet:
    exp1_ridge: Ridge
    exp2_lgb: lgb.LGBMRegressor
    exp3_huber: HuberRegressor
    exp4_stress: lgb.LGBMRegressor
    imputer: SimpleImputer
    scaler: StandardScaler
    train_residuals: Dict[str, np.ndarray]


def _clean_matrix(X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Convert matrix to float64 and replace inf/-inf with nan."""
    if isinstance(X, pd.DataFrame):
        arr = X.values.astype(np.float64)
    else:
        arr = np.asarray(X, dtype=np.float64)
    arr[~np.isfinite(arr)] = np.nan
    return arr


def train_specialist_suite(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    seed: int = 42
) -> SpecialistTrainedSet:
    """Train all 4 orthogonal specialist models on training slice with strict isolation."""
    X_clean = _clean_matrix(X_train)
    
    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_tr_imp = imp.fit_transform(X_clean)
    X_tr_sc = scaler.fit_transform(X_tr_imp)
    y_arr = np.asarray(y_train, dtype=np.float64)

    # 1. Ridge Trend Expert
    exp1 = Ridge(alpha=50.0, random_state=seed)
    exp1.fit(X_tr_sc, y_arr)
    res1 = np.abs(y_arr - exp1.predict(X_tr_sc))

    # 2. LightGBM Deep Quad Expert
    exp2 = lgb.LGBMRegressor(
        n_estimators=180, learning_rate=0.03, max_depth=5,
        num_leaves=24, min_child_samples=25, subsample=0.8,
        colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=1.0,
        random_state=seed, verbose=-1, n_jobs=-1, objective="regression_l1"
    )
    exp2.fit(X_tr_imp, y_arr)
    res2 = np.abs(y_arr - exp2.predict(X_tr_imp))

    # 3. Robust Huber Outlier Expert
    exp3 = HuberRegressor(max_iter=300, alpha=10.0)
    exp3.fit(X_tr_sc, y_arr)
    res3 = np.abs(y_arr - exp3.predict(X_tr_sc))

    # 4. Stress Downturn Specialist (Weights negative growth shocks heavily)
    p25 = np.percentile(y_arr, 25)
    sample_weights = np.where(y_arr < p25, 3.0, 1.0)
    exp4 = lgb.LGBMRegressor(
        n_estimators=120, learning_rate=0.04, max_depth=4,
        num_leaves=16, min_child_samples=20, subsample=0.8,
        random_state=seed, verbose=-1, n_jobs=-1, objective="regression_l1"
    )
    exp4.fit(X_tr_imp, y_arr, sample_weight=sample_weights)
    res4 = np.abs(y_arr - exp4.predict(X_tr_imp))

    residuals = {
        "ridge": res1,
        "lgbm": res2,
        "huber": res3,
        "stress": res4
    }

    return SpecialistTrainedSet(
        exp1_ridge=exp1,
        exp2_lgb=exp2,
        exp3_huber=exp3,
        exp4_stress=exp4,
        imputer=imp,
        scaler=scaler,
        train_residuals=residuals
    )


def predict_specialist_suite(
    suite: SpecialistTrainedSet,
    X_test: pd.DataFrame | np.ndarray
) -> Dict[str, np.ndarray]:
    """Generate individual specialist predictions for test samples."""
    X_clean = _clean_matrix(X_test)
    X_te_imp = suite.imputer.transform(X_clean)
    X_te_sc = suite.scaler.transform(X_te_imp)

    return {
        "ridge": suite.exp1_ridge.predict(X_te_sc),
        "lgbm": suite.exp2_lgb.predict(X_te_imp),
        "huber": suite.exp3_huber.predict(X_te_sc),
        "stress": suite.exp4_stress.predict(X_te_imp)
    }
