import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from src.models import build_preprocessor
from src.config import COMPUTE, CV_FOLDS, RANDOM_STATE

EXTRA_NUMERIC = [
    'Water_Stress', 'ET_Proxy', 'Soil_Health', 'Irrigation_Efficiency',
    'Water_Demand', 'THI', 'Rain_ET_Balance', 'Growth_Stage_Num',
    'Season_Num',
    'Rainfall_mm_vs_group', 'Rainfall_mm_zscore',
    'Temperature_C_vs_group', 'Temperature_C_zscore',
    'Soil_Moisture_vs_group', 'Soil_Moisture_zscore',
    'ET_Proxy_vs_group', 'ET_Proxy_zscore',
]


def build_xgb_pipe():
    model = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        tree_method='hist',
        device='cpu',
        eval_metric='mlogloss',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    return Pipeline([
        ('preprocessor', build_preprocessor(extra_numeric_cols=EXTRA_NUMERIC)),
        ('model', model)
    ])


def build_lgbm_pipe():
    model = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1
    )
    return Pipeline([
        ('preprocessor', build_preprocessor(extra_numeric_cols=EXTRA_NUMERIC)),
        ('model', model)
    ])


def build_catboost_pipe():
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        random_seed=RANDOM_STATE,
        verbose=0
    )
    return Pipeline([
        ('preprocessor', build_preprocessor(extra_numeric_cols=EXTRA_NUMERIC)),
        ('model', model)
    ])


def soft_vote(proba_list):
    """
    Average probabilities across models.
    proba_list: list of (n_samples, n_classes) arrays
    """
    return np.mean(proba_list, axis=0)


def hard_vote(proba_list):
    """
    Majority class vote across models.
    """
    preds = [np.argmax(p, axis=1) for p in proba_list]
    preds = np.stack(preds, axis=1)
    return np.apply_along_axis(
        lambda x: np.bincount(x, minlength=3).argmax(),
        axis=1,
        arr=preds
    )


def run_ensemble(X_train, X_test, y_train, sample_weights=None, voting='soft'):
    """
    Fits all three models and returns final predictions on X_test.
    """
    if sample_weights is None:
        sample_weights = compute_sample_weight('balanced', y=y_train)

    pipes = {
        'xgb': build_xgb_pipe(),
        'lgbm': build_lgbm_pipe(),
        'catboost': build_catboost_pipe()
    }

    proba_list = []
    for name, pipe in pipes.items():
        print(f"Fitting {name}...")
        pipe.fit(X_train, y_train, model__sample_weight=sample_weights)
        proba_list.append(pipe.predict_proba(X_test))
        print(f"{name} done")

    if voting == 'soft':
        avg_proba = soft_vote(proba_list)
        final_preds = np.argmax(avg_proba, axis=1)
    elif voting == 'hard':
        final_preds = hard_vote(proba_list)

    return final_preds, pipes


def run_ensemble_cv(X, y, sample_weights=None, compute: str = COMPUTE):
    """
    CV evaluation of the ensemble using soft voting.
    """
    if sample_weights is None:
        sample_weights = compute_sample_weight('balanced', y=y)

    cv = StratifiedKFold(
        n_splits=CV_FOLDS[compute],
        shuffle=True,
        random_state=RANDOM_STATE
    )

    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]
        fold_weights = sample_weights[train_idx]

        fold_preds, _ = run_ensemble(
            X_fold_train, X_fold_val,
            y_fold_train, fold_weights
        )

        from sklearn.metrics import balanced_accuracy_score
        score = balanced_accuracy_score(y_fold_val, fold_preds)
        print(f"Fold {fold + 1} Balanced Accuracy: {score:.4f}")
        fold_scores.append(score)

    print(f"\nEnsemble CV Balanced Accuracy: {np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}")
    return fold_scores