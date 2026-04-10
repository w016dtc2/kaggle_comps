import optuna
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils import resample
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from src.models import build_preprocessor
from src.config import CV_FOLDS, RANDOM_STATE

optuna.logging.set_verbosity(optuna.logging.WARNING)

EXTRA_NUMERIC = [
    'Water_Stress', 'ET_Proxy', 'Soil_Health', 'Irrigation_Efficiency',
    'Water_Demand', 'THI', 'Rain_ET_Balance', 'Growth_Stage_Num',
    'Season_Num',
    'Rainfall_mm_vs_group', 'Rainfall_mm_zscore',
    'Temperature_C_vs_group', 'Temperature_C_zscore',
    'Soil_Moisture_vs_group', 'Soil_Moisture_zscore',
    'ET_Proxy_vs_group', 'ET_Proxy_zscore',
]


def objective(trial, X, y):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "tree_method": "hist",
        "device": "cpu",
        "eval_metric": "mlogloss",
        "random_state": RANDOM_STATE,
        "n_jobs": -1
    }

    pipe = Pipeline([
        ('preprocessor', build_preprocessor(extra_numeric_cols=EXTRA_NUMERIC)),
        ('model', XGBClassifier(**params))
    ])

    sample_weights = compute_sample_weight('balanced', y=y)
    cv = StratifiedKFold(
        n_splits=CV_FOLDS["tune"],
        shuffle=True,
        random_state=RANDOM_STATE
    )

    scores = cross_val_score(
        pipe, X, y,
        cv=cv,
        scoring='balanced_accuracy',
        params={'model__sample_weight': sample_weights}
    )

    return scores.mean()


def run_tuning(X, y, n_trials: int = 50):
    # Subsample for speed
    print(f"Subsampling to 50k rows for tuning...")
    X_sample, y_sample = resample(
        X, y,
        n_samples=350000,
        random_state=RANDOM_STATE,
        stratify=y
    )

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X_sample, y_sample),
        n_trials=n_trials,
        show_progress_bar=True
    )

    print(f"\nBest CV score: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    return study.best_params