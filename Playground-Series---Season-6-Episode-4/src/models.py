import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from src.config import (
    COMPUTE,
    MODEL_PARAMS,
    CV_FOLDS,
    NUMERIC_COLS,
    ORDINAL_COLS,
    BINARY_COLS,
    NOMINAL_COLS,
    RANDOM_STATE
)


def encode_target(y):
    """
    Encodes string target labels to integers for XGBoost.
    Returns encoded y and the fitted encoder for inverse transform later.
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"Class mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    return y_encoded, le


def build_preprocessor(extra_numeric_cols: list = None):
    """
    Builds the ColumnTransformer preprocessor.
    extra_numeric_cols handles engineered features added by features.py
    """
    numeric = NUMERIC_COLS.copy()
    if extra_numeric_cols:
        numeric += extra_numeric_cols

    ordinal_transformers = [
        (f'ord_{col}', OrdinalEncoder(categories=[cats]), [col])
        for col, cats in ORDINAL_COLS.items()
    ]

    transformers = [
        ('num', 'passthrough', numeric),
        ('bin', OrdinalEncoder(categories=[['No', 'Yes']]), BINARY_COLS),
        ('nom', OneHotEncoder(drop='first', handle_unknown='ignore',
                              sparse_output=False), NOMINAL_COLS),
    ] + ordinal_transformers

    return ColumnTransformer(transformers=transformers)


def build_pipeline(compute: str = COMPUTE):
    """
    Builds the full pipeline for the given compute environment.
    """
    extra_numeric = [
        'Water_Stress', 'ET_Proxy', 'Soil_Health', 'Irrigation_Efficiency',
        'Water_Demand', 'THI', 'Rain_ET_Balance', 'Growth_Stage_Num',
        'Season_Num',
        # group stat features
        'Rainfall_mm_vs_group', 'Rainfall_mm_zscore',
        'Temperature_C_vs_group', 'Temperature_C_zscore',
        'Soil_Moisture_vs_group', 'Soil_Moisture_zscore',
        'ET_Proxy_vs_group', 'ET_Proxy_zscore',
    ]

    preprocessor = build_preprocessor(extra_numeric_cols=extra_numeric)

    model = XGBClassifier(
        **MODEL_PARAMS[compute],
        n_jobs=-1
    )

    return Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])


def run_cv(pipe, X, y_encoded, compute: str = COMPUTE):
    """
    Runs stratified CV. Expects y to already be label encoded.
    """
    sample_weights = compute_sample_weight('balanced', y=y_encoded)
    cv = StratifiedKFold(
        n_splits=CV_FOLDS[compute],
        shuffle=True,
        random_state=RANDOM_STATE
    )

    scores = cross_val_score(
        pipe, X, y_encoded,
        cv=cv,
        scoring='balanced_accuracy',
        params={'model__sample_weight': sample_weights}
    )

    print(f"CV Balanced Accuracy: {scores.mean():.4f} +/- {scores.std():.4f}")
    return scores