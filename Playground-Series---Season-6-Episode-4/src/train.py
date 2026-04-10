import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, classification_report

from src.config import (
    COMPUTE,
    MODE,
    DATA_PATH,
    TEST_PATH,
    SUBMISSION_PATH,
    TARGET,
    ID_COL,
    RANDOM_STATE,
    TEST_SIZE
)
from src.features import create_features
from src.models import build_pipeline, run_cv, encode_target


def load_data(compute: str = COMPUTE):
    print(f"Running on: {compute}")
    train = pd.read_csv(DATA_PATH[compute])
    test = pd.read_csv(TEST_PATH[compute])
    return train, test


def prepare(df: pd.DataFrame):
    df = create_features(df)
    return df


def main():
    # ─── Load ─────────────────────────────────────────────────────────────────
    train, test = load_data()

    # ─── Feature engineering ──────────────────────────────────────────────────
    train = prepare(train)
    test = prepare(test)

    # ─── Split off ID and target ──────────────────────────────────────────────
    test_ids = test[ID_COL]
    X = train.drop(columns=[ID_COL, TARGET])
    y = train[TARGET]
    X_test = test.drop(columns=[ID_COL])

    # ─── Encode target ────────────────────────────────────────────────────────
    y_encoded, le = encode_target(y)

    # ─── Train / holdout split ────────────────────────────────────────────────
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y_encoded,
        test_size=TEST_SIZE,
        stratify=y_encoded,
        random_state=RANDOM_STATE
    )

    # ─── Build pipeline ───────────────────────────────────────────────────────
    if MODE == "tune":
        from src.tune import run_tuning
        from xgboost import XGBClassifier
        from src.models import build_preprocessor
        from sklearn.pipeline import Pipeline

        print("\nRunning Optuna HPO...")
        best_params = run_tuning(X_train, y_train)

        pipe = Pipeline([
            ('preprocessor', build_preprocessor(extra_numeric_cols=[
                'Water_Stress', 'ET_Proxy', 'Soil_Health', 'Irrigation_Efficiency',
                'Water_Demand', 'THI', 'Rain_ET_Balance', 'Growth_Stage_Num',
                'Season_Num',
                'Rainfall_mm_vs_group', 'Rainfall_mm_zscore',
                'Temperature_C_vs_group', 'Temperature_C_zscore',
                'Soil_Moisture_vs_group', 'Soil_Moisture_zscore',
                'ET_Proxy_vs_group', 'ET_Proxy_zscore',
            ])),
            ('model', XGBClassifier(**best_params, n_jobs=-1))
        ])

    elif MODE == "train":
        print("\nBuilding pipeline with config params...")
        pipe = build_pipeline()

    # ─── CV ───────────────────────────────────────────────────────────────────
    print("\nRunning CV...")
    run_cv(pipe, X_train, y_train)

    # ─── Holdout eval ─────────────────────────────────────────────────────────
    print("\nFitting on train, evaluating on holdout...")
    pipe.fit(X_train, y_train)
    holdout_preds = pipe.predict(X_holdout)
    holdout_score = balanced_accuracy_score(y_holdout, holdout_preds)
    print(f"Holdout Balanced Accuracy: {holdout_score:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        y_holdout, holdout_preds,
        target_names=le.classes_
    ))

    # ─── Refit on full training data ──────────────────────────────────────────
    print("\nRefitting on full training data...")
    pipe.fit(X, y_encoded)

    # ─── Generate submission ──────────────────────────────────────────────────
    print("\nGenerating submission...")
    submission_preds = le.inverse_transform(pipe.predict(X_test))
    submission = pd.DataFrame({
        ID_COL: test_ids,
        TARGET: submission_preds
    })

    submission.to_csv(SUBMISSION_PATH[COMPUTE], index=False)
    print(f"Submission saved to: {SUBMISSION_PATH[COMPUTE]}")
    print(submission.head())
    print(submission[TARGET].value_counts())

if __name__ == "__main__":
    main() 


