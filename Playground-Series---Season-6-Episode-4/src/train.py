import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight

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
        from src.tune import run_tuning, compute_aggressive_weights
        from xgboost import XGBClassifier
        from src.models import build_preprocessor
        from sklearn.pipeline import Pipeline

        print("\nRunning Optuna HPO...")
        best_params = run_tuning(X_train, y_train)

        high_multiplier = best_params.pop('high_multiplier')
        print(f"Best high_multiplier: {high_multiplier:.4f}")

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

        train_weights = compute_aggressive_weights(y_train, high_multiplier)
        full_weights = compute_aggressive_weights(y_encoded, high_multiplier)

    elif MODE == "train":
        high_multiplier = 3.0
        train_weights = compute_aggressive_weights(y_train, high_multiplier)
        full_weights = compute_aggressive_weights(y_encoded, high_multiplier)

        if MODEL_MODE == "single":
            print("\nBuilding single pipeline with config params...")
            pipe = build_pipeline()

        elif MODEL_MODE == "ensemble":
            from src.ensemble import run_ensemble, run_ensemble_cv
            print("\nRunning ensemble...")

    # ─── CV + Holdout ─────────────────────────────────────────────────────────────
    if MODEL_MODE == "single" or MODE == "tune":
        print("\nRunning CV...")
        run_cv(pipe, X_train, y_train, sample_weights=train_weights)

        print("\nFitting on train, evaluating on holdout...")
        pipe.fit(X_train, y_train, model__sample_weight=train_weights)
        holdout_preds = pipe.predict(X_holdout)
        holdout_score = balanced_accuracy_score(y_holdout, holdout_preds)
        print(f"Holdout Balanced Accuracy: {holdout_score:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_holdout, holdout_preds, target_names=le.classes_))

        print("\nRefitting on full training data...")
        pipe.fit(X, y_encoded, model__sample_weight=full_weights)
        final_preds = pipe.predict(X_test)
        submission_preds = le.inverse_transform(final_preds)

    elif MODEL_MODE == "ensemble" and MODE == "train":
        print("\nRunning ensemble CV...")
        run_ensemble_cv(X_train, y_train, sample_weights=train_weights)

        print("\nFitting ensemble on full data...")
        final_preds, pipes = run_ensemble(
            X, X_test, y_encoded,
            sample_weights=full_weights,
            voting='soft'
        )
        submission_preds = le.inverse_transform(final_preds)

    # ─── Generate submission ──────────────────────────────────────────────────────
    print("\nGenerating submission...")
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
