# Experiment Log

## baseline-rf (Random Forest, no feature engineering)
- **Branch:** playground-s6e4-baseline
- **Model:** RandomForestClassifier
- **CV Balanced Accuracy:** 0.9554 +/- 0.0014
- **Holdout Balanced Accuracy:** 0.9544
- **LB Score:** TBD
- **Notes:** Baseline, no feature engineering, class_weight='balanced'

## xgb-feateng-v1 (XGBoost + feature engineering)
- **Branch:** playground-s6e4-baseline
- **Model:** XGBClassifier
- **CV Balanced Accuracy:** .9674 +/- 0.0013
- **Holdout Balanced Accuracy:** 0.9655
- **LB Score:** 0.95979
- **Notes:** Added domain features, group stats, ordinal cycle features