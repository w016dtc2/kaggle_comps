# ─── Single dial to switch between environments ───────────────────────────────
COMPUTE = "cpu"  # change to "gpu" for Kaggle runs

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH = {
    "cpu": "data/sample.csv",  # small local sample for fast iteration
    "gpu": "/kaggle/input/competitions/playground-series-s6e4/train.csv"
}

TEST_PATH = {
    "cpu": "data/sample_test.csv",
    "gpu": "/kaggle/input/competitions/playground-series-s6e4/test.csv"
}

SUBMISSION_PATH = {
    "cpu": "submissions/submission_local.csv",
    "gpu": "/kaggle/working/submission.csv"
}

# ─── Columns ──────────────────────────────────────────────────────────────────
TARGET = "Irrigation_Need"
ID_COL = "id"

NUMERIC_COLS = [
    'Soil_pH', 'Soil_Moisture', 'Organic_Carbon', 'Electrical_Conductivity',
    'Temperature_C', 'Humidity', 'Rainfall_mm', 'Sunlight_Hours',
    'Wind_Speed_kmh', 'Field_Area_hectare', 'Previous_Irrigation_mm'
]

ORDINAL_COLS = {
    'Crop_Growth_Stage': ['Sowing', 'Vegetative', 'Flowering', 'Harvest'],
    'Season': ['Rabi', 'Zaid', 'Kharif']
}

BINARY_COLS = ['Mulching_Used']
NOMINAL_COLS = ['Soil_Type', 'Crop_Type', 'Irrigation_Type', 'Water_Source', 'Region']

# ─── Model params ─────────────────────────────────────────────────────────────
MODEL_PARAMS = {
    "cpu": {
        "n_estimators": 100,
        "tree_method": "hist",
        "device": "cpu",
        "learning_rate": 0.05,
        "random_state": 42,
        "eval_metric": "mlogloss"
    },
    "gpu": {
        "n_estimators": 1000,
        "tree_method": "hist",
        "device": "cuda",
        "learning_rate": 0.05,
        "random_state": 42,
        "eval_metric": "mlogloss"
    }
}

# ─── CV settings ──────────────────────────────────────────────────────────────
CV_FOLDS = {
    "cpu": 3,
    "gpu": 5
}

RANDOM_STATE = 42
TEST_SIZE = 0.2