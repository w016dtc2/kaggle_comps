import pandas as pd
import numpy as np
from src.config import NUMERIC_COLS

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    All feature engineering in one place.
    Called identically on train and test data.
    """
    df = df.copy()

    # ─── Ordinal cycle features ───────────────────────────────────────────────

    # Crop growth stage as numeric cycle position
    growth_order = ['Sowing', 'Vegetative', 'Flowering', 'Harvest']
    df['Growth_Stage_Num'] = df['Crop_Growth_Stage'].map(
        {stage: i for i, stage in enumerate(growth_order)}
    )

    # Season as numeric cycle position
    season_order = ['Rabi', 'Zaid', 'Kharif']
    df['Season_Num'] = df['Season'].map(
        {season: i for i, season in enumerate(season_order)}
    )

    # ─── Domain-driven interaction features ───────────────────────────────────

    # Water stress — how much rain relative to temp and moisture demand
    df['Water_Stress'] = df['Rainfall_mm'] / (
        df['Temperature_C'] * df['Soil_Moisture'] + 1
    )

    # Evapotranspiration proxy — heat + sun drives water loss
    df['ET_Proxy'] = (
        df['Temperature_C'] * df['Sunlight_Hours'] * (1 - df['Humidity'] / 100)
    )

    # Soil health composite
    df['Soil_Health'] = (
        df['Organic_Carbon'] * df['Soil_pH'] / (df['Electrical_Conductivity'] + 1)
    )

    # Previous irrigation efficiency — did past irrigation actually raise moisture?
    df['Irrigation_Efficiency'] = (
        df['Soil_Moisture'] / (df['Previous_Irrigation_mm'] + 1)
    )

    # Field scale water demand
    df['Water_Demand'] = df['Field_Area_hectare'] * df['ET_Proxy']

    # Temperature humidity index — heat stress on crops
    df['THI'] = df['Temperature_C'] * (1 - 0.55 * (1 - df['Humidity'] / 100))

    # Rainfall vs evapotranspiration balance
    df['Rain_ET_Balance'] = df['Rainfall_mm'] - df['ET_Proxy']

    # ─── Group-level statistical features ─────────────────────────────────────
    group_cols = ['Region', 'Crop_Type', 'Season']
    stat_cols = ['Rainfall_mm', 'Temperature_C', 'Soil_Moisture', 'ET_Proxy']

    for col in stat_cols:
        group_mean = df.groupby(group_cols)[col].transform('mean')
        group_std = df.groupby(group_cols)[col].transform('std').fillna(0)

        # How does this field compare to its regional/crop/season context
        df[f'{col}_vs_group'] = df[col] - group_mean
        df[f'{col}_zscore'] = (df[col] - group_mean) / (group_std + 1e-5)

    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    """
    Returns all feature column names after engineering.
    Excludes id and target.
    """
    exclude = ['id', 'Irrigation_Need']
    return [c for c in df.columns if c not in exclude]