"""Data loading, cleaning and feature engineering for the bridge condition project.

Everything the model needs between the raw columns and the classifier lives here,
so training and prediction always use exactly the same steps.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "bridge_data.csv"

TARGET = "bridge_condition"  # 1 = Good, 0 = Poor

# The only inputs a user has to provide
RAW_FEATURES = ["age_of_bridge", "traffic_volume"]

# What the classifier is trained on (selected in Sprint 3)
MODEL_FEATURES = ["age_of_bridge", "traffic_volume", "traffic_per_year", "old_and_busy"]

# Domain thresholds behind the `old_and_busy` flag (found in the Sprint 1 EDA):
# no bridge in the data is in good condition unless it is both older than
# AGE_THRESHOLD years and carries more than TRAFFIC_THRESHOLD vehicles.
AGE_THRESHOLD = 50
TRAFFIC_THRESHOLD = 2000


def load_raw_data(path=RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV file."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names, fix category spelling and drop duplicate rows."""
    out = df.copy()
    out.columns = out.columns.str.strip().str.lower().str.replace(" ", "_")
    if "maintenance_level" in out.columns:
        out["maintenance_level"] = out["maintenance_level"].replace(
            {"No-Maintainance": "No-Maintenance"}
        )
    return out.drop_duplicates().reset_index(drop=True)


def engineer_features(X) -> pd.DataFrame:
    """Turn raw age and traffic into the model features.

    Works on a DataFrame with the columns in RAW_FEATURES, so it can be used
    both while training and on a single new bridge.
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=RAW_FEATURES)

    age = X["age_of_bridge"].astype(float)
    traffic = X["traffic_volume"].astype(float)

    out = pd.DataFrame(index=X.index)
    out["age_of_bridge"] = age
    out["traffic_volume"] = traffic
    out["traffic_per_year"] = traffic / age.clip(lower=1)  # ratio feature
    out["old_and_busy"] = ((age > AGE_THRESHOLD) & (traffic > TRAFFIC_THRESHOLD)).astype(int)
    return out[MODEL_FEATURES]


def get_preprocessor() -> Pipeline:
    """Feature engineering followed by standardization (fitted on training data only)."""
    return Pipeline(
        [
            ("features", FunctionTransformer(engineer_features, validate=False)),
            ("scaler", StandardScaler()),
        ]
    )


def split_data(df: pd.DataFrame, test_size: float = 0.20, random_state: int = 42):
    """Stratified train/test split on the cleaned data."""
    X = df[RAW_FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
