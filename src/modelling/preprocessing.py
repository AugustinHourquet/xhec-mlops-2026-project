# This module loads and cleans the abalone data and builds the model features (see notebooks/eda.ipynb).

import pandas as pd

TARGET = "Rings"


def load_data(path: str) -> pd.DataFrame:
    """Load the abalone dataset."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop impossible values, then feature outliers beyond 3*IQR. The target is never filtered."""
    features = df.select_dtypes("number").columns.drop(TARGET)
    df = df[(df[features] > 0).all(axis=1)]

    q1 = df[features].quantile(0.25)
    q3 = df[features].quantile(0.75)
    iqr = q3 - q1
    outliers = ((df[features] < q1 - 3 * iqr) | (df[features] > q3 + 3 * iqr)).any(
        axis=1
    )
    return df[~outliers].reset_index(drop=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode Sex as infant vs adult and drop the target if present."""
    return df.drop(columns=[TARGET, "Sex"], errors="ignore").assign(
        is_infant=(df["Sex"] == "I").astype(int)
    )


def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Clean the raw data and split the features from the target."""
    df = clean_data(df)
    return build_features(df), df[TARGET]
