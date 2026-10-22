# This module trains the Random Forest and evaluates it (see notebooks/modelling.ipynb).

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

RANDOM_STATE = 42

# best configuration of the MLflow hyperparameter search in notebooks/modelling.ipynb
BEST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 8,
    "min_samples_leaf": 5,
    "max_features": 1.0,
}


def train_model(
    X: pd.DataFrame, y: pd.Series, params: dict | None = None
) -> RandomForestRegressor:
    """Fit a Random Forest with the given hyperparameters (the best ones by default)."""
    model = RandomForestRegressor(
        random_state=RANDOM_STATE, n_jobs=-1, **(params or BEST_PARAMS)
    )
    model.fit(X, y)
    return model


def evaluate_model(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """MAE and RMSE on predictions rounded to the nearest integer number of rings."""
    y_pred = np.round(y_pred)
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
    }
