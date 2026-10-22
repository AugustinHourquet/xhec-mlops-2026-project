# This module predicts the number of rings of abalones with a trained model.

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin


def predict(model: RegressorMixin, X: pd.DataFrame) -> np.ndarray:
    """Predict the number of rings, rounded to the nearest integer."""
    return np.round(model.predict(X)).astype(int)
