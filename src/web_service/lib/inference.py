import pandas as pd
from sklearn.base import BaseEstimator

#from src.modelling.preprocessing import preprocess  # agree on this name with step 2
from src.web_service.lib.models import AbaloneInput

COLUMN_NAMES = {
    "sex": "Sex",
    "length": "Length",
    "diameter": "Diameter",
    "height": "Height",
    "whole_weight": "Whole weight",
    "shucked_weight": "Shucked weight",
    "viscera_weight": "Viscera weight",
    "shell_weight": "Shell weight",
}

def run_inference(payload: AbaloneInput, model: BaseEstimator) -> float:
    """Preprocess one abalone's measurements and return the predicted rings."""
    df = pd.DataFrame([payload.model_dump()]).rename(columns=COLUMN_NAMES)
    X = preprocess(df)
    return float(model.predict(X)[0])
