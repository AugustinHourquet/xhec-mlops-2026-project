# Code with FastAPI (app = FastAPI(...))
from pathlib import Path
from fastapi import FastAPI

from src.web_service.lib.inference import run_inference
from src.web_service.lib.models import AbaloneInput, AbalonePrediction
from src.web_service.utils import load_object
# Other imports

app = FastAPI(title="Abalone Age Prediction Endpoint", description="Predict the number of rings of an abalone " \
"from its physical measurements")

MODEL_PATH = Path(__file__).parent / "local_objects" / "model.pkl"
model = load_object(MODEL_PATH)

@app.get("/")
def home() -> dict:
    return {"health_check": "App up and running!"}


@app.post("/predict", response_model=AbalonePrediction, status_code=201)
def predict(payload: AbaloneInput) -> dict:
    # TODO: complete and replace the "InsertHereAPydanticClass" with the correct Pydantic classes defined in web_service/lib/models.py
    return {"rings": run_inference(payload, model)}