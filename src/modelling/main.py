# This module is the training flow: it reads the data, preprocesses it, trains a model and saves it.

import argparse
from pathlib import Path

from predicting import predict
from preprocessing import load_data, preprocess_data
from sklearn.model_selection import train_test_split
from training import RANDOM_STATE, evaluate_model, train_model
from utils import pickle_object

MODEL_PATH = Path(__file__).parents[1] / "web_service" / "local_objects" / "model.pkl"


def main(trainset_path: Path) -> None:
    """Train a model using the data at the given path and save the model (pickle)."""
    # Read data
    df = load_data(trainset_path)

    # Preprocess data
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # Train model
    model = train_model(X_train, y_train)
    print(f"Test metrics: {evaluate_model(y_test, predict(model, X_test))}")

    # Pickle model
    pickle_object(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model using the data at the given path."
    )
    parser.add_argument("trainset_path", type=str, help="Path to the training set")
    args = parser.parse_args()
    main(Path(args.trainset_path))
