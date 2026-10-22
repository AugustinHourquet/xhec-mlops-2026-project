# This module pickles and unpickles objects such as the trained model.

import pickle
from pathlib import Path
from typing import Any


def pickle_object(obj: Any, path: Path) -> None:
    """Save an object as a pickle file, creating the parent folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path) -> Any:
    """Load an object from a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)
