# Utils file. TODO: add a `load_object` function to load pickle objects
import pickle
from pathlib import Path
from typing import Any

def load_object(data:Path) -> Any:

    with open(data, 'rb') as file:
        loaded_data = pickle.load(file)
        return loaded_data