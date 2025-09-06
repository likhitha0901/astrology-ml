import os
import joblib
import json

def save_model(model, path: str):
    """
    Save trained model to disk.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"✅ Model saved at {path}")

def load_model(path: str):
    """
    Load trained model from disk.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Model not found at {path}")
    return joblib.load(path)

def save_json(data: dict, filepath: str):
    """
    Save dictionary to JSON file.
    """
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

def load_json(filepath: str) -> dict:
    """
    Load dictionary from JSON file.
    """
    with open(filepath, "r") as f:
        return json.load(f)
