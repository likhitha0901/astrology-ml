import os
import joblib
import json

def save_pipeline(model, scaler, label_encoder, path="models/"):
    os.makedirs(path, exist_ok=True)
    joblib.dump(model, os.path.join(path, "astrology_model.pkl"))
    joblib.dump(scaler, os.path.join(path, "scaler.pkl"))
    joblib.dump(label_encoder, os.path.join(path, "label_encoder.pkl"))
    print(f"✅ Model, scaler, and label encoder saved to {path}")

def load_pipeline(path="models/"):
    model = joblib.load(os.path.join(path, "astrology_model.pkl"))
    scaler = joblib.load(os.path.join(path, "scaler.pkl"))
    label_encoder = joblib.load(os.path.join(path, "label_encoder.pkl"))
    return model, scaler, label_encoder

def save_json(data: dict, filepath: str):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

def load_json(filepath: str) -> dict:
    with open(filepath, "r") as f:
        return json.load(f)
