import pandas as pd
import joblib

# Load model & scaler
model = joblib.load("models/astrology_model.pkl")
scaler = joblib.load("models/scaler.pkl")

FEATURES = [
    "palm_length_cm",
    "palm_width_cm",
    "index_to_ring_ratio",
    "heart_line_depth",
    "head_line_length",
    "life_line_length",
    "fate_line_presence",
    "mount_apollo",
    "mount_mars",
    "heart_line_breaks"
]

def predict(features_dict):
    """
    Predict zodiac from palmistry features.
    Args:
        features_dict (dict): Dictionary with palmistry features.
    Returns:
        str: Predicted zodiac label.
    """
    df = pd.DataFrame([features_dict])[FEATURES]
    df_scaled = scaler.transform(df)
    prediction = model.predict(df_scaled)
    return prediction[0]

# Example usage
if __name__ == "__main__":
    sample = {
        "palm_length_cm": 18.5,
        "palm_width_cm": 8.0,
        "index_to_ring_ratio": 0.95,
        "heart_line_depth": 7.0,
        "head_line_length": 10.2,
        "life_line_length": 12.3,
        "fate_line_presence": 1,
        "mount_apollo": 3,
        "mount_mars": 2,
        "heart_line_breaks": 0
    }

    result = predict(sample)
    print("🔮 Predicted Zodiac:", result)

