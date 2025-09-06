import pandas as pd
import joblib
import argparse
import json

# Load model
model = joblib.load("models/astrology_model.pkl")

def predict(input_data):
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)
    return prediction[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input JSON file")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        input_data = json.load(f)

    result = predict(input_data)
    print("🔮 Prediction:", result)
import pandas as pd
import joblib
import argparse
import json

# Load model
model = joblib.load("models/astrology_model.pkl")

def predict(input_data):
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)
    return prediction[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input JSON file")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        input_data = json.load(f)

    result = predict(input_data)
    print("🔮 Prediction:", result)
