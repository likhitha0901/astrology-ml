import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

FEATURES = [
    "palm_length_cm", "palm_width_cm", "index_to_ring_ratio",
    "heart_line_depth", "head_line_length", "life_line_length",
    "fate_line_presence", "mount_apollo", "mount_mars", "heart_line_breaks"
]

TARGET = "zodiac_label"

def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna()

def encode_target(df: pd.DataFrame):
    le = LabelEncoder()
    df[TARGET] = le.fit_transform(df[TARGET])
    return df, le

def scale_features(X: pd.DataFrame):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def preprocess_data(filepath: str):
    df = load_data(filepath)
    df = clean_data(df)
    df, le = encode_target(df)
    X = df[FEATURES]
    y = df[TARGET]
    X_scaled, scaler = scale_features(X)
    return X_scaled, y, df, scaler, le
