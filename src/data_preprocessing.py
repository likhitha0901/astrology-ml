import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Feature and target definitions
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

TARGET = "zodiac_label"

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load dataset from CSV file.
    """
    return pd.read_csv(filepath)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values by dropping rows (simple method).
    More advanced: you could use imputation here.
    """
    return df.dropna()

def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode the zodiac_label column into numeric values.
    """
    le = LabelEncoder()
    df[TARGET] = le.fit_transform(df[TARGET])
    return df, le

def scale_features(X: pd.DataFrame) -> (pd.DataFrame, StandardScaler):
    """
    Standardize numerical features.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def preprocess_data(filepath: str):
    """
    Complete preprocessing pipeline.
    Returns:
        X_scaled: Scaled feature matrix
        y: Encoded target labels
        df: Cleaned DataFrame
        scaler: Fitted scaler
        label_encoder: Fitted label encoder
    """
    # Load
    df = load_data(filepath)

    # Clean
    df = clean_data(df)

    # Encode target
    df, label_encoder = encode_target(df)

    # Split features and target
    X = df[FEATURES]
    y = df[TARGET]

    # Scale features
    X_scaled, scaler = scale_features(X)

    return X_scaled, y, df, scaler, label_encoder
