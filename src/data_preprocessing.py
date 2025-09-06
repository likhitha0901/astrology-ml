import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load dataset from CSV file.
    """
    return pd.read_csv(filepath)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values and basic cleaning.
    """
    df = df.dropna()  # simple approach (can replace with imputation later)
    return df

def encode_features(df: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
    """
    Encode categorical features using LabelEncoder.
    """
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    return df

def scale_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Scale numerical features using StandardScaler.
    """
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df

def preprocess_data(filepath: str, categorical_cols: list, feature_cols: list, label_col: str):
    """
    Complete preprocessing pipeline.
    """
    df = load_data(filepath)
    df = clean_data(df)
    df = encode_features(df, categorical_cols)
    df = scale_features(df, feature_cols)
    X = df[feature_cols]
    y = df[label_col]
    return X, y, df
