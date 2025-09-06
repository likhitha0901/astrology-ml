import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# ----------------------------
# 1. Load dataset
# ----------------------------
DATA_PATH = "data/palmistry_data.csv"
df = pd.read_csv(DATA_PATH)

print("✅ Dataset loaded. Shape:", df.shape)
print(df.head())

# ----------------------------
# 2. Define features and target
# ----------------------------
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

X = df[FEATURES]
y = df[TARGET]

# ----------------------------
# 3. Train/Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# 4. Feature Scaling
# ----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# 5. Train Model
# ----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# ----------------------------
# 6. Evaluate
# ----------------------------
y_pred = model.predict(X_test_scaled)

print("\n🎯 Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ----------------------------
# 7. Save Model + Scaler
# ----------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/astrology_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Model and scaler saved in 'models/'")

