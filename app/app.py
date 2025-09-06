import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/astrology_model.pkl")

st.title("🌌 Astrology Predictor")
st.write("Enter your features to get astrology-based predictions.")

# Example input fields (update as per dataset)
feature1 = st.number_input("Feature 1", min_value=0, max_value=10, value=5)
feature2 = st.number_input("Feature 2", min_value=0, max_value=10, value=3)

if st.button("Predict"):
    input_data = pd.DataFrame([[feature1, feature2]], columns=["feature1", "feature2"])
    prediction = model.predict(input_data)[0]
    st.success(f"🔮 Prediction: {prediction}")
