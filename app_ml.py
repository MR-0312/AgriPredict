import streamlit as st
import joblib
import numpy as np

# Load trained model & preprocessors
model = joblib.load("final_rf_crop_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit UI
st.title("🌱 Crop Recommendation System - ML Model")
st.write("Enter soil parameters to get the best crop recommendation using the trained Random Forest model.")

# User Input Fields
N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, step=1, value=50)
P = st.number_input("Phosphorus (P)", min_value=0, max_value=200, step=1, value=50)
K = st.number_input("Potassium (K)", min_value=0, max_value=200, step=1, value=50)
pH = st.number_input("pH Level", min_value=3.0, max_value=9.0, step=0.1, value=6.5)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=2000.0, step=1.0, value=500.0)
temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, step=0.1, value=25.0)

# Predict Crop
if st.button("Predict Crop"):
    # Prepare input
    input_features = np.array([[N, P, K, pH, rainfall, temperature]])
    input_features = scaler.transform(input_features)
    
    # Predict
    predicted_class = model.predict(input_features)[0]
    predicted_crop = label_encoder.inverse_transform([predicted_class])[0]

    st.success(f"✅ Recommended Crop: **{predicted_crop}**")
