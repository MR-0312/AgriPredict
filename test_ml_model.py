import joblib
import numpy as np

# Load trained model & preprocessors
model = joblib.load("final_rf_crop_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")

# Test Sample (Modify these values for testing)
test_sample = np.array([[50, 60, 40, 6.5, 500, 25]])  # N, P, K, pH, Rainfall, Temperature

# Scale input
test_sample = scaler.transform(test_sample)

# Predict crop
predicted_class = model.predict(test_sample)[0]
predicted_crop = label_encoder.inverse_transform([predicted_class])[0]

print(f"Predicted Crop: {predicted_crop}")
