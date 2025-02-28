import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# Load trained model & preprocessors
model = joblib.load("final_rf_crop_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")

# Load test dataset
test_df = pd.read_csv("Test Dataset.csv").drop(columns=["Unnamed: 0"])

# Extract features and target
feature_columns = ["N", "P", "K", "pH", "rainfall", "temperature"]
X_test = test_df[feature_columns]
y_test = label_encoder.transform(test_df["Crop"])  # Encode target

# Scale test data (convert back to DataFrame to keep feature names)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_columns)

# Predict on test data
y_pred = model.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Final Test Accuracy: {accuracy:.4f}")

# Show sample predictions
test_df["Predicted Crop"] = label_encoder.inverse_transform(y_pred)
print("\n🔹 Sample Predictions:")
print(test_df[["N", "P", "K", "pH", "rainfall", "temperature", "Predicted Crop"]].head(10))
