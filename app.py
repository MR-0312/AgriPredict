import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataset for preprocessing
train_df = pd.read_csv("Train Dataset.csv")
train_df = train_df.drop(columns=["Unnamed: 0"])

# Prepare encoders and scalers
label_encoder = LabelEncoder()
train_df["Crop"] = label_encoder.fit_transform(train_df["Crop"])

scaler = StandardScaler()
feature_columns = ["N", "P", "K", "pH", "rainfall", "temperature"]
train_df[feature_columns] = scaler.fit_transform(train_df[feature_columns])

# Define Model
class CropClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CropClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CropClassifier(len(feature_columns), len(label_encoder.classes_))
model.load_state_dict(torch.load("crop_classifier.pth", map_location=device))
model.to(device)
model.eval()

# Streamlit UI
st.title("🌾 Crop Recommendation System")
st.write("Enter soil parameters to get the best crop recommendation.")

# Input fields
N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, step=1, value=50)
P = st.number_input("Phosphorus (P)", min_value=0, max_value=200, step=1, value=50)
K = st.number_input("Potassium (K)", min_value=0, max_value=200, step=1, value=50)
pH = st.number_input("pH Level", min_value=3.0, max_value=9.0, step=0.1, value=6.5)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=2000.0, step=1.0, value=500.0)
temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, step=0.1, value=25.0)

# Normalize input
input_features = np.array([[N, P, K, pH, rainfall, temperature]])
input_features = scaler.transform(input_features)
input_tensor = torch.tensor(input_features, dtype=torch.float32).to(device)

# Predict
if st.button("Predict Crop"):
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
        predicted_crop = label_encoder.inverse_transform([predicted_class])[0]
    st.success(f"✅ Recommended Crop: **{predicted_crop}**")
