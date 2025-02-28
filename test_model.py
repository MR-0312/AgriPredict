import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch.nn as nn

# Load dataset (for scaling & encoding)
test_df = pd.read_csv("Test Dataset.csv")
test_df = test_df.drop(columns=["Unnamed: 0"])

# Load the same encoder and scaler used in training
label_encoder = LabelEncoder()
test_df["Crop"] = label_encoder.fit_transform(test_df["Crop"])

scaler = StandardScaler()
feature_columns = ["N", "P", "K", "pH", "rainfall", "temperature"]
test_df[feature_columns] = scaler.fit_transform(test_df[feature_columns])

# Load trained model
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CropClassifier(len(feature_columns), len(label_encoder.classes_))
model.load_state_dict(torch.load("crop_classifier.pth"))
model.to(device)
model.eval()

# Function to predict a crop
def predict_crop(features):
    features = torch.tensor(features, dtype=torch.float32).to(device)
    features = features.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(features)
        predicted_class = torch.argmax(output, dim=1).item()
    return label_encoder.inverse_transform([predicted_class])[0]

# Test on a sample from the dataset
sample = test_df.iloc[0][feature_columns].values
predicted_crop = predict_crop(sample)
print(f"Predicted Crop: {predicted_crop}")
