import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load datasets
train_df = pd.read_csv("Train Dataset.csv")
test_df = pd.read_csv("Test Dataset.csv")

# Drop unnecessary index column
train_df = train_df.drop(columns=["Unnamed: 0"])
test_df = test_df.drop(columns=["Unnamed: 0"])

# Encode crop labels
label_encoder = LabelEncoder()
train_df["Crop"] = label_encoder.fit_transform(train_df["Crop"])
test_df["Crop"] = label_encoder.transform(test_df["Crop"])

# Normalize numerical features
scaler = StandardScaler()
feature_columns = ["N", "P", "K", "pH", "rainfall", "temperature"]
train_df[feature_columns] = scaler.fit_transform(train_df[feature_columns])
test_df[feature_columns] = scaler.transform(test_df[feature_columns])

# Convert to PyTorch tensors
X_train = torch.tensor(train_df[feature_columns].values, dtype=torch.float32)
y_train = torch.tensor(train_df["Crop"].values, dtype=torch.long)
X_test = torch.tensor(test_df[feature_columns].values, dtype=torch.float32)
y_test = torch.tensor(test_df["Crop"].values, dtype=torch.long)

# PyTorch Dataset
class CropDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Dataloaders
batch_size = 64
train_loader = DataLoader(CropDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(CropDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

# Define Neural Network
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

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CropClassifier(len(feature_columns), len(label_encoder.classes_)).to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

# Save the model
torch.save(model.state_dict(), "crop_classifier.pth")
print("Model saved as crop_classifier.pth")
