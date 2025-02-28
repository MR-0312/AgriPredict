import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import accuracy_score

# Load dataset
train_df = pd.read_csv("Train Dataset.csv")
test_df = pd.read_csv("Test Dataset.csv")

# Drop unnecessary column
train_df = train_df.drop(columns=["Unnamed: 0"])
test_df = test_df.drop(columns=["Unnamed: 0"])

# Encode crop labels
label_encoder = LabelEncoder()
train_df["Crop"] = label_encoder.fit_transform(train_df["Crop"])
test_df["Crop"] = label_encoder.transform(test_df["Crop"])

# Normalize features
scaler = StandardScaler()
feature_columns = ["N", "P", "K", "pH", "rainfall", "temperature"]
train_df[feature_columns] = scaler.fit_transform(train_df[feature_columns])
test_df[feature_columns] = scaler.transform(test_df[feature_columns])

# Split into training and testing sets
X_train, y_train = train_df[feature_columns], train_df["Crop"]
X_test, y_test = test_df[feature_columns], test_df["Crop"]

# Define models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='rbf', C=10, gamma='scale'),
    "XGBoost": xgb.XGBClassifier(objective="multi:softmax", num_class=len(label_encoder.classes_), eval_metric="mlogloss")
}

# Train and evaluate models
results = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"{name} Accuracy: {accuracy:.2f}")

    # Save the model
    joblib.dump(model, f"{name.lower().replace(' ', '_')}_crop_model.pkl")
    print(f"Model saved as {name.lower().replace(' ', '_')}_crop_model.pkl\n")

# Print final comparison
print("\n==== Model Performance Comparison ====")
for name, acc in results.items():
    print(f"{name}: {acc:.2f}")

# Recommend the best model
best_model = max(results, key=results.get)
print(f"\n🎯 Best Model: {best_model} with {results[best_model]:.2f} accuracy")
