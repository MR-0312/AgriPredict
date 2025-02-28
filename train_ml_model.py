import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
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

# Train-Test Split
X_train, y_train = train_df[feature_columns], train_df["Crop"]
X_test, y_test = test_df[feature_columns], test_df["Crop"]

# Define base Random Forest model
rf = RandomForestClassifier(random_state=42)

# Define hyperparameter grid (Improved Tuning)
param_dist = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 5, 10, 15],
    "min_samples_leaf": [1, 2, 4, 6],
    "bootstrap": [True, False]
}

# Randomized Search (Faster Tuning)
random_search = RandomizedSearchCV(
    rf, param_distributions=param_dist, 
    n_iter=15, cv=3, scoring="accuracy", n_jobs=-1, verbose=1, random_state=42
)

# Train with hyperparameter tuning
random_search.fit(X_train, y_train)

# Best model after tuning
best_rf = random_search.best_estimator_
print(f"Best Hyperparameters: {random_search.best_params_}")

# Evaluate on test set
y_pred = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Final Tuned RF Accuracy: {accuracy:.4f}")  # More precision

# Save best model & encoders
joblib.dump(best_rf, "final_rf_crop_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Final model & encoders saved!")
