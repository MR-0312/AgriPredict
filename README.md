# AgriPredict
# 🌾 Crop Recommendation System (ML)

A **Machine Learning-based Crop Recommendation System** that predicts the best crop to grow based on soil parameters like **Nitrogen, Phosphorus, Potassium, pH, Rainfall, and Temperature**.

## ✅ Features
- **Trained using Random Forest (Optimized)**
- **Accuracy: ~93.86%** after Hyperparameter Tuning
- **Includes a Web App (Streamlit) for easy usage**

---

## 📂 Project Structure
```
├── Train Dataset.csv        # Training data for model
├── Test Dataset.csv         # Test data for evaluation
├── train_ml_model.py        # Script to train & save ML model
├── test_ml_model.py         # Evaluate model on test dataset
├── app_ml.py                # Streamlit UI for predictions
├── final_rf_crop_model.pkl  # Saved trained model
├── label_encoder.pkl        # Label encoder for crop classes
├── scaler.pkl               # Scaler for data normalization
├── requirements.txt         # Dependencies
├── README.md                # Documentation
```

---

## ⚡ Installation & Setup

### 1️⃣ Install Dependencies
First, install the required Python packages:
```bash
pip install -r requirements.txt
```

### 2️⃣ Train the Model (If not already trained)
Run the script to train the Random Forest model on the dataset:
```bash
python train_ml_model.py
```
✅ Trains & tunes the model  
✅ Saves the final trained model

### 🚀 Running the Tests
To evaluate the trained model on the test dataset, run:
```bash
python test_ml_model.py
```
✅ Displays final test accuracy  
✅ Shows sample predictions

### 🌍 Running the Streamlit Web App
Launch the web app using:
```bash
streamlit run app_ml.py
```
➡️ Open the browser URL (usually http://localhost:8501)  
➡️ Enter soil parameters  
➡️ Get crop recommendations instantly!

---

## 📊 Model Performance
| Model | Accuracy |
|--------|----------|
| Random Forest (Optimized) | **93.86%** |

🚀 **Final Choice:** Random Forest (Balanced Performance & Efficiency)

---

## 🛠 Future Improvements
- 🔹 Improve hyperparameter tuning for higher accuracy
- 🔹 Add more soil parameters for better predictions
- 🔹 Deploy the model using Flask or FastAPI

---

## 💡 Author & Credits
Developed by **[Your Name]**  
Inspired by **Agriculture + AI** solutions 🚀

---

## 📌 Now, Run It! 🚀
```bash
python train_ml_model.py   # Train the model
python test_ml_model.py    # Evaluate the model
streamlit run app_ml.py    # Run the web app
```

🌱 **Happy Farming with AI!** 🌾

