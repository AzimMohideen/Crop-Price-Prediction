import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from flask import Flask, request, jsonify

# Directory containing crop CSV files
crop_files = ["Arhar.csv", "Cotton.csv", "Paddy.csv", "Ragi.csv", "Moong.csv", "Maize.csv", "Soyabean.csv", "Wheat.csv", "Barley.csv", "Sugarcane.csv"]
updated_data_path = "updated_dataset.csv"

# Load the updated dataset
updated_data = pd.read_csv(updated_data_path)

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

def train_crop_model(crop_name, crop_data):
    print(f"Training model for {crop_name}...")
    
    # Merge with updated dataset on Year and Month
    merged_data = pd.merge(crop_data, updated_data, on=['Year', 'Month'], how='left')
    merged_data.dropna(inplace=True)  # Remove null values
    
    # Define Features and Target
    X = merged_data[['Month', 'Year', 'Rainfall', 'Rainfall_Deviation', 'Flood_Flag']]
    y = merged_data['WPI']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100, random_state=42))
    model.fit(X_train, y_train)
    
    # Save model
    model_filename = f"models/{crop_name}_model.pkl"
    joblib.dump(model, model_filename)
    print(f"Model saved: {model_filename}\n")

# Train models for each crop
for crop_file in crop_files:
    crop_name = crop_file.split('.')[0]
    crop_data = pd.read_csv(f"C:/Users/azimm/OneDrive/Documents/cpp2/cropsdataset/{crop_file}")
    train_crop_model(crop_name, crop_data)

print("All crop models trained and saved successfully!")

# Flask Web Application
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    crop_name = data.get("crop")
    model_path = f"models/{crop_name}_model.pkl"
    
    if not os.path.exists(model_path):
        return jsonify({"error": "Model not found for the specified crop"})
    
    model = joblib.load(model_path)
    df = pd.DataFrame([data])
    df = df[['Month', 'Year', 'Rainfall', 'Rainfall_Deviation', 'Flood_Flag']]
    prediction = model.predict(df)[0]
    
    return jsonify({'Predicted WPI': prediction})

if __name__ == '__main__':
    app.run(debug=True)
