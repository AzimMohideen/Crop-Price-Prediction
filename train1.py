import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error

# Load dataset
dataset_path = "merged.csv"  # Ensure it's in your working directory
df = pd.read_csv(dataset_path)

# Fix duplicate column issue
df = df.loc[:, ~df.columns.duplicated()]

# Create standardized column names to match what was used in training
if 'Rainfall_x' in df.columns and 'Rainfall' not in df.columns:
    df['Rainfall'] = df['Rainfall_x']
if 'WPI_x' in df.columns and 'WPI' not in df.columns:
    df['WPI'] = df['WPI_x']
if 'WPI_y' in df.columns and 'WPI' not in df.columns:
    df['WPI'] = df['WPI_y']

# Add missing columns if they don't exist
if 'Rainfall_Deviation' not in df.columns:
    # Calculate deviation from mean rainfall for each crop
    crop_mean_rainfalls = df.groupby('Crop')['Rainfall'].transform('mean')
    df['Rainfall_Deviation'] = df['Rainfall'] - crop_mean_rainfalls

if 'Flood_Flag' not in df.columns:
    # Create a simple flood flag based on high rainfall
    crop_rainfall_std = df.groupby('Crop')['Rainfall'].transform('std')
    df['Flood_Flag'] = ((df['Rainfall'] > (df.groupby('Crop')['Rainfall'].transform('mean') + 1.5 * crop_rainfall_std)) * 1)

# Get available models
model_dir = "models"
crop_models = {f.split("_rainfall_model.pkl")[0]: os.path.join(model_dir, f) 
               for f in os.listdir(model_dir) if f.endswith("_rainfall_model.pkl")}

# Dictionary to store accuracy results
accuracy_results = {}

for crop, model_path in crop_models.items():
    if crop not in df['Crop'].unique():
        continue  # Skip crops not present in the dataset

    print(f"Evaluating model for {crop}...")

    # Load model
    model = joblib.load(model_path)

    # Filter dataset for the crop
    crop_data = df[df['Crop'] == crop].copy()

    # Load thresholds to get the feature names used during training
    threshold_path = os.path.join(model_dir, f"{crop}_thresholds.pkl")
    if os.path.exists(threshold_path):
        thresholds = joblib.load(threshold_path)
        
        # Add Rainfall_Category based on thresholds
        mean_rainfall = thresholds.get('mean_rainfall', crop_data['Rainfall'].mean())
        std_rainfall = thresholds.get('std_rainfall', crop_data['Rainfall'].std())
        deficient_threshold = thresholds.get('deficient_threshold', mean_rainfall - 0.75 * std_rainfall)
        excessive_threshold = thresholds.get('excessive_threshold', mean_rainfall + 0.75 * std_rainfall)
        
        crop_data['Rainfall_Category'] = 0  # Default to normal
        crop_data.loc[crop_data['Rainfall'] > excessive_threshold, 'Rainfall_Category'] = 1  # Excessive
        crop_data.loc[crop_data['Rainfall'] < deficient_threshold, 'Rainfall_Category'] = -1  # Deficient
    else:
        # If no thresholds file, create a simple category
        crop_data['Rainfall_Category'] = 0  # Default to normal

    # Define features based on what the model expects
    try:
        if hasattr(model, 'feature_names_in_'):
            # For scikit-learn 1.0+ models, use the feature names from the model
            features = model.feature_names_in_
            X = crop_data[features]
        else:
            # Default features
            X = crop_data[['Month', 'Year', 'Rainfall', 'Rainfall_Category', 'Rainfall_Deviation']]
        
        # Use WPI as target (try different column names)
        if 'WPI' in crop_data.columns:
            y_actual = crop_data['WPI']
        elif 'WPI_y' in crop_data.columns:
            y_actual = crop_data['WPI_y']
        elif 'WPI_x' in crop_data.columns:
            y_actual = crop_data['WPI_x']
        else:
            print(f"No WPI column found for {crop}, skipping...")
            continue

        # Predict using the model
        y_predicted = model.predict(X)

        # Calculate accuracy using MAPE
        mape = mean_absolute_percentage_error(y_actual, y_predicted)
        accuracy = 100 - (mape * 100)
        accuracy_results[crop] = accuracy
        
        print(f"Accuracy for {crop}: {accuracy:.2f}%")
        
    except Exception as e:
        print(f"Error evaluating {crop}: {str(e)}")
        import traceback
        traceback.print_exc()

# Only create plots if we have results
if accuracy_results:
    # Sort crops alphabetically for better visualization
    sorted_crops = sorted(accuracy_results.keys())
    sorted_accuracies = [accuracy_results[crop] for crop in sorted_crops]

    # *Line Graph - Accuracy Trend*
    plt.figure(figsize=(10, 5))
    plt.plot(sorted_crops, sorted_accuracies, marker='o', linestyle='-', color='blue', label="Accuracy Trend")
    plt.xlabel("Crop")
    plt.ylabel("Accuracy (%)")
    plt.title("Crop Price Prediction Model Accuracy (Line Graph)")
    plt.xticks(rotation=45)
    plt.ylim(min(sorted_accuracies) - 5, 100)  # Dynamic lower bound
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('accuracy_line.png')
    plt.show()

    # *Scatter Plot - Accuracy Distribution*
    plt.figure(figsize=(10, 5))
    plt.scatter(sorted_crops, sorted_accuracies, color='red', label="Accuracy Points", s=100)
    plt.xlabel("Crop")
    plt.ylabel("Accuracy (%)")
    plt.title("Crop Price Prediction Model Accuracy (Scatter Plot)")
    plt.xticks(rotation=45)
    plt.ylim(min(sorted_accuracies) - 5, 100)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('accuracy_scatter.png')
    plt.show()
else:
    print("No accuracy results to display. Check for errors above.")