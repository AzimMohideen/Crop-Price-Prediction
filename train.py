import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# MSP data for 2023 (base prices)
MSP_DATA = {
    'Ragi': 4290,
    'Paddy': 2300,
    'Cotton': 6620,  # Medium Staple
    'Wheat': 2275,
    'Barley': 1850
}

# Optimal rainfall ranges for each crop (mm)
RAINFALL_RANGES = {
    'Wheat': {'min': 450, 'max': 650, 'optimal': 550},
    'Barley': {'min': 400, 'max': 500, 'optimal': 450},
    'Cotton': {'min': 500, 'max': 700, 'optimal': 600},
    'Paddy': {'min': 1000, 'max': 1200, 'optimal': 1100},
    'Ragi': {'min': 500, 'max': 900, 'optimal': 700}
}

# Annual growth rates (estimated)
ANNUAL_GROWTH_RATES = {
    'Wheat': 0.04,  # 4% annual growth
    'Barley': 0.035,  # 3.5% annual growth
    'Cotton': 0.05,  # 5% annual growth
    'Paddy': 0.045,  # 4.5% annual growth
    'Ragi': 0.042  # 4.2% annual growth
}

# Rainfall impact factors
RAINFALL_IMPACT = {
    'excessive': 0.35,  # 35% price increase for excessive rainfall
    'deficient': 0.40,  # 40% price increase for deficient rainfall
}

def train_crop_model(crop_name, crop_data):
    """Train a model for a specific crop and save it."""
    print(f"Training model for {crop_name}...")
    
    # Determine which columns to use
    rainfall_col = 'Rainfall_x' if 'Rainfall_x' in crop_data.columns else 'Rainfall'
    wpi_col = 'WPI_y' if 'WPI_y' in crop_data.columns else ('WPI' if 'WPI' in crop_data.columns else 'WPI_x')
    
    # Calculate rainfall statistics for thresholds
    mean_rainfall = crop_data[rainfall_col].mean()
    std_rainfall = crop_data[rainfall_col].std()
    min_rainfall = crop_data[rainfall_col].min()
    max_rainfall = crop_data[rainfall_col].max()
    
    # Calculate more precise thresholds based on historical data
    deficient_threshold = mean_rainfall - 0.75 * std_rainfall
    excessive_threshold = mean_rainfall + 0.75 * std_rainfall
    
    # Add rainfall category as a feature
    crop_data['Rainfall_Category'] = 0  # Default to normal
    crop_data.loc[crop_data[rainfall_col] > excessive_threshold, 'Rainfall_Category'] = 1  # Excessive
    crop_data.loc[crop_data[rainfall_col] < deficient_threshold, 'Rainfall_Category'] = -1  # Deficient
    
    # Add rainfall deviation from mean as a feature
    crop_data['Rainfall_Deviation'] = crop_data[rainfall_col] - mean_rainfall
    
    # Get min and max years for future predictions
    min_year = crop_data['Year'].min()
    max_year = crop_data['Year'].max()
    
    # Calculate average price for reference
    avg_price = crop_data[wpi_col].mean()
    
    # Prepare features and target
    features = ['Month', 'Year', rainfall_col, 'Rainfall_Category', 'Rainfall_Deviation']
    X = crop_data[features]
    y = crop_data[wpi_col]
    
    # Train a Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Save the model
    joblib.dump(model, f'models/{crop_name}_rainfall_model.pkl')
    
    # Save thresholds and statistics
    thresholds = {
        'mean_rainfall': mean_rainfall,
        'std_rainfall': std_rainfall,
        'min_rainfall': min_rainfall,
        'max_rainfall': max_rainfall,
        'deficient_threshold': deficient_threshold,
        'excessive_threshold': excessive_threshold,
        'least_threshold': deficient_threshold,
        'avg_price': avg_price,
        'min_year': min_year,
        'max_year': max_year
    }
    joblib.dump(thresholds, f'models/{crop_name}_thresholds.pkl')
    
    # Generate future predictions (2018-2028)
    generate_future_predictions(crop_name, model, thresholds, features)
    
    # Save feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    joblib.dump(feature_importance, f'models/{crop_name}_importance.pkl')
    
    print(f"Model for {crop_name} trained and saved successfully!")
    return model, thresholds

def generate_future_predictions(crop_name, model, thresholds, features):
    """Generate future price predictions for years 2018-2028."""
    print(f"Generating future predictions for {crop_name}...")
    
    # Get the optimal rainfall for this crop
    optimal_rainfall = RAINFALL_RANGES.get(crop_name, {'optimal': thresholds['mean_rainfall']})['optimal']
    
    # Get the annual growth rate for this crop
    annual_growth = ANNUAL_GROWTH_RATES.get(crop_name, 0.04)  # Default to 4% if not specified
    
    # Get the MSP (Minimum Support Price) for this crop
    msp = MSP_DATA.get(crop_name, None)
    
    # Create a dataframe for future years (2018-2028)
    future_years = range(2018, 2029)
    future_data = []
    
    for year in future_years:
        # For each month in the year
        for month in range(1, 13):
            # Use optimal rainfall for prediction
            rainfall = optimal_rainfall
            
            # Create a row with features
            row = {
                'Year': year,
                'Month': month,
                features[2]: rainfall,  # Rainfall column
                'Rainfall_Category': 0,  # Normal rainfall
                'Rainfall_Deviation': 0  # No deviation from mean
            }
            
            future_data.append(row)
    
    future_df = pd.DataFrame(future_data)
    
    # Make predictions using the model
    base_predictions = model.predict(future_df[features])
    future_df['Base_WPI'] = base_predictions
    
    # Apply time series adjustment for future years
    base_year = thresholds['max_year']
    future_df['Years_Beyond_Base'] = future_df['Year'] - base_year
    
    # Calculate adjusted WPI with annual growth
    future_df['Predicted_WPI'] = future_df.apply(
        lambda row: row['Base_WPI'] * (1 + annual_growth) ** max(0, row['Years_Beyond_Base']), 
        axis=1
    )
    
    # If we have MSP data, ensure predictions align with it
    if msp is not None:
        # Convert MSP to WPI (approximate)
        msp_wpi = msp / 25  # Using the same conversion factor as in web_interface.py
        
        # Find the average predicted WPI for 2023
        avg_wpi_2023 = future_df[future_df['Year'] == 2023]['Predicted_WPI'].mean()
        
        # Calculate adjustment factor to align with MSP
        if avg_wpi_2023 > 0:
            adjustment_factor = msp_wpi / avg_wpi_2023
            
            # Apply adjustment to all predictions
            future_df['Predicted_WPI'] = future_df['Predicted_WPI'] * adjustment_factor
    
    # Generate predictions for different rainfall scenarios
    # Normal rainfall (already in Predicted_WPI)
    # Excessive rainfall
    future_df['Excessive_WPI'] = future_df['Predicted_WPI'] * (1 + RAINFALL_IMPACT['excessive'])
    # Deficient rainfall
    future_df['Deficient_WPI'] = future_df['Predicted_WPI'] * (1 + RAINFALL_IMPACT['deficient'])
    
    # Save the future predictions
    joblib.dump(future_df, f'models/{crop_name}_future_predictions.pkl')
    
    print(f"Future predictions for {crop_name} generated successfully!")
    return future_df

def train_all_models():
    """Train models for all crops in the dataset."""
    try:
        # Load the merged dataset
        merged_data = pd.read_csv("merged.csv")
        
        # Check if 'Crop' column exists
        if 'Crop' in merged_data.columns:
            # Get unique crop names
            crops = merged_data['Crop'].unique()
            
            for crop in crops:
                # Filter data for this crop
                crop_data = merged_data[merged_data['Crop'] == crop]
                
                # Train model for this crop
                train_crop_model(crop, crop_data)
        else:
            # If no Crop column, assume single crop dataset
            crop_name = os.path.basename(os.getcwd())  # Use directory name as crop name
            train_crop_model(crop_name, merged_data)
        
        print("All models trained successfully!")
    except Exception as e:
        print(f"Error training models: {str(e)}")

if __name__ == "__main__":
    train_all_models()
