from flask import Flask, render_template, request, redirect, url_for, jsonify
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json

app = Flask(__name__)

# Get actual crop names from the models directory
def get_crop_names():
    # Look for both old and new model formats
    model_files = [f for f in os.listdir('models') if f.endswith('_model.pkl') or f.endswith('_rainfall_model.pkl')]
    
    # Extract crop names without the suffix
    crop_names = []
    for f in model_files:
        if f.endswith('_rainfall_model.pkl'):
            crop_names.append(f.split('_rainfall_model.pkl')[0])
        else:
            crop_names.append(f.split('_model.pkl')[0])
    
    # Filter out names with "_rainfall" suffix - these are duplicates
    filtered_names = [name for name in crop_names if not name.endswith('_rainfall')]
    
    return filtered_names

# Load models and thresholds for each crop
def load_models():
    models = {}
    thresholds = {}
    future_predictions = {}
    crop_names = get_crop_names()
    
    for crop in crop_names:
        rainfall_model_path = f'models/{crop}_rainfall_model.pkl'
        threshold_path = f'models/{crop}_thresholds.pkl'
        future_pred_path = f'models/{crop}_future_predictions.pkl'
        
        if os.path.exists(rainfall_model_path):
            models[crop] = joblib.load(rainfall_model_path)
            
            if os.path.exists(threshold_path):
                thresholds[crop] = joblib.load(threshold_path)
            else:
                thresholds[crop] = None
                
            if os.path.exists(future_pred_path):
                future_predictions[crop] = joblib.load(future_pred_path)
            else:
                future_predictions[crop] = None
                
    return models, thresholds, future_predictions, crop_names

models, thresholds, future_predictions, crop_names = load_models()

# Load historical data for visualization
def load_historical_data(crop_name):
    try:
        data = pd.read_csv("merged.csv")
        if 'Crop' in data.columns:
            crop_data = data[data['Crop'] == crop_name].copy()
        else:
            crop_data = data.copy()
        
        # Determine which columns to use
        rainfall_col = 'Rainfall_x' if 'Rainfall_x' in crop_data.columns else 'Rainfall'
        wpi_col = 'WPI_y' if 'WPI_y' in crop_data.columns else ('WPI' if 'WPI' in crop_data.columns else 'WPI_x')
        
        # Sort by year and month
        if 'Year' in crop_data.columns and 'Month' in crop_data.columns:
            crop_data['Date'] = pd.to_datetime(crop_data['Year'].astype(str) + '-' + crop_data['Month'].astype(str), format='%Y-%m')
            crop_data = crop_data.sort_values('Date')
        
        return crop_data, rainfall_col, wpi_col
    except Exception as e:
        print(f"Error loading historical data: {e}")
        return None, None, None

@app.route('/', methods=['GET', 'POST'])
def index():
    error_message = None
    prediction = None
    confidence_interval = None
    price_change = None
    rainfall_category = None
    display_thresholds = None
    
    # Get current year for default value
    current_year = pd.Timestamp.now().year
    years_range = range(2018, current_year + 10)  # Allow predictions up to 10 years in the future
    prediction_year = current_year
    
    if request.method == 'POST':
        try:
            crop_name = request.form['crop']
            rainfall = float(request.form['rainfall'])
            
            # Get selected year from form
            prediction_year = int(request.form.get('year', current_year))
            
            if crop_name not in models:
                error_message = f"No model found for {crop_name}. Please train the model first."
                return render_template('index.html', crops=crop_names, error_message=error_message, 
                                      years_range=years_range, current_year=current_year)
            
            model = models[crop_name]
            crop_thresholds = thresholds.get(crop_name)
            
            # Determine rainfall category with more precise thresholds
            if crop_thresholds:
                # Calculate more precise thresholds based on historical data
                mean_rainfall = crop_thresholds['mean_rainfall']
                std_rainfall = crop_thresholds['std_rainfall']
                
                # Adjust thresholds to be more realistic
                deficient_threshold = mean_rainfall - 0.75 * std_rainfall
                excessive_threshold = mean_rainfall + 0.75 * std_rainfall
                
                if rainfall > excessive_threshold:
                    rainfall_category = "Excessive"
                    category_value = 1
                elif rainfall < deficient_threshold:
                    rainfall_category = "Deficient"
                    category_value = -1
                else:
                    rainfall_category = "Normal"
                    category_value = 0
                
                # Create a dictionary with thresholds for display in template
                display_thresholds = {
                    'deficient_threshold': deficient_threshold,
                    'excessive_threshold': excessive_threshold,
                    'mean_rainfall': mean_rainfall
                }
            else:
                # Default behavior if thresholds not available
                rainfall_category = "Unknown"
                category_value = 0
                display_thresholds = {
                    'deficient_threshold': 0,
                    'excessive_threshold': 0,
                    'mean_rainfall': 0
                }
            
            # STEP 1: Get base prediction from future predictions
            if crop_name in future_predictions and future_predictions[crop_name] is not None:
                future_df = future_predictions[crop_name]
                
                # Filter for the selected year
                year_data = future_df[future_df['Year'] == prediction_year]
                
                if len(year_data) > 0:
                    # Get the average prediction for the year
                    if rainfall_category == "Excessive":
                        base_prediction = year_data['Excessive_WPI'].mean()
                    elif rainfall_category == "Deficient":
                        base_prediction = year_data['Deficient_WPI'].mean()
                    else:
                        base_prediction = year_data['Predicted_WPI'].mean()
                else:
                    # Year not in future predictions, use extrapolation
                    error_message = f"Year {prediction_year} is outside the prediction range. Please select a year between 2018 and 2028."
                    return render_template('index.html', crops=crop_names, error_message=error_message,
                                          years_range=years_range, current_year=current_year)
            else:
                # No future predictions available, use the model directly
                # Prepare input data for prediction
                current_date = pd.Timestamp.now()
                month = current_date.month
                year = prediction_year
                
                # Load historical data to get column names
                crop_data, rainfall_col, wpi_col = load_historical_data(crop_name)
                
                if crop_data is None:
                    error_message = "Could not load historical data for prediction."
                    return render_template('index.html', crops=crop_names, error_message=error_message,
                                          years_range=years_range, current_year=current_year)
                
                # Create input DataFrame with the same columns as training data
                input_data = pd.DataFrame({
                    'Month': [month],
                    'Year': [year],
                    rainfall_col: [rainfall],
                    'Rainfall_Category': [category_value],
                    'Rainfall_Deviation': [rainfall - crop_thresholds['mean_rainfall']]
                })
                
                # Make prediction
                if hasattr(model, 'feature_names_in_'):
                    # For scikit-learn 1.0+ models
                    feature_names = model.feature_names_in_
                    input_features = input_data[feature_names]
                else:
                    # For older scikit-learn versions
                    feature_columns = ['Month', 'Year', rainfall_col, 'Rainfall_Category', 'Rainfall_Deviation']
                    input_features = input_data[feature_columns]
                
                # Get base prediction
                base_prediction = model.predict(input_features)[0]
                
                # Apply rainfall adjustments
                if rainfall_category == "Excessive":
                    base_prediction *= 1.25  # 25% increase
                elif rainfall_category == "Deficient":
                    base_prediction *= 1.30  # 30% increase
            
            # STEP 2: Final prediction is the base prediction (already adjusted for rainfall)
            prediction = base_prediction
            
            # STEP 3: Calculate derived values
            # Convert WPI to price per quintal (100kg)
            wpi_to_quintal_factor = 25
            price_per_quintal = prediction * wpi_to_quintal_factor
            
            # Apply 2022 inflation adjustment (11.03%)
            inflation_adjusted_price = price_per_quintal * (1 + 11.03/100)
            
            # Calculate confidence interval based on quintal price
            confidence_range = price_per_quintal * 0.15
            confidence_interval = [price_per_quintal - confidence_range, price_per_quintal + confidence_range]
            
            # STEP 4: Determine price trend explanation
            if crop_thresholds:
                avg_price_wpi = crop_thresholds['avg_price']
                avg_price_quintal = avg_price_wpi * wpi_to_quintal_factor
                
                if rainfall_category == "Excessive" or rainfall_category == "Deficient":
                    if price_per_quintal > avg_price_quintal:
                        price_change = f"Price is likely above average (Avg: ₹{avg_price_quintal:.2f})"
                    else:
                        price_change = f"Price is projected to be ₹{price_per_quintal:.2f}, which is unusual given the {rainfall_category.lower()} rainfall"
                else:
                    if price_per_quintal > avg_price_quintal:
                        price_change = f"Price is higher than average (Avg: ₹{avg_price_quintal:.2f})"
                    else:
                        price_change = f"Price is likely near average (Avg: ₹{avg_price_quintal:.2f})"
            
            # STEP 5: Store prediction data for display
            prediction_data = {
                'wpi': prediction,
                'per_quintal': price_per_quintal,
                'inflation_adjusted': inflation_adjusted_price,
                'year': prediction_year
            }
            
        except Exception as e:
            error_message = f"Error making prediction: {str(e)}"
            print(error_message)
            import traceback
            traceback.print_exc()
            
        return render_template('index.html', 
                              crops=crop_names, 
                              prediction=prediction_data,
                              confidence_interval=confidence_interval,
                              price_change=price_change,
                              rainfall_category=rainfall_category,
                              thresholds=display_thresholds,
                              error_message=error_message,
                              years_range=years_range,
                              current_year=current_year)
    
    return render_template('index.html', 
                          crops=crop_names, 
                          years_range=years_range,
                          current_year=current_year)

@app.route('/visualization/<crop_name>')
def visualization(crop_name):
    crop_data, rainfall_col, wpi_col = load_historical_data(crop_name)
    
    if crop_data is None or len(crop_data) == 0:
        return render_template('error.html', message=f"No data available for {crop_name}")
    
    # Create historical price trend visualization
    fig1 = go.Figure()
    
    # Add price line
    fig1.add_trace(
        go.Scatter(
            x=crop_data['Date'], 
            y=crop_data[wpi_col], 
            mode='lines+markers',
            name='Price (WPI)',
            line=dict(color='#1f77b4', width=2)
        )
    )
    
    # Add trend line
    if len(crop_data) > 2:
        # Calculate trend
        crop_data['trend_x'] = np.arange(len(crop_data))
        trend_model = np.polyfit(crop_data['trend_x'], crop_data[wpi_col], 1)
        crop_data['trend_y'] = trend_model[0] * crop_data['trend_x'] + trend_model[1]
        
        fig1.add_trace(
            go.Scatter(
                x=crop_data['Date'],
                y=crop_data['trend_y'],
                mode='lines',
                name='Price Trend',
                line=dict(color='red', width=1, dash='dash')
            )
        )
        
        # Add annotations for significant price changes - FIX HERE
        try:
            price_changes = crop_data[wpi_col].pct_change() * 100
            # Use boolean indexing instead of .index to get positions
            significant_changes = crop_data[abs(price_changes) > 10]
            
            # Only add annotations if there are significant changes
            if len(significant_changes) > 0:
                for _, row in significant_changes.iterrows():
                    # Get the corresponding price change
                    change_val = price_changes.loc[row.name]
                    fig1.add_annotation(
                        x=row['Date'],
                        y=row[wpi_col],
                        text=f"{change_val:.1f}% change",
                        showarrow=True,
                        arrowhead=1
                    )
        except Exception as e:
            print(f"Error adding price change annotations: {e}")
            # Continue without annotations if there's an error
    
    # Add seasonal patterns if data spans multiple years
    years = crop_data['Year'].unique()
    if len(years) > 1:
        # Calculate average price by month
        monthly_avg = crop_data.groupby('Month')[wpi_col].mean()
        if len(monthly_avg) > 0:  # Make sure we have data
            peak_month = monthly_avg.idxmax()
            low_month = monthly_avg.idxmin()
            
            fig1.add_annotation(
                x=0.5, y=1.05,
                text=f"Peak prices typically in month {peak_month}, lowest in month {low_month}",
                showarrow=False,
                xref="paper", yref="paper"
            )
    
    # Update layout
    fig1.update_layout(
        title=f'Historical Price Trend for {crop_name}',
        xaxis_title='Date',
        yaxis_title='Price (WPI)',
        hovermode='x unified',
        template='plotly_white'
    )
    
    price_graph = fig1.to_html(full_html=False, include_plotlyjs=False)
    
    # Create improved rainfall vs price correlation visualization
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add rainfall bars with color coding
    crop_data['rainfall_color'] = 'rgba(0, 128, 255, 0.6)'  # Default blue
    
    if crop_name in thresholds and thresholds[crop_name] is not None:
        thresh = thresholds[crop_name]
        # Color code based on rainfall thresholds
        crop_data.loc[crop_data[rainfall_col] < thresh['least_threshold'], 'rainfall_color'] = 'rgba(255, 165, 0, 0.6)'  # Orange for deficient
        crop_data.loc[crop_data[rainfall_col] > thresh['excessive_threshold'], 'rainfall_color'] = 'rgba(128, 0, 128, 0.6)'  # Purple for excessive
    
    fig2.add_trace(
        go.Bar(
            x=crop_data['Date'], 
            y=crop_data[rainfall_col], 
            name='Rainfall (mm)',
            marker_color=crop_data['rainfall_color'].tolist()
        ),
        secondary_y=False
    )
    
    # Add price line with markers
    fig2.add_trace(
        go.Scatter(
            x=crop_data['Date'], 
            y=crop_data[wpi_col], 
            name='Price (WPI)', 
            line=dict(color='red', width=2),
            mode='lines+markers'
        ),
        secondary_y=True
    )
    
    # Add correlation annotation
    if len(crop_data) > 2:
        corr = crop_data[[rainfall_col, wpi_col]].corr().iloc[0,1]
        relationship = "negative" if corr < 0 else "positive"
        strength = "strong" if abs(corr) > 0.7 else ("moderate" if abs(corr) > 0.3 else "weak")
        
        fig2.add_annotation(
            x=0.5, y=1.05,
            text=f"Correlation: {corr:.2f} ({strength} {relationship} relationship)",
            showarrow=False,
            xref="paper", yref="paper"
        )
    
    # Update layout
    fig2.update_layout(
        title=f'Rainfall vs Price Relationship for {crop_name}',
        hovermode='x unified',
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    # Update axes
    fig2.update_xaxes(title_text='Date')
    fig2.update_yaxes(title_text='Rainfall (mm)', secondary_y=False)
    fig2.update_yaxes(title_text='Price (WPI)', secondary_y=True)
    
    correlation_graph = fig2.to_html(full_html=False, include_plotlyjs=False)
    
    # Create price prediction guide
    if crop_name in thresholds and thresholds[crop_name] is not None:
        crop_thresholds = thresholds[crop_name]
        
        # Create a scatter plot with rainfall on x-axis and price on y-axis
        fig3 = px.scatter(crop_data, x=rainfall_col, y=wpi_col, 
                         title=f'Rainfall vs Price Relationship for {crop_name}',
                         trendline='ols')
        
        # Add vertical lines for thresholds
        fig3.add_vline(x=crop_thresholds['least_threshold'], line_dash='dash', line_color='orange',
                      annotation_text='Deficient Rainfall Threshold')
        fig3.add_vline(x=crop_thresholds['excessive_threshold'], line_dash='dash', line_color='blue',
                      annotation_text='Excessive Rainfall Threshold')
        
        # Update layout
        fig3.update_layout(
            xaxis_title='Rainfall (mm)',
            yaxis_title='Price (WPI)',
            template='plotly_white'
        )
        
        prediction_guide = fig3.to_html(full_html=False, include_plotlyjs=False)
    else:
        prediction_guide = None
    
    # Get feature importance if available
    importance_path = f'models/{crop_name}_importance.pkl'
    if os.path.exists(importance_path):
        feature_importance = joblib.load(importance_path)
        
        # Create feature importance bar chart
        fig4 = px.bar(feature_importance, x='feature', y='importance', 
                     title=f'Feature Importance for {crop_name} Price Prediction')
        
        fig4.update_layout(
            xaxis_title='Feature',
            yaxis_title='Importance',
            template='plotly_white'
        )
        
        importance_graph = fig4.to_html(full_html=False, include_plotlyjs=False)
    else:
        importance_graph = None
    
    return render_template('visualization.html', 
                          crop_name=crop_name,
                          price_graph=price_graph,
                          correlation_graph=correlation_graph,
                          prediction_guide=prediction_guide,
                          importance_graph=importance_graph)

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    message = None
    
    if request.method == 'POST':
        crop_name = request.form['crop']
        try:
            # Import the training function from train.py
            from train import train_crop_model
            
            # Load the data for the specific crop
            merged_data = pd.read_csv("merged.csv")
            if 'Crop' in merged_data.columns:
                crop_data = merged_data[merged_data['Crop'] == crop_name]
            else:
                crop_data = merged_data
            
            # Retrain the model
            train_crop_model(crop_name, crop_data)
            
            # Reload the models
            global models, thresholds, future_predictions, crop_names
            models, thresholds, future_predictions, crop_names = load_models()
            
            message = f"Model for {crop_name} retrained successfully!"
        except Exception as e:
            message = f"Error retraining model: {str(e)}"

    return render_template('admin.html', crops=crop_names, message=message)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    # Create templates and static directories if they don't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    app.run(debug=True) 