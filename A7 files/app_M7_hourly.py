# http://127.0.0.1:5003/predict
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS # <--- ADDED: Import CORS
from prophet import Prophet
import joblib  # <-- Using joblib to load
import warnings
import os

warnings.filterwarnings("ignore")

# --- 1. Initialize App and Load Artifacts ---
app = Flask(__name__)
CORS(app)
print("--- Loading M7 (Prophet, Hourly) Model ---")

MODEL_NAME = "A7 files\\M7_hourly.joblib" # <-- Loading the .joblib file

try:
    # Load the trained Prophet model using joblib
    MODEL = joblib.load(MODEL_NAME)
    print(f"Successfully loaded {MODEL_NAME}")
    
except Exception as e:
    print(f"FATAL: Could not load model artifacts. Run train_M7_hourly.py first. Error: {e}")
    MODEL = None

# --- 2. Define Prediction Endpoint ---
@app.route('/predict', methods=['GET'])
def predict():
    """Predicts the next HOUR's call count for A7 using Prophet."""
    
    if MODEL is None:
        return jsonify({"error": "Model is not loaded. Check server logs."}), 500

    try:
        # 1. Create a dataframe for the next 1 hour
        # Prophet's `make_future_dataframe` uses the history stored in the model
        future_df = MODEL.make_future_dataframe(periods=1, freq='H')

        # 2. Predict
        prediction_df = MODEL.predict(future_df)
        
        # 3. Get the last row (which is our new prediction)
        last_row = prediction_df.iloc[-1]
        
        prediction_raw = last_row['yhat']
        target_timestamp = last_row['ds'] 
        
        # 4. Clean and return the prediction
        final_prediction = int(np.round(prediction_raw).clip(0))
        
        # Prophet's features are internal (trend, daily, weekly)
        # We cast to standard float/int for JSON serialization
        features_dict = {
            "trend": float(last_row['trend']),
            "daily_seasonality": float(last_row['daily']),
            "weekly_seasonality": float(last_row['weekly']),
            "forecast_component_sum": float(last_row['yhat'])
        }
        
        return jsonify({
            "api_code": "A7",
            "model_type": "Prophet_Hourly",
            "forecast_for_timestamp": target_timestamp.isoformat(),
            "predicted_call_count": final_prediction,
            "features_used": features_dict
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": f"An error occurred during prediction: {e}"}), 500

# --- 3. Run the App ---
if __name__ == '__main__':
    app.run(debug=True, port=5003)