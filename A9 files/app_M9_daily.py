# http://127.0.0.1:5007/predict
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS # <--- ADDED: Import CORS
import warnings
import os

warnings.filterwarnings("ignore")

# --- 1. Initialize App and Load Artifacts ---
app = Flask(__name__)
CORS(app)
print("--- Loading M9_daily Model and Data ---")

# Define file paths
MODEL_NAME = "A9 files\\M9_daily.joblib"
DATA_NAME = "A9 files\\A9_daily_calls.csv"

try:
    # Load the trained model
    MODEL = joblib.load(MODEL_NAME)
    print(f"Successfully loaded {MODEL_NAME}")
    
    # Load the historical data, which we need to calculate features
    DATA = pd.read_csv(DATA_NAME, index_col='datetime', parse_dates=True)
    DATA = DATA.asfreq('D')
    print(f"Successfully loaded {DATA_NAME}. Last data point is from: {DATA.index.max()}")
    
except Exception as e:
    print(f"FATAL: Could not load model artifacts. Run train_M9_daily.py first. Error: {e}")
    MODEL = None

# --- 2. Define Prediction Endpoint ---
@app.route('/predict', methods=['GET'])
def predict():
    """Predicts the next DAY's call count for A9."""
    
    if MODEL is None:
        return jsonify({"error": "Model is not loaded. Check server logs."}), 500

    try:
        # 1. Determine the target timestamp to predict
        # This is the day after our last known data point
        last_known_date = DATA.index.max()
        target_timestamp = last_known_date + pd.Timedelta(days=1)

        print(f"Prediction requested for: {target_timestamp.isoformat()}")

        # 2. Get data windows needed for features
        # For lag_7
        lag_7_timestamp = target_timestamp - pd.Timedelta(days=7)
        
        # For rolling_mean_7 (mean of 7 days *before* the target)
        rolling_window_start = target_timestamp - pd.Timedelta(days=7)
        rolling_window_end = target_timestamp - pd.Timedelta(days=1)

        # 3. Calculate features manually
        features = []
        
        # Check if we have enough historical data
        if lag_7_timestamp not in DATA.index:
             return jsonify({"error": f"Not enough historical data. Need data from {lag_7_timestamp} to predict."}), 400
        if rolling_window_start not in DATA.index:
             return jsonify({"error": f"Not enough historical data. Need data from {rolling_window_start} to predict."}), 400

        # Features from the target_timestamp itself
        features.append(target_timestamp.dayofweek)  # dayofweek
        features.append(target_timestamp.month)      # month
        features.append(target_timestamp.dayofyear)  # dayofyear
        features.append(target_timestamp.quarter)    # quarter
        
        # Lag feature
        lag_7_value = DATA.loc[lag_7_timestamp]['call_count']
        features.append(lag_7_value)                 # lag_7
        
        # Rolling feature
        rolling_data = DATA.loc[rolling_window_start:rolling_window_end]['call_count']
        rolling_mean_7_value = rolling_data.mean()
        features.append(rolling_mean_7_value)        # rolling_mean_7

        # 4. Predict
        # We must reshape features to (1, 6) because model expects a 2D array
        feature_vector = np.array(features).reshape(1, -1)
        
        prediction_raw = MODEL.predict(feature_vector)
        
        # 5. Clean and return the prediction
        final_prediction = int(np.round(prediction_raw[0]).clip(0))
        
        # Cast all features to standard python types for JSON serialization
        features_dict = {
            'dayofweek': int(features[0]),
            'month': int(features[1]),
            'dayofyear': int(features[2]),
            'quarter': int(features[3]),
            'lag_7': int(features[4]),
            'rolling_mean_7': float(features[5])
        }
        
        return jsonify({
            "api_code": "A9",
            "model_type": "XGBoost_Daily",
            "forecast_for_timestamp": target_timestamp.isoformat(),
            "predicted_call_count": final_prediction,
            "features_used": features_dict
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": f"An error occurred during prediction: {e}"}), 500

# --- 3. Run the App ---
if __name__ == '__main__':
    app.run(debug=True, port=5007) 