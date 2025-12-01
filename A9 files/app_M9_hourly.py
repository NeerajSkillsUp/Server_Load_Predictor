# http://127.0.0.1:5006/predict
import joblib
import pandas as pd
import numpy as np
import statsmodels.api as sm  # <-- Using statsmodels
from flask import Flask, request, jsonify
from flask_cors import CORS # <--- ADDED: Import CORS
import warnings
import os

warnings.filterwarnings("ignore")

# --- 1. Initialize App and Load Artifacts ---
app = Flask(__name__)
CORS(app)
print("--- Loading M9 (Hourly) Model and Data ---")

# Define file paths
MODEL_NAME = "A9 files\\M9_hourly.joblib"
DATA_NAME = "A9 files\\A9_hourly_calls.csv"

try:
    # Load the trained model
    MODEL = joblib.load(MODEL_NAME)
    print(f"Successfully loaded {MODEL_NAME}")
    
    # Load the historical data, which we need to calculate features
    DATA = pd.read_csv(DATA_NAME, index_col='datetime', parse_dates=True)
    DATA = DATA.asfreq('H')
    print(f"Successfully loaded {DATA_NAME}. Last data point is from: {DATA.index.max()}")
    
except Exception as e:
    print(f"FATAL: Could not load model artifacts. Run train_M9_hourly.py first. Error: {e}")
    MODEL = None

# --- 2. Define Prediction Endpoint ---
@app.route('/predict', methods=['GET'])
def predict():
    """Predicts the next HOUR's call count for A9."""
    
    if MODEL is None:
        return jsonify({"error": "Model is not loaded. Check server logs."}), 500

    try:
        # 1. Determine the target timestamp to predict
        last_known_date = DATA.index.max()
        target_timestamp = last_known_date + pd.Timedelta(hours=1)

        print(f"Prediction requested for: {target_timestamp.isoformat()}")

        # 2. Get data windows needed for features
        lag_24_timestamp = target_timestamp - pd.Timedelta(hours=24)
        rolling_window_start = target_timestamp - pd.Timedelta(hours=24)
        rolling_window_end = target_timestamp - pd.Timedelta(hours=1)

        # 3. Calculate features manually
        features = []
        
        # Check if we have enough historical data
        if lag_24_timestamp not in DATA.index:
             return jsonify({"error": f"Not enough historical data. Need data from {lag_24_timestamp} to predict."}), 400
        if rolling_window_start not in DATA.index:
             return jsonify({"error": f"Not enough historical data. Need data from {rolling_window_start} to predict."}), 400

        # Features from the target_timestamp itself
        features.append(target_timestamp.hour)       # hour
        features.append(target_timestamp.dayofweek)  # dayofweek
        features.append(target_timestamp.month)      # month
        
        # Lag feature
        lag_24_value = DATA.loc[lag_24_timestamp]['call_count']
        features.append(lag_24_value)                # lag_24
        
        # Rolling feature
        rolling_data = DATA.loc[rolling_window_start:rolling_window_end]['call_count']
        rolling_mean_24_value = rolling_data.mean()
        features.append(rolling_mean_24_value)       # rolling_mean_24

        # 4. Predict
        # We must reshape features to (1, 5)
        feature_vector = np.array(features).reshape(1, -1)
        
        # IMPORTANT: Add the constant to the feature vector for prediction
        # The column order must match training: [const, hour, dayofweek, month, lag_24, rolling_mean_24]
        # Our `create_features` function puts features in that order, so `add_constant`
        # will prepend the constant, which is correct.
        feature_vector_with_const = sm.add_constant(feature_vector, has_constant='add')
        
        prediction_raw = MODEL.predict(feature_vector_with_const)
        
        # 5. Clean and return the prediction
        final_prediction = int(np.round(prediction_raw[0]).clip(0))
        
        # Cast all features to standard python types for JSON serialization
        features_dict = {
            'hour': int(features[0]),
            'dayofweek': int(features[1]),
            'month': int(features[2]),
            'lag_24': int(features[3]),
            'rolling_mean_24': float(features[4])
        }
        
        return jsonify({
            "api_code": "A9",
            "model_type": "Poisson_GLM_Hourly",
            "forecast_for_timestamp": target_timestamp.isoformat(),
            "predicted_call_count": final_prediction,
            "features_used": features_dict
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": f"An error occurred during prediction: {e}"}), 500

# --- 3. Run the App ---
if __name__ == '__main__':
    app.run(debug=True, port=5006)