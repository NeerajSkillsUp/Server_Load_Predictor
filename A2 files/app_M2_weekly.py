# http://127.0.0.1:5002/predict
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS # <--- ADDED: Import CORS
import warnings

warnings.filterwarnings("ignore")

# --- 1. Initialize App and Load Artifacts ---
app = Flask(__name__)
CORS(app)
print("--- Loading M2_weekly Model and Data ---")

MODEL_NAME = "A2 files\\M2_weekly.joblib"
DATA_NAME = "A2 files\\A2_weekly_calls.csv"

try:
    # Load the trained model
    MODEL = joblib.load(MODEL_NAME)
    print(f"Successfully loaded {MODEL_NAME}")
    
    # Load the historical data, which we need to calculate features
    DATA = pd.read_csv(DATA_NAME, index_col='datetime', parse_dates=True)
    DATA = DATA.asfreq('W') # 'W' = Week ending Sunday
    print(f"Successfully loaded {DATA_NAME}. Last data point is from: {DATA.index.max()}")
    
except Exception as e:
    print(f"FATAL: Could not load model artifacts. Run train_M2_weekly.py first. Error: {e}")
    MODEL = None

# --- 2. Define Prediction Endpoint ---
@app.route('/predict', methods=['GET'])
def predict():
    """Predicts the next WEEK's call count for A2."""
    
    if MODEL is None:
        return jsonify({"error": "Model is not loaded. Check server logs."}), 500

    try:
        # 1. Determine the target timestamp to predict
        # This is the week after our last known data point
        last_known_date = DATA.index.max()
        target_timestamp = last_known_date + pd.Timedelta(weeks=1)

        print(f"Prediction requested for week ending: {target_timestamp.isoformat()}")

        # 2. Get data windows needed for features
        
        # For lag_4
        lag_4_timestamp = target_timestamp - pd.Timedelta(weeks=4)
        
        # For rolling_mean_4 (mean of 4 weeks *before* the target)
        rolling_window_start = target_timestamp - pd.Timedelta(weeks=4)
        rolling_window_end = target_timestamp - pd.Timedelta(weeks=1)

        # 3. Calculate features manually
        features = []
        
        # Check if we have enough historical data
        if lag_4_timestamp not in DATA.index:
             return jsonify({"error": f"Not enough historical data. Need data from {lag_4_timestamp} to predict."}), 400
        if rolling_window_start not in DATA.index:
             return jsonify({"error": f"Not enough historical data. Need data from {rolling_window_start} to predict."}), 400

        # Features from the target_timestamp itself
        features.append(target_timestamp.month)                 # month
        features.append(target_timestamp.isocalendar().week)    # weekofyear
        features.append(target_timestamp.quarter)               # quarter
        
        # Lag feature
        lag_4_value = DATA.loc[lag_4_timestamp]['call_count']
        features.append(lag_4_value)                            # lag_4
        
        # Rolling feature
        rolling_data = DATA.loc[rolling_window_start:rolling_window_end]['call_count']
        rolling_mean_4_value = rolling_data.mean()
        features.append(rolling_mean_4_value)                   # rolling_mean_4

        # 4. Predict
        # We must reshape features to (1, 5) because model expects a 2D array
        feature_vector = np.array(features).reshape(1, -1)
        
        prediction_raw = MODEL.predict(feature_vector)
        
        # 5. Clean and return the prediction
        final_prediction = int(np.round(prediction_raw[0]).clip(0))
        
        # Cast all features to standard python types for JSON serialization
        features_dict = {
            'month': int(features[0]),
            'weekofyear': int(features[1]),
            'quarter': int(features[2]),
            'lag_4': int(features[3]),
            'rolling_mean_4': float(features[4])
        }
        
        return jsonify({
            "api_code": "A2",
            "model_type": "RandomForest_Weekly",
            "forecast_for_timestamp": target_timestamp.isoformat(),
            "predicted_call_count": final_prediction,
            "features_used": features_dict
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": f"An error occurred during prediction: {e}"}), 500

# --- 3. Run the App ---
if __name__ == '__main__':
    app.run(debug=True, port=5002) 