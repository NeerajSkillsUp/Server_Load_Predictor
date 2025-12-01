# http://127.0.0.1:5008/predict
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
print("--- Loading M9_weekly (SARIMA) Model ---")

# Define file paths
MODEL_NAME = "A9 files\\M9_weekly.joblib"
# --- FIX: We must load the data file to find the last date ---
DATA_NAME = "A9 files\\A9_weekly_calls.csv"

try:
    # Load the trained model
    MODEL = joblib.load(MODEL_NAME)
    print(f"Successfully loaded {MODEL_NAME}")
    
    # Load the historical data *only* to find the last date
    DATA = pd.read_csv(DATA_NAME, index_col='datetime', parse_dates=True)
    DATA = DATA.asfreq('W')
    print(f"Successfully loaded {DATA_NAME}. Last data point is from: {DATA.index.max()}")
    
except Exception as e:
    print(f"FATAL: Could not load model artifacts. Run train_M9_weekly.py first. Error: {e}")
    MODEL = None
    DATA = None

# --- 2. Define Prediction Endpoint ---
@app.route('/predict', methods=['GET'])
def predict():
    """Predicts the next WEEK's call count for A9."""
    
    if MODEL is None or DATA is None:
        return jsonify({"error": "Model or data is not loaded. Check server logs."}), 500

    try:
        # 1. Determine the target timestamp to predict
        # --- FIX ---
        # Get the last date from the DATA, not the model
        last_known_date = DATA.index.max()
        # --- END FIX ---
        
        target_timestamp = last_known_date + pd.Timedelta(weeks=1)

        print(f"Prediction requested for week ending: {target_timestamp.isoformat()}")

        # 2. Predict
        # SARIMA's .forecast() method will automatically predict the next step
        # after the data it was trained on (which we know ends at last_known_date)
        prediction_raw = MODEL.forecast(steps=1)
        
        # 3. Clean and return the prediction
        # prediction_raw is a pandas Series, so .iloc[0] is correct
        final_prediction = int(np.round(prediction_raw.iloc[0]).clip(0))
        
        return jsonify({
            "api_code": "A9",
            "model_type": "SARIMA_Weekly(m=4)",
            "forecast_for_timestamp": target_timestamp.isoformat(),
            "predicted_call_count": final_prediction,
            "features_used": "SARIMA internal states (ar, ma, sar, sma)"
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": f"An error occurred during prediction: {e}"}), 500

# --- 3. Run the App ---
if __name__ == '__main__':
    app.run(debug=True, port=5008) # Running on port 5008