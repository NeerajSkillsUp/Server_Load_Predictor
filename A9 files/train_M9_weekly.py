import pandas as pd
import numpy as np
from statsmodels.tsa.api import SARIMAX # <-- Using SARIMAX
import joblib
import warnings
import os

warnings.filterwarnings("ignore")

print("--- Starting Final M9 (SARIMA, Weekly) Training ---")

# Define file paths
MODEL_NAME = "A9 files\\M9_weekly.joblib"
DATA_SOURCE_RAW = "A9 files\\A9.csv"
DATA_SOURCE_PROCESSED = "A9 files\\A9_weekly_calls.csv"

# --- 1. Preprocessing ---
print(f"Loading and preprocessing {DATA_SOURCE_RAW}...")
try:
    df_raw = pd.read_csv(DATA_SOURCE_RAW)
    df_raw['datetime'] = pd.to_datetime(df_raw['Time of Call'], format='%d-%m-%Y %H:%M')
    df_raw = df_raw.set_index('datetime')
    
    # Resample to weekly counts ('W' = week ending Sunday)
    df_weekly = df_raw.resample('W').size().to_frame('call_count')
    df_weekly = df_weekly.asfreq(freq='W', fill_value=0)
    
    # Drop the last row (incomplete week)
    if not df_weekly.empty:
        df_weekly = df_weekly.iloc[:-1]
    
    # Save the weekly data
    df_weekly.to_csv(DATA_SOURCE_PROCESSED)
    print(f"Preprocessed and saved '{DATA_SOURCE_PROCESSED}'. Total {len(df_weekly)} weeks.")
    
except Exception as e:
    print(f"Error during preprocessing: {e}")
    raise

# --- 2. Feature Engineering ---
# (Not needed for SARIMA, the model handles seasonality internally)
print("Loading final data for SARIMA training...")
# We use the weekly data we just saved
y = df_weekly['call_count']


if y.empty or len(y) < 8: # m*2
    print(f"FATAL: Not enough data for SARIMA m=4 (needs at least 8 weeks). Found only {len(y)}.")
else:
    # --- 3. Model Training ---
    print("Training SARIMA (m=4) model on full weekly dataset...")
    
    # Instantiate SARIMA model (using params from your notebook)
    # m=4 (monthly seasonality on weekly data)
    sarima_model = SARIMAX(y,
                           order=(1, 0, 1), # Non-seasonal p,d,q
                           seasonal_order=(1, 0, 1, 4), # Seasonal P,D,Q,m
                           enforce_stationarity=False,
                           enforce_invertibility=False)
    
    # Fit the model
    sarima_fit = sarima_model.fit(disp=False)
    print("Model training complete.")

    # --- 4. Save Model ---
    print(f"Saving model to '{MODEL_NAME}'...")
    joblib.dump(sarima_fit, MODEL_NAME) # Save the *fitted* model
    print(f"Successfully saved '{MODEL_NAME}'.")

print("--- Weekly Training Script Finished ---")