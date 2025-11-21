import pandas as pd
import numpy as np
from prophet import Prophet
import joblib  # <-- Using joblib to save
import warnings
import os

warnings.filterwarnings("ignore")

print("--- Starting Final M7 (Prophet, Hourly) Training ---")

# Define file paths
MODEL_NAME = "A7 files\\M7_hourly.joblib"
DATA_SOURCE_RAW = "A7 files\\A7.csv"
DATA_SOURCE_PROCESSED = "A7 files\\A7_hourly_calls.csv"

# --- 1. Preprocessing ---
print(f"Loading and preprocessing {DATA_SOURCE_RAW}...")
try:
    df_raw = pd.read_csv(DATA_SOURCE_RAW)
    df_raw['datetime'] = pd.to_datetime(df_raw['Time of Call'], format='%d-%m-%Y %H:%M')
    df_raw = df_raw.set_index('datetime')
    
    df_hourly = df_raw.resample('H').size().to_frame('call_count')
    df_hourly = df_hourly.asfreq(freq='H', fill_value=0)
    
    # Drop the last row (incomplete hour)
    if not df_hourly.empty:
        df_hourly = df_hourly.iloc[:-1]
    
    # Save the hourly data
    df_hourly.to_csv(DATA_SOURCE_PROCESSED)
    print(f"Preprocessed and saved '{DATA_SOURCE_PROCESSED}'. Total {len(df_hourly)} hours.")
    
except Exception as e:
    print(f"Error during preprocessing: {e}")
    raise

# --- 2. Prepare Data for Prophet ---
print("Preparing data for Prophet model...")
df_prophet = df_hourly.reset_index().rename(columns={'datetime': 'ds', 'call_count': 'y'})
print(f"Prepared Prophet input with {len(df_prophet)} records.")


if df_prophet.empty or len(df_prophet) < 2:
    print("FATAL: Not enough data for Prophet to train (needs at least 2 data points).")
else:
    # --- 3. Train Prophet Model ---
    print("Training final Prophet model (M7) for A7...")
    try:
        # Instantiate Prophet model (as per your notebook)
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False
        )

        model.fit(df_prophet)
        print("Fit complete.")
    except Exception as e:
        print(f"Error during Prophet model training: {e}")
        raise

    # --- 4. Save Model Artifact ---
    try:
        joblib.dump(model, MODEL_NAME) # <-- Using joblib.dump as you requested
        print(f"Successfully saved Prophet model to {MODEL_NAME}")
    except Exception as e:
        print(f"Error saving model: {e}")
        raise

print("\n--- M7 (Prophet) Training for A7 Complete. Ready for deployment! ---")