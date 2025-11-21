import pandas as pd
import numpy as np
from xgboost import XGBRegressor # <-- Using XGBoost
import joblib
import warnings
import os

warnings.filterwarnings("ignore")

print("--- Starting Final M7 (XGBoost, Daily) Training ---")

# Define file paths
MODEL_NAME = "A7 files\\M7_daily.joblib"
DATA_SOURCE_RAW = "A7 files\\A7.csv"
DATA_SOURCE_PROCESSED = "A7 files\\A7_daily_calls.csv"

# --- 0. Feature Creation Function (for DAILY data) ---
def create_features(df, target_col):
    """Create time-based and lag features for DAILY data."""
    df_feat = df.copy()
    
    df_feat['dayofweek'] = df_feat.index.dayofweek
    df_feat['month'] = df_feat.index.month
    df_feat['dayofyear'] = df_feat.index.dayofyear
    df_feat['quarter'] = df_feat.index.quarter

    # Lag feature based on weekly seasonality (7 days)
    df_feat['lag_7'] = df_feat[target_col].shift(7)
    
    # Also add a weekly rolling mean
    df_feat['rolling_mean_7'] = df_feat[target_col].shift(1).rolling(window=7).mean()

    # Drop rows with NaNs created by feature engineering
    df_feat = df_feat.dropna()
    
    return df_feat.drop(target_col, axis=1), df_feat[target_col]

# --- 1. Preprocessing ---
print(f"Loading and preprocessing {DATA_SOURCE_RAW}...")
try:
    df_raw = pd.read_csv(DATA_SOURCE_RAW)
    df_raw['datetime'] = pd.to_datetime(df_raw['Time of Call'], format='%d-%m-%Y %H:%M')
    df_raw = df_raw.set_index('datetime')
    
    df_daily = df_raw.resample('D').size().to_frame('call_count')
    df_daily = df_daily.asfreq(freq='D', fill_value=0)
    
    # Drop the last row (incomplete day)
    if not df_daily.empty:
        df_daily = df_daily.iloc[:-1]
    
    # Save the daily data
    df_daily.to_csv(DATA_SOURCE_PROCESSED)
    print(f"Preprocessed and saved '{DATA_SOURCE_PROCESSED}'. Total {len(df_daily)} days.")
    
except Exception as e:
    print(f"Error during preprocessing: {e}")
    raise

# --- 2. Feature Engineering ---
print("Creating features from 100% of the daily data...")
# We use the daily data we just saved
X, y = create_features(df_daily, 'call_count')
print(f"Training features shape: {X.shape}")
print(f"Training target shape: {y.shape}")

if X.empty:
    print("FATAL: No features were created. This happens if data is shorter than your lag period (7 days).")
else:
    # --- 3. Model Training ---
    print("Training XGBoost model on full daily dataset...")
    
    # Instantiate XGBoost model (using params from your notebook)
    model = XGBRegressor(n_estimators=1000,
                         # early_stopping_rounds=50, # Not used in final training
                         objective='count:poisson',
                         eval_metric='rmse',
                         n_jobs=-1,
                         random_state=42)
    
    # Fit the model
    # Note: We don't use eval_set in the final training, we train on 100% of data
    model.fit(X, y)
    print("Model training complete.")

    # --- 4. Save Model ---
    print(f"Saving model to '{MODEL_NAME}'...")
    joblib.dump(model, MODEL_NAME)
    print(f"Successfully saved '{MODEL_NAME}'.")

print("--- Daily Training Script Finished ---")