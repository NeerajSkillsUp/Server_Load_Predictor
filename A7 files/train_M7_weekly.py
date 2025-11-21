import pandas as pd
import numpy as np
from xgboost import XGBRegressor # <-- Using XGBoost
import joblib
import warnings
import os

warnings.filterwarnings("ignore")

print("--- Starting Final M7 (XGBoost, Weekly) Training ---")

# Define file paths
MODEL_NAME = "A7 files\\M7_weekly.joblib"
DATA_SOURCE_RAW = "A7 files\\A7.csv"
DATA_SOURCE_PROCESSED = "A7 files\\A7_weekly_calls.csv"

# --- 0. Feature Creation Function (for WEEKLY data) ---
def create_features(df, target_col):
    """Create time-based and lag features for WEEKLY data."""
    df_feat = df.copy()
    
    df_feat['month'] = df_feat.index.month
    # Use .isocalendar().week to get week of year, convert to int
    df_feat['weekofyear'] = df_feat.index.isocalendar().week.astype(int)
    df_feat['quarter'] = df_feat.index.quarter

    # Lag feature based on "monthly" seasonality (4 weeks)
    df_feat['lag_4'] = df_feat[target_col].shift(4)
    
    # Also add a "monthly" rolling mean
    df_feat['rolling_mean_4'] = df_feat[target_col].shift(1).rolling(window=4).mean()

    # Drop rows with NaNs created by feature engineering
    df_feat = df_feat.dropna()
    
    return df_feat.drop(target_col, axis=1), df_feat[target_col]

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
print("Creating features from 100% of the weekly data...")
# We use the weekly data we just saved
X, y = create_features(df_weekly, 'call_count')
print(f"Training features shape: {X.shape}")
print(f"Training target shape: {y.shape}")

if X.empty:
    print("FATAL: No features were created. This happens if data is shorter than your lag period (4 weeks).")
else:
    # --- 3. Model Training ---
    print("Training XGBoost model on full weekly dataset...")
    
    # Instantiate XGBoost model (using params from your notebook)
    model = XGBRegressor(n_estimators=1000,
                         # early_stopping_rounds=20, # Not used in final training
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

print("--- Weekly Training Script Finished ---")