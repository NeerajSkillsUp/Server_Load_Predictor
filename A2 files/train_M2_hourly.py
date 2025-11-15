import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor # <-- Changed to RandomForest
import joblib
import warnings

warnings.filterwarnings("ignore")

print("--- Starting Final M2 (RandomForest, Hourly) Training ---")
MODEL_NAME = "A2 files\\M2_hourly.joblib"
DATA_SOURCE_RAW = "A2 files\\A2.csv"
DATA_SOURCE_PROCESSED = "A2 files\\A2_hourly_calls.csv"

# --- 0. Feature Creation Function (for HOURLY data) ---
def create_features(df, target_col):
    """Create time-based and lag features for HOURLY data."""
    df_feat = df.copy()
    
    df_feat['hour'] = df_feat.index.hour
    df_feat['dayofweek'] = df_feat.index.dayofweek
    df_feat['month'] = df_feat.index.month
    
    # Lag feature based on 24h seasonality
    df_feat['lag_24'] = df_feat[target_col].shift(24)
    # Also add a rolling mean
    df_feat['rolling_mean_24'] = df_feat[target_col].shift(1).rolling(window=24).mean()

    # Drop rows with NaNs created by feature engineering
    df_feat = df_feat.dropna()
    
    return df_feat.drop(target_col, axis=1), df_feat[target_col]

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

# --- 2. Feature Engineering ---
print("Creating features from 100% of the hourly data...")
# We use the hourly data we just saved
X, y = create_features(df_hourly, 'call_count')
print(f"Training features shape: {X.shape}")
print(f"Training target shape: {y.shape}")

if X.empty:
    print("FATAL: No features were created. This happens if data is shorter than your lag period (24 hours).")
else:
    # --- 3. Model Training ---
    print("Training RandomForest model on full hourly dataset...")
    
    # Instantiate RandomForest model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    
    # Fit the model
    model.fit(X, y)
    print("Model training complete.")

    # --- 4. Save Model ---
    print(f"Saving model to '{MODEL_NAME}'...")
    joblib.dump(model, MODEL_NAME)
    print(f"Successfully saved '{MODEL_NAME}'.")

print("--- Hourly Training Script Finished ---")