# --- Best Model (M2, A9 Daily): XGBoost ---
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.dates as mdates

# --- 1. Load and Preprocess Data ---
print("--- Loading and Preprocessing A9.csv for DAILY analysis ---")
try:
    df = pd.read_csv('A9.csv')
    print("Original data sample:")
    print(df.head())

    # Parse time. The format is DD-MM-YYYY HH:MM
    df['datetime'] = pd.to_datetime(df['Time of Call'], format='%d-%m-%Y %H:%M')

    # Set index
    df = df.set_index('datetime')

    # Resample to daily counts
    # .size() counts all rows in the bin.
    # .asfreq(freq='D') ensures all days are present, filling with NA
    # .fillna(0) ensures that days with no calls are recorded as 0.
    df_daily = df.resample('D').size().to_frame('call_count')
    df_daily = df_daily.asfreq(freq='D', fill_value=0) # Ensure all days are present

    # --- MODIFICATION ---
    # Drop the last row, which is the incomplete, current day.
    if not df_daily.empty:
        df_daily = df_daily.iloc[:-1]
        print(f"\n--- Dropped last row (assumed incomplete day) ---")
    # --- END MODIFICATION ---

    print("\n--- Resampled Daily Data (A9) ---")
    print(df_daily.head())
    print(f"\nData shape: {df_daily.shape}")
    print(f"Time range from {df_daily.index.min()} to {df_daily.index.max()}")
    print("\nData Info:")
    df_daily.info()

    # Save the processed data to a CSV for potential future use
    df_daily.to_csv('A9_daily_calls.csv')
    print("\nSaved 'A9_daily_calls.csv'")

except Exception as e:
    print(f"Error during data loading or preprocessing: {e}")
    raise

# --- 2. EDA: Time Series Plot ---
print("\n--- Generating Time Series Plot (Daily) ---")
try:
    plt.figure(figsize=(18, 7))
    plt.plot(df_daily.index, df_daily['call_count'], label='Daily Calls')
    plt.title('Daily API Calls for A9 (A9_Daily_calls.png)', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Number of Calls', fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('a9_daily_calls.png')
    print("Saved 'a9_daily_calls.png'")
    plt.close() # Close the figure
except Exception as e:
    print(f"Error generating time series plot: {e}")

# --- 3. EDA: Stationarity Test (ADF) ---
print("\n--- Performing Stationarity Test (ADF) on Daily Data ---")
try:
    adf_result = adfuller(df_daily['call_count'])
    print(f'ADF Statistic: {adf_result[0]}')
    print(f'p-value: {adf_result[1]}')
    print('Critical Values:')
    for key, value in adf_result[4].items():
        print(f'\t{key}: {value}')

    if adf_result[1] < 0.05:
        print("\nResult: The series is likely stationary (p-value < 0.05).")
    else:
        print("\nResult: The series is likely non-stationary (p-value >= 0.05).")
except Exception as e:
    print(f"Error during ADF test: {e}")

# --- 4. EDA: Autocorrelation Plots ---
print("\n--- Generating Autocorrelation Plots (Daily) ---")
try:
    # Determine a reasonable number of lags, e.g., 60 days
    n_lags = min(60, len(df_daily) // 2 - 1)

    if n_lags <= 0 and len(df_daily) > 2:
         n_lags = len(df_daily) // 2 - 1 # Fallback for very short series

    if n_lags > 0:
        # ACF Plot
        fig, ax = plt.subplots(figsize=(12, 5))
        plot_acf(df_daily['call_count'], lags=n_lags, ax=ax)
        ax.set_title('Autocorrelation Function (ACF) for A9 (a9_acf_daily.png)')
        ax.set_xlabel('Lags (days)')
        ax.set_ylabel('ACF')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('a9_acf_daily.png')
        print("Saved 'a9_acf_daily.png'")
        plt.close(fig)

        # PACF Plot
        fig, ax = plt.subplots(figsize=(12, 5))
        plot_pacf(df_daily['call_count'], lags=n_lags, ax=ax)
        ax.set_title('Partial Autocorrelation Function (PACF) for A9 (a9_pacf_daily.png)')
        ax.set_xlabel('Lags (days)')
        ax.set_ylabel('PACF')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('a9_pacf_daily.png')
        print("Saved 'a9_pacf_daily.png'")
        plt.close(fig)
    else:
        print("Not enough data to generate valid autocorrelation plots.")

except Exception as e:
    print(f"Error generating autocorrelation plots: {e}")

print("\n--- EDA for A9 (Daily) complete. ---")


# ==================================================================
#                  --- MODELING PART ---
# ==================================================================


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.api import SARIMAX
import statsmodels.api as sm
from xgboost import XGBRegressor
from prophet import Prophet
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

print("\n\n--- M2 Model Competition (A9 Daily Data) ---")

# --- 0. Function Definition ---
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


# --- 1. Load Preprocessed Data ---
try:
    df = pd.read_csv('A9_daily_calls.csv', index_col='datetime', parse_dates=True)
    df = df.asfreq('D') # Ensure daily frequency
    print(f"Loaded 'A9_daily_calls.csv'. Total days: {len(df)}")
except Exception as e:
    print(f"Error loading data: {e}")
    print("Please make sure 'A9_daily_calls.csv' exists (run the EDA script first).")
    raise

# --- 2. Train/Test Split ---
test_size_days = 31  # Use 31 days for testing

if len(df) <= test_size_days:
    print(f"Warning: Dataset is too small for the planned test size. Adjusting test size.")
    test_size_days = int(len(df) * 0.2) # Fallback to 20%

# Ensure enough training data for seasonality
MIN_TRAIN_LEN_SARIMA = (7 * 2) # SARIMA needs at least 2 full cycles
MIN_TRAIN_LEN_NAIVE = 7 # Naive needs 1 full cycle
MIN_TRAIN_LEN_ML = 8 # For lag_7

if len(df) - test_size_days < MIN_TRAIN_LEN_SARIMA:
     print(f"Warning: Training data is very short ({len(df) - test_size_days} days).")
     print("Models with 7-day seasonality (SARIMA, Naive, ML) may fail or be inaccurate.")
     # Adjust test size to ensure at least 2 full seasons for training
     if len(df) > MIN_TRAIN_LEN_SARIMA + 7: # Keep at least 1 week for testing
         test_size_days = max(7, len(df) - MIN_TRAIN_LEN_SARIMA) 
         print(f"Adjusting test size to {test_size_days} days to allow for 7-day seasonality.")
     else:
         print("Cannot use 7-day seasonality. Models will likely be poor.")


train_df = df.iloc[:-test_size_days]
test_df = df.iloc[-test_size_days:]

print(f"Training data: {len(train_df)} days (from {train_df.index.min()} to {train_df.index.max()})")
print(f"Testing data:  {len(test_df)} days (from {test_df.index.min()} to {test_df.index.max()})")

# Dictionary to store model performance
model_performance = {}

# --- 3. Model 1: Seasonal Naïve (Baseline) ---
print("\n--- Training Model 1: Seasonal Naïve (s=7) ---")
try:
    if len(train_df) < MIN_TRAIN_LEN_NAIVE:
        print(f"Training data ({len(train_df)}) too short for s=7. Skipping model.")
        model_performance['Seasonal Naive (s=7)'] = {'MAE': np.nan, 'RMSE': np.nan}
        y_pred_naive = pd.Series(np.nan, index=test_df.index)
    else:
        last_season_of_train = train_df['call_count'].iloc[-7:].values
        num_repeats = int(np.ceil(len(test_df) / 7))
        naive_forecast_values = np.tile(last_season_of_train, num_repeats)[:len(test_df)]
        y_pred_naive = pd.Series(naive_forecast_values, index=test_df.index)

        mae_naive = mean_absolute_error(test_df['call_count'], y_pred_naive)
        rmse_naive = np.sqrt(mean_squared_error(test_df['call_count'], y_pred_naive))

        model_performance['Seasonal Naive (s=7)'] = {'MAE': mae_naive, 'RMSE': rmse_naive}
        print(f"Seasonal Naive (s=7) MAE: {mae_naive:.4f}, RMSE: {rmse_naive:.4f}")
except Exception as e:
    print(f"Error in Seasonal Naive model: {e}")
    y_pred_naive = pd.Series(np.nan, index=test_df.index)


# --- 4. Model 2: SARIMA ---
print("\n--- Training Model 2: SARIMA (m=7) ---")
try:
    if len(train_df) < MIN_TRAIN_LEN_SARIMA:
        print(f"Training data ({len(train_df)}) too short for SARIMA m=7. Skipping model.")
        model_performance['SARIMA (m=7)'] = {'MAE': np.nan, 'RMSE': np.nan}
        y_pred_sarima = pd.Series(np.nan, index=test_df.index)
    else:
        sarima_model = SARIMAX(train_df['call_count'],
                               order=(1, 0, 1), # Non-seasonal p,d,q
                               seasonal_order=(1, 0, 1, 7), # Seasonal P,D,Q,m
                               enforce_stationarity=False,
                               enforce_invertibility=False)

        print("Fitting SARIMA model... (This may take a minute)")
        sarima_fit = sarima_model.fit(disp=False)
        print("Fit complete.")
        y_pred_sarima = sarima_fit.get_forecast(steps=len(test_df)).predicted_mean

        mae_sarima = mean_absolute_error(test_df['call_count'], y_pred_sarima)
        rmse_sarima = np.sqrt(mean_squared_error(test_df['call_count'], y_pred_sarima))

        model_performance['SARIMA (m=7)'] = {'MAE': mae_sarima, 'RMSE': rmse_sarima}
        print(f"SARIMA (m=7) MAE: {mae_sarima:.4f}, RMSE: {rmse_sarima:.4f}")
except Exception as e:
    print(f"Error in SARIMA model: {e}")
    y_pred_sarima = pd.Series(np.nan, index=test_df.index)


# --- 5. Create Features for ML Models ---
print("\n--- Creating features for ML models (Daily) ---")
try:
    df_full_features = pd.concat([train_df, test_df])
    X_full, y_full = create_features(df_full_features, 'call_count')

    # Align indices after feature creation (which drops NaNs)
    X_train_feat = X_full.loc[train_df.index.intersection(X_full.index)]
    y_train_feat = y_full.loc[train_df.index.intersection(y_full.index)]
    X_test_feat = X_full.loc[test_df.index.intersection(X_full.index)]
    y_test_feat = y_full.loc[test_df.index.intersection(y_full.index)]

    # Check if feature creation left any data
    if X_train_feat.empty or X_test_feat.empty:
        print("Not enough data to create lagged features for train/test split. ML models will be skipped.")
        ML_MODELS_ENABLED = False
    else:
        ML_MODELS_ENABLED = True
        print(f"Feature-based training data shape: {X_train_feat.shape}")
        print(f"Feature-based testing data shape: {X_test_feat.shape}")

except Exception as e:
    print(f"Error creating features: {e}")
    ML_MODELS_ENABLED = False

# Initialize ML preds
y_pred_rf_rounded = pd.Series(np.nan, index=test_df.index)
y_pred_glm_rounded = pd.Series(np.nan, index=test_df.index)
y_pred_xgb_rounded = pd.Series(np.nan, index=test_df.index)


if ML_MODELS_ENABLED:
    # --- 6. Model 3: RandomForest ---
    print("\n--- Training Model 3: RandomForest ---")
    try:
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        print("Fitting RandomForest model...")
        rf_model.fit(X_train_feat, y_train_feat)
        print("Fit complete.")

        y_pred_rf = rf_model.predict(X_test_feat)
        y_pred_rf = pd.Series(y_pred_rf, index=y_test_feat.index)
        y_pred_rf_rounded = np.round(y_pred_rf.clip(0)) # Clip at 0, then round

        mae_rf = mean_absolute_error(y_test_feat, y_pred_rf_rounded)
        rmse_rf = np.sqrt(mean_squared_error(y_test_feat, y_pred_rf_rounded))

        model_performance['RandomForest'] = {'MAE': mae_rf, 'RMSE': rmse_rf}
        print(f"RandomForest MAE: {mae_rf:.4f}, RMSE: {rmse_rf:.4f}")
    except Exception as e:
        print(f"Error in RandomForest model: {e}")
        model_performance['RandomForest'] = {'MAE': np.nan, 'RMSE': np.nan}


    # --- 7. Model 4: Poisson GLM ---
    print("\n--- Training Model 4: Poisson GLM ---")
    try:
        # Add constant for intercept
        X_train_glm = sm.add_constant(X_train_feat, has_constant='add')
        X_test_glm = sm.add_constant(X_test_feat, has_constant='add')
        y_train_glm = y_train_feat.loc[X_train_glm.index] # Align y

        glm_model = sm.GLM(y_train_glm, X_train_glm, family=sm.families.Poisson())
        print("Fitting Poisson GLM model...")
        glm_fit = glm_model.fit()
        print("Fit complete.")

        y_pred_glm = glm_fit.predict(X_test_glm)
        y_pred_glm_rounded = np.round(y_pred_glm.clip(0))

        mae_glm = mean_absolute_error(y_test_feat, y_pred_glm_rounded)
        rmse_glm = np.sqrt(mean_squared_error(y_test_feat, y_pred_glm_rounded))

        model_performance['Poisson GLM'] = {'MAE': mae_glm, 'RMSE': rmse_glm}
        print(f"Poisson GLM MAE: {mae_glm:.4f}, RMSE: {rmse_glm:.4f}")
    except Exception as e:
        print(f"Error in Poisson GLM model: {e}")
        model_performance['Poisson GLM'] = {'MAE': np.nan, 'RMSE': np.nan}

    # --- 8. Model 5: XGBoost ---
    print("\n--- Training Model 5: XGBoost ---")
    try:
        xgb_model = XGBRegressor(n_estimators=1000,
                                 early_stopping_rounds=50,
                                 objective='count:poisson', # Better for count data
                                 eval_metric='rmse',
                                 n_jobs=-1,
                                 random_state=42)

        print("Fitting XGBoost model...")
        xgb_model.fit(X_train_feat, y_train_feat,
                      eval_set=[(X_test_feat, y_test_feat)],
                      verbose=False)
        print("Fit complete.")

        y_pred_xgb = xgb_model.predict(X_test_feat)
        y_pred_xgb = pd.Series(y_pred_xgb, index=y_test_feat.index)
        y_pred_xgb_rounded = np.round(y_pred_xgb.clip(0))

        mae_xgb = mean_absolute_error(y_test_feat, y_pred_xgb_rounded)
        rmse_xgb = np.sqrt(mean_squared_error(y_test_feat, y_pred_xgb_rounded))

        model_performance['XGBoost'] = {'MAE': mae_xgb, 'RMSE': rmse_xgb}
        print(f"XGBoost MAE: {mae_xgb:.4f}, RMSE: {rmse_xgb:.4f}")
    except Exception as e:
        print(f"Error in XGBoost model: {e}")
        model_performance['XGBoost'] = {'MAE': np.nan, 'RMSE': np.nan}
else:
    print("\nSkipping RandomForest, GLM, and XGBoost due to insufficient data for feature creation.")
    model_performance['RandomForest'] = {'MAE': np.nan, 'RMSE': np.nan}
    model_performance['Poisson GLM'] = {'MAE': np.nan, 'RMSE': np.nan}
    model_performance['XGBoost'] = {'MAE': np.nan, 'RMSE': np.nan}


# --- 9. Model 6: Prophet ---
print("\n--- Training Model 6: Prophet ---")
try:
    if len(train_df) < 2: # Prophet needs at least 2 data points
        print(f"Training data ({len(train_df)}) too short for Prophet. Skipping model.")
        model_performance['Prophet'] = {'MAE': np.nan, 'RMSE': np.nan}
        y_pred_prophet_rounded = pd.Series(np.nan, index=test_df.index)
    else:
        # Prophet requires columns 'ds' and 'y'
        train_df_prophet = train_df.reset_index().rename(columns={'datetime': 'ds', 'call_count': 'y'})

        # Prophet will auto-detect daily seasonality
        prophet_model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
        prophet_model.add_seasonality(name='monthly', period=30.5, fourier_order=5)


        print("Fitting Prophet model...")
        prophet_model.fit(train_df_prophet)
        print("Fit complete.")

        future_df = prophet_model.make_future_dataframe(periods=len(test_df), freq='D')
        y_pred_prophet_df = prophet_model.predict(future_df)

        y_pred_prophet = y_pred_prophet_df['yhat'].iloc[-len(test_df):]
        y_pred_prophet.index = test_df.index
        y_pred_prophet_rounded = np.round(y_pred_prophet.clip(0))

        mae_prophet = mean_absolute_error(test_df['call_count'], y_pred_prophet_rounded)
        rmse_prophet = np.sqrt(mean_squared_error(test_df['call_count'], y_pred_prophet_rounded))

        model_performance['Prophet'] = {'MAE': mae_prophet, 'RMSE': rmse_prophet}
        print(f"Prophet MAE: {mae_prophet:.4f}, RMSE: {rmse_prophet:.4f}")
except Exception as e:
    print(f"Error in Prophet model: {e}")
    model_performance['Prophet'] = {'MAE': np.nan, 'RMSE': np.nan}
    y_pred_prophet_rounded = pd.Series(np.nan, index=test_df.index)

# --- 10. Comparison and Selection ---
print("\n--- M2 Model Competition Results (A9 Daily) ---")
perf_df = pd.DataFrame(model_performance).T.sort_values(by='MAE').dropna(how='all')
print(perf_df)

if perf_df.empty:
    print("\n--- No models were successfully trained. Cannot select a best model. ---")
    best_model_name = "None"
else:
    best_model_name = perf_df.index[0]
    print(f"\n--- Best Model (M2, A9 Daily): {best_model_name} ---")

# Store predictions of all models for analysis
predictions_df = pd.DataFrame(test_df['call_count'])
predictions_df.rename(columns={'call_count': 'Actual'}, inplace=True)
try:
    predictions_df['Seasonal_Naive_Pred'] = y_pred_naive
    predictions_df['SARIMA_Pred'] = y_pred_sarima
    # Use reindex to align predictions from ML models, filling with NaN if they failed
    predictions_df['RandomForest_Pred'] = y_pred_rf_rounded.reindex(predictions_df.index)
    predictions_df['Poisson_GLM_Pred'] = y_pred_glm_rounded.reindex(predictions_df.index)
    predictions_df['XGBoost_Pred'] = y_pred_xgb_rounded.reindex(predictions_df.index)
    predictions_df['Prophet_Pred'] = y_pred_prophet_rounded.reindex(predictions_df.index)
except Exception as e:
    print(f"Error creating predictions df (likely due to index mismatch from low data): {e}")

# Save predictions
predictions_df.to_csv('A9_M2_predictions_DAILY.csv')
print("\nSaved all model predictions to 'A9_M2_predictions_DAILY.csv'")

# --- 11. Plot Best Model Forecast ---
print("Generating forecast plot for best model (Daily)...")
try:
    plt.figure(figsize=(18, 7))
    plt.plot(train_df['call_count'].iloc[-120:], label='Recent History (120 days)')
    plt.plot(test_df['call_count'], label='Actual Test Data', color='orange')

    # Get the best model's predictions
    best_pred = None
    if best_model_name.startswith('Seasonal Naive'):
        best_pred = y_pred_naive
    elif best_model_name.startswith('SARIMA'):
        best_pred = y_pred_sarima
    elif best_model_name == 'RandomForest':
        best_pred = y_pred_rf_rounded
    elif best_model_name == 'Poisson GLM':
        best_pred = y_pred_glm_rounded
    elif best_model_name == 'XGBoost':
        best_pred = y_pred_xgb_rounded
    elif best_model_name == 'Prophet':
        best_pred = y_pred_prophet_rounded

    if best_pred is not None and not best_pred.isnull().all():
        # Ensure prediction index matches test data for plotting
        best_pred = best_pred.loc[test_df.index.intersection(best_pred.index)]
        plt.plot(best_pred, label=f'Best Model: {best_model_name}', color='green', linestyle='--')

    plt.title(f'M2 (A9, Daily) - Best Model Forecast ({best_model_name}) vs. Actual', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Number of Calls')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('m2_A9_best_model_forecast_DAILY.png')
    print("Saved 'm2_A9_best_model_forecast_DAILY.png'")
    plt.close()

except Exception as e:
    print(f"Error generating plot: {e}")

print("\n--- M2 Full Competition Analysis (A9 Daily) Complete. ---")