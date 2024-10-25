import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging

logging.getLogger('tensorflow').setLevel(logging.FATAL)

import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning  # Import ConvergenceWarning

warnings.filterwarnings("ignore")  # Ignore all warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)  # Ignore ConvergenceWarnings
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler


FORECAST_MONTHS = 6
LSTM_EPOCHS = 200
WINDOW_SIZE = 6

# Helper function to calculate MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def is_stationary(series):
    p_value = adfuller(series.dropna())[1]
    return p_value < 0.05

def seasonal_differencing(series, period):
    return series.diff(period).dropna()

# Data Preprocessing: Handle missing values and outliers
def preprocess_data(data):
    data = data.fillna(method='ffill')
    data = data.clip(lower=0)  # Ensure no negative values
    return data

def sarima_grid_search(data):
    p_values = d_values = q_values = range(0, 2)
    P_values = D_values = Q_values = range(0, 2)

    best_aic = np.inf
    best_order = None
    best_model = None

    for p in p_values:
        for d in d_values:
            for q in q_values:
                for P in P_values:
                    for D in D_values:
                        for Q in Q_values:
                            try:
                                model = SARIMAX(data,
                                                order=(p, d, q),
                                                seasonal_order=(P, D, Q, 12),
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                                results = model.fit(disp=False, maxiter=200, method='lbfgs')  # Added parameters
                                if results.aic < best_aic:
                                    best_aic = results.aic
                                    best_order = (p, d, q, P, D, Q)
                                    best_model = results
                            except Exception as e:
                                print(f"Error fitting SARIMA({p},{d},{q})x({P},{D},{Q},12): {e}")

    return best_model, best_order

# Create the LSTM model
def create_rnn_model(input_shape):
    model = Sequential([
        LSTM(128, activation='relu', return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Prepare RNN data with sliding window
def prepare_rnn_data(data, window_size=WINDOW_SIZE):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

# Fit SARIMA Model with Grid Search
def fit_sarima_model(data):
    best_sarima, best_order = sarima_grid_search(data)
    if best_sarima is not None:
        forecast = best_sarima.forecast(steps=FORECAST_MONTHS)
        return forecast, best_sarima
    return None, None

# Improved RNN forecast
def run_rnn_forecast(monthly_demand):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(monthly_demand.values.reshape(-1, 1))
    X, y = prepare_rnn_data(scaled_data, window_size=WINDOW_SIZE)

    if len(X) == 0 or len(y) == 0:
        raise ValueError("Insufficient data for RNN training")

    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = create_rnn_model((X.shape[1], 1))

    # Use early stopping to avoid overfitting
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    mc = ModelCheckpoint('best_model.keras', save_best_only=True)  # Changed to .keras

    model.fit(X, y, epochs=LSTM_EPOCHS, validation_split=0.2, verbose=1, callbacks=[es, mc])

    last_input = scaled_data[-WINDOW_SIZE:].reshape((1, WINDOW_SIZE, 1))
    forecast_rnn = []

    for _ in range(FORECAST_MONTHS):
        predicted = model.predict(last_input, verbose=0)
        forecast_rnn.append(predicted[0, 0])
        last_input = np.append(last_input[:, 1:, :], predicted.reshape(1, 1, 1), axis=1)

    forecast = scaler.inverse_transform(np.array(forecast_rnn).reshape(-1, 1)).flatten()
    return forecast

# Combine forecasts from SARIMA and RNN
def ensemble_forecast(sarima_forecast, rnn_forecast):
    sarima_weight = 0.7  # Adjust weights based on model performance
    rnn_weight = 0.3
    return sarima_weight * np.array(sarima_forecast) + rnn_weight * np.array(rnn_forecast)

def predict_demand(df, item_id):
    item_data = df[df['item_id'] == item_id].copy()
    item_data.sort_values('transaction_date', inplace=True)

    item_data['year_month'] = item_data['transaction_date'].dt.to_period('M')
    monthly_demand = item_data.groupby('year_month')['quantity'].sum().reset_index()
    monthly_demand.set_index('year_month', inplace=True)

    monthly_demand = preprocess_data(monthly_demand.asfreq('M', fill_value=0))

    if not is_stationary(monthly_demand['quantity']):
        monthly_demand['quantity'] = seasonal_differencing(monthly_demand['quantity'], period=12)

    forecast_sarima, _ = fit_sarima_model(monthly_demand['quantity'])
    forecast_rnn = run_rnn_forecast(monthly_demand['quantity'])

    # Calculate MAPE for both models
    if len(monthly_demand['quantity']) > FORECAST_MONTHS:
        actual_values = monthly_demand['quantity'][-FORECAST_MONTHS:].values
        sarima_mape = mean_absolute_percentage_error(actual_values, forecast_sarima)
        rnn_mape = mean_absolute_percentage_error(actual_values, forecast_rnn)

        print(f"SARIMA MAPE: {sarima_mape:.2f}%")
        print(f"RNN MAPE: {rnn_mape:.2f}%")

    forecast = ensemble_forecast(forecast_sarima, forecast_rnn)

    future_dates = [monthly_demand.index[-1].to_timestamp() + pd.DateOffset(months=i) for i in
                    range(1, FORECAST_MONTHS + 1)]
    return future_dates, forecast

def check_stock_and_alert(df, item_id, predicted_demand, future_months):
    item_data = df[df['item_id'] == item_id]
    current_stock = item_data['quantity'].sum()

    alerts = []
    for month, demand in zip(future_months, predicted_demand):
        if demand > current_stock:
            alerts.append(
                f"Alert: You may need to reorder {demand - current_stock:.2f} units of item ID {item_id} by {month.strftime('%Y-%m')}.")
        else:
            alerts.append(
                f"No reorder necessary for item ID {item_id} in {month.strftime('%Y-%m')}. Sufficient stock available.")

    return alerts