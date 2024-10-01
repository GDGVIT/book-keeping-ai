import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import warnings
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress warnings
warnings.filterwarnings("ignore")

# Constants
FORECAST_MONTHS = 6
LSTM_EPOCHS = 50
WINDOW_SIZE = 1


# Check for stationarity
def is_stationary(series):
    """Check if a time series is stationary."""
    p_value = adfuller(series.dropna())[1]
    return p_value < 0.05


# Seasonal differencing
def seasonal_differencing(series, period):
    """Perform seasonal differencing on the series."""
    return series.diff(period).dropna()


# RNN Model Creation
def create_rnn_model(input_shape):
    """Create and compile the RNN model."""
    model = Sequential([
        LSTM(50, activation='relu', input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# Prepare data for RNN
def prepare_rnn_data(data, window_size=WINDOW_SIZE):
    """Prepare data for RNN training."""
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)


# Fit ARIMA model
def fit_arima_model(data, seasonal, forecast_months):
    """Fit an ARIMA model to the data and return the forecast."""
    try:
        arima_model = auto_arima(data, seasonal=seasonal, stepwise=True, m=12 if seasonal else 1)
        forecast, conf_int = arima_model.predict(n_periods=forecast_months, return_conf_int=True)
        logging.info(f"{'Seasonal' if seasonal else 'Non-seasonal'} ARIMA model fitted successfully.")
        return forecast
    except Exception as e:
        logging.error(f"ARIMA fitting failed: {e}")
        return None


# Forecast demand using ARIMA
def forecast_with_arima(monthly_demand, forecast_months):
    """Forecast demand using the fitted ARIMA model."""
    seasonal = len(monthly_demand) >= 24
    forecast = fit_arima_model(monthly_demand['quantity'], seasonal, forecast_months)

    if forecast is not None:
        mse_arima = mean_squared_error(monthly_demand['quantity'][-forecast_months:], forecast)
        r2_arima = r2_score(monthly_demand['quantity'][-forecast_months:], forecast)
        logging.info(f"ARIMA Model MSE: {mse_arima:.2f}, R² Score: {r2_arima:.2f}")
        return forecast
    return None


# Forecast demand using RNN
def run_rnn_forecast(monthly_demand, forecast_months):
    """Forecast demand using the RNN model."""
    try:
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(monthly_demand['quantity'].values.reshape(-1, 1))
        X, y = prepare_rnn_data(scaled_data, window_size=WINDOW_SIZE)

        if len(X) == 0 or len(y) == 0:
            raise ValueError("Insufficient data for RNN training")

        X = X.reshape((X.shape[0], X.shape[1], 1))

        model = create_rnn_model((X.shape[1], 1))
        model.fit(X, y, epochs=LSTM_EPOCHS, verbose=0)

        last_input = scaled_data[-WINDOW_SIZE:].reshape((1, WINDOW_SIZE, 1))
        forecast_rnn = []

        for _ in range(forecast_months):
            predicted = model.predict(last_input, verbose=0)
            forecast_rnn.append(predicted[0, 0])
            last_input = np.append(last_input[:, 1:, :], predicted.reshape(1, 1, 1), axis=1)

        forecast = scaler.inverse_transform(np.array(forecast_rnn).reshape(-1, 1)).flatten()
        logging.info("RNN model fitted successfully.")
        return forecast
    except Exception as e:
        logging.error(f"RNN fitting failed: {e}")
        return None


# Main function for predicting demand
def predict_demand(df, item_id, forecast_months=FORECAST_MONTHS):
    """Predict future demand for a specific item."""
    item_data = df[df['item_id'] == item_id].copy()
    item_data.sort_values('transaction_date', inplace=True)

    # Prepare data for modeling
    item_data['year_month'] = item_data['transaction_date'].dt.to_period('M')
    monthly_demand = item_data.groupby('year_month')['quantity'].sum().reset_index()
    monthly_demand.set_index('year_month', inplace=True)

    # Handle missing values
    monthly_demand = monthly_demand.asfreq('M', fill_value=0)  # Fill missing months with 0
    monthly_demand['quantity'] = monthly_demand['quantity'].interpolate(
        method='time')  # Interpolate to handle NaN values
    monthly_demand['quantity'].fillna(0, inplace=True)  # Replace remaining NaNs with 0

    # Apply seasonal differencing if necessary
    if not is_stationary(monthly_demand['quantity']):
        monthly_demand['quantity'] = seasonal_differencing(monthly_demand['quantity'], period=12)

    # Ensure there are no NaN values before fitting ARIMA
    monthly_demand['quantity'].fillna(0, inplace=True)

    # Forecast with ARIMA or RNN
    forecast = None
    if len(monthly_demand) < 24:
        forecast = forecast_with_arima(monthly_demand, forecast_months)
        if forecast is None:
            forecast = run_rnn_forecast(monthly_demand, forecast_months)
    else:
        forecast = forecast_with_arima(monthly_demand, forecast_months)
        if forecast is None:
            forecast = run_rnn_forecast(monthly_demand, forecast_months)

    if forecast is None:
        logging.error(f"Forecasting failed for item ID {item_id}.")
        return None, None

    future_dates = [monthly_demand.index[-1].to_timestamp() + pd.DateOffset(months=i) for i in
                    range(1, forecast_months + 1)]
    return future_dates, forecast


# Check stock and create reorder alerts
def check_stock_and_alert(df, item_id, predicted_demand, future_months):
    """Check current stock against predicted demand and generate alerts."""
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
