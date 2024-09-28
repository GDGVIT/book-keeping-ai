import pandas as pd
import numpy as np
from bokeh.plotting import figure, show, output_file
from bokeh.layouts import column
from bokeh.models import ColumnDataSource
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Check for stationarity
def is_stationary(series):
    result = adfuller(series)
    p_value = result[1]
    return p_value < 0.05

# Predict future demand using Linear Regression, Exponential Smoothing, or ARIMA
def predict_demand(df, item_id, forecast_months=6):
    item_data = df[df['item_id'] == item_id].copy()
    item_data = item_data.sort_values('transaction_date')

    # Prepare data for modeling
    item_data['year_month'] = item_data['transaction_date'].dt.to_period('M')
    monthly_demand = item_data.groupby('year_month')['quantity'].sum().reset_index()
    monthly_demand.set_index('year_month', inplace=True)

    # Check if we have enough data
    if len(monthly_demand) < 2:
        print(f"Not enough data available for item ID {item_id}.")
        return None, None

    # Handle missing values
    monthly_demand = monthly_demand.asfreq('M', fill_value=0)

    # Check for stationarity and differencing if necessary
    if not is_stationary(monthly_demand['quantity']):
        print("The time series is non-stationary. Differencing the series.")
        monthly_demand['quantity'] = monthly_demand['quantity'].diff().dropna()

    # Attempt to fit the Linear Regression model
    try:
        monthly_demand['time_index'] = np.arange(len(monthly_demand))
        X = monthly_demand['time_index'].values.reshape(-1, 1)
        y = monthly_demand['quantity'].values

        model = LinearRegression()
        model.fit(X, y)

        # Generate future dates for prediction
        future_index = np.arange(len(monthly_demand), len(monthly_demand) + forecast_months).reshape(-1, 1)
        forecast = model.predict(future_index)

        # Evaluate model performance
        mse = mean_squared_error(y, model.predict(X))
        r2 = r2_score(y, model.predict(X))
        print(f"Linear Regression - MSE: {mse:.2f}, R-squared: {r2:.2f}")

    except Exception as e:
        print(f"Linear Regression fitting failed: {e}. Trying Exponential Smoothing as fallback.")
        try:
            model = ExponentialSmoothing(monthly_demand['quantity'], seasonal='add', seasonal_periods=12)
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=forecast_months)
            print("Exponential Smoothing model fitted successfully.")
        except Exception as e:
            print(f"Exponential Smoothing fitting failed: {e}. Trying ARIMA as fallback.")
            try:
                model = ARIMA(monthly_demand['quantity'], order=(1, 1, 1))
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=forecast_months)
                print("ARIMA model fitted successfully.")
            except Exception as e:
                print(f"ARIMA fitting failed: {e}.")
                return None, None

    # Generate future dates for the forecast
    future_dates = [monthly_demand.index[-1].to_timestamp() + pd.DateOffset(months=i) for i in range(1, forecast_months + 1)]
    print("Predicted Demand for future months:")
    for date, demand in zip(future_dates, forecast):
        print(f"{date.strftime('%Y-%m')}: {demand:.2f}")

    return future_dates, forecast

# Check stock and create reorder alerts
def check_stock_and_alert(df, item_id, predicted_demand, future_months):
    item_data = df[df['item_id'] == item_id]
    current_stock = item_data['quantity'].sum()

    alerts = []
    for month, demand in zip(future_months, predicted_demand):
        if demand > current_stock:
            alerts.append(
                f"Alert: You may need to reorder {demand - current_stock:.2f} units of item ID {item_id} by {month.strftime('%Y-%m')} as the stock might run out."
            )
        else:
            alerts.append(
                f"No reorder necessary for item ID {item_id} in {month.strftime('%Y-%m')}. Sufficient stock available."
            )

    return alerts

# Create Bokeh plots for actual and predicted demand
def create_bokeh_plots(df, item_id, future_months, predicted_demand):
    item_data = df[df['item_id'] == item_id].copy()
    item_data['year_month'] = item_data['transaction_date'].dt.to_period('M')
    actual_demand = item_data.groupby('year_month')['quantity'].sum().reset_index()

    actual_source = ColumnDataSource(
        data=dict(month=actual_demand['year_month'].dt.to_timestamp(), quantity=actual_demand['quantity']))
    predicted_source = ColumnDataSource(data=dict(month=future_months, quantity=predicted_demand))

    actual_plot = figure(title=f'Actual Demand for Item ID {item_id}', x_axis_label='Date', y_axis_label='Quantity',
                         x_axis_type='datetime')
    actual_plot.line('month', 'quantity', source=actual_source, line_width=2, color='blue',
                     legend_label='Actual Demand')
    actual_plot.scatter('month', 'quantity', source=actual_source, size=8, color='blue')

    predicted_plot = figure(title=f'Predicted Demand for Item ID {item_id}', x_axis_label='Date',
                            y_axis_label='Quantity', x_axis_type='datetime')
    predicted_plot.line('month', 'quantity', source=predicted_source, line_width=2, color='orange',
                        legend_label='Predicted Demand')
    predicted_plot.scatter('month', 'quantity', source=predicted_source, size=8, color='orange')

    output_file("demand_forecasting.html")
    show(column(actual_plot, predicted_plot))
