import pandas as pd
import numpy as np
from bokeh.plotting import figure, show, output_file
from bokeh.layouts import column
from bokeh.models import ColumnDataSource
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Load data
def load_data(data_path):
    df = pd.read_csv(data_path)
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    return df

# Function to check for stationarity
def is_stationary(series):
    series = series.dropna()
    if series.nunique() <= 1:  # Check if series is constant
        print("Series is constant.")
        return False
    result = adfuller(series)
    return result[1] < 0.05  # p-value less than 0.05 indicates stationarity

# Function to predict future demand using ARIMA or Exponential Smoothing model
def predict_demand(df, item_id, forecast_months=6):
    item_data = df[df['item_id'] == item_id].copy()
    item_data = item_data.sort_values('transaction_date')

    # Prepare the data for modeling
    item_data['year_month'] = item_data['transaction_date'].dt.to_period('M')
    monthly_demand = item_data.groupby('year_month')['quantity'].sum().reset_index()

    # Set the year_month as the index
    monthly_demand.set_index('year_month', inplace=True)
    num_observations = len(monthly_demand)

    if num_observations < 2:
        print(f"Not enough data available for item ID {item_id}.")
        return None, None

    # Handle missing values
    monthly_demand['quantity'].replace([np.inf, -np.inf], np.nan, inplace=True)
    monthly_demand.dropna(inplace=True)

    # Attempt to fit the ARIMA model
    try:
        model = ARIMA(monthly_demand['quantity'], order=(1, 1, 1))  # Adjust order as needed
        model_fit = model.fit()

        # Generate future dates
        last_period = monthly_demand.index[-1].to_timestamp()
        future_dates = [last_period + pd.DateOffset(months=i) for i in range(1, forecast_months + 1)]

        # Generate predictions
        forecast = model_fit.forecast(steps=forecast_months)

        # Evaluate model performance
        y_train = monthly_demand['quantity'].values
        y_pred_train = model_fit.fittedvalues.values

        mse = mean_squared_error(y_train, y_pred_train)
        r2 = r2_score(y_train, y_pred_train)

        print(f"Model Mean Squared Error: {mse:.2f}")
        print(f"Model R-squared: {r2:.2f}")

    except Exception as e:
        print(f"ARIMA fitting failed: {e}. Using Exponential Smoothing as a fallback.")

        # Adjusted to allow Exponential Smoothing for more cases
        if num_observations >= 5:
            model = ExponentialSmoothing(monthly_demand['quantity'], seasonal='add', seasonal_periods=12)
            model_fit = model.fit()

            forecast = model_fit.forecast(steps=forecast_months)
            future_dates = [monthly_demand.index[-1].to_timestamp() + pd.DateOffset(months=i) for i in
                            range(1, forecast_months + 1)]
        else:
            print(f"Not enough data for Exponential Smoothing. Item ID {item_id} needs at least 5 months of data.")
            return None, None

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

# Main function to run the analysis
def main():
    data_path = 'data/orders2.csv'  # Change to your CSV path
    df = load_data(data_path)

    item_id = int(input("Enter item ID for demand forecast: "))

    # Automatically determine the number of months for forecasting based on available data
    latest_date = df['transaction_date'].max()
    num_forecast_months = 12  # or any other number if you want more future predictions

    future_months, predicted_demand = predict_demand(df, item_id, forecast_months=num_forecast_months)

    if predicted_demand is not None:
        alerts = check_stock_and_alert(df, item_id, predicted_demand, future_months)
        for alert in alerts:
            print(alert)

        create_bokeh_plots(df, item_id, future_months, predicted_demand)
    else:
        print(f"Could not generate forecasts for item ID {item_id}.")


if __name__ == "__main__":
    main()
