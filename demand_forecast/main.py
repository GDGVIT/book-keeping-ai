from forecasting import predict_demand, check_stock_and_alert
from bokeh_forecast import create_bokeh_plots
from utils import load_data

def main():
    data_path = 'data/orders4.csv'
    df = load_data(data_path)

    item_id = int(input("Enter item ID for demand forecast: "))
    num_forecast_months = 12

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
