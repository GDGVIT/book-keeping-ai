from bokeh.plotting import figure, show, output_file
from bokeh.layouts import column
from bokeh.models import ColumnDataSource







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
