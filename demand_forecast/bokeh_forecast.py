from bokeh.plotting import figure, show, output_file
from bokeh.layouts import column
from bokeh.models import ColumnDataSource


def create_bokeh_plots(df, item_id, future_months, predicted_demand):
    # Filter data for the specified item_id
    item_data = df[df['item_id'] == item_id].copy()

    # Convert transaction date to year-month format
    item_data['year_month'] = item_data['transaction_date'].dt.to_period('M')

    # Aggregate actual demand data by month
    actual_demand = item_data.groupby('year_month')['quantity'].sum().reset_index()

    # Prepare ColumnDataSource for actual demand
    actual_source = ColumnDataSource(data=dict(
        month=actual_demand['year_month'].dt.to_timestamp(),
        quantity=actual_demand['quantity']
    ))

    # Prepare ColumnDataSource for predicted demand
    predicted_source = ColumnDataSource(data=dict(
        month=future_months,
        quantity=predicted_demand
    ))

    # Create the actual demand plot
    actual_plot = figure(
        title=f'Actual Demand for Item ID {item_id}',
        x_axis_label='Date',
        y_axis_label='Quantity',
        x_axis_type='datetime',
        width=800,  # Changed from plot_width to width
        height=400,  # Use height instead of plot_height
        toolbar_location='above',
        background_fill_color='#f9f9f9'  # Light background for better visibility
    )

    actual_plot.line(
        'month', 'quantity',
        source=actual_source,
        line_width=2,
        color='blue',
        legend_label='Actual Demand'
    )

    actual_plot.scatter(
        'month', 'quantity',
        source=actual_source,
        size=8,
        color='blue'
    )

    # Create the predicted demand plot
    predicted_plot = figure(
        title=f'Predicted Demand for Item ID {item_id}',
        x_axis_label='Date',
        y_axis_label='Quantity',
        x_axis_type='datetime',
        width=800,  # Changed from plot_width to width
        height=400,  # Use height instead of plot_height
        toolbar_location='above',
        background_fill_color='#f9f9f9'
    )

    predicted_plot.line(
        'month', 'quantity',
        source=predicted_source,
        line_width=2,
        color='orange',
        legend_label='Predicted Demand'
    )

    predicted_plot.scatter(
        'month', 'quantity',
        source=predicted_source,
        size=8,
        color='orange'
    )

    # Output the Bokeh plots to an HTML file
    output_file("demand_forecasting.html")

    # Show both plots in a vertical column layout
    show(column(actual_plot, predicted_plot))
