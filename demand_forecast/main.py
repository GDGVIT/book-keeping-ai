from flask import Flask, request, jsonify
from forecasting import predict_demand, check_stock_and_alert
from bokeh_forecast import create_bokeh_plots
from utils import load_data
import os

app = Flask(__name__)

# Directory to store uploaded files
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
data_file_path = None


@app.route('/upload', methods=['POST'])
def upload_file():
    global data_file_path
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Save the uploaded file
    data_file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(data_file_path)
    return jsonify({"message": "File uploaded successfully", "file_path": data_file_path})


@app.route('/forecast', methods=['POST'])
def forecast():
    global data_file_path
    if not data_file_path:
        return jsonify({"error": "No data file uploaded yet. Please upload a file first."}), 400

    data = request.get_json()
    item_id = data.get('item_id')

    # Validate item_id
    if item_id is None:
        return jsonify({"error": "Item ID is required."}), 400

    # Load data
    df = load_data(data_file_path)

    # Check if item_id exists in the dataset
    if item_id not in df['item_id'].values:  # Adjust 'item_id' to match your DataFrame column name
        print(f"Item ID {item_id} not found in the dataset")  # Log for debugging
        return jsonify({"error": f"Item ID {item_id} not found in the dataset."}), 404

    # Make predictions
    future_months, predicted_demand = predict_demand(df, item_id)

    if predicted_demand is not None:
        alerts = check_stock_and_alert(df, item_id, predicted_demand, future_months)

        # Optional: create and save Bokeh plots if needed
        create_bokeh_plots(df, item_id, future_months, predicted_demand)

        # Convert arrays to lists for JSON serialization
        response = {
            "future_months": future_months.tolist() if hasattr(future_months, 'tolist') else future_months,
            "predicted_demand": predicted_demand.tolist() if hasattr(predicted_demand, 'tolist') else predicted_demand,
            "alerts": alerts
        }
    else:
        response = {
            "error": f"Could not generate forecasts for item ID {item_id}."
        }

    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
