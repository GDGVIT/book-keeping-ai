from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from pathlib import Path
from forecasting import predict_demand, check_stock_and_alert
from bokeh_forecast import create_bokeh_plots
from utils import load_data
import os
import logging

app = Flask(__name__)

# Enable CORS
CORS(app, resources={r"/*": {"origins": "*"}})

# Configure upload folder and file size limit
UPLOAD_FOLDER = Path(os.getenv('UPLOAD_FOLDER', 'uploads'))
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

# Logging setup
app.logger.setLevel(logging.INFO)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}
data_file_path = None


def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return "Welcome to the Bookkeeping AI API! Use /upload to upload a file and /forecast for forecasting."


@app.route('/upload', methods=['POST'])
def upload_file():
    global data_file_path

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only .csv and .xlsx files are allowed"}), 400

    try:
        # Save the new file and overwrite the existing one
        data_file_path = UPLOAD_FOLDER / file.filename
        file.save(data_file_path)

        # Cleanup old plots
        for plot in UPLOAD_FOLDER.glob("*.html"):
            plot.unlink()

        app.logger.info(f"File uploaded successfully: {data_file_path}")
        return jsonify({"message": "File uploaded successfully", "file_path": str(data_file_path)})

    except Exception as e:
        app.logger.error(f"File upload error: {e}")
        return jsonify({"error": "Failed to upload file"}), 500


@app.route('/forecast', methods=['POST'])
def forecast():
    global data_file_path
    if not data_file_path:
        return jsonify({"error": "No data file uploaded yet. Please upload a file first."}), 400

    data = request.get_json()
    item_id = data.get('item_id')

    if item_id is None:
        return jsonify({"error": "Item ID is required."}), 400

    try:
        # Load data
        df = load_data(data_file_path)

        # Validate item ID
        if item_id not in df['item_id'].values:
            app.logger.error(f"Item ID {item_id} not found in the dataset")
            return jsonify({"error": f"Item ID {item_id} not found in the dataset."}), 404

        # Generate predictions
        future_months, predicted_demand = predict_demand(df, item_id)

        if predicted_demand is None:
            return jsonify({"error": f"Could not generate forecasts for item ID {item_id}."}), 500

        # Generate alerts and plots
        alerts = check_stock_and_alert(df, item_id, predicted_demand, future_months)
        plot_path = create_bokeh_plots(df, item_id, future_months, predicted_demand)

        # Safely convert variables for JSON serialization
        response = {
            "future_months": future_months if isinstance(future_months, list) else future_months.tolist(),
            "predicted_demand": predicted_demand if isinstance(predicted_demand, list) else predicted_demand.tolist(),
            "alerts": alerts,
            "plot_url": f"/plot/{item_id}"
        }

        app.logger.info(f"Forecast successfully generated for item ID {item_id}")
        return jsonify(response)

    except Exception as e:
        app.logger.error(f"Error during forecasting: {e}")
        return jsonify({"error": "An error occurred during forecasting. Please try again."}), 500


@app.route('/plot/<item_id>')
def plot(item_id):
    plot_path = UPLOAD_FOLDER / f"demand_forecast_{item_id}.html"
    if plot_path.exists():
        return send_file(plot_path)
    else:
        return jsonify({"error": "Plot not found"}), 404


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
