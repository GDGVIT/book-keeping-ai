import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import time
import joblib
import os

# Load and preprocess data
try:
    print("Loading data...")
    data = pd.read_csv('data/orders.csv')
    print("Data loaded successfully.")
except Exception as e:
    print(f"Error loading data: {e}")
    raise

# Handle date conversion and feature engineering
try:
    print("Converting dates and engineering features...")
    date_format = "%d-%b-%y"
    data['Date Order was placed'] = pd.to_datetime(data['Date Order was placed'], format=date_format)
    data['Delivery Date'] = pd.to_datetime(data['Delivery Date'], format=date_format)
    data['DayOfWeek'] = data['Date Order was placed'].dt.dayofweek
    data['Month'] = data['Date Order was placed'].dt.month
    data['DaysToDelivery'] = (data['Delivery Date'] - data['Date Order was placed']).dt.days
    print("Date conversion and feature engineering completed.")
except Exception as e:
    print(f"Error during date conversion or feature engineering: {e}")
    raise

# Encode categorical variables
try:
    print("Encoding categorical variables...")
    data_encoded = pd.get_dummies(data, columns=['Product ID', 'Customer Status'], drop_first=True)
    print("Categorical encoding completed.")
except Exception as e:
    print(f"Error during categorical encoding: {e}")
    raise

# Define features and target
try:
    print("Separating features and target variable...")
    X = data_encoded.drop(
        columns=['Quantity Ordered', 'Date Order was placed', 'Delivery Date', 'Order ID', 'Customer ID'])
    y = data_encoded['Quantity Ordered']
    print("Feature and target variable separation completed.")
except Exception as e:
    print(f"Error separating features and target variable: {e}")
    raise

# Split data
try:
    print("Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Data splitting completed.")
except Exception as e:
    print(f"Error during data splitting: {e}")
    raise

# Scale the data
try:
    print("Scaling data...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    print("Data scaling completed.")
except Exception as e:
    print(f"Error during data scaling: {e}")
    raise

# Model filename
model_filename = 'demand_forecast_model.joblib'

# Check if the model file exists
if os.path.exists(model_filename):
    try:
        print("Loading existing model...")
        model = joblib.load(model_filename)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
else:
    # Fit a Random Forest model
    try:
        print("Training model...")
        start_time = time.time()

        model = RandomForestRegressor(n_estimators=10, random_state=42)

        print("Fitting model...")
        model.fit(X_train, y_train)

        # Save the trained model
        joblib.dump(model, model_filename)
        print(f"Model saved as {model_filename}.")

        end_time = time.time()
        print(f"Model training completed in {end_time - start_time:.2f} seconds.")
    except Exception as e:
        print(f"Error during model training: {e}")
        raise

# Predict on validation set
try:
    print("Predicting...")
    start_time = time.time()
    y_pred = model.predict(X_val)
    end_time = time.time()
    print(f"Prediction completed in {end_time - start_time:.2f} seconds.")
except Exception as e:
    print(f"Error during prediction: {e}")
    raise

# Evaluate the model
try:
    print("Evaluating model...")
    start_time = time.time()
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    end_time = time.time()
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R² Score: {r2:.2f}")
    print(f"Model evaluation completed in {end_time - start_time:.2f} seconds.")
except Exception as e:
    print(f"Error during model evaluation: {e}")
    raise

# Cross-validation with reduced complexity
try:
    print("Performing cross-validation with fewer folds...")
    start_time = time.time()

    # Test with a smaller subset of data
    X_small, _, y_small, _ = train_test_split(X, y, test_size=0.9, random_state=42)

    # Perform cross-validation with fewer folds
    cross_val_scores = cross_val_score(model, X_small, y_small, cv=3, scoring='r2')

    # Calculate mean and standard deviation of cross-validation scores
    mean_score = cross_val_scores.mean()
    std_dev = cross_val_scores.std()

    end_time = time.time()

    print(f"Cross-Validation R² Scores: {cross_val_scores}")
    print(f"Average Cross-Validation R² Score: {mean_score:.2f}")
    print(f"Standard Deviation of Cross-Validation R² Scores: {std_dev:.2f}")
    print(f"Cross-validation completed in {end_time - start_time:.2f} seconds.")
except Exception as e:
    print(f"Error during cross-validation: {e}")
    raise

# Predict manual data
try:
    print("Predicting manual data...")
    start_time = time.time()
    manual_data = pd.DataFrame({
        'Product ID': [220101000001, 210201000001, 230101000001],
        'Current Stock': [3, 2, 1]
    })
    manual_data_encoded = pd.get_dummies(manual_data, columns=['Product ID'], drop_first=True)

    # Ensure all columns from X are present in manual_data_encoded with default value of 0
    manual_data_encoded = pd.concat(
        [manual_data_encoded, pd.DataFrame(0, index=manual_data_encoded.index, columns=[col for col in X.columns if col not in manual_data_encoded.columns])],
        axis=1
    )

    # Reorder columns to match X
    manual_data_encoded = manual_data_encoded[X.columns]

    # Scale manual data
    manual_data_encoded = scaler.transform(manual_data_encoded)

    # Predict
    predicted_demand = model.predict(manual_data_encoded)
    end_time = time.time()
    print(f"Manual data prediction completed in {end_time - start_time:.2f} seconds.")

    # Output predictions
    BUFFER = 0.10  # Adjust the buffer as needed

    for i, row in manual_data.iterrows():
        product_id = row['Product ID']
        current_stock = row['Current Stock']
        predicted_demand_value = predicted_demand[i]

        if predicted_demand_value > current_stock + BUFFER:
            print(f"Warning: Low Stock! Consider reordering Product ID: {product_id}")
        else:
            print(f"Stock is sufficient for Product ID: {product_id}")
except Exception as e:
    print(f"Error during manual data prediction: {e}")
    raise

# Plot the actual vs predicted values
try:
    print("Plotting actual vs predicted values...")
    start_time = time.time()
    plt.figure(figsize=(14, 8))
    plt.plot(y_val.values, label='Actual Values', color='blue')
    plt.plot(y_pred, label='Predicted Values', color='red')
    plt.title(f'Actual vs Predicted Quantity Ordered\nR² Score: {r2:.2f}')
    plt.xlabel('Samples Index')
    plt.ylabel('Quantity Ordered')
    plt.legend()
    plt.grid(True)
    plt.show()
    end_time = time.time()
    print(f"Actual vs Predicted plot displayed in {end_time - start_time:.2f} seconds.")
except Exception as e:
    print(f"Error during plotting actual vs predicted values: {e}")

# Plot the distribution of predictions
try:
    print("Plotting distribution of predicted demand...")
    start_time = time.time()
    plt.figure(figsize=(12, 6))
    plt.hist(predicted_demand, bins=50, alpha=0.7, color='green')
    plt.title('Distribution of Predicted Demand')
    plt.xlabel('Predicted Demand')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    end_time = time.time()
    print(f"Distribution of predicted demand plot displayed in {end_time - start_time:.2f} seconds.")
except Exception as e:
    print(f"Error during plotting predicted demand distribution: {e}")

# Analyze residuals
try:
    print("Analyzing residuals...")
    start_time = time.time()
    residuals = y_val - y_pred
    plt.figure(figsize=(12, 6))
    plt.hist(residuals, bins=50, alpha=0.7, color='purple')
    plt.title('Residuals Distribution')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    end_time = time.time()
    print(f"Residuals analysis completed in {end_time - start_time:.2f} seconds.")
except Exception as e:
    print(f"Error during residuals analysis: {e}")
