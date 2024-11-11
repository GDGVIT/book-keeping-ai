import pandas as pd

# Load data
def load_data(data_path):
    df = pd.read_csv(data_path)
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    return df
