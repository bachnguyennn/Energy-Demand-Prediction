import pandas as pd
import numpy as np
import os

RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "AEP_hourly.csv")
PROCESSED_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "AEP_hourly_preprocessed.csv")

def load_data(filepath=RAW_DATA_PATH):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at {filepath}. Please download it first.")
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    # Rename columns for convenience
    df = df.rename(columns={'Datetime': 'datetime', 'AEP_MW': 'load_mw'})
    
    # Convert to datetime and sort
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Check for duplicates and keep first
    df = df.drop_duplicates(subset=['datetime'], keep='first')
    
    # Set datetime as index to resample and fill missing hours
    df = df.set_index('datetime')
    df = df.resample('H').mean()
    
    # Fill missing values using linear interpolation
    if df['load_mw'].isnull().sum() > 0:
        print(f"Filled {df['load_mw'].isnull().sum()} missing values via interpolation.")
        df['load_mw'] = df['load_mw'].interpolate(method='linear')
        
    df = df.reset_index()
    return df

def main():
    print("Loading raw data...")
    try:
        df_raw = load_data()
        print("Preprocessing data...")
        df_clean = preprocess_data(df_raw)
        
        os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
        df_clean.to_csv(PROCESSED_DATA_PATH, index=False)
        print(f"Preprocessed data saved to {PROCESSED_DATA_PATH}")
    except Exception as e:
        print(f"Error during preprocessing: {e}")

if __name__ == "__main__":
    main()
