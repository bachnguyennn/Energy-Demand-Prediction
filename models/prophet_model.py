import pandas as pd
import numpy as np
from prophet import Prophet
import os
import sys

# Add parent directory to path to import evaluate module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.evaluate import walk_forward_split, calculate_metrics, save_metrics, print_metrics

FEATURES_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "AEP_hourly_features.csv")

def prepare_prophet_data(df):
    """Rename columns to 'ds' and 'y' for Prophet."""
    prophet_df = df[['datetime', 'load_mw', 'is_holiday', 'is_weekend', 'hour']].copy()
    prophet_df = prophet_df.rename(columns={'datetime': 'ds', 'load_mw': 'y'})
    return prophet_df

def run_prophet_validation():
    print("Loading feature data for Prophet...")
    if not os.path.exists(FEATURES_DATA_PATH):
        print("Data not found. Run preprocess.py and feature_engineering.py first.")
        return
        
    df = pd.read_csv(FEATURES_DATA_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    prophet_df = prepare_prophet_data(df)
    
    n_splits = 3
    fold_metrics = []
    
    print("Starting Walk-forward validation for Prophet...")
    for fold, (train_idx, test_idx) in enumerate(walk_forward_split(df, n_splits=n_splits, test_size_days=30)):
        print(f"--- Fold {fold + 1}/{n_splits} ---")
        
        train_df = prophet_df.iloc[train_idx]
        test_df = prophet_df.iloc[test_idx]
        
        # Initialize and fit model
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
        # Add regressors
        model.add_regressor('is_holiday')
        model.add_regressor('is_weekend')
        
        model.fit(train_df)
        
        # Predict
        forecast = model.predict(test_df.drop(columns=['y']))
        y_pred = forecast['yhat'].values
        y_true = test_df['y'].values
        
        metrics = calculate_metrics(y_true, y_pred)
        print_metrics(f"Prophet Fold {fold+1}", metrics)
        fold_metrics.append(metrics)
        
    # Aggregate metrics
    avg_metrics = {
        k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0].keys()
    }
    
    print_metrics("Prophet Average", avg_metrics)
    save_metrics("Prophet", avg_metrics)

if __name__ == "__main__":
    run_prophet_validation()
