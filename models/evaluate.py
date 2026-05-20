import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error
import os
import json

METRICS_PATH = os.path.join(os.path.dirname(__file__), "..", "reports", "metrics.json")

def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(root_mean_squared_error(y_true, y_pred)),
        "MAPE": float(mean_absolute_percentage_error(y_true, y_pred))
    }

def walk_forward_split(df, n_splits=3, test_size_days=30):
    """
    Generator that yields (train_idx, test_idx) for walk-forward validation.
    Assumes df is sorted by datetime.
    test_size_days: Number of days in each test split.
    n_splits: Number of walk-forward folds.
    """
    df = df.copy().reset_index(drop=True)
    test_size = test_size_days * 24 # assuming hourly data
    
    total_len = len(df)
    if total_len < (n_splits + 1) * test_size:
        raise ValueError("Dataset is too small for the requested number of splits and test size.")
        
    for i in range(n_splits):
        # We start from the end to ensure the last test set is the most recent data
        test_end = total_len - i * test_size
        test_start = test_end - test_size
        train_end = test_start
        
        train_idx = df.index[:train_end].to_list()
        test_idx = df.index[test_start:test_end].to_list()
        
        yield train_idx, test_idx

def save_metrics(model_name, metrics):
    """Save metrics to a JSON file."""
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {}
        
    all_metrics[model_name] = metrics
    
    with open(METRICS_PATH, 'w') as f:
        json.dump(all_metrics, f, indent=4)
        
    print(f"Metrics for {model_name} saved successfully.")

def print_metrics(model_name, metrics):
    print(f"\n--- {model_name} Evaluation ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
