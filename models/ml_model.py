import pandas as pd
import numpy as np
import xgboost as xgb
import os
import sys

# Add parent directory to path to import evaluate module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.evaluate import walk_forward_split, calculate_metrics, save_metrics, print_metrics

FEATURES_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "AEP_hourly_features.csv")

def run_xgboost_validation():
    print("Loading feature data for XGBoost...")
    if not os.path.exists(FEATURES_DATA_PATH):
        print("Data not found. Run preprocess.py and feature_engineering.py first.")
        return
        
    df = pd.read_csv(FEATURES_DATA_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Define features and target
    target = 'load_mw'
    drop_cols = ['datetime', target]
    features = [c for c in df.columns if c not in drop_cols]
    
    n_splits = 3
    fold_metrics = []
    
    print(f"Using {len(features)} features for training.")
    print("Starting Walk-forward validation for XGBoost...")
    
    for fold, (train_idx, test_idx) in enumerate(walk_forward_split(df, n_splits=n_splits, test_size_days=30)):
        print(f"--- Fold {fold + 1}/{n_splits} ---")
        
        X_train, y_train = df.loc[train_idx, features], df.loc[train_idx, target]
        X_test, y_test = df.loc[test_idx, features], df.loc[test_idx, target]
        
        model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        y_pred = model.predict(X_test)
        
        metrics = calculate_metrics(y_test, y_pred)
        print_metrics(f"XGBoost Fold {fold+1}", metrics)
        fold_metrics.append(metrics)
        
    # Aggregate metrics
    avg_metrics = {
        k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0].keys()
    }
    
    print_metrics("XGBoost Average", avg_metrics)
    save_metrics("XGBoost", avg_metrics)
    
    # Save a dummy model or feature importances later if needed for notebook
    
if __name__ == "__main__":
    run_xgboost_validation()
