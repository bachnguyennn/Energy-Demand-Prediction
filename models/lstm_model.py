import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os
import sys

# Add parent directory to path to import evaluate module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.evaluate import walk_forward_split, calculate_metrics, save_metrics, print_metrics

FEATURES_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "AEP_hourly_features.csv")

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, seq_len=24):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.X) - self.seq_len
        
    def __getitem__(self, idx):
        return (self.X[idx : idx + self.seq_len], self.y[idx + self.seq_len])

class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1):
        super(LSTMForecaster, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) # Take the last time step
        return out.squeeze()

def train_lstm(model, dataloader, epochs=5, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
    return model

def predict_lstm(model, dataloader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for X_batch, _ in dataloader:
            y_pred = model(X_batch)
            predictions.extend(y_pred.numpy())
    return np.array(predictions)

def run_lstm_validation():
    print("Loading feature data for LSTM...")
    if not os.path.exists(FEATURES_DATA_PATH):
        print("Data not found. Run preprocess.py and feature_engineering.py first.")
        return
        
    df = pd.read_csv(FEATURES_DATA_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    target = 'load_mw'
    drop_cols = ['datetime', target]
    features = [c for c in df.columns if c not in drop_cols]
    
    seq_len = 24
    n_splits = 3
    fold_metrics = []
    
    print("Starting Walk-forward validation for LSTM...")
    for fold, (train_idx, test_idx) in enumerate(walk_forward_split(df, n_splits=n_splits, test_size_days=30)):
        print(f"--- Fold {fold + 1}/{n_splits} ---")
        
        # In order to predict the test set properly using sequence of length seq_len,
        # we need to include the last seq_len items from train set into the test set inputs.
        extended_test_idx = list(range(test_idx[0] - seq_len, test_idx[-1] + 1))
        
        X_train_raw = df.loc[train_idx, features].values
        y_train_raw = df.loc[train_idx, target].values
        
        X_test_raw = df.loc[extended_test_idx, features].values
        y_test_raw = df.loc[extended_test_idx, target].values
        
        # Scale data
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train = scaler_X.fit_transform(X_train_raw)
        y_train = scaler_y.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
        
        X_test = scaler_X.transform(X_test_raw)
        y_test = scaler_y.transform(y_test_raw.reshape(-1, 1)).flatten()
        
        train_dataset = TimeSeriesDataset(X_train, y_train, seq_len=seq_len)
        test_dataset = TimeSeriesDataset(X_test, y_test, seq_len=seq_len)
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        model = LSTMForecaster(input_dim=len(features), hidden_dim=64, num_layers=2)
        model = train_lstm(model, train_loader, epochs=5)
        
        y_pred_scaled = predict_lstm(model, test_loader)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_true = y_test_raw[seq_len:] # Drop the overlapping part
        
        metrics = calculate_metrics(y_true, y_pred)
        print_metrics(f"LSTM Fold {fold+1}", metrics)
        fold_metrics.append(metrics)
        
    avg_metrics = {
        k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0].keys()
    }
    
    print_metrics("LSTM Average", avg_metrics)
    save_metrics("LSTM", avg_metrics)

if __name__ == "__main__":
    run_lstm_validation()
