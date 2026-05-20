import os
import pandas as pd
import numpy as np
import warnings
import sys

# Add parent directory to path to import evaluate module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.evaluate import walk_forward_split, calculate_metrics, save_metrics, print_metrics

try:
    import torch
    import lightning.pytorch as pl
    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.metrics import QuantileLoss
    HAS_PYTORCH_FORECASTING = True
except ImportError:
    HAS_PYTORCH_FORECASTING = False

FEATURES_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "AEP_hourly_features.csv")

def prepare_tft_data(df):
    """Format data for pytorch_forecasting."""
    tft_df = df.copy()
    # Add a dummy group column because TFT expects at least one group
    tft_df['group'] = 'AEP'
    # Create an integer time index
    tft_df['time_idx'] = np.arange(len(tft_df))
    # Convert categorical to string
    tft_df['is_holiday'] = tft_df['is_holiday'].astype(str)
    tft_df['is_weekend'] = tft_df['is_weekend'].astype(str)
    
    return tft_df

def run_tft_validation():
    if not HAS_PYTORCH_FORECASTING:
        print("pytorch-forecasting is not installed. Please install it to run TFT model.")
        return
        
    print("Loading feature data for TFT...")
    if not os.path.exists(FEATURES_DATA_PATH):
        print("Data not found. Run preprocess.py and feature_engineering.py first.")
        return
        
    df = pd.read_csv(FEATURES_DATA_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])
    tft_df = prepare_tft_data(df)
    
    n_splits = 3
    fold_metrics = []
    
    max_prediction_length = 24
    max_encoder_length = 72
    
    print("Starting Walk-forward validation for TFT...")
    for fold, (train_idx, test_idx) in enumerate(walk_forward_split(tft_df, n_splits=n_splits, test_size_days=30)):
        print(f"--- Fold {fold + 1}/{n_splits} ---")
        
        # Test split needs past data for encoding
        train_df = tft_df.iloc[train_idx]
        test_df = tft_df.iloc[list(range(test_idx[0] - max_encoder_length, test_idx[-1] + 1))]
        
        training = TimeSeriesDataSet(
            train_df,
            time_idx="time_idx",
            target="load_mw",
            group_ids=["group"],
            min_encoder_length=max_encoder_length // 2,
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
            static_categoricals=["group"],
            time_varying_known_categoricals=["is_holiday", "is_weekend"],
            time_varying_known_reals=["time_idx", "hour_sin", "hour_cos", "month_sin", "month_cos"],
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=["load_mw"],
            target_normalizer=GroupNormalizer(groups=["group"], transformation="softplus"),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
        
        validation = TimeSeriesDataSet.from_dataset(training, test_df, predict=True, stop_randomization=True)
        
        train_dataloader = training.to_dataloader(train=True, batch_size=64, num_workers=0)
        val_dataloader = validation.to_dataloader(train=False, batch_size=64, num_workers=0)
        
        pl.seed_everything(42)
        trainer = pl.Trainer(
            max_epochs=3,
            accelerator="auto",
            gradient_clip_val=0.1,
            limit_train_batches=30, # Limit batches for faster demo
            logger=False,
            enable_checkpointing=False
        )
        
        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=0.03,
            hidden_size=16,
            attention_head_size=1,
            dropout=0.1,
            hidden_continuous_size=8,
            output_size=7,  # 7 quantiles by default
            loss=QuantileLoss(),
            log_interval=10,
            reduce_on_plateau_patience=4,
        )
        
        trainer.fit(
            tft,
            train_dataloaders=train_dataloader,
        )
        
        # Predict on validation dataloader
        predictions = tft.predict(val_dataloader, return_y=True)
        
        # predictions.output contains the point predictions (median) if using QuantileLoss
        y_pred = predictions.output.cpu().numpy().flatten()
        y_true = predictions.y[0].cpu().numpy().flatten()
        
        # Ensure they match in length, sometimes there's a slight mismatch depending on batching
        min_len = min(len(y_pred), len(y_true))
        metrics = calculate_metrics(y_true[:min_len], y_pred[:min_len])
        print_metrics(f"TFT Fold {fold+1}", metrics)
        fold_metrics.append(metrics)
        
    avg_metrics = {
        k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0].keys()
    }
    
    print_metrics("TFT Average", avg_metrics)
    save_metrics("TFT", avg_metrics)

if __name__ == "__main__":
    run_tft_validation()
