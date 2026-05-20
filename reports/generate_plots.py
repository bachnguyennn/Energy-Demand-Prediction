import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import shap

# Set styling
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.titlesize': 16})

METRICS_PATH = os.path.join(os.path.dirname(__file__), "metrics.json")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
FEATURES_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "AEP_hourly_features.csv")

def generate_model_comparison_plot():
    print("Generating Model Comparison Plot...")
    if not os.path.exists(METRICS_PATH):
        print(f"Error: {METRICS_PATH} not found. Run model training first.")
        return
        
    with open(METRICS_PATH, 'r') as f:
        metrics = json.load(f)
        
    df_metrics = pd.DataFrame(metrics).T
    
    # Plot MAE & RMSE (using secondary y-axis for MAPE)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color_mae = '#4c72b0'
    color_rmse = '#c44e52'
    
    width = 0.35
    x = np.arange(len(df_metrics))
    
    ax1.bar(x - width/2, df_metrics['MAE'], width, label='MAE (MW)', color=color_mae)
    ax1.bar(x + width/2, df_metrics['RMSE'], width, label='RMSE (MW)', color=color_rmse)
    
    ax1.set_ylabel('Error (MW)')
    ax1.set_title('Model Performance Comparison (MAE & RMSE)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_metrics.index)
    ax1.legend(loc='upper left')
    
    # Overlay MAPE on secondary axis
    ax2 = ax1.twinx()
    color_mape = '#55a868'
    ax2.plot(x, df_metrics['MAPE'] * 100, color=color_mape, marker='o', linewidth=2, label='MAPE (%)')
    ax2.set_ylabel('MAPE (%)', color=color_mape)
    ax2.tick_params(axis='y', labelcolor=color_mape)
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "model_comparison.png"), dpi=300)
    plt.close()
    print("Saved model_comparison.png")

def generate_predictions_and_shap_plots():
    print("Generating Predictions & SHAP Plots...")
    if not os.path.exists(FEATURES_DATA_PATH):
        print(f"Error: {FEATURES_DATA_PATH} not found. Run preprocessing and feature engineering first.")
        return
        
    df = pd.read_csv(FEATURES_DATA_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    target = 'load_mw'
    drop_cols = ['datetime', target]
    features = [c for c in df.columns if c not in drop_cols]
    
    # Split into train and test (last 7 days of the dataset for visualization)
    test_days = 7
    test_hours = test_days * 24
    
    train_df = df.iloc[:-test_hours]
    test_df = df.iloc[-test_hours:]
    
    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]
    
    # Train XGBoost
    model = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    # 1. Predictions vs Actuals Plot
    plt.figure(figsize=(14, 6))
    plt.plot(test_df['datetime'], y_test, label='Actual Load', color='black', alpha=0.8, linewidth=1.5)
    plt.plot(test_df['datetime'], y_pred, label='XGBoost Predicted', color='#c44e52', linestyle='--', linewidth=1.5)
    
    # Simulate a simpler representation of Prophet and LSTM for comparative visualization
    # (smoothing the curve to represent the seasonal-only Prophet predictions)
    y_smooth = test_df['rolling_mean_24h'].values
    plt.plot(test_df['datetime'], y_smooth, label='Prophet (Seasonal Baseline Mock)', color='#4c72b0', alpha=0.6, linestyle=':')
    
    plt.title('Hourly Electricity Load: Predictions vs Actuals (7-Day Slice)')
    plt.xlabel('Date & Time')
    plt.ylabel('Electricity Load (MW)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "predictions_vs_actuals.png"), dpi=300)
    plt.close()
    print("Saved predictions_vs_actuals.png")
    
    # 2. SHAP Plot
    # Keep subset of data for SHAP computation speed
    X_shap = X_test.iloc[:100] 
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_shap, show=False)
    plt.title('SHAP Feature Importance (XGBoost)', fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "xgboost_shap.png"), dpi=300)
    plt.close()
    print("Saved xgboost_shap.png")

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    generate_model_comparison_plot()
    generate_predictions_and_shap_plots()

if __name__ == "__main__":
    main()
