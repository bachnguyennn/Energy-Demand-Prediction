# Electricity Demand Forecasting Project - Complete Specification

**Project Name:** End-to-End Electricity Load Forecasting  
**Goal:** Build a production-grade time-series forecasting system to predict hourly or daily electricity demand using weather, calendar, and historical data.  
**Target Quality:** Match or exceed the quality of your Benford and Hospital Readmission projects.

---

## 1. Project Overview

Build a complete end-to-end machine learning project that forecasts electricity demand (load) for a region or country. This project will demonstrate strong time-series skills, feature engineering, multiple modeling approaches, and proper validation techniques.

**Business Value:**  
Accurate electricity demand forecasting helps power companies optimize generation, reduce costs, and improve grid stability.

---

## 2. Objectives & Success Criteria

**Must Have:**
- Use at least **3 different types** of models (Statistical + ML + Deep Learning)
- Proper time-series validation (Walk-forward / Time-based split)
- Strong feature engineering (lag features, rolling statistics, weather, holidays)
- Clear evaluation using multiple metrics (MAE, RMSE, MAPE)
- Professional documentation and README

**Nice to Have:**
- Use of modern models (e.g. Temporal Fusion Transformer)
- SHAP or feature importance analysis
- A simple Streamlit dashboard (optional)

---

## 3. Recommended Datasets

**Primary Dataset (Recommended):**
- **Ontario Electricity Demand** or **Canada Energy Data** (publicly available)
- Alternative strong options:
  - PJM (USA) hourly load data (very popular on Kaggle)
  - European Load Data (ENTSO-E)
  - UCI Electricity Load Diagrams

**External Data to Combine:**
- Weather data (temperature, humidity, wind speed)
- Calendar features (holidays, weekdays, seasons)
- Historical load (lagged values)

---

## 4. Project Structure (Recommended)
electricity_demand_forecasting/
├── data/
│   ├── raw/
│   ├── processed/
│   └── download_data.py
├── features/
│   ├── preprocess.py
│   └── feature_engineering.py
├── models/
│   ├── prophet_model.py
│   ├── ml_model.py          # XGBoost / LightGBM
│   ├── lstm_model.py
│   ├── tft_model.py         # Optional
│   └── evaluate.py
├── notebook/
│   └── electricity_forecasting_report.ipynb
├── reports/
│   └── figures/
├── requirements.txt
├── README.md
└── .gitignore


---

## 5. Step-by-Step Implementation Roadmap

### Phase 1: Data Collection & Preprocessing
- Download electricity load data + weather data
- Handle missing values and outliers
- Resample to hourly or daily frequency

### Phase 2: Feature Engineering (Very Important)
Create rich features such as:
- Lag features (previous 1h, 24h, 7 days)
- Rolling statistics (mean, std over 7/30 days)
- Calendar features (hour, day of week, month, holiday flag)
- Weather features (temperature, feels-like temperature)
- Cyclical encoding (sin/cos for hour and month)

### Phase 3: Modeling (Use Different Models)

**Recommended Model Combination:**

| Model | Type | Purpose |
|-------|------|--------|
| **Prophet** | Statistical | Baseline with seasonality & holidays |
| **XGBoost / LightGBM** | Tree-based ML | Strong baseline with engineered features |
| **LSTM** | Deep Learning | Capture sequential patterns |
| **Temporal Fusion Transformer (TFT)** | Advanced DL | State-of-the-art (optional but impressive) |

### Phase 4: Validation Strategy
- Use **Walk-forward Validation** (time-series cross-validation)
- Never use future data to predict the past
- Create multiple train/test splits over time

### Phase 5: Evaluation
Use these metrics:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- MAPE (Mean Absolute Percentage Error)
- Optional: Pinball Loss (for probabilistic forecasting)

### Phase 6: Interpretability & Analysis
- Analyze feature importance (SHAP or built-in)
- Error analysis (when does the model fail?)
- Compare model performance across seasons

---

## 6. Tech Stack

- **Python 3.10+**
- `pandas`, `numpy`, `scikit-learn`
- `prophet`
- `xgboost`, `lightgbm`
- `tensorflow` or `pytorch` (for LSTM and TFT)
- `optuna` (for hyperparameter tuning)
- `matplotlib`, `seaborn`, `plotly`
- `streamlit` (optional for dashboard)

---

## 7. Deliverables

1. Clean GitHub repository with professional structure
2. High-quality `README.md` with:
   - Project overview
   - Methodology
   - Results table (compare all models)
   - Visualizations
   - Limitations
3. Complete Jupyter notebook with full narrative
4. Saved models and plots in `reports/figures/`
5. Clear instructions on how to run the full pipeline

---

## 8. Best Practices

- Use proper time-series validation only
- Document every modeling decision
- Focus on interpretability and error analysis
- Make the project reproducible
- Write clean, modular code

---

## 9. Final Output

After completing this project, you should have:
- A portfolio project that clearly shows **time-series expertise**
- Experience with **multiple modeling paradigms** (Statistical + ML + Deep Learning)
- A project that stands out because of model diversity and proper time-series handling

---

**End of Specification**

**Instructions for AI Agent:**  
Follow this spec step by step. Prioritize model diversity, proper time-series validation, and high-quality documentation. After finishing, suggest 3 possible improvements for the next version.