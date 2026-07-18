# End-to-End Retail Demand Forecasting

A comprehensive machine learning project to forecast daily store-item sales across a multi-series dataset. 

This repository documents the entire journey from initial data exploration and baseline modeling to a production-grade machine learning pipeline. It demonstrates how to handle a complex multi-series time-series forecasting problem by capturing both series-specific dynamics and shared temporal structures.

---

## 📖 The Business Problem

A retail chain needs to forecast daily demand for 50 items across 10 different stores over a 90-day horizon.

**Why does this matter?** Accurate demand forecasting sits at the heart of supply chain optimization. Overestimating demand ties up capital in stagnant inventory and increases storage costs (or spoilage for perishables). Underestimating demand leads to stockouts, lost revenue, and dissatisfied customers. 

**The Challenge:** We aren't just forecasting a single aggregate time series. We have 500 distinct but related time series (10 stores × 50 items) that exhibit both unique local behaviors and shared macroeconomic trends. Our goal is to predict 45,000 distinct daily values for Q1 2018 based on 5 years of historical data.

---

## 🔍 Data Understanding & EDA

Before writing a single line of modeling code, extensive Exploratory Data Analysis (EDA) was performed (documented in our notebooks) to understand the underlying data generation process:

- **Macro Trends:** We observed a consistent 5–10% year-over-year growth across all stores and items. The growth curves are roughly parallel, suggesting that items grow with the overall brand rather than cannibalizing each other.
- **Seasonality Patterns:** 
  - *Weekly:* A strong 7-day cycle exists, with noticeable dips on weekdays and peaks on weekends.
  - *Monthly/Yearly:* Sales consistently peak in mid-summer (July) and dip significantly in January-February.
- **Store-Item Interactions:** We discovered that store-item effects are largely additive. Certain items consistently outsell others uniformly across all stores, which informed our decision to use mean-based target encodings later on.
- **Autocorrelation:** Strong autocorrelation spikes at lag-7 (weekly) and lag-364 (yearly).
- **Data Quality:** The data was exceptionally clean—no missing dates, minimal zero-sale days, and only a light right tail of outliers (which we handled via robust clipping at the 99.9th percentile).

---

## 🧪 Experimentation & Modeling

This project didn't start with complex models. Instead, we followed a principled, iterative approach:

### 1. Baselines
We began with simple interpretable baselines to establish a performance floor:
- Naive historical mean.
- Seasonal naive (predicting the same day last year).
- Multiplicative decomposition (annual growth × monthly × day-of-week factors).

The multiplicative decomposition proved surprisingly strong (achieving ~15-17% SMAPE), confirming that any advanced model *must* successfully capture both the global trend and local seasonality.

### 2. Feature Engineering
Driven by our EDA, we engineered a robust feature set specifically designed to avoid data leakage given the 90-day forecast horizon:
- **Lags & Rolling Windows:** We anchored our lag features starting at `lag-91` (up to `lag-728`). We built 7, 14, 28, 56, and 91-day rolling means/stds on top of `lag-91` to capture local momentum without leaking the future.
- **Cross-Series Features:** We aggregated daily total store sales and total item sales, shifted them by 91 days, and fed them back into the model to capture shared temporal shocks.
- **Cyclical Date Encodings:** Sine/Cosine transformations of day-of-week and day-of-year to model continuous seasonal transitions.
- **Holidays & Events:** Integrated US Federal Holidays and calculated "days to nearest holiday" to capture local demand spikes.
- **Exponential Moving Averages (EMA):** Smoothed trend indicators ($\alpha=0.90, 0.95$).

### 3. Model Selection
We evaluated multiple approaches via a strict 3-fold Walk-Forward Validation strategy:
- **Ridge Regression:** Failed to capture complex non-linear interactions.
- **Random Forest:** Strong performance but computationally expensive to tune and train.
- **Deep Learning (LSTM/DeepAR):** Explored but ultimately discarded. Training was 10-50x slower with no material accuracy gain over gradient boosting for this specific structured tabular data.
- **LightGBM:** Selected as the final champion. It provided the best balance of speed, accuracy, and interpretability (handling 500 series simultaneously with ease).

---

## 🚀 The Final Solution

Our production pipeline trains a **Global LightGBM Regressor**. Instead of training 500 separate models, one global model learns from all 500 series simultaneously, allowing it to leverage cross-series patterns and avoid overfitting on sparse items.

**Key Features of the Pipeline:**
- **Point Forecasting:** An MAE-optimized model predicts the exact expected sales (Average CV SMAPE: **~11.6%**).
- **Quantile Forecasting:** Two additional models are trained with Quantile loss ($\alpha=0.05$ and $\alpha=0.95$) to generate predictive intervals. This directly solves the business problem of conservative vs. optimistic inventory risk planning.
- **Robust Preprocessing:** Automated schema validation, data type casting, and outlier clipping.
- **Guardrails:** Automated post-processing checks ensure point forecasts never violate the $Q05 \le \text{Point} \le Q95$ physical reality constraint.

---

## 🛠 Repository Structure

```
Retail-Demand-Forecasting/
├── notebooks/
│   ├── 01_EDA_Sales_Patterns.ipynb          ← Exploratory data analysis
│   ├── 02_Baseline_Models.ipynb             ← Naive + multiplicative baselines
│   ├── 03_Feature_Engineering_Analysis.ipynb ← Lag/rolling/encoding analysis
│   ├── 04_Model_Experiments.ipynb           ← Ridge, RF, LightGBM comparison
│   └── 05_LightGBM_Final_Pipeline.ipynb     ← Final production model
├── src/
│   ├── data_loader.py     ← Data ingestion
│   ├── preprocessing.py   ← Schema checks, outlier clipping, gap-filling
│   ├── features.py        ← Advanced feature engineering (lags, holidays, cyclical)
│   ├── train.py           ← LightGBM model fitting & CV
│   ├── predict.py         ← Inference generation
│   ├── guard.py           ← Output validation constraints
│   ├── metrics.py         ← Evaluation (SMAPE, MAE, Pinball Loss, Coverage)
│   ├── backtesting.py     ← 3-Fold Walk-forward CV framework
│   ├── explain.py         ← Diagnostic plots and feature importance
│   ├── pipeline.py        ← End-to-end orchestration
│   └── utils.py           ← Shared helpers
├── config/config.yaml     ← Pipeline configuration parameters
├── data/raw/              ← train.csv, test.csv, sample_submission.csv
├── data/processed/        ← Cached feature matrices
├── models/                ← Saved .joblib models + metadata.json
├── outputs/               ← predictions.parquet, feature_importance.csv, submission.csv
├── tests/                 ← Pytest unit test suite
├── demo/app.py            ← Streamlit forecast explorer dashboard
├── Makefile               ← Task runner
└── requirements.txt       ← Dependencies
```

---

## 💻 How to Run

### 1. Setup the Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Supply Data
Ensure the raw Kaggle dataset files are placed in `data/raw/`:
- `data/raw/train.csv`
- `data/raw/test.csv`
- `data/raw/sample_submission.csv`

### 3. Execute the Pipeline
You can run the end-to-end pipeline (preprocessing → walk-forward CV → final training → inference) using:
```bash
make train
```
Or run it directly via Python:
```bash
python -m src.pipeline
```

### 4. Run Tests & Validation
Execute the unit test suite to verify module integrity and guardrails:
```bash
make test
```

### 5. Explore the Demo
Launch the interactive Streamlit dashboard to visually explore the forecasts and prediction intervals:
```bash
make serve
```
