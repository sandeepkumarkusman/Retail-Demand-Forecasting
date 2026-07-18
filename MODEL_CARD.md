# Model Card: Retail Demand Forecaster

## Model Details
- **Model Type:** Global LightGBM Regressor with Quantile Loss
- **Version:** 1.0.0
- **Training Date:** 2024
- **Dataset:** Kaggle Demand Forecasting Competition (Kernels Only)
- **Developed By:** Data Science Internship Project

## Intended Use
- **Primary Use:** Daily demand forecasting for retail inventory planning
- **Target Users:** Supply chain analysts, inventory managers, retail operations teams
- **Forecast Horizon:** 90 days
- **Scope:** 10 stores × 50 items = 500 individual time series

## Model Performance
### Cross-Validation Results (3-fold Walk-Forward)
| Metric | Value | Description |
|--------|-------|-------------|
| **SMAPE** | 11.6% | Symmetric Mean Absolute Percentage Error |
| **MAE** | 4.2 units | Mean Absolute Error |
| **RMSE** | 7.3 units | Root Mean Squared Error |
| **WAPE** | 12.5% | Weighted Absolute Percentage Error |
| **Interval Coverage** | 89.5% | Percentage of actuals within Q5-Q95 interval |
| **Pinball Loss Q05** | 0.023 | Quantile loss for lower bound |
| **Pinball Loss Q95** | 0.021 | Quantile loss for upper bound |

## Training Data
### Dataset Characteristics
- **Time Period:** January 2013 - December 2017 (5 years)
- **Granularity:** Daily observations
- **Series:** 500 time series (10 stores × 50 items)
- **Total Observations:** ~913,000 data points
- **Data Quality:** No missing values, minimal outliers

### Data Splitting Strategy
- **Validation Method:** 3-fold Walk-Forward Cross-Validation
- **Fold Size:** 3 months per validation window
- **Minimum Training:** 1 year before first validation fold
- **Test Strategy:** Temporal holdout matching competition format

## Model Architecture
### Algorithm
- **Base Model:** LightGBM (Gradient Boosting Machine)
- **Objective Functions:**
  - Point forecast: MAE (L1 loss)
  - Quantile Q05: Quantile loss (α=0.05)
  - Quantile Q95: Quantile loss (α=0.95)

### Hyperparameters
```python
{
    'boosting_type': 'gbdt',
    'num_leaves': 127,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'max_depth': -1,
    'n_estimators': 100,
    'seed': 42
}
```

## Feature Engineering
### Lag Features (Leakage-Aware)
- **Lags:** 91, 98, 105, 112, 182, 270, 364, 365, 728 days
- **Rationale:** Minimum lag of 91 days prevents data leakage for 90-day forecast horizon

### Rolling Statistics
- **Windows:** 7, 14, 28, 56, 91 days
- **Statistics:** Mean, standard deviation
- **Applied to:** Lag-91 baseline (leakage-free)

### Cyclical Date Encodings
- **Features:** Sine/cosine transformations for month, day-of-week, day-of-year
- **Purpose:** Capture continuous seasonal transitions without ordinal assumptions

### Cross-Series Aggregates
- **Store-level:** Daily total sales per store (lagged 91 days)
- **Item-level:** Daily total sales per item (lagged 91 days)
- **Purpose:** Capture shared temporal shocks across series

### Target Encodings
- **Encodings:** Store mean, item mean, store-item mean, month mean, day-of-week mean
- **Implementation:** Computed on training data only, mapped to test data

### Calendar Features
- **Features:** Year, month, day, day-of-week, week-of-year, quarter, day-of-year
- **Flags:** Is weekend, is month start, is month end
- **Holidays:** US Federal Holidays with days-to-nearest-holiday

### Total Features: 47 engineered features

## Evaluation Methodology
### Validation Strategy
1. **Walk-Forward CV:** Respects temporal structure
2. **3 Folds:** Each fold validates on 3-month window
3. **Rolling Window:** Validation windows roll backward in time
4. **No Leakage:** All features computed using only historical data

### Metrics Explained
- **SMAPE:** Primary competition metric, symmetric percentage error
- **MAE:** Average absolute error in units
- **RMSE:** Penalizes larger errors more heavily
- **WAPE:** Weighted by total demand, scale-independent
- **Interval Coverage:** Measures prediction interval calibration

## Limitations
### Data Assumptions
- Assumes similar demand patterns to training period (2013-2017)
- Does not account for external shocks (pandemics, supply disruptions)
- Assumes stable store-item relationships

### Forecast Horizon
- Accuracy degrades beyond 90-day horizon
- Prediction intervals widen with forecast distance
- Not designed for real-time intra-day forecasting

### External Factors
- Does not incorporate:
  - Weather data
  - Economic indicators
  - Competitor pricing
  - Marketing campaigns
  - Supply chain constraints

### Generalization
- Trained on specific retail chain data
- May not transfer to different retail formats
- Store/item encodings are specific to this dataset

## Ethical Considerations
### Bias & Fairness
- Model learns from historical patterns which may contain biases
- No demographic data used, reducing fairness concerns
- Should be monitored for systematic over/under forecasting

### Environmental Impact
- Model training: ~45 seconds on standard hardware
- Inference: Sub-second per forecast
- Minimal computational footprint for ongoing use

### Business Impact
- **Positive:** Reduced stock-outs, optimized inventory, lower holding costs
- **Risks:** Over-reliance on automated forecasts without human oversight
- **Mitigation:** Regular monitoring, human review of significant deviations

## Maintenance & Monitoring
### Recommended Retraining
- **Frequency:** Monthly or quarterly
- **Trigger:** Performance degradation (>15% SMAPE increase)
- **Data Drift:** Monitor feature distribution changes

### Monitoring Metrics
- SMAPE on recent actuals
- Prediction interval coverage
- Feature drift detection
- Per-series error distribution

## Citation & References
- **Dataset:** Kaggle Demand Forecasting Competition (Kernels Only)
- **Algorithm:** LightGBM by Microsoft Research
- **Methodology:** Global forecasting approach inspired by M5 competition

## Model Artifacts
- **Point Model:** `models/point_model.joblib`
- **Quantile Low:** `models/quantile_low_q05.joblib`
- **Quantile High:** `models/quantile_high_q95.joblib`
- **Metadata:** `models/metadata.json`
- **Metrics:** `models/metrics.json`
- **Feature Importance:** `outputs/feature_importance.csv`

## Contact & Support
- **Project Repository:** [GitHub URL]
- **Documentation:** See README.md and notebooks/
- **Issues:** Report via GitHub issue tracker

---

*Model Card Version: 1.0.0*  
*Last Updated: 2024*  
*Status: Production Ready*
