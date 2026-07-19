"""Baseline models for comparison with LightGBM (Amazon Forecast-style model comparison)."""

import numpy as np
import pandas as pd
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


def fit_arima_baseline(
    train_df: pd.DataFrame,
    config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Fit ARIMA baseline model (Amazon Forecast-style statistical baseline).
    
    Simple ARIMA(1,1,1) model for comparison with LightGBM.
    Trains on aggregated data by store-item to reduce computational cost.
    """
    if not STATSMODELS_AVAILABLE:
        print("statsmodels not available. Skipping ARIMA baseline.")
        return {'model': None, 'error': 'statsmodels not available'}
    
    if config is None:
        config = {}
    
    # Aggregate by date for a single time series (simplified baseline)
    # In production, would train per store-item
    train_agg = train_df.groupby('date')['sales'].sum().reset_index()
    train_agg = train_agg.sort_values('date')
    
    # Use last 365 days for faster training
    train_agg = train_agg.tail(365)
    
    try:
        model = ARIMA(train_agg['sales'], order=(1, 1, 1))
        fitted_model = model.fit()
        
        return {
            'model': fitted_model,
            'type': 'ARIMA',
            'order': (1, 1, 1),
            'train_dates': (train_agg['date'].min(), train_agg['date'].max())
        }
    except Exception as e:
        return {'model': None, 'error': str(e)}


def predict_arima_baseline(
    val_df: pd.DataFrame,
    model_dict: Dict[str, Any]
) -> pd.DataFrame:
    """Generate predictions from ARIMA baseline model."""
    if model_dict.get('model') is None:
        return pd.DataFrame()
    
    model = model_dict['model']
    n_steps = len(val_df)
    
    try:
        forecast = model.forecast(steps=n_steps)
        
    except Exception as e:
        print(f"ARIMA prediction error: {e}")
        forecast = np.zeros(n_steps)
    
    # Create prediction DataFrame
    pred_df = val_df[['id', 'date']].copy()
    pred_df['sales'] = forecast.values if hasattr(forecast, 'values') else forecast
    pred_df['q05'] = pred_df['sales'] * 0.8  # Simple quantile approximation
    pred_df['q95'] = pred_df['sales'] * 1.2
    
    return pred_df


def fit_prophet_baseline(
    train_df: pd.DataFrame,
    config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Fit Prophet baseline model (Amazon Forecast-style baseline).
    
    Prophet model with yearly seasonality for comparison with LightGBM.
    Trains on aggregated data by store-item to reduce computational cost.
    """
    if not PROPHET_AVAILABLE:
        print("Prophet not available. Skipping Prophet baseline.")
        return {'model': None, 'error': 'Prophet not available'}
    
    if config is None:
        config = {}
    
    # Aggregate by date for a single time series (simplified baseline)
    train_agg = train_df.groupby('date')['sales'].sum().reset_index()
    train_agg = train_agg.sort_values('date')
    
    # Use last 365 days for faster training
    train_agg = train_agg.tail(365)
    
    # Prophet requires columns named 'ds' and 'y'
    prophet_df = train_agg.rename(columns={'date': 'ds', 'sales': 'y'})
    
    try:
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='multiplicative'
        )
        fitted_model = model.fit(prophet_df)
        
        return {
            'model': fitted_model,
            'type': 'Prophet',
            'train_dates': (train_agg['date'].min(), train_agg['date'].max())
        }
    except Exception as e:
        return {'model': None, 'error': str(e)}


def predict_prophet_baseline(
    val_df: pd.DataFrame,
    model_dict: Dict[str, Any]
) -> pd.DataFrame:
    """Generate predictions from Prophet baseline model."""
    if model_dict.get('model') is None:
        return pd.DataFrame()
    
    model = model_dict['model']
    
    # Create future dates for prediction
    future_dates = val_df[['date']].copy()
    future_dates = future_dates.rename(columns={'date': 'ds'})
    
    try:
        forecast = model.predict(future_dates)
        
        # Prophet returns 'yhat' as point forecast
        predictions = forecast['yhat'].values
        lower_bound = forecast['yhat_lower'].values
        upper_bound = forecast['yhat_upper'].values
        
    except Exception as e:
        print(f"Prophet prediction error: {e}")
        predictions = np.zeros(len(val_df))
        lower_bound = predictions * 0.8
        upper_bound = predictions * 1.2
    
    # Create prediction DataFrame
    pred_df = val_df[['id', 'date']].copy()
    pred_df['sales'] = predictions
    pred_df['q05'] = lower_bound
    pred_df['q95'] = upper_bound
    
    return pred_df


def fit_naive_baseline(
    train_df: pd.DataFrame,
    config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Fit naive baseline (last value repeated).
    
    Simplest baseline: forecast = last observed value.
    """
    if config is None:
        config = {}
    
    # Check if data is already a single series or needs aggregation
    if 'sales' not in train_df.columns:
        raise ValueError("DataFrame must have 'sales' column")
    
    # Sort by date and get last value
    train_sorted = train_df.sort_values('date')
    last_value = train_sorted['sales'].iloc[-1]
    
    return {
        'model': {'last_value': last_value},
        'type': 'Naive',
        'last_value': last_value,
        'train_dates': (train_sorted['date'].min(), train_sorted['date'].max())
    }


def predict_naive_baseline(
    val_df: pd.DataFrame,
    model_dict: Dict[str, Any]
) -> pd.DataFrame:
    """Generate predictions from naive baseline model."""
    last_value = model_dict['model']['last_value']
    n_steps = len(val_df)
    
    # Create prediction DataFrame
    pred_df = val_df[['id', 'date']].copy()
    pred_df['sales'] = last_value
    pred_df['q05'] = last_value * 0.8
    pred_df['q95'] = last_value * 1.2
    
    return pred_df


def compare_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    lightgbm_model_dict: Dict[str, Any],
    lightgbm_predict_fn: callable
) -> pd.DataFrame:
    """
    Compare LightGBM against baseline models (Amazon Forecast-style model comparison).
    
    Returns a DataFrame with metrics for each model.
    """
    from src.metrics import calculate_ml_metrics
    
    results = []
    
    # LightGBM predictions
    try:
        lightgbm_pred = lightgbm_predict_fn(val_df, val_df[['id']], lightgbm_model_dict)
        merged = val_df[['id', 'sales']].merge(lightgbm_pred, on='id')
        y_true = merged['sales'].values
        y_pred = merged['sales_y'].values if 'sales_y' in merged.columns else merged['sales'].values
        y_q05 = merged['q05'].values if 'q05' in merged.columns else None
        y_q95 = merged['q95'].values if 'q95' in merged.columns else None
        
        lgbm_metrics = calculate_ml_metrics(y_true, y_pred, y_q05, y_q95)
        lgbm_metrics['model'] = 'LightGBM'
        results.append(lgbm_metrics)
    except Exception as e:
        print(f"LightGBM prediction error: {e}")
    
    # Naive baseline
    try:
        naive_model = fit_naive_baseline(train_df)
        naive_pred = predict_naive_baseline(val_df, naive_model)
        merged = val_df[['id', 'sales']].merge(naive_pred, on='id')
        y_true = merged['sales'].values
        y_pred = merged['sales_y'].values if 'sales_y' in merged.columns else merged['sales'].values
        y_q05 = merged['q05'].values if 'q05' in merged.columns else None
        y_q95 = merged['q95'].values if 'q95' in merged.columns else None
        
        naive_metrics = calculate_ml_metrics(y_true, y_pred, y_q05, y_q95)
        naive_metrics['model'] = 'Naive'
        results.append(naive_metrics)
    except Exception as e:
        print(f"Naive baseline error: {e}")
    
    # ARIMA baseline
    try:
        arima_model = fit_arima_baseline(train_df)
        if arima_model.get('model') is not None:
            arima_pred = predict_arima_baseline(val_df, arima_model)
            if len(arima_pred) > 0:
                merged = val_df[['id', 'sales']].merge(arima_pred, on='id')
                y_true = merged['sales'].values
                y_pred = merged['sales_y'].values if 'sales_y' in merged.columns else merged['sales'].values
                y_q05 = merged['q05'].values if 'q05' in merged.columns else None
                y_q95 = merged['q95'].values if 'q95' in merged.columns else None
                
                arima_metrics = calculate_ml_metrics(y_true, y_pred, y_q05, y_q95)
                arima_metrics['model'] = 'ARIMA'
                results.append(arima_metrics)
    except Exception as e:
        print(f"ARIMA baseline error: {e}")
    
    # Prophet baseline
    try:
        prophet_model = fit_prophet_baseline(train_df)
        if prophet_model.get('model') is not None:
            prophet_pred = predict_prophet_baseline(val_df, prophet_model)
            if len(prophet_pred) > 0:
                merged = val_df[['id', 'sales']].merge(prophet_pred, on='id')
                y_true = merged['sales'].values
                y_pred = merged['sales_y'].values if 'sales_y' in merged.columns else merged['sales'].values
                y_q05 = merged['q05'].values if 'q05' in merged.columns else None
                y_q95 = merged['q95'].values if 'q95' in merged.columns else None
                
                prophet_metrics = calculate_ml_metrics(y_true, y_pred, y_q05, y_q95)
                prophet_metrics['model'] = 'Prophet'
                results.append(prophet_metrics)
    except Exception as e:
        print(f"Prophet baseline error: {e}")
    
    comparison_df = pd.DataFrame(results)
    
    if not comparison_df.empty:
        comparison_df = comparison_df.set_index('model')
    
    return comparison_df
