"""Evaluation metrics for the forecasting solution."""

from typing import Union, Optional
import numpy as np
import pandas as pd


def calculate_ml_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    y_q05: Optional[np.ndarray] = None,
    y_q95: Optional[np.ndarray] = None
) -> dict[str, float]:
    """Calculate a comprehensive suite of modern forecasting metrics."""
    # Handle pandas series
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    if isinstance(y_q05, pd.Series):
        y_q05 = y_q05.values
    if isinstance(y_q95, pd.Series):
        y_q95 = y_q95.values
        
    metrics = {}
    
    # MAE
    metrics['mae'] = float(np.mean(np.abs(y_true - y_pred)))
    
    # RMSE
    metrics['rmse'] = float(np.sqrt(np.mean(np.square(y_true - y_pred))))
    
    # SMAPE
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    # Avoid division by zero
    diff = np.abs(y_true - y_pred) / np.maximum(denominator, 1e-8)
    metrics['smape'] = float(np.mean(diff) * 100)
    
    # WAPE (Weighted Absolute Percentage Error)
    sum_y = np.sum(np.abs(y_true))
    if sum_y > 0:
        metrics['wape'] = float(np.sum(np.abs(y_true - y_pred)) / sum_y * 100)
    else:
        metrics['wape'] = 0.0
        
    # MASE (Simplified: MAE divided by naive in-sample MAE of 1-step forecast)
    # This requires history usually, but we approximate by taking naive shift of y_true
    if len(y_true) > 1:
        naive_mae = np.mean(np.abs(y_true[1:] - y_true[:-1]))
        if naive_mae > 0:
            metrics['mase'] = float(metrics['mae'] / naive_mae)
        else:
            metrics['mase'] = 0.0
    else:
        metrics['mase'] = 0.0
        
    # Quantile metrics
    if y_q05 is not None and y_q95 is not None:
        # Pinball loss for Q05
        diff_05 = y_true - y_q05
        metrics['pinball_q05'] = float(np.mean(np.where(diff_05 >= 0, 0.05 * diff_05, -0.95 * diff_05)))
        
        # Pinball loss for Q95
        diff_95 = y_true - y_q95
        metrics['pinball_q95'] = float(np.mean(np.where(diff_95 >= 0, 0.95 * diff_95, -0.05 * diff_95)))
        
        # Interval coverage
        covered = (y_true >= y_q05) & (y_true <= y_q95)
        metrics['interval_coverage_pct'] = float(np.mean(covered) * 100)
        
    return metrics
