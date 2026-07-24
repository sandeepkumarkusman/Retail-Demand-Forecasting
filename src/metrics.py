"""Evaluation metrics for the forecasting solution."""

from typing import Callable, Optional, Union

import numpy as np
import pandas as pd


def calculate_ml_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_q05: Optional[np.ndarray] = None,
    y_q95: Optional[np.ndarray] = None,
    custom_metric_fn: Optional[Callable] = None,
) -> dict[str, float]:
    """
    Calculate a comprehensive suite of modern forecasting metrics.

    Includes Amazon Forecast-style metrics: MAE, RMSE, SMAPE, WAPE, MASE, RMSLE, MAPE,
    plus quantile metrics and optional custom business metrics.
    """
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

    # MAE (Mean Absolute Error)
    metrics["mae"] = float(np.mean(np.abs(y_true - y_pred)))

    # RMSE (Root Mean Squared Error)
    metrics["rmse"] = float(np.sqrt(np.mean(np.square(y_true - y_pred))))

    # SMAPE (Symmetric Mean Absolute Percentage Error)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / np.maximum(denominator, 1e-8)
    metrics["smape"] = float(np.mean(diff) * 100)

    # WAPE (Weighted Absolute Percentage Error)
    sum_y = np.sum(np.abs(y_true))
    if sum_y > 0:
        metrics["wape"] = float(np.sum(np.abs(y_true - y_pred)) / sum_y * 100)
    else:
        metrics["wape"] = 0.0

    # MASE (Mean Absolute Scaled Error)
    if len(y_true) > 1:
        naive_mae = np.mean(np.abs(y_true[1:] - y_true[:-1]))
        if naive_mae > 0:
            metrics["mase"] = float(metrics["mae"] / naive_mae)
        else:
            metrics["mase"] = 0.0
    else:
        metrics["mase"] = 0.0

    # RMSLE (Root Mean Squared Logarithmic Error) - Amazon Forecast metric
    # Add small constant to avoid log(0)
    y_true_log = np.log1p(np.maximum(y_true, 0))
    y_pred_log = np.log1p(np.maximum(y_pred, 0))
    metrics["rmsle"] = float(np.sqrt(np.mean(np.square(y_true_log - y_pred_log))))

    # MAPE (Mean Absolute Percentage Error) - Amazon Forecast metric
    mape_denominator = np.maximum(np.abs(y_true), 1e-8)
    metrics["mape"] = float(np.mean(np.abs(y_true - y_pred) / mape_denominator) * 100)

    # R² (Coefficient of Determination)
    ss_res = np.sum(np.square(y_true - y_pred))
    ss_tot = np.sum(np.square(y_true - np.mean(y_true)))
    if ss_tot > 0:
        metrics["r2"] = float(1 - (ss_res / ss_tot))
    else:
        metrics["r2"] = 0.0

    # Quantile metrics
    if y_q05 is not None and y_q95 is not None:
        # Pinball loss for Q05
        diff_05 = y_true - y_q05
        metrics["pinball_q05"] = float(np.mean(np.where(diff_05 >= 0, 0.05 * diff_05, -0.95 * diff_05)))

        # Pinball loss for Q95
        diff_95 = y_true - y_q95
        metrics["pinball_q95"] = float(np.mean(np.where(diff_95 >= 0, 0.95 * diff_95, -0.05 * diff_95)))

        # Interval coverage
        covered = (y_true >= y_q05) & (y_true <= y_q95)
        metrics["interval_coverage_pct"] = float(np.mean(covered) * 100)

    # Custom metric (Amazon Forecast-style business-specific metrics)
    if custom_metric_fn is not None:
        try:
            custom_result = custom_metric_fn(y_true, y_pred)
            if isinstance(custom_result, dict):
                metrics.update(custom_result)
            else:
                metrics["custom_metric"] = float(custom_result)
        except Exception as e:
            metrics["custom_metric_error"] = str(e)

    return metrics


def calculate_business_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_type: str = "stockout_cost",
    holding_cost_per_unit: float = 1.0,
    stockout_cost_per_unit: float = 10.0,
    price_per_unit: float = 50.0,
) -> float:
    """
    Calculate business-specific metrics (Amazon Forecast-style).

    Parameters
    ----------
    metric_type : str
        Type of business metric:
        - 'stockout_cost': Cost of stockouts (under-forecasting)
        - 'holding_cost': Cost of excess inventory (over-forecasting)
        - 'total_cost': Combined stockout + holding cost
        - 'revenue_loss': Lost revenue from stockouts
        - 'service_level': Percentage of demand met
    """
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values

    # Calculate under-forecast (stockouts) and over-forecast (excess)
    under_forecast = np.maximum(y_true - y_pred, 0)
    over_forecast = np.maximum(y_pred - y_true, 0)

    if metric_type == "stockout_cost":
        return float(np.sum(under_forecast * stockout_cost_per_unit))
    elif metric_type == "holding_cost":
        return float(np.sum(over_forecast * holding_cost_per_unit))
    elif metric_type == "total_cost":
        stockout_cost = np.sum(under_forecast * stockout_cost_per_unit)
        holding_cost = np.sum(over_forecast * holding_cost_per_unit)
        return float(stockout_cost + holding_cost)
    elif metric_type == "revenue_loss":
        return float(np.sum(under_forecast * price_per_unit))
    elif metric_type == "service_level":
        total_demand = np.sum(y_true)
        if total_demand > 0:
            met_demand = np.sum(np.minimum(y_pred, y_true))
            return float((met_demand / total_demand) * 100)
        else:
            return 0.0
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")
