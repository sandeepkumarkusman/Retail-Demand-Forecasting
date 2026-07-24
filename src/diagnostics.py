"""ML diagnostics and analysis utilities for forecasting model."""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def analyze_feature_stability(
    feature_importance_dfs: List[pd.DataFrame],
) -> pd.DataFrame:
    """
    Analyze how feature importance changes across CV folds.

    Parameters
    ----------
    feature_importance_dfs : List[pd.DataFrame]
        List of feature importance DataFrames from each CV fold

    Returns
    -------
    pd.DataFrame
        DataFrame with stability metrics for each feature
    """
    # Combine all fold data
    combined = pd.concat(feature_importance_dfs, keys=range(len(feature_importance_dfs)))

    stability_scores = {}
    for feature in combined["feature"].unique():
        feature_data = combined[combined["feature"] == feature]["importance"]
        stability_scores[feature] = {
            "mean_importance": feature_data.mean(),
            "std_importance": feature_data.std(),
            "min_importance": feature_data.min(),
            "max_importance": feature_data.max(),
            "cv": (feature_data.std() / feature_data.mean() if feature_data.mean() > 0 else 0),
            "num_folds": len(feature_data),
        }

    return pd.DataFrame(stability_scores).T.sort_values("mean_importance", ascending=False)


def analyze_per_series_performance(predictions: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze performance by store and item combinations.

    Parameters
    ----------
    predictions : pd.DataFrame
        DataFrame with predictions including store, item, and predicted values
    actual : pd.DataFrame
        DataFrame with actual values including store, item, and actual sales

    Returns
    -------
    pd.DataFrame
        Performance metrics per store-item combination
    """
    from src.metrics import calculate_ml_metrics

    results = []

    # Merge predictions with actuals
    merged = predictions.merge(actual, on=["store", "item", "date"], how="inner", suffixes=("_pred", "_actual"))

    for store in merged["store"].unique():
        for item in merged["item"].unique():
            subset = merged[(merged["store"] == store) & (merged["item"] == item)]
            if len(subset) > 0:
                y_true = subset["sales_actual"].values
                y_pred = subset["sales_pred"].values

                # Get quantile predictions if available
                y_q05 = subset["q05"].values if "q05" in subset.columns else None
                y_q95 = subset["q95"].values if "q95" in subset.columns else None

                metrics = calculate_ml_metrics(y_true, y_pred, y_q05, y_q95)

                results.append(
                    {
                        "store": store,
                        "item": item,
                        "n_predictions": len(subset),
                        **metrics,
                    }
                )

    return pd.DataFrame(results)


def residual_analysis(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """
    Comprehensive residual diagnostics.

    Parameters
    ----------
    actual : np.ndarray
        Actual values
    predicted : np.ndarray
        Predicted values

    Returns
    -------
    Dict[str, float]
        Dictionary of residual diagnostic metrics
    """
    residuals = actual - predicted

    return {
        "mean_residual": float(residuals.mean()),
        "std_residual": float(residuals.std()),
        "skewness": float(stats.skew(residuals)),
        "kurtosis": float(stats.kurtosis(residuals)),
        "min_residual": float(residuals.min()),
        "max_residual": float(residuals.max()),
        "median_residual": float(np.median(residuals)),
    }


def analyze_temporal_errors(predictions: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze how errors vary by time periods (day of week, month, etc.).

    Parameters
    ----------
    predictions : pd.DataFrame
        DataFrame with predictions and date column
    actual : pd.DataFrame
        DataFrame with actual values and date column

    Returns
    -------
    pd.DataFrame
        Error metrics grouped by time periods
    """
    from src.metrics import calculate_ml_metrics

    # Merge predictions with actuals
    merged = predictions.merge(actual, on=["store", "item", "date"], how="inner", suffixes=("_pred", "_actual"))
    merged["date"] = pd.to_datetime(merged["date"])

    results = []

    # By day of week
    for dow in range(7):
        subset = merged[merged["date"].dt.dayofweek == dow]
        if len(subset) > 0:
            metrics = calculate_ml_metrics(subset["sales_actual"], subset["sales_pred"])
            results.append(
                {
                    "time_period": "day_of_week",
                    "period_value": dow,
                    "n_samples": len(subset),
                    **metrics,
                }
            )

    # By month
    for month in range(1, 13):
        subset = merged[merged["date"].dt.month == month]
        if len(subset) > 0:
            metrics = calculate_ml_metrics(subset["sales_actual"], subset["sales_pred"])
            results.append(
                {
                    "time_period": "month",
                    "period_value": month,
                    "n_samples": len(subset),
                    **metrics,
                }
            )

    return pd.DataFrame(results)


def analyze_forecast_bias(predictions: pd.DataFrame, actual: pd.DataFrame) -> Dict[str, float]:
    """
    Analyze systematic bias in forecasts.

    Parameters
    ----------
    predictions : pd.DataFrame
        DataFrame with predictions
    actual : pd.DataFrame
        DataFrame with actual values

    Returns
    -------
    Dict[str, float]
        Bias analysis metrics
    """
    # Merge predictions with actuals
    merged = predictions.merge(actual, on=["store", "item", "date"], how="inner", suffixes=("_pred", "_actual"))

    residuals = merged["sales_actual"] - merged["sales_pred"]

    # Percentage bias
    pct_errors = (merged["sales_actual"] - merged["sales_pred"]) / merged["sales_actual"]
    pct_errors = pct_errors.replace([np.inf, -np.inf], np.nan).dropna()

    return {
        "mean_bias": float(residuals.mean()),
        "median_bias": float(residuals.median()),
        "mean_pct_bias": float(pct_errors.mean() * 100),
        "median_pct_bias": float(pct_errors.median() * 100),
        "underforecast_pct": float((residuals > 0).mean() * 100),  # Actual > Pred
        "overforecast_pct": float((residuals < 0).mean() * 100),  # Actual < Pred
    }


def analyze_interval_calibration(predictions: pd.DataFrame, actual: pd.DataFrame) -> Dict[str, float]:
    """
    Analyze prediction interval calibration.

    Parameters
    ----------
    predictions : pd.DataFrame
        DataFrame with predictions including q05 and q95 columns
    actual : pd.DataFrame
        DataFrame with actual values

    Returns
    -------
    Dict[str, float]
        Interval calibration metrics
    """
    # Merge predictions with actuals
    merged = predictions.merge(actual, on=["store", "item", "date"], how="inner", suffixes=("_pred", "_actual"))

    if "q05" not in merged.columns or "q95" not in merged.columns:
        return {"error": "Quantile columns not found"}

    # Check coverage
    within_interval = (merged["sales_actual"] >= merged["q05"]) & (merged["sales_actual"] <= merged["q95"])

    # Interval width statistics
    interval_width = merged["q95"] - merged["q05"]

    return {
        "coverage_pct": float(within_interval.mean() * 100),
        "mean_interval_width": float(interval_width.mean()),
        "median_interval_width": float(interval_width.median()),
        "std_interval_width": float(interval_width.std()),
        "below_q05_pct": float((merged["sales_actual"] < merged["q05"]).mean() * 100),
        "above_q95_pct": float((merged["sales_actual"] > merged["q95"]).mean() * 100),
    }


def generate_diagnostic_report(predictions: pd.DataFrame, actual: pd.DataFrame) -> Dict:
    """
    Generate comprehensive diagnostic report.

    Parameters
    ----------
    predictions : pd.DataFrame
        DataFrame with predictions
    actual : pd.DataFrame
        DataFrame with actual values

    Returns
    -------
    Dict
        Comprehensive diagnostic report
    """
    from src.metrics import calculate_ml_metrics

    # Merge predictions with actuals
    merged = predictions.merge(actual, on=["store", "item", "date"], how="inner", suffixes=("_pred", "_actual"))

    y_true = merged["sales_actual"].values
    y_pred = merged["sales_pred"].values
    y_q05 = merged["q05"].values if "q05" in merged.columns else None
    y_q95 = merged["q95"].values if "q95" in merged.columns else None

    report = {
        "overall_metrics": calculate_ml_metrics(y_true, y_pred, y_q05, y_q95),
        "residual_analysis": residual_analysis(y_true, y_pred),
        "bias_analysis": analyze_forecast_bias(predictions, actual),
        "n_samples": len(merged),
        "n_series": merged[["store", "item"]].drop_duplicates().shape[0],
    }

    if y_q05 is not None and y_q95 is not None:
        report["interval_calibration"] = analyze_interval_calibration(predictions, actual)

    return report
