"""Model accuracy tracking over time (Amazon Forecast-style continuous monitoring)."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


class ModelAccuracyTracker:
    """
    Track model accuracy over time (Amazon Forecast-style).

    Automatically tracks accuracy as new data is imported and quantifies
    deviation from initial quality metrics to make informed decisions
    about keeping, retraining, or rebuilding the model.
    """

    def __init__(self, tracking_file: str = "outputs/model_accuracy_history.json"):
        """
        Initialize the tracker.

        Parameters
        ----------
        tracking_file : str
            Path to JSON file storing accuracy history.
        """
        self.tracking_file = Path(tracking_file)
        self.tracking_file.parent.mkdir(parents=True, exist_ok=True)
        self.history = self._load_history()

    def _load_history(self) -> Dict[str, Any]:
        """Load existing accuracy history from file."""
        if self.tracking_file.exists():
            try:
                with open(self.tracking_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading accuracy history: {e}")
                return {"runs": [], "baseline_metrics": None}
        else:
            return {"runs": [], "baseline_metrics": None}

    def _save_history(self):
        """Save accuracy history to file."""
        with open(self.tracking_file, "w") as f:
            json.dump(self.history, f, indent=2, default=str)

    def record_training_run(self, metrics: Dict[str, float], model_version: str = "1.0.0", notes: str = ""):
        """
        Record a training run with its metrics.

        Parameters
        ----------
        metrics : dict
            Dictionary of metrics (smape, mae, rmse, etc.)
        model_version : str
            Version identifier for the model
        notes : str
            Optional notes about this training run
        """
        run_record = {
            "timestamp": datetime.now().isoformat(),
            "model_version": model_version,
            "metrics": metrics,
            "notes": notes,
        }

        self.history["runs"].append(run_record)

        # Set baseline if this is the first run
        if self.history["baseline_metrics"] is None:
            self.history["baseline_metrics"] = metrics.copy()

        self._save_history()

    def get_accuracy_trend(self, metric: str = "smape") -> pd.DataFrame:
        """
        Get accuracy trend for a specific metric over time.

        Parameters
        ----------
        metric : str
            Metric to track (e.g., 'smape', 'mae', 'rmse')

        Returns
        -------
        DataFrame with timestamp and metric value
        """
        if not self.history["runs"]:
            return pd.DataFrame()

        data = []
        for run in self.history["runs"]:
            if metric in run["metrics"]:
                data.append(
                    {
                        "timestamp": run["timestamp"],
                        "model_version": run["model_version"],
                        metric: run["metrics"][metric],
                        "notes": run.get("notes", ""),
                    }
                )

        df = pd.DataFrame(data)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        return df

    def detect_drift(self, metric: str = "smape", threshold: float = 0.15, window_size: int = 5) -> Dict[str, Any]:
        """
        Detect model drift by comparing recent performance to baseline.

        Parameters
        ----------
        metric : str
            Metric to monitor
        threshold : float
            Percentage threshold for drift detection (e.g., 0.15 = 15%)
        window_size : int
            Number of recent runs to average

        Returns
        -------
        Dictionary with drift detection results
        """
        if self.history["baseline_metrics"] is None:
            return {"drift_detected": False, "reason": "No baseline established"}

        if len(self.history["runs"]) < window_size:
            return {
                "drift_detected": False,
                "reason": f"Need at least {window_size} runs for drift detection",
            }

        baseline_value = self.history["baseline_metrics"].get(metric)
        if baseline_value is None:
            return {
                "drift_detected": False,
                "reason": f"Metric {metric} not in baseline",
            }

        # Calculate average of recent runs
        recent_runs = self.history["runs"][-window_size:]
        recent_values = [run["metrics"].get(metric) for run in recent_runs if metric in run["metrics"]]

        if not recent_values:
            return {
                "drift_detected": False,
                "reason": f"Metric {metric} not in recent runs",
            }

        recent_avg = np.mean(recent_values)

        # Calculate percentage change
        pct_change = abs(recent_avg - baseline_value) / baseline_value

        drift_detected = pct_change > threshold

        return {
            "drift_detected": drift_detected,
            "baseline_value": baseline_value,
            "recent_average": recent_avg,
            "percentage_change": pct_change * 100,
            "threshold": threshold * 100,
            "metric": metric,
            "window_size": window_size,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of model accuracy history."""
        if not self.history["runs"]:
            return {"total_runs": 0, "baseline_metrics": None}

        latest_run = self.history["runs"][-1]

        return {
            "total_runs": len(self.history["runs"]),
            "baseline_metrics": self.history["baseline_metrics"],
            "latest_metrics": latest_run["metrics"],
            "latest_timestamp": latest_run["timestamp"],
            "latest_version": latest_run["model_version"],
        }


def subset_forecast_by_importance(df: pd.DataFrame, top_n: int = 100, importance_metric: str = "total_sales") -> pd.DataFrame:
    """
    Filter to forecast only important items (Amazon Forecast-style subset forecasting).

    Reduces compute costs by focusing on items most important to business objectives.

    Parameters
    ----------
    df : DataFrame
        Full dataset with store, item, sales columns
    top_n : int
        Number of top items to forecast
    importance_metric : str
        Metric to determine importance:
        - 'total_sales': Items with highest total sales
        - 'avg_sales': Items with highest average sales
        - 'variance': Items with highest sales variance (most volatile)

    Returns
    -------
    Filtered DataFrame with only top N important items
    """
    if importance_metric == "total_sales":
        # Calculate total sales per store-item
        importance = df.groupby(["store", "item"])["sales"].sum().reset_index()
        importance.columns = ["store", "item", "total_sales"]
        importance = importance.sort_values("total_sales", ascending=False)
        top_items = importance.head(top_n)

    elif importance_metric == "avg_sales":
        # Calculate average sales per store-item
        importance = df.groupby(["store", "item"])["sales"].mean().reset_index()
        importance.columns = ["store", "item", "avg_sales"]
        importance = importance.sort_values("avg_sales", ascending=False)
        top_items = importance.head(top_n)

    elif importance_metric == "variance":
        # Calculate sales variance per store-item (most volatile)
        importance = df.groupby(["store", "item"])["sales"].var().reset_index()
        importance.columns = ["store", "item", "variance"]
        importance = importance.sort_values("variance", ascending=False)
        top_items = importance.head(top_n)

    else:
        raise ValueError(f"Unknown importance metric: {importance_metric}")

    # Filter original dataframe to only top items
    filtered_df = df.merge(top_items[["store", "item"]], on=["store", "item"], how="inner")

    return filtered_df, top_items
