"""Walk-forward cross-validation utilities for time-series evaluation."""

from pathlib import Path
from typing import Generator, Callable, Dict, Any

import numpy as np
import pandas as pd

from src.metrics import calculate_ml_metrics


def walk_forward_splits(
    df: pd.DataFrame,
    date_col: str = "date",
    n_folds: int = 3,
    fold_size_months: int = 3,
    min_train_years: int = 1,
) -> Generator[tuple[pd.DataFrame, pd.DataFrame], None, None]:
    """
    Yield (train, validation) DataFrame splits using a walk-forward strategy.

    Each validation window is `fold_size_months` months wide and is taken from the
    tail of the available training data, rolling backwards by one fold per iteration.

    Parameters
    ----------
    df : DataFrame with at least `date_col` and `sales` columns.
    n_folds : Number of CV folds to generate.
    fold_size_months : Width of each validation window in months.
    min_train_years : Minimum training years before the first fold.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    max_date = df[date_col].max()
    min_date = df[date_col].min()

    for fold_idx in range(n_folds):
        # Validation window ends at max_date minus (fold_idx * fold_size_months)
        val_end = max_date - pd.DateOffset(months=fold_idx * fold_size_months)
        val_start = val_end - pd.DateOffset(months=fold_size_months) + pd.Timedelta(days=1)

        if val_start <= min_date + pd.DateOffset(years=min_train_years):
            break

        train_mask = df[date_col] < val_start
        val_mask = (df[date_col] >= val_start) & (df[date_col] <= val_end)

        yield df[train_mask].copy(), df[val_mask].copy()


def evaluate_walk_forward(
    df: pd.DataFrame,
    config: Dict[str, Any],
    model_fit_fn: Callable,
    model_predict_fn: Callable,
    n_folds: int = 3,
    fold_size_months: int = 3,
) -> pd.DataFrame:
    """
    Run walk-forward CV with caller-supplied fit/predict callables and return
    per-fold SMAPE, MAE, RMSE, and quantile metrics in a summary DataFrame.
    
    Amazon Forecast-style: Configurable number of backtest windows to evaluate
    model accuracy over different start dates.

    Parameters
    ----------
    df : Prepared feature DataFrame (output of prepare_ml_data).
    config : Configuration dictionary.
    model_fit_fn : Callable (train_df, config) → model_dict
    model_predict_fn : Callable (val_df, sample_sub, model_dict) → DataFrame of predictions
    n_folds : Number of walk-forward folds (Amazon Forecast-style configurable).
    fold_size_months : Width of each validation window in months.
    """
    results = []
    for fold_idx, (train_fold, val_fold) in enumerate(
        walk_forward_splits(df, n_folds=n_folds, fold_size_months=fold_size_months)
    ):
        model_dict = model_fit_fn(train_fold, config)
        
        # Predict expects sample_sub, we can pass None if it doesn't strictly need it for ordering,
        # but to be safe, we'll pass a dummy DataFrame with 'id' column if it exists in val_fold.
        sample_sub = val_fold[['id']] if 'id' in val_fold.columns and not val_fold['id'].isna().all() else pd.DataFrame({'id': range(len(val_fold))})
        if 'id' not in val_fold.columns or val_fold['id'].isna().all():
            val_fold['id'] = range(len(val_fold))
            
        pred_df = model_predict_fn(val_fold, sample_sub, model_dict)
        
        # Merge true sales with predictions to ensure alignment
        merged = val_fold[['id', 'date', 'sales']].merge(pred_df, on='id')
        y_true = merged["sales_x"].values if "sales_x" in merged.columns else merged["sales"].values
        y_pred = merged["sales_y"].values if "sales_y" in merged.columns else merged["sales"].values
        
        y_q05 = pred_df.set_index('id').loc[merged['id']]['q05'].values if 'q05' in pred_df.columns else None
        y_q95 = pred_df.set_index('id').loc[merged['id']]['q95'].values if 'q95' in pred_df.columns else None

        metrics = calculate_ml_metrics(y_true, y_pred, y_q05, y_q95)
        metrics["fold"] = fold_idx
        metrics["val_start"] = val_fold["date"].min()
        metrics["val_end"] = val_fold["date"].max()
        results.append(metrics)

    return pd.DataFrame(results).set_index("fold")
