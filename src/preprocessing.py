"""Data preprocessing and quality checks for the forecasting pipeline."""

import numpy as np
import pandas as pd


def validate_schema(df: pd.DataFrame, expected_cols: list[str]) -> None:
    """Raise if any expected column is missing from df."""
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure consistent column types before feature engineering."""
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    if "store" in df.columns:
        df["store"] = df["store"].astype(int)
    if "item" in df.columns:
        df["item"] = df["item"].astype(int)
    if "sales" in df.columns:
        df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    return df


def clip_sales_outliers(
    df: pd.DataFrame,
    lower_quantile: float = 0.001,
    upper_quantile: float = 0.999,
) -> pd.DataFrame:
    """
    Clip sales values to the [lower_quantile, upper_quantile] range.

    This removes extreme outliers that are likely data entry errors. Applied
    only to training rows (where is_train == 1 or sales is not NaN).
    """
    df = df.copy()
    if "sales" not in df.columns:
        return df

    train_mask = df["sales"].notna()
    lo = df.loc[train_mask, "sales"].quantile(lower_quantile)
    hi = df.loc[train_mask, "sales"].quantile(upper_quantile)
    df.loc[train_mask, "sales"] = df.loc[train_mask, "sales"].clip(lo, hi)
    return df


def fill_missing_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure every (store, item) pair has a continuous daily date range.

    Missing dates are filled with NaN sales and will be imputed downstream.
    This guards against sparse time series with missing days.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    full_dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D")

    reindexed_parts = []
    for (store, item), group in df.groupby(["store", "item"]):
        group = group.set_index("date").reindex(full_dates)
        group["store"] = store
        group["item"] = item
        group.index.name = "date"
        reindexed_parts.append(group.reset_index())

    return pd.concat(reindexed_parts, ignore_index=True)


def preprocess(df: pd.DataFrame, clip_outliers: bool = True) -> pd.DataFrame:
    """
    Full preprocessing pass: schema check → type casting → outlier clipping.

    Parameters
    ----------
    df : Raw combined DataFrame (train + test).
    clip_outliers : Whether to clip extreme sales values in training rows.
    """
    validate_schema(df, expected_cols=["date", "store", "item"])
    df = cast_types(df)
    if clip_outliers and "sales" in df.columns:
        df = clip_sales_outliers(df)
    return df
