"""Feature creation for the forecasting solution."""

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar


def expand_ml_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract standard date components for ML."""
    df = df.copy()
    if "date" not in df.columns and df.index.name == "date":
        df = df.reset_index()
    if "date" not in df.columns:
        return df

    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.dayofweek
    df["dayofyear"] = df["date"].dt.dayofyear
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)

    # Cyclical encodings
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)

    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    df["quarter"] = df["date"].dt.quarter

    # Holidays
    cal = USFederalHolidayCalendar()
    # We create a date range to check holidays
    min_date = df["date"].min()
    max_date = df["date"].max()
    if pd.isna(min_date):
        return df
    holidays = cal.holidays(start=min_date, end=max_date)
    df["is_holiday"] = df["date"].isin(holidays).astype(int)

    # Days to nearest holiday
    holiday_dates = pd.DataFrame({"holiday_date": holidays})
    if len(holiday_dates) > 0:
        df_sorted = df[["date"]].copy().sort_values("date")
        df_merged = pd.merge_asof(
            df_sorted,
            holiday_dates,
            left_on="date",
            right_on="holiday_date",
            direction="forward",
        )
        # We need to assign it back properly based on the original index
        df_merged.index = df_sorted.index
        df["days_to_holiday"] = (
            (df_merged["holiday_date"] - df_merged["date"])
            .dt.days.fillna(999)
            .astype(int)
        )
    else:
        df["days_to_holiday"] = 999

    df["days_since_start"] = (df["date"] - min_date).dt.days
    return df


def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add advanced lag features and rolling statistics based on the notebook logic.
    For a 90-day forecast horizon, the minimum safe lag is 91.
    """
    df = df.copy()
    df.sort_values(["store", "item", "date"], inplace=True)

    # Lag features
    lag_days = [91, 98, 105, 112, 182, 270, 364, 365, 728]
    for lag in lag_days:
        df[f"sales_lag_{lag}"] = df.groupby(["store", "item"])["sales"].shift(lag)

    # Cross-series aggregate lags
    df_store_daily = (
        df.groupby(["store", "date"])["sales"]
        .sum()
        .reset_index()
        .rename(columns={"sales": "store_total_sales"})
    )
    df_item_daily = (
        df.groupby(["item", "date"])["sales"]
        .sum()
        .reset_index()
        .rename(columns={"sales": "item_total_sales"})
    )

    df_store_daily["store_total_sales_lag_91"] = df_store_daily.groupby("store")[
        "store_total_sales"
    ].shift(91)
    df_item_daily["item_total_sales_lag_91"] = df_item_daily.groupby("item")[
        "item_total_sales"
    ].shift(91)

    df = df.merge(
        df_store_daily[["store", "date", "store_total_sales_lag_91"]],
        on=["store", "date"],
        how="left",
    )
    df = df.merge(
        df_item_daily[["item", "date", "item_total_sales_lag_91"]],
        on=["item", "date"],
        how="left",
    )
    df.sort_values(["store", "item", "date"], inplace=True)

    # Rolling window features on lag-91
    windows = [7, 14, 28, 56, 91]
    for w in windows:
        df[f"rolling_mean_{w}"] = df.groupby(["store", "item"])["sales"].transform(
            lambda x: x.shift(91).rolling(window=w, min_periods=1).mean()
        )
        df[f"rolling_std_{w}"] = df.groupby(["store", "item"])["sales"].transform(
            lambda x: x.shift(91).rolling(window=w, min_periods=1).std()
        )

    # Local Trend
    df["local_trend"] = df["rolling_mean_28"] - df["rolling_mean_91"]

    # EMA features on lag-91
    df["ema_090"] = df.groupby(["store", "item"])["sales"].transform(
        lambda x: x.shift(91).ewm(alpha=0.90, min_periods=1).mean()
    )
    df["ema_095"] = df.groupby(["store", "item"])["sales"].transform(
        lambda x: x.shift(91).ewm(alpha=0.95, min_periods=1).mean()
    )

    # Expanding mean
    df["expanding_mean"] = df.groupby(["store", "item"])["sales"].transform(
        lambda x: x.shift(91).expanding(min_periods=1).mean()
    )

    # Year-over-Year Growth Features
    df["yoy_ratio"] = df["sales_lag_364"] / df["sales_lag_728"].replace(0, np.nan)
    df["sales_same_month_last_year"] = df["sales_lag_364"]

    return df


def add_target_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Target encoding across categorical interactions using training data only."""
    df = df.copy()
    train_mask = df["is_train"] == 1

    encodings = {
        "store_mean": ["store"],
        "item_mean": ["item"],
        "store_item_mean": ["store", "item"],
        "month_mean": ["month"],
        "dow_mean": ["dayofweek"],
        "store_month_mean": ["store", "month"],
        "item_month_mean": ["item", "month"],
        "item_dow_mean": ["item", "dayofweek"],
        "store_dow_mean": ["store", "dayofweek"],
    }

    for feature_name, group_cols in encodings.items():
        mean_series = df.loc[train_mask].groupby(group_cols)["sales"].mean()
        if len(group_cols) == 1:
            df[feature_name] = df[group_cols[0]].map(mean_series)
        else:
            df[feature_name] = df.set_index(group_cols).index.map(mean_series)

    return df


def prepare_ml_data(data: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """Full feature engineering pipeline for ML."""
    df = expand_ml_date_features(data)

    df["store"] = df["store"].astype(int)
    df["item"] = df["item"].astype(int)

    # We expect 'is_train' to be passed in or we infer it
    if "is_train" not in df.columns:
        df["is_train"] = int(is_train)

    if "sales" in df.columns:
        df = add_lag_and_rolling_features(df)
        df = add_target_encoding(df)

    if is_train and "sales" in df.columns:
        df.dropna(subset=["sales"], inplace=True)

    return df
