"""Feature creation for the primary XYZT solution only."""

import pandas as pd
import numpy as np


def expand_xyzt_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Copy a date-indexed frame and add XYZT cell-9 date features exactly."""
    data = df.copy()
    data["day"] = data.index.day
    data["month"] = data.index.month
    data["year"] = data.index.year
    data["dayofweek"] = data.index.dayofweek
    return data


def prepare_fourth_place_data(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Reproduce 4th_place_sol_n.py lines 38-51 exactly."""
    columns = list(train.columns)
    data = pd.concat([train, test], axis=0).reset_index(drop=True)
    data = data.loc[:, columns]
    data.index = pd.to_datetime(data.date)
    data.drop("date", axis=1, inplace=True)
    data["year"] = data.index.year - min(data.index.year) + 1
    data["month"] = data.index.month
    data["weekday"] = data.index.weekday
    data = data[data.year > 1]
    return data


def add_polyfit_showcase_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce store-item-polyfit-showcase.ipynb cell 1."""
    df["date"] = pd.to_datetime(df["date"])
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["day_of_week"] = df["date"].dt.weekday
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["date"].dt.quarter
    df["day_of_year"] = df["date"].dt.dayofyear
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = 0.5 * (np.cos(2 * np.pi * (df["month"] + 5) / 12) + 1)
    df["week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


def expand_store_prediction_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce store-prediction.ipynb cell 4 with a pandas compatibility shim."""
    data = df.copy()
    data["day"] = data.index.day
    data["month"] = data.index.month
    data["year"] = data.index.year
    data["dayofweek"] = data.index.dayofweek
    data["dayofyear"] = data.index.dayofyear
    try:
        data["weekofyear"] = data.index.weekofyear
    except AttributeError:
        data["weekofyear"] = data.index.isocalendar().week.astype(int)
    return data
