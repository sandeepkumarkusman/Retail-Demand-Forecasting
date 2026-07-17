"""Feature creation for the primary XYZT solution only."""

import pandas as pd
import numpy as np
import re


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


def add_datepart_prophet_reference(
    df: pd.DataFrame,
    fldname: str,
    inplace: bool = False,
    drop: bool = False,
) -> pd.DataFrame | None:
    """Reproduce Prophet/dumb reference notebook cell 16.

    ``Week`` and ``Weekofyear`` use the identical ISO-week calculation on
    current pandas versions where the source accessors were removed.
    """
    if not inplace:
        df = df.copy()
    fld = df[fldname]
    fld_dtype = fld.dtype
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64
    if not np.issubdtype(fld_dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    targ_pre = re.sub("[Dd]ate$", "", fldname)

    attributes = ["Year", "Month", "Week", "Day", "Dayofweek", "Dayofyear", "Weekofyear"]
    for name in attributes:
        if name in ("Week", "Weekofyear"):
            df[targ_pre + name] = fld.dt.isocalendar().week.astype(int)
        else:
            df[targ_pre + name] = getattr(fld.dt, name.lower())
    if drop:
        df.drop(fldname, axis=1, inplace=True)
    if not inplace:
        return df


def add_total_sales_reference_features(
    train: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    """Reproduce eda-of-total-sales.ipynb cells 4 and 18."""
    from pandas.tseries.holiday import USFederalHolidayCalendar as cal1

    cal = cal1()
    holidays = cal.holidays(start=train.index.min(), end=train.index.max())
    train["year"] = train.index.year
    train["month"] = train.index.month
    train["DoM"] = train.index.day
    train["DoW"] = train.index.dayofweek
    train["DoY"] = train.index.dayofyear
    train["Holiday"] = train.index.isin(holidays)
    train_holidays = train[train.Holiday == True]
    return train, train_holidays, holidays
