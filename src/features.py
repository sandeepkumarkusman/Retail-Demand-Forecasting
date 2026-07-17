"""Feature creation for the primary XYZT solution only."""

import pandas as pd


def expand_xyzt_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Copy a date-indexed frame and add XYZT cell-9 date features exactly."""
    data = df.copy()
    data["day"] = data.index.day
    data["month"] = data.index.month
    data["year"] = data.index.year
    data["dayofweek"] = data.index.dayofweek
    return data
