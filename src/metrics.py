"""Notebook-specific evaluation metrics kept isolated by source."""

from typing import Union

import numpy as np
import pandas as pd


def smape_prophet_dumb(
    y: Union[np.ndarray, float],
    yhat: Union[np.ndarray, float],
    average: bool = True,
    signed: bool = False,
) -> float:
    """Reproduce the SMAPE function from the Prophet/dumb reference notebook."""
    if signed:
        result = 2.0 * (yhat - y) / (np.abs(y) + np.abs(yhat)) * 100
    else:
        result = 2.0 * np.abs(yhat - y) / (np.abs(y) + np.abs(yhat)) * 100
    if average:
        return np.mean(result)
    return result


def smape_df_prophet_dumb(
    df: pd.DataFrame,
    average: bool = True,
    signed: bool = False,
) -> pd.DataFrame:
    """Apply the Prophet/dumb SMAPE to a frame exposing y and yhat columns."""
    return smape_prophet_dumb(df.y, df.yhat, average=average, signed=signed)


def smape_fourth_place(y_true, y_pred):
    """Reproduce the fourth-place script's zero-denominator SMAPE function."""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 200.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.nanmean(diff)
