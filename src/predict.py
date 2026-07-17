"""Inference for the primary XYZT awesome predictor only."""

import numpy as np
import pandas as pd


def predict_xyzt_awesome(
    test: pd.DataFrame,
    submission: pd.DataFrame,
    model: dict[str, object],
) -> pd.DataFrame:
    """Apply XYZT cell 35's awesome predictor without rounding."""
    submission[["sales"]] = submission[["sales"]].astype(np.float64)
    for _, row in test.iterrows():
        dow, month, year = row.name.dayofweek, row.name.month, row.name.year
        item, store = row["item"], row["store"]
        base_sales = model["dow_item_table"].at[dow, item]
        mul = model["month_table"].at[month, "sales"] * model["store_table"].at[store, "sales"]
        pred_sales = base_sales * mul * model["annual_growth"](year)
        submission.at[row["id"], "sales"] = pred_sales
    return submission


def round_xyzt_awesome_predictions(prediction: pd.DataFrame) -> pd.DataFrame:
    """Copy and round XYZT cell 35 predictions exactly."""
    rounded = prediction.copy()
    rounded["sales"] = np.round(rounded["sales"]).astype(int)
    return rounded
