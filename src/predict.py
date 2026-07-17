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


def predict_fourth_place(data: pd.DataFrame) -> pd.DataFrame:
    """Reproduce 4th_place_sol_n.py lines 84-87."""
    data["smry_product"] = np.prod(
        data.loc[:, ["month_mod", "year_mod", "weekday_mod", "store_item_mod"]] + 1,
        axis=1,
    )
    data["sales_mod_pred"] = np.round(data.smry_product * np.round(np.nanmean(data.sales), 1))
    return data


def create_fourth_place_submission(
    sample_submission: pd.DataFrame,
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Reproduce 4th_place_sol_n.py lines 109-110."""
    sample_submission["sales"] = data[data.year == 6].sales_mod_pred.reset_index(drop=True)
    return sample_submission


def predict_polyfit_showcase(test: pd.DataFrame, model: dict[str, object]) -> pd.DataFrame:
    """Reproduce store-item-polyfit-showcase.ipynb cell 2 inference."""
    predicted_sales = []
    for _, row in test.iterrows():
        day_of_week, month, year, item, store = (
            row.day_of_week,
            row.month,
            row.year,
            row["item"],
            row.store,
        )
        base_sales = model["dow_item"].loc[day_of_week, item]
        month_multiplier = model["month_df"].loc[month, "sales"]
        store_multiplier = model["store_df"].loc[store, "sales"]
        predicted_sales.append(
            base_sales * month_multiplier * store_multiplier * model["year_factor"]
        )
    submission = pd.DataFrame(test["id"])
    submission["sales"] = np.round(predicted_sales).astype(int)
    return submission


def predict_store_prediction_slightly_better(
    test: pd.DataFrame, submission: pd.DataFrame, model: dict[str, object]
) -> pd.DataFrame:
    """Reproduce store-prediction.ipynb cell 12 with its current annual model."""
    submission[["sales"]] = submission[["sales"]].astype(np.float64)
    for _, row in test.iterrows():
        dow, month, year = row.name.dayofweek, row.name.month, row.name.year
        item, store = row["item"], row["store"]
        base_sales = model["store_item_table"].at[store, item]
        mul = model["month_table"].at[month, "sales"] * model["dow_table"].at[dow, "sales"]
        submission.at[row["id"], "sales"] = base_sales * mul * model["annual_growth"](year)
    return submission


def predict_store_prediction_weighted(
    test: pd.DataFrame, submission: pd.DataFrame, model: dict[str, object]
) -> pd.DataFrame:
    """Reproduce store-prediction.ipynb cell 24."""
    return predict_store_prediction_slightly_better(test, submission, model)


def round_store_prediction_submission(prediction: pd.DataFrame) -> pd.DataFrame:
    """Reproduce store-prediction.ipynb cells 21 and 25 rounding."""
    rounded = prediction.copy()
    rounded["sales"] = np.round(rounded["sales"]).astype(int)
    return rounded


def predict_unavailable_artifact_fallback_baseline(
    test: pd.DataFrame,
    sample_submission: pd.DataFrame,
    model: dict[str, object],
) -> pd.DataFrame:
    """Predict the explicitly non-notebook-derived missing-artifact fallback.

    Values are looked up by test ``id`` so the sample-submission row ordering and
    column ordering are retained. Missing store-item combinations receive the
    historical global mean. This function must not be used to claim a Prophet
    or external-ensemble reproduction.
    """
    key_index = pd.MultiIndex.from_frame(test[["store", "item"]])
    values = model["store_item_mean"].reindex(key_index).to_numpy()
    values = np.where(pd.isna(values), model["global_mean"], values)
    values_by_id = pd.Series(values, index=test["id"].to_numpy())

    submission = sample_submission.copy()
    submission["sales"] = submission["id"].map(values_by_id)
    submission["sales"] = submission["sales"].fillna(model["global_mean"])
    return submission


def round_unavailable_artifact_fallback_submission(
    prediction: pd.DataFrame,
) -> pd.DataFrame:
    """Round fallback output to the Store Item Demand integer submission format."""
    submission = prediction.copy()
    submission["sales"] = np.round(submission["sales"]).astype(int)
    return submission


def blend_store_prediction_candidates(
    sub1: pd.DataFrame,
    sub2: pd.DataFrame,
    sub3: pd.DataFrame,
    sub4: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduce store-prediction.ipynb cells 29-31, including aliasing."""
    del sub4
    sub5 = sub1
    sub5.sales = sub1.sales * 0.5 + sub2.sales * 0.5
    sub6 = sub1
    sub6.sales = sub1.sales * 0.4 + sub2.sales * 0.5 + sub3.sales * 0.1
    return sub5, sub6


def blend32_weighted_average(df_sub: pd.DataFrame) -> pd.DataFrame:
    """Reproduce blend-boosting notebook cell 4's 32-column weighted blend."""
    df_sub["weighted_avg"] = (
        4 * (df_sub["9"] + df_sub["17"] + df_sub["23"]) / 3
        + (
            df_sub["2"] + df_sub["5"] + df_sub["15"] + df_sub["16"]
            + df_sub["18"] + df_sub["20"] + df_sub["21"] + df_sub["22"]
            + df_sub["26"] + df_sub["27"] + df_sub["29"] + df_sub["30"]
        ) / 12
        + (
            df_sub["3"] + df_sub["4"] + df_sub["6"] + df_sub["7"]
            + df_sub["8"] + df_sub["10"] + df_sub["11"] + df_sub["12"]
            + df_sub["13"] + df_sub["14"] + df_sub["19"] + df_sub["24"]
            + df_sub["25"] + df_sub["28"] + df_sub["31"]
        ) * 3 / 15
        + 6 * (df_sub["0"] + df_sub["1"]) / 2
    ) / 14
    return df_sub


def create_blend32_submission(df_sub: pd.DataFrame) -> pd.DataFrame:
    """Reproduce blend-boosting notebook cell 4 submission generation."""
    return pd.DataFrame(
        {"id": [*range(45000)], "sales": df_sub["weighted_avg"].round().astype("int")}
    )
