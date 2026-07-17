"""Training components for the primary XYZT awesome predictor only."""

import numpy as np
import pandas as pd


def fit_xyzt_awesome_model(data: pd.DataFrame) -> dict[str, object]:
    """Fit XYZT cells 23, 30, and 34 in their original execution order."""
    grand_avg = data.sales.mean()

    store_item_table = pd.pivot_table(
        data,
        index="store",
        columns="item",
        values="sales",
        aggfunc=np.mean,
    )

    month_table = pd.pivot_table(data, index="month", values="sales", aggfunc=np.mean)
    month_table.sales /= grand_avg

    dow_table = pd.pivot_table(data, index="dayofweek", values="sales", aggfunc=np.mean)
    dow_table.sales /= grand_avg

    year_table = pd.pivot_table(data, index="year", values="sales", aggfunc=np.mean)
    year_table /= grand_avg

    years = np.arange(2013, 2019)
    annual_sales_avg = year_table.values.squeeze()
    p1 = np.poly1d(np.polyfit(years[:-1], annual_sales_avg, 1))
    p2 = np.poly1d(np.polyfit(years[:-1], annual_sales_avg, 2))
    annual_growth = p2

    years = np.arange(2013, 2019)
    annual_sales_avg = year_table.values.squeeze()
    weights = np.exp((years - 2018) / 5)
    annual_growth = np.poly1d(
        np.polyfit(years[:-1], annual_sales_avg, 2, w=weights[:-1])
    )

    dow_item_table = pd.pivot_table(
        data,
        index="dayofweek",
        columns="item",
        values="sales",
        aggfunc=np.mean,
    )

    store_table = pd.pivot_table(data, index="store", values="sales", aggfunc=np.mean)
    store_table.sales /= grand_avg

    return {
        "grand_avg": grand_avg,
        "store_item_table": store_item_table,
        "month_table": month_table,
        "dow_table": dow_table,
        "year_table": year_table,
        "years": years,
        "annual_sales_avg": annual_sales_avg,
        "p1": p1,
        "p2": p2,
        "weights": weights,
        "annual_growth": annual_growth,
        "dow_item_table": dow_item_table,
        "store_table": store_table,
    }
