"""Training components for the primary XYZT awesome predictor only."""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

from src.utils import suppress_stdout, timer_gen


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


def fit_fourth_place_factors(data: pd.DataFrame) -> pd.DataFrame:
    """Reproduce 4th_place_sol_n.py factor fitting in lines 53-84."""
    month_smry = (
        (data.groupby(["month"]).agg([np.nanmean]).sales - np.nanmean(data.sales))
        / np.nanmean(data.sales)
    ).rename(columns={"nanmean": "month_mod"})
    data = data.join(month_smry, how="left", on="month")

    year_smry = (
        (data.groupby(["year"]).agg([np.nanmean]).sales - np.nanmean(data.sales))
        / np.nanmean(data.sales)
    ).rename(columns={"nanmean": "year_mod"})
    cagr = (
        data[data.year == 5].groupby(["store", "item"]).agg(np.nanmean).sales
        / data[data.year == 2].groupby(["store", "item"]).agg(np.nanmean).sales
    ) ** (1 / 4) - 1
    print((np.mean(cagr), np.std(cagr)))
    year_smry.loc[6, :] = np.mean(cagr) * 3
    data = data.join(year_smry, how="left", on="year")

    weekday_smry = (
        (data.groupby(["weekday"]).agg([np.nanmean]).sales - np.nanmean(data.sales))
        / np.nanmean(data.sales)
    ).rename(columns={"nanmean": "weekday_mod"})
    data = data.join(weekday_smry, how="left", on="weekday")

    store_item_smry = (
        (data.groupby(["store", "item"]).agg([np.nanmean]).sales - np.nanmean(data.sales))
        / np.nanmean(data.sales)
    ).rename(columns={"nanmean": "store_item_mod"})
    data = data.join(store_item_smry, how="left", on=["store", "item"])
    return data


def fit_polyfit_showcase_model(train: pd.DataFrame) -> dict[str, object]:
    """Reproduce store-item-polyfit-showcase.ipynb cell 2."""
    train_subset_2 = train[(train["store"] == 1) & (train["item"] == 20)]
    trunc_data = train[train["year"] >= 2013]
    overall_mean = trunc_data["sales"].mean()
    dow_item = pd.pivot_table(
        trunc_data,
        index="day_of_week",
        columns="item",
        values="sales",
        aggfunc=np.mean,
    )
    month_df = pd.pivot_table(
        trunc_data, index="month", values="sales", aggfunc=np.mean
    ) / overall_mean
    store_df = pd.pivot_table(
        trunc_data, index="store", values="sales", aggfunc=np.mean
    ) / overall_mean
    year_df = pd.pivot_table(train, index="year", values="sales", aggfunc=np.mean) / overall_mean
    rate = 2.5
    years = np.arange(2013, 2018)
    annual_growth = np.poly1d(
        np.polyfit(
            years,
            year_df.values.squeeze(),
            2,
            w=np.exp((years - 2018) / rate),
        )
    )
    year_factor = round(annual_growth(2018), 30)
    return {
        "train_subset_2": train_subset_2,
        "trunc_data": trunc_data,
        "overall_mean": overall_mean,
        "dow_item": dow_item,
        "month_df": month_df,
        "store_df": store_df,
        "year_df": year_df,
        "rate": rate,
        "years": years,
        "annual_growth": annual_growth,
        "year_factor": year_factor,
    }


def fit_store_prediction_unweighted_model(data: pd.DataFrame) -> dict[str, object]:
    """Reproduce store-prediction.ipynb cells 13-18."""
    store_item_table = pd.pivot_table(
        data, index="store", columns="item", values="sales", aggfunc=np.mean
    )
    grand_avg = data.sales.mean()
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
    return {
        "store_item_table": store_item_table,
        "grand_avg": grand_avg,
        "month_table": month_table,
        "dow_table": dow_table,
        "year_table": year_table,
        "years": years,
        "annual_sales_avg": annual_sales_avg,
        "p1": p1,
        "p2": p2,
        "annual_growth": annual_growth,
    }


def fit_store_prediction_weighted_model(data: pd.DataFrame) -> dict[str, object]:
    """Reproduce store-prediction.ipynb cell 22 after cells 13-18."""
    model = fit_store_prediction_unweighted_model(data)
    years = np.arange(2013, 2019)
    annual_sales_avg = model["year_table"].values.squeeze()
    weights = np.exp((years - 2018) / 6)
    annual_growth = np.poly1d(
        np.polyfit(years[:-1], annual_sales_avg, 2, w=weights[:-1])
    )
    print(f"2018 Relative Sales by Weighted Fit = {annual_growth(2018)}")
    model["years"] = years
    model["annual_sales_avg"] = annual_sales_avg
    model["weights"] = weights
    model["annual_growth"] = annual_growth
    return model


def fit_store_prediction_model(data: pd.DataFrame) -> dict[str, object]:
    """Return the final weighted state after store-prediction cells 13-22."""
    return fit_store_prediction_weighted_model(data)


class DumbBase(ABC):
    """Reference dumb base model from the Prophet/dumb notebook cell 51."""

    name = "dumb_base"

    def __init__(self, growth="linear", fit_window_years=None):
        self.growth = growth
        self.fit_window_years = fit_window_years
        self.verbose = True

    @staticmethod
    def expand_data(data):
        data_exp = data.copy()
        data_exp["day"] = data["date"].apply(lambda x: x.day)
        data_exp["month"] = data["date"].apply(lambda x: x.month)
        data_exp["year"] = data["date"].apply(lambda x: x.year)
        data_exp["dayofweek"] = data["date"].apply(lambda x: x.dayofweek)
        data_exp["weekofyear"] = data["date"].apply(
            lambda x: x.weekofyear if hasattr(x, "weekofyear") else x.isocalendar().week
        )
        data_exp["dayofyear"] = data["date"].apply(lambda x: x.dayofyear)
        return data_exp

    def _fit_annual_sales(self):
        if isinstance(self.growth, pd.DataFrame):
            self._annual_sales = lambda x: self.growth.loc[x, "sales"]
        else:
            print("Dumb fit: functional annual growth")
            year_table = pd.pivot_table(self.data, index="year", values="sales", aggfunc=np.mean)
            years = year_table.index.values
            annual_sales_avg = year_table.values.squeeze()

            if growth == "linear":
                self._annual_sales = np.poly1d(np.polyfit(years, annual_sales_avg, 1))
            elif growth == "quadratic":
                self._annual_sales = np.poly1d(np.polyfit(years, annual_sales_avg, 2))
            else:
                raise KeyError

    @abstractmethod
    def _fit_base_seasonality(self):
        pass

    def fit(self, data):
        if "year" in data.columns:
            self.data = data.copy()
        else:
            print("Dumb fit: Expand data")
            self.data = self.expand_data(data)

        if self.fit_window_years is not None:
            date_max = self.data["date"].max()
            date_min = date_max.replace(year=date_max.year - self.fit_window_years)
            self.data = self.data.query("date > @date_min")

        self.data["sales"] /= self.data["sales"].mean()
        self._fit_base_seasonality()
        self._fit_annual_sales()

    @abstractmethod
    def _predict_base_seasonality(self, item, store, date):
        pass

    def _predict_annual_sales(self, year):
        return self._annual_sales(year)

    def predict(self, data):
        data = data.assign(sales_hat=.001)
        with suppress_stdout(not self.verbose):
            timer = timer_gen()
            count = 1
            for i, row in data.iterrows():
                if count % 100000 == 0:
                    print("dumb predict {} {}".format(count, next(timer), end=" | "))
                date, item, store = row["date"], row["item"], row["store"]
                pred_sales = self._predict_base_seasonality(item, store, date) * self._predict_annual_sales(date.year)
                data.at[i, "sales_hat"] = pred_sales
                count += 1
        return data


class DumbOriginal(DumbBase):
    """Original dumb model from reference notebook cell 51."""

    name = "dumb_original"

    def _fit_base_seasonality(self):
        self.store_item_table = pd.pivot_table(
            self.data, index="store", columns="item", values="sales", aggfunc=np.mean
        )
        self.month_table = pd.pivot_table(self.data, index="month", values="sales", aggfunc=np.mean)
        self.dow_table = pd.pivot_table(self.data, index="dayofweek", values="sales", aggfunc=np.mean)

    def _predict_base_seasonality(self, item, store, date):
        dow, month, year = date.dayofweek, date.month, date.year
        base_sales = self.store_item_table.at[store, item]
        seasonal_sales = self.month_table.at[month, "sales"] * self.dow_table.at[dow, "sales"]
        return base_sales * seasonal_sales


class DumbItemDayofweek(DumbBase):
    """Item-weekday dumb variation from reference notebook cell 61."""

    name = "dumb_item_dow"

    def _fit_base_seasonality(self):
        self.store_table = pd.pivot_table(self.data, index="store", values="sales", aggfunc=np.mean)
        self.month_table = pd.pivot_table(self.data, index="month", values="sales", aggfunc=np.mean)
        self.dow_item_table = pd.pivot_table(
            self.data, index="dayofweek", columns="item", values="sales", aggfunc=np.mean
        )

    def _predict_base_seasonality(self, item, store, date):
        dow, month, year = date.dayofweek, date.month, date.year
        base_sales = self.store_table.at[store, "sales"]
        seasonal_sales = self.month_table.at[month, "sales"] * self.dow_item_table.at[dow, item]
        return base_sales * seasonal_sales


class DumbStoreDayofweek(DumbBase):
    """Store-weekday dumb variation from reference notebook cell 61."""

    name = "dumb_store_dow"

    def _fit_base_seasonality(self):
        self.item_table = pd.pivot_table(self.data, index="item", values="sales", aggfunc=np.mean)
        self.month_table = pd.pivot_table(self.data, index="month", values="sales", aggfunc=np.mean)
        self.dow_store_table = pd.pivot_table(
            self.data, index="dayofweek", columns="store", values="sales", aggfunc=np.mean
        )

    def _predict_base_seasonality(self, item, store, date):
        dow, month, year = date.dayofweek, date.month, date.year
        base_sales = self.item_table.at[item, "sales"]
        seasonal_sales = self.month_table.at[month, "sales"] * self.dow_store_table.at[dow, store]
        return base_sales * seasonal_sales


class DumbItemStoreDayofweek(DumbBase):
    """Item-store-weekday dumb variation from reference notebook cell 61."""

    name = "dumb_item_store_dow"

    def _fit_base_seasonality(self):
        self.base_scalar = 1.0
        self.month_table = pd.pivot_table(self.data, index="month", values="sales", aggfunc=np.mean)
        self.dow_item_store_table = pd.pivot_table(
            self.data,
            index=["item", "store"],
            columns="dayofweek",
            values="sales",
            aggfunc=np.mean,
        )

    def _predict_base_seasonality(self, item, store, date):
        dow, month, year = date.dayofweek, date.month, date.year
        base_sales = self.base_scalar
        seasonal_sales = self.month_table.at[month, "sales"] * self.dow_item_store_table.at[(item, store), dow]
        return base_sales * seasonal_sales
