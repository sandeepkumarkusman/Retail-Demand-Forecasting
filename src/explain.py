"""Diagnostic plots for retail demand exploratory data analysis."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_year_patterns(data: pd.DataFrame) -> None:
    """Plot yearly relative sales by item and by store."""
    agg_year_item = pd.pivot_table(data, index="year", columns="item", values="sales", aggfunc=np.mean).values
    agg_year_store = pd.pivot_table(data, index="year", columns="store", values="sales", aggfunc=np.mean).values

    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.plot(agg_year_item / agg_year_item.mean(0)[np.newaxis])
    plt.title("Items")
    plt.xlabel("Year")
    plt.ylabel("Relative Sales")
    plt.subplot(122)
    plt.plot(agg_year_store / agg_year_store.mean(0)[np.newaxis])
    plt.title("Stores")
    plt.xlabel("Year")
    plt.ylabel("Relative Sales")
    plt.show()


def plot_month_patterns(data: pd.DataFrame) -> None:
    """Plot monthly relative sales by item and by store."""
    agg_month_item = pd.pivot_table(data, index="month", columns="item", values="sales", aggfunc=np.mean).values
    agg_month_store = pd.pivot_table(data, index="month", columns="store", values="sales", aggfunc=np.mean).values

    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.plot(agg_month_item / agg_month_item.mean(0)[np.newaxis])
    plt.title("Items")
    plt.xlabel("Month")
    plt.ylabel("Relative Sales")
    plt.subplot(122)
    plt.plot(agg_month_store / agg_month_store.mean(0)[np.newaxis])
    plt.title("Stores")
    plt.xlabel("Month")
    plt.ylabel("Relative Sales")
    plt.show()


def plot_dayofweek_patterns(data: pd.DataFrame) -> None:
    """Plot day-of-week relative sales by item and by store."""
    agg_dow_item = pd.pivot_table(data, index="dayofweek", columns="item", values="sales", aggfunc=np.mean).values
    agg_dow_store = pd.pivot_table(data, index="dayofweek", columns="store", values="sales", aggfunc=np.mean).values

    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.plot(agg_dow_item / agg_dow_item.mean(0)[np.newaxis])
    plt.title("Items")
    plt.xlabel("Day of Week")
    plt.ylabel("Relative Sales")
    plt.subplot(122)
    plt.plot(agg_dow_store / agg_dow_store.mean(0)[np.newaxis])
    plt.title("Stores")
    plt.xlabel("Day of Week")
    plt.ylabel("Relative Sales")
    plt.show()


def plot_calendar_interactions(data: pd.DataFrame) -> None:
    """Plot interactions between calendar components (DOW, Month, Year)."""
    agg_dow_month = pd.pivot_table(data, index="dayofweek", columns="month", values="sales", aggfunc=np.mean).values
    agg_month_year = pd.pivot_table(data, index="month", columns="year", values="sales", aggfunc=np.mean).values
    agg_dow_year = pd.pivot_table(data, index="dayofweek", columns="year", values="sales", aggfunc=np.mean).values

    plt.figure(figsize=(18, 5))
    plt.subplot(131)
    plt.plot(agg_dow_month / agg_dow_month.mean(0)[np.newaxis])
    plt.title("Months")
    plt.xlabel("Day of Week")
    plt.ylabel("Relative Sales")
    plt.subplot(132)
    plt.plot(agg_month_year / agg_month_year.mean(0)[np.newaxis])
    plt.title("Years")
    plt.xlabel("Months")
    plt.ylabel("Relative Sales")
    plt.subplot(133)
    plt.plot(agg_dow_year / agg_dow_year.mean(0)[np.newaxis])
    plt.title("Years")
    plt.xlabel("Day of Week")
    plt.ylabel("Relative Sales")
    plt.show()


def plot_store_item_patterns(data: pd.DataFrame) -> None:
    """Plot relative sales across stores and items."""
    agg_store_item = pd.pivot_table(data, index="store", columns="item", values="sales", aggfunc=np.mean).values

    plt.figure(figsize=(14, 5))
    plt.subplot(121)
    plt.plot(agg_store_item / agg_store_item.mean(0)[np.newaxis])
    plt.title("Items")
    plt.xlabel("Store")
    plt.ylabel("Relative Sales")
    plt.subplot(122)
    plt.plot(agg_store_item.T / agg_store_item.T.mean(0)[np.newaxis])
    plt.title("Stores")
    plt.xlabel("Item")
    plt.ylabel("Relative Sales")
    plt.show()
