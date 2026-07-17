"""Diagnostic plots from the XYZT source notebook only."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_xyzt_year_patterns(data: pd.DataFrame) -> None:
    """Reproduce XYZT notebook cell 11."""
    agg_year_item = pd.pivot_table(
        data, index="year", columns="item", values="sales", aggfunc=np.mean
    ).values
    agg_year_store = pd.pivot_table(
        data, index="year", columns="store", values="sales", aggfunc=np.mean
    ).values

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


def plot_xyzt_month_patterns(data: pd.DataFrame) -> None:
    """Reproduce XYZT notebook cell 13."""
    agg_month_item = pd.pivot_table(
        data, index="month", columns="item", values="sales", aggfunc=np.mean
    ).values
    agg_month_store = pd.pivot_table(
        data, index="month", columns="store", values="sales", aggfunc=np.mean
    ).values

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


def plot_xyzt_dayofweek_patterns(data: pd.DataFrame) -> None:
    """Reproduce XYZT notebook cell 15."""
    agg_dow_item = pd.pivot_table(
        data, index="dayofweek", columns="item", values="sales", aggfunc=np.mean
    ).values
    agg_dow_store = pd.pivot_table(
        data, index="dayofweek", columns="store", values="sales", aggfunc=np.mean
    ).values

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


def plot_xyzt_calendar_interactions(data: pd.DataFrame) -> None:
    """Reproduce XYZT notebook cell 17."""
    agg_dow_month = pd.pivot_table(
        data, index="dayofweek", columns="month", values="sales", aggfunc=np.mean
    ).values
    agg_month_year = pd.pivot_table(
        data, index="month", columns="year", values="sales", aggfunc=np.mean
    ).values
    agg_dow_year = pd.pivot_table(
        data, index="dayofweek", columns="year", values="sales", aggfunc=np.mean
    ).values

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


def plot_xyzt_store_item_patterns(data: pd.DataFrame) -> None:
    """Reproduce XYZT notebook cell 20."""
    agg_store_item = pd.pivot_table(
        data, index="store", columns="item", values="sales", aggfunc=np.mean
    ).values

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
