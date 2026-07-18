"""Shared utilities and helper functions."""

import datetime as dt
import calendar
import os
import sys
from contextlib import contextmanager

import pandas as pd


class timer_gen:
    """A simple timer generator."""

    def __init__(self):
        self.t0 = dt.datetime.now()
        self.t1, self.t2 = None, self.t0

    def __iter__(self):
        return self

    def __next__(self):
        self.t1, self.t2 = self.t2, dt.datetime.now()
        return "<timer = {} ({})>".format(self.t2 - self.t1, self.t2 - self.t0)


@contextmanager
def suppress_stdout(on: bool = True):
    """Context manager to optionally suppress stdout."""
    if on:
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout
    else:
        yield


def display_side_by_side(*args):
    """Helper to display pandas DataFrames side-by-side in notebooks."""
    from IPython.display import display_html

    html_str = ""
    for df in args:
        if type(df) == pd.Series:
            df = pd.DataFrame(df, columns=["value"])
        html_str += df.to_html()
    display_html(html_str.replace("table", 'table style="display:inline"'), raw=True)


def days_in_month(my_date):
    """Returns the number of days in the month for a given date."""
    my_year = my_date.year
    my_month = my_date.month
    return calendar.monthrange(my_year, my_month)[1]
