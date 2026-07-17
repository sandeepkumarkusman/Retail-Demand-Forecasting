"""Data loading for the primary XYZT solution only."""

from pathlib import Path

import pandas as pd


def load_xyzt_data(data_dir: str | Path = "data/raw") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load competition inputs using XYZT notebook parsing and indexing behavior.

    This reproduces cell 3 of ``keeping-it-simple-by-xyzt.ipynb`` with the
    project raw-data directory in place of the notebook's ``../input`` path.
    No validation or transformation is performed here.
    """
    raw_path = Path(data_dir)
    train = pd.read_csv(
        raw_path / "train.csv",
        low_memory=False,
        parse_dates=["date"],
        index_col=["date"],
    )
    test = pd.read_csv(
        raw_path / "test.csv",
        low_memory=False,
        parse_dates=["date"],
        index_col=["date"],
    )
    sample_submission = pd.read_csv(raw_path / "sample_submission.csv")
    return train, test, sample_submission
