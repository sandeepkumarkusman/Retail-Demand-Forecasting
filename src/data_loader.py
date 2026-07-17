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


def load_fourth_place_data(
    data_dir: str | Path = "data/raw",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load 4th_place_sol_n.py inputs without XYZT date parsing."""
    raw_path = Path(data_dir)
    train = pd.read_csv(raw_path / "train.csv")
    test = pd.read_csv(raw_path / "test.csv")
    sample_submission = pd.read_csv(raw_path / "sample_submission.csv")
    return train, test, sample_submission


def load_polyfit_showcase_data(
    data_dir: str | Path = "data/raw",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load store-item-polyfit-showcase inputs as raw CSV frames."""
    raw_path = Path(data_dir)
    return pd.read_csv(raw_path / "train.csv"), pd.read_csv(raw_path / "test.csv")


def load_store_prediction_candidates(
    output_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load store-prediction.ipynb cells 27's four external submissions."""
    source_path = Path(output_dir)
    return tuple(
        pd.read_csv(source_path / f"weight_predictor_{number}.csv")
        for number in range(1, 5)
    )


def load_blend32_candidates(candidate_path: str | Path) -> pd.DataFrame:
    """Load blend-boosting notebook cell 1's 32-prediction matrix."""
    return pd.read_csv(candidate_path)
