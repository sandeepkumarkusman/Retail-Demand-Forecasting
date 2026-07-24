"""Data loading for the retail demand forecasting solution."""

from pathlib import Path
import pandas as pd


def load_forecasting_data(
    data_dir: str | Path = "data/raw",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, test, and sample_submission inputs."""
    raw_path = Path(data_dir)

    train = pd.read_csv(raw_path / "train.csv", parse_dates=["date"])
    test = pd.read_csv(raw_path / "test.csv", parse_dates=["date"])
    sample_sub = pd.read_csv(raw_path / "sample_submission.csv")

    return train, test, sample_sub
