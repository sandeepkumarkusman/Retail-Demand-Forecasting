"""Read-only input guards for the primary XYZT solution."""

from pathlib import Path

import pandas as pd


XYZT_REQUIRED_FILES = ("train.csv", "test.csv", "sample_submission.csv")
XYZT_TRAIN_COLUMNS = ("store", "item", "sales")
XYZT_TEST_COLUMNS = ("id", "store", "item")
XYZT_SUBMISSION_COLUMNS = ("id", "sales")


def validate_xyzt_required_files(data_dir: str | Path = "data/raw") -> None:
    """Raise when a primary-XYZT raw input file is missing."""
    raw_path = Path(data_dir)
    missing_files = [filename for filename in XYZT_REQUIRED_FILES if not (raw_path / filename).is_file()]
    if missing_files:
        raise FileNotFoundError(
            "Missing XYZT input file(s) in "
            f"{raw_path}: {', '.join(missing_files)}"
        )


def _validate_xyzt_frame_schema(
    frame: pd.DataFrame,
    *,
    frame_name: str,
    expected_columns: tuple[str, ...],
    requires_date_index: bool,
) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{frame_name} must be a pandas DataFrame")
    if tuple(frame.columns) != expected_columns:
        raise ValueError(
            f"{frame_name} columns must be {expected_columns} in that order; "
            f"received {tuple(frame.columns)}"
        )
    if requires_date_index and frame.index.name != "date":
        raise ValueError(f"{frame_name} index must be named 'date'")
    if frame.empty:
        raise ValueError(f"{frame_name} must contain at least one row")


def validate_xyzt_data(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample_submission: pd.DataFrame,
) -> None:
    """Validate XYZT input contracts without mutating any provided frame."""
    _validate_xyzt_frame_schema(
        train,
        frame_name="train",
        expected_columns=XYZT_TRAIN_COLUMNS,
        requires_date_index=True,
    )
    _validate_xyzt_frame_schema(
        test,
        frame_name="test",
        expected_columns=XYZT_TEST_COLUMNS,
        requires_date_index=True,
    )
    _validate_xyzt_frame_schema(
        sample_submission,
        frame_name="sample_submission",
        expected_columns=XYZT_SUBMISSION_COLUMNS,
        requires_date_index=False,
    )

    if train.reset_index().duplicated(["date", "store", "item"]).any():
        raise ValueError("train contains duplicate date/store/item rows")
    if test.reset_index().duplicated(["date", "store", "item"]).any():
        raise ValueError("test contains duplicate date/store/item rows")
    if test["id"].duplicated().any():
        raise ValueError("test contains duplicate IDs")
    if sample_submission["id"].duplicated().any():
        raise ValueError("sample_submission contains duplicate IDs")
    if len(test) != len(sample_submission):
        raise ValueError(
            "test and sample_submission row counts must match; "
            f"received {len(test)} and {len(sample_submission)}"
        )
    if not pd.Index(test["id"]).equals(pd.Index(sample_submission["id"])):
        raise ValueError("test and sample_submission IDs must match in the same order")
