"""Tests for input and output validation guards."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.guard import validate_required_files, validate_ml_predictions


class TestGuard(unittest.TestCase):
    def test_validates_required_files(self) -> None:
        """It raises FileNotFoundError when required files are missing."""
        with TemporaryDirectory() as temp_dir:
            raw_dir = Path(temp_dir)

            with self.assertRaisesRegex(FileNotFoundError, "Missing input file"):
                validate_required_files(raw_dir)

            (raw_dir / "train.csv").touch()
            (raw_dir / "test.csv").touch()
            (raw_dir / "sample_submission.csv").touch()

            # Should not raise
            validate_required_files(raw_dir)

    def test_validate_ml_predictions_bounds(self):
        """It validates that point predictions are within expected bounds."""
        preds = pd.DataFrame(
            {
                "id": [1, 2],
                "sales": [-5, 500],  # -5 is invalid
                "q05": [0, 450],
                "q95": [10, 550],
            }
        )

        with self.assertRaisesRegex(ValueError, "outside acceptable range"):
            validate_ml_predictions(preds)

    def test_validate_ml_predictions_ordering(self):
        """It validates quantile ordering."""
        preds = pd.DataFrame(
            {
                "id": [1],
                "sales": [500],
                "q05": [600],  # invalid, q05 > sales
                "q95": [700],
            }
        )
        with self.assertRaisesRegex(ValueError, "where q05 > point prediction"):
            validate_ml_predictions(preds)
