from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.backtesting import (
    aggregate_prophet_cv_smape_reference,
    create_dumb_cv_smape_aggregate_reference,
    load_dumb_cv_reference,
    load_prophet_cv_reference,
)


def _cv_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cv_fold": ["2017-04-01", "2017-04-01", "2017-04-01", "2017-04-01"],
            "sample": ["in", "in", "oos", "oos"],
            "y": [10.0, 20.0, 10.0, 20.0],
            "yhat": [10.0, 10.0, 20.0, 20.0],
        }
    )


class TestReferenceBacktesting(unittest.TestCase):
    def test_cv_loaders_preserve_four_level_index_behavior(self) -> None:
        source = pd.DataFrame(
            {
                "cv_fold": ["2017-04-01"],
                "sample": ["oos"],
                "num": [1],
                "ds": ["2017-01-01"],
                "y": [10.0],
                "yhat": [11.0],
            }
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "cv.csv"
            source.to_csv(path, index=False)

            prophet_loaded = load_prophet_cv_reference(path)
            dumb_loaded = load_dumb_cv_reference(path)

        self.assertEqual(prophet_loaded.index.nlevels, 4)
        pd.testing.assert_frame_equal(prophet_loaded, dumb_loaded)

    def test_prophet_fold_sample_aggregation_matches_source_formula(self) -> None:
        result = aggregate_prophet_cv_smape_reference(_cv_frame())

        self.assertAlmostEqual(result.loc[("2017-04-01", "in")], 100.0 / 3.0)
        self.assertAlmostEqual(result.loc[("2017-04-01", "oos")], 100.0 / 3.0)

    def test_dumb_cv_aggregate_shifts_fold_dates_back_one_day(self) -> None:
        result = create_dumb_cv_smape_aggregate_reference(
            {"Dumb A": _cv_frame(), "Dumb B": _cv_frame()}
        )

        self.assertEqual(list(result.columns), ["Dumb A", "Dumb B"])
        self.assertEqual(result.index[0], pd.Timestamp("2017-03-31"))
        self.assertAlmostEqual(result.iloc[0, 0], 100.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
