import unittest

import numpy as np
import pandas as pd

from src.metrics import (
    smape_df_prophet_dumb,
    smape_fourth_place,
    smape_prophet_dumb,
)


class TestNotebookSpecificMetrics(unittest.TestCase):
    def test_prophet_dumb_smape_matches_source_formula(self) -> None:
        actual = smape_prophet_dumb(np.array([10.0, 20.0]), np.array([20.0, 10.0]))

        self.assertAlmostEqual(actual, 200.0 / 3.0)

    def test_prophet_dumb_signed_smape_matches_source_formula(self) -> None:
        actual = smape_prophet_dumb(
            np.array([10.0, 20.0]),
            np.array([20.0, 10.0]),
            signed=True,
        )

        self.assertEqual(actual, 0.0)

    def test_prophet_dumb_dataframe_wrapper_uses_y_and_yhat(self) -> None:
        frame = pd.DataFrame({"y": [10.0, 20.0], "yhat": [20.0, 10.0]})

        self.assertAlmostEqual(smape_df_prophet_dumb(frame), 200.0 / 3.0)

    def test_fourth_place_smape_preserves_zero_denominator_handling(self) -> None:
        actual = smape_fourth_place(
            np.array([0.0, 10.0]),
            np.array([0.0, 20.0]),
        )

        self.assertEqual(actual, 100.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
