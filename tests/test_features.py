import unittest

import pandas as pd

from src.features import expand_xyzt_date_features


class TestXYZTDateFeatures(unittest.TestCase):
    def test_expands_xyzt_date_features_without_mutating_source(self) -> None:
        source = pd.DataFrame(
            {"store": [1, 2], "item": [1, 2], "sales": [13, 11]},
            index=pd.DatetimeIndex(["2013-01-01", "2016-02-29"], name="date"),
        )
        original = source.copy(deep=True)

        result = expand_xyzt_date_features(source)

        pd.testing.assert_frame_equal(source, original)
        self.assertIsNot(result, source)
        self.assertEqual(
            list(result.columns),
            ["store", "item", "sales", "day", "month", "year", "dayofweek"],
        )
        self.assertEqual(result["day"].tolist(), [1, 29])
        self.assertEqual(result["month"].tolist(), [1, 2])
        self.assertEqual(result["year"].tolist(), [2013, 2016])
        self.assertEqual(result["dayofweek"].tolist(), [1, 0])


if __name__ == "__main__":
    unittest.main()
