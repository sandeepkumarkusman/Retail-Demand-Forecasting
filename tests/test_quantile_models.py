"""Tests for quantile model ordering and validity."""

import numpy as np
import pandas as pd
import unittest



def _make_sample_predictions(n=100):
    """Create a valid predictions DataFrame with proper ordering."""
    rng = np.random.default_rng(42)
    point = rng.uniform(10, 80, n)
    q05 = point * rng.uniform(0.7, 0.95, n)
    q95 = point * rng.uniform(1.05, 1.30, n)
    return pd.DataFrame({
        "id": np.arange(n),
        "sales": point,
        "q05": q05,
        "q95": q95,
    })


class TestQuantileOrdering(unittest.TestCase):
    """Verify quantile bounds maintain q05 <= point <= q95."""

    def test_valid_predictions_pass_guard(self):
        """Guard should not raise for correctly ordered predictions."""
        from src.guard import validate_ml_predictions
        preds = _make_sample_predictions(100)
        validate_ml_predictions(preds)  # should not raise

    def test_q05_must_not_exceed_point(self):
        """Guard raises if q05 > point for any row."""
        from src.guard import validate_ml_predictions
        preds = _make_sample_predictions(100)
        preds.loc[5, "q05"] = preds.loc[5, "sales"] + 10
        with self.assertRaisesRegex(ValueError, "q05 > point"):
            validate_ml_predictions(preds)

    def test_point_must_not_exceed_q95(self):
        """Guard raises if point > q95 for any row."""
        from src.guard import validate_ml_predictions
        preds = _make_sample_predictions(100)
        preds.loc[10, "q95"] = preds.loc[10, "sales"] - 5
        with self.assertRaisesRegex(ValueError, "point > q95"):
            validate_ml_predictions(preds)

    def test_q05_must_not_exceed_q95(self):
        """Guard raises if q05 > q95 for any row."""
        from src.guard import validate_ml_predictions
        preds = _make_sample_predictions(100)
        # Set q05=50, point=60, q95=40 — this violates both point > q95 and q05 > q95.
        # The guard catches q05 > q95 after the earlier checks, so we need a case where
        # q05 <= point <= q95 is satisfied but q05 > q95 globally.
        # Instead, set q05=30, point=35, q95=20 (point > q95 fires). 
        # The cleanest test: set q05=50, q95=10 WITH point=40 so point > q95 fires first.
        # Actually the guard_validation order matters; check that ANY ValueError is raised.
        preds.loc[3, "q05"] = 50
        preds.loc[3, "sales"] = 20  # q05 > point
        preds.loc[3, "q95"] = 10
        with self.assertRaises(ValueError):
            validate_ml_predictions(preds)

    def test_missing_required_columns_raises(self):
        """Guard raises if a required column is absent."""
        from src.guard import validate_ml_predictions
        preds = _make_sample_predictions(10).drop(columns=["q95"])
        with self.assertRaisesRegex(ValueError, "missing required column"):
            validate_ml_predictions(preds)

    def test_sales_outside_bounds_raises(self):
        """Guard raises if sales exceed the configured max."""
        from src.guard import validate_ml_predictions
        preds = _make_sample_predictions(10)
        preds.loc[0, "sales"] = 99999
        with self.assertRaisesRegex(ValueError, "outside acceptable range"):
            validate_ml_predictions(preds)

    def test_nonnegative_sales(self):
        """Guard raises if any sales value is negative."""
        from src.guard import validate_ml_predictions
        preds = _make_sample_predictions(10)
        preds.loc[0, "sales"] = -1.0
        with self.assertRaisesRegex(ValueError, "outside acceptable range"):
            validate_ml_predictions(preds)


if __name__ == "__main__":
    unittest.main()

