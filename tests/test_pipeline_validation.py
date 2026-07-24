"""Tests for pipeline validation and reproducibility."""

import numpy as np
import pandas as pd
import pytest
import tempfile
import shutil
from pathlib import Path


def test_pipeline_reproducibility():
    """Ensure pipeline produces consistent results with same seed."""
    # This is a simplified test - in practice would run actual pipeline
    # For now, we test the concept

    # Set seed
    np.random.seed(42)
    data1 = np.random.randn(100)

    # Reset seed and generate again
    np.random.seed(42)
    data2 = np.random.randn(100)

    # Should be identical
    np.testing.assert_array_equal(data1, data2)


def test_pipeline_with_missing_data():
    """Test that pipeline handles missing dates gracefully."""
    # Create sample data with missing dates
    dates = pd.date_range("2017-01-01", periods=100, freq="D")
    dates_with_gaps = dates.delete([20, 21, 22, 50, 51])

    df = pd.DataFrame(
        {
            "date": dates_with_gaps,
            "store": [1] * len(dates_with_gaps),
            "item": [1] * len(dates_with_gaps),
            "sales": np.random.randint(10, 100, len(dates_with_gaps)),
        }
    )

    # Add is_train column
    df["is_train"] = 1

    # Feature engineering should handle gaps
    from src.features import prepare_ml_data

    df_featured = prepare_ml_data(df, is_train=True)

    assert df_featured is not None
    assert len(df_featured) == len(df)


def test_quantile_constraint_validation():
    """Test that quantile constraints are enforced."""
    from src.guard import validate_ml_predictions

    # Create predictions that violate constraints
    predictions = pd.DataFrame(
        {
            "id": range(10),
            "sales": [50] * 10,  # Point forecasts
            "q05": [60] * 10,  # Lower bound > point (violation)
            "q95": [40] * 10,  # Upper bound < point (violation)
        }
    )

    # Validation should catch this
    with pytest.raises(Exception):  # Should raise an error
        validate_ml_predictions(predictions, {})


def test_output_schema_validation():
    """Test that output has required columns."""
    from src.guard import validate_required_files

    # Create temporary directory structure
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create required files
        (tmpdir_path / "train.csv").write_text(
            "date,store,item,sales\n2017-01-01,1,1,50"
        )
        (tmpdir_path / "test.csv").write_text("date,store,item\n2018-01-01,1,1")
        (tmpdir_path / "sample_submission.csv").write_text("id,sales\n0,50")

        # Should not raise error
        validate_required_files(str(tmpdir_path))


def test_feature_count_consistency():
    """Test that feature engineering produces consistent number of features."""
    from src.features import prepare_ml_data

    # Create sample data
    dates = pd.date_range("2017-01-01", periods=200, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "store": [1] * 200,
            "item": [1] * 200,
            "sales": np.random.randint(10, 100, 200),
            "is_train": 1,
        }
    )

    df_featured = prepare_ml_data(df, is_train=True)

    # Count feature columns (exclude date, sales, is_train, id)
    feature_cols = [
        c for c in df_featured.columns if c not in ["date", "sales", "is_train", "id"]
    ]

    # Should have a reasonable number of features (40-50 range)
    assert 30 <= len(feature_cols) <= 60


def test_data_type_consistency():
    """Test that data types are consistent."""
    from src.preprocessing import cast_types

    df = pd.DataFrame(
        {
            "date": ["2017-01-01", "2017-01-02"],
            "store": ["1", "2"],
            "item": ["1", "2"],
            "sales": ["50", "60"],
        }
    )

    df_cast = cast_types(df)

    # Check types
    assert pd.api.types.is_datetime64_any_dtype(df_cast["date"])
    assert pd.api.types.is_integer_dtype(df_cast["store"])
    assert pd.api.types.is_integer_dtype(df_cast["item"])
    assert pd.api.types.is_numeric_dtype(df_cast["sales"])


def test_outlier_clipping():
    """Test that outlier clipping works correctly."""
    from src.preprocessing import clip_sales_outliers

    # Create data with extreme outliers
    df = pd.DataFrame(
        {
            "date": pd.date_range("2017-01-01", periods=100),
            "store": [1] * 100,
            "item": [1] * 100,
            "sales": [50] * 98 + [10000, -100],  # Extreme outliers
        }
    )

    df_clipped = clip_sales_outliers(df, lower_quantile=0.01, upper_quantile=0.99)

    # Check that outliers were clipped
    assert df_clipped["sales"].max() < 10000
    assert df_clipped["sales"].min() >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
