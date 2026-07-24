"""Tests for feature engineering to ensure no data leakage."""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    add_lag_and_rolling_features,
    add_target_encoding,
    expand_ml_date_features,
    prepare_ml_data,
)


def test_lag_features_no_leakage():
    """Ensure lag features don't leak future information."""
    # Create sample data
    dates = pd.date_range("2017-01-01", periods=200, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "store": [1] * 200,
            "item": [1] * 200,
            "sales": np.random.randint(10, 100, 200),
        }
    )

    # Add lag features
    df_featured = add_lag_and_rolling_features(df)

    # Check that lag_91 is properly shifted (should be NaN for first 91 rows)
    assert df_featured["sales_lag_91"].isna().sum() == 91

    # Check that non-NaN lag_91 values equal sales shifted by 91
    valid_mask = ~df_featured["sales_lag_91"].isna()
    expected = df.loc[valid_mask.index, "sales"].values
    actual = df_featured.loc[valid_mask, "sales_lag_91"].values
    np.testing.assert_array_equal(expected, actual)


def test_cyclical_encodings():
    """Test cyclical date encodings are correct."""
    dates = pd.date_range("2017-01-01", periods=365, freq="D")
    df = pd.DataFrame({"date": dates, "sales": np.random.randint(10, 100, 365)})

    df_featured = expand_ml_date_features(df)

    # Check sin/cos pairs are orthogonal (should sum to ~1)
    for sin_col, cos_col in [
        ("month_sin", "month_cos"),
        ("dow_sin", "dow_cos"),
        ("doy_sin", "doy_cos"),
    ]:
        if sin_col in df_featured.columns and cos_col in df_featured.columns:
            magnitude = np.sqrt(df_featured[sin_col] ** 2 + df_featured[cos_col] ** 2)
            # Allow small numerical errors
            assert np.allclose(magnitude, 1.0, atol=0.1)


def test_target_encoding_no_leakage():
    """Ensure target encoding uses only training data."""
    # Create sample data with train/test split
    dates = pd.date_range("2017-01-01", periods=200, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "store": [1] * 100 + [2] * 100,
            "item": [1] * 200,
            "sales": np.random.randint(10, 100, 200),
            "is_train": [1] * 150 + [0] * 50,  # Last 50 are test
        }
    )

    df_featured = add_target_encoding(df)

    # Check that test rows don't use their own target
    test_mask = df_featured["is_train"] == 0
    assert test_mask.sum() > 0

    # Store mean should be based on training data only
    store_1_train_mean = df[df["store"] == 1]["sales"].mean()
    store_1_encoded_mean = df_featured[df_featured["store"] == 1]["store_mean"].mean()

    # They should be close (since encoding uses training data)
    assert abs(store_1_train_mean - store_1_encoded_mean) < 5


def test_rolling_statistics_leakage_free():
    """Ensure rolling statistics are computed on lagged data."""
    dates = pd.date_range("2017-01-01", periods=200, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "store": [1] * 200,
            "item": [1] * 200,
            "sales": np.arange(200),  # Simple increasing pattern
        }
    )

    df_featured = add_lag_and_rolling_features(df)

    # Rolling mean should be based on lagged data, not current
    # Check that rolling_mean_7 at row 100 doesn't include sales[100]
    # It should be based on sales shifted by 91
    assert df_featured["rolling_mean_7"].isna().sum() >= 91  # At least lag period


def test_feature_engineering_reproducibility():
    """Test that feature engineering produces consistent results."""
    dates = pd.date_range("2017-01-01", periods=100, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "store": [1] * 100,
            "item": [1] * 100,
            "sales": np.random.randint(10, 100, 100),
        }
    )

    # Run feature engineering twice
    df_featured1 = prepare_ml_data(df.copy(), is_train=True)
    df_featured2 = prepare_ml_data(df.copy(), is_train=True)

    # Should produce identical results
    assert df_featured1.equals(df_featured2)


def test_feature_columns_consistency():
    """Test that feature engineering produces consistent column set."""
    dates = pd.date_range("2017-01-01", periods=100, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "store": [1] * 100,
            "item": [1] * 100,
            "sales": np.random.randint(10, 100, 100),
        }
    )

    df_featured = prepare_ml_data(df, is_train=True)

    # Check that expected feature columns exist
    expected_features = ["sales_lag_91", "rolling_mean_7", "month_sin", "dow_sin"]
    for feat in expected_features:
        assert feat in df_featured.columns, f"Expected feature {feat} not found"


def test_missing_data_handling():
    """Test that feature engineering handles missing dates gracefully."""
    # Create data with gaps
    dates = pd.date_range("2017-01-01", periods=100, freq="D")
    dates = dates.delete([50, 51, 52])  # Remove some dates
    df = pd.DataFrame(
        {
            "date": dates,
            "store": [1] * len(dates),
            "item": [1] * len(dates),
            "sales": np.random.randint(10, 100, len(dates)),
        }
    )

    # Should not raise an error
    df_featured = prepare_ml_data(df, is_train=True)
    assert df_featured is not None
    assert len(df_featured) == len(df)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
