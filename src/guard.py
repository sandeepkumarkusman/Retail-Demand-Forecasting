"""Read-only input guards and validation for the forecasting solution."""

from pathlib import Path
import pandas as pd


REQUIRED_FILES = ("train.csv", "test.csv", "sample_submission.csv")


def validate_required_files(data_dir: str | Path = "data/raw") -> None:
    """Raise when a raw input file is missing."""
    raw_path = Path(data_dir)
    missing_files = [filename for filename in REQUIRED_FILES if not (raw_path / filename).is_file()]
    if missing_files:
        raise FileNotFoundError(
            "Missing input file(s) in "
            f"{raw_path}: {', '.join(missing_files)}"
        )


def validate_ml_predictions(predictions: pd.DataFrame, config: dict = None) -> None:
    """
    Validate that the generated ML predictions conform to expected bounds
    and that quantiles maintain strict ordering: q05 <= point <= q95
    """
    if config is None:
        config = {}
        
    required_cols = ['id', 'sales', 'q05', 'q95']
    for col in required_cols:
        if col not in predictions.columns:
            raise ValueError(f"Predictions missing required column: {col}")
            
    # Check bounds
    thresholds = config.get('guard_thresholds', {'min_sales': 0, 'max_sales': 10000})
    min_sales = thresholds.get('min_sales', 0)
    max_sales = thresholds.get('max_sales', 10000)
    
    if (predictions['sales'] < min_sales).any() or (predictions['sales'] > max_sales).any():
        raise ValueError(f"Point predictions fall outside acceptable range [{min_sales}, {max_sales}]")
        
    # Check Quantile Ordering
    invalid_lower = predictions[predictions['q05'] > predictions['sales']]
    if not invalid_lower.empty:
        raise ValueError(f"Found {len(invalid_lower)} rows where q05 > point prediction")
        
    invalid_upper = predictions[predictions['sales'] > predictions['q95']]
    if not invalid_upper.empty:
        raise ValueError(f"Found {len(invalid_upper)} rows where point > q95 prediction")
        
    invalid_overall = predictions[predictions['q05'] > predictions['q95']]
    if not invalid_overall.empty:
        raise ValueError(f"Found {len(invalid_overall)} rows where q05 > q95 prediction")
