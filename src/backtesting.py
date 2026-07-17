"""Precomputed CV analysis from the Prophet/dumb reference notebook.

The source notebook does not contain CV model-generation logic. These functions
only load and aggregate already-generated prediction files.
"""

from pathlib import Path
from typing import Mapping

import pandas as pd

from src.metrics import smape_df_prophet_dumb


def load_prophet_cv_reference(cv_path: str | Path) -> pd.DataFrame:
    """Load Prophet CV predictions exactly as reference notebook cell 31."""
    return pd.read_csv(cv_path, index_col=[0, 1, 2, 3])


def load_dumb_cv_reference(cv_path: str | Path) -> pd.DataFrame:
    """Load dumb-model CV predictions using the reference notebook convention."""
    return pd.read_csv(cv_path, index_col=[0, 1, 2, 3])


def aggregate_prophet_cv_smape_reference(data_cv: pd.DataFrame) -> pd.Series:
    """Reproduce the fold/sample SMAPE aggregation in reference notebook cell 32."""
    return data_cv.groupby(["cv_fold", "sample"]).apply(smape_df_prophet_dumb)


def create_dumb_cv_smape_aggregate_reference(
    df_cv_dict: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Reproduce reference notebook cell 54's dumb-model CV comparison table."""
    df_smape_aggr = pd.concat(
        (
            value.groupby(["cv_fold"])
            .apply(smape_df_prophet_dumb)
            .rename(key)
            for key, value in df_cv_dict.items()
        ),
        axis=1,
    )
    df_smape_aggr.index = [
        value - pd.Timedelta(days=1)
        for value in pd.to_datetime(df_smape_aggr.index)
    ]
    return df_smape_aggr
