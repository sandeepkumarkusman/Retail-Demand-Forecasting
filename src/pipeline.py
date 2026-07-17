"""Orchestration for the primary XYZT awesome solution."""

from pathlib import Path

import pandas as pd

from src.data_loader import load_xyzt_data
from src.features import expand_xyzt_date_features
from src.predict import predict_xyzt_awesome, round_xyzt_awesome_predictions
from src.train import fit_xyzt_awesome_model


def run_xyzt_awesome_pipeline(data_dir: str | Path = "data/raw") -> pd.DataFrame:
    """Run XYZT cells 3, 9, 23, 30, 34, and 35 through rounded predictions."""
    train, test, sample_sub = load_xyzt_data(data_dir)
    data = expand_xyzt_date_features(train)
    model = fit_xyzt_awesome_model(data)
    pred = predict_xyzt_awesome(test, sample_sub.copy(), model)
    return round_xyzt_awesome_predictions(pred)
