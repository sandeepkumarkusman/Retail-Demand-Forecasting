"""Inference components for the LightGBM forecasting solution."""

import numpy as np
import pandas as pd


def predict_lightgbm_model(
    test: pd.DataFrame, 
    sample_submission: pd.DataFrame, 
    model_dict: dict[str, object]
) -> pd.DataFrame:
    """
    Generate Point, Q05, and Q95 predictions using the multi-model dictionary.
    """
    models = model_dict['models']
    features = model_dict['features']
    
    # We assume `test` contains the necessary feature columns because `pipeline.py` prepared it.
    X_test = test[features]
    
    point_preds = models['point_model'].predict(X_test)
    qlow_preds = models['quantile_low'].predict(X_test)
    qhigh_preds = models['quantile_high'].predict(X_test)
    
    # Clip to 0
    point_preds = np.clip(point_preds, 0, None)
    qlow_preds = np.clip(qlow_preds, 0, None)
    qhigh_preds = np.clip(qhigh_preds, 0, None)
    
    submission = pd.DataFrame()
    submission['id'] = test['id'].astype(int)
    submission['sales'] = point_preds
    submission['q05'] = qlow_preds
    submission['q95'] = qhigh_preds
    
    # Sort back by ID in case test was sorted by store/item/date
    submission = submission.sort_values('id').reset_index(drop=True)
    
    return submission
