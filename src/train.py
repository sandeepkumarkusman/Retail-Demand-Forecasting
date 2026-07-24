"""Training components for the LightGBM forecasting solution."""

import numpy as np
import pandas as pd
import lightgbm as lgb
from src.features import prepare_ml_data


def smape_lgb(y_true, y_pred):
    """Custom SMAPE evaluation metric for LightGBM sklearn API."""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 200.0
    diff = np.abs(y_true - y_pred) / np.maximum(denominator, 1e-8)
    return "smape", np.nanmean(diff), False


def fit_lightgbm_model(data: pd.DataFrame, config: dict = None) -> dict[str, object]:
    """
    Fits three LightGBM models (Point, Quantile Low, Quantile High).
    Accepts either raw train data (will featurize internally) or pre-featurized data.
    """
    if config is None:
        config = {}

    # Check if data is already featurized (has lag features)
    if "sales_lag_91" not in data.columns:
        print("Preparing ML features for training...")
        train_df = prepare_ml_data(data, is_train=True)
    else:
        print("Using pre-featurized data...")
        train_df = data[data["sales"].notna()].copy()

    # Validation: Walk-forward matching test seasonality (last 90 days of available training data)
    # This avoids hard-coding '2017-01-01'
    max_date = train_df["date"].max()
    val_start = max_date - pd.Timedelta(days=89)  # 90 days total
    val_mask = train_df["date"] >= val_start

    val_split = train_df[val_mask]
    train_split = train_df[~val_mask]

    features = [
        c for c in train_df.columns if c not in ["date", "sales", "is_train", "id"]
    ]

    X_train, y_train = train_split[features], train_split["sales"]
    X_val, y_val = val_split[features], val_split["sales"]

    print(
        f"Training LightGBM on {len(X_train)} samples, validating on {len(X_val)} samples."
    )
    print(f"Features ({len(features)}): {features[:5]}...")

    base_params = {
        "boosting_type": "gbdt",
        "num_leaves": 127,
        "learning_rate": config.get("learning_rate", 0.05),
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "max_depth": -1,
        "verbose": -1,
        "n_jobs": -1,
        "seed": config.get("random_seed", 42),
    }

    n_estimators = config.get("n_estimators", 100)
    alphas = config.get("quantile_alphas", [0.05, 0.95])

    models = {}

    # 1. Train Point Model (MAE)
    print("--- Training Point Model (MAE) ---")
    point_params = base_params.copy()
    point_params["objective"] = "regression_l1"
    point_params["metric"] = "mae"

    point_model = lgb.LGBMRegressor(**point_params, n_estimators=n_estimators)
    point_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=smape_lgb,
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    models["point_model"] = point_model

    # 2. Train Quantile Low (q05)
    print(f"--- Training Quantile Low Model (alpha={alphas[0]}) ---")
    qlow_params = base_params.copy()
    qlow_params["objective"] = "quantile"
    qlow_params["alpha"] = alphas[0]

    qlow_model = lgb.LGBMRegressor(**qlow_params, n_estimators=n_estimators)
    qlow_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    models["quantile_low"] = qlow_model

    # 3. Train Quantile High (q95)
    print(f"--- Training Quantile High Model (alpha={alphas[1]}) ---")
    qhigh_params = base_params.copy()
    qhigh_params["objective"] = "quantile"
    qhigh_params["alpha"] = alphas[1]

    qhigh_model = lgb.LGBMRegressor(**qhigh_params, n_estimators=n_estimators)
    qhigh_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    models["quantile_high"] = qhigh_model

    return {
        "models": models,
        "features": features,
        "y_val_true": y_val.values,
        "X_val": X_val,
        "val_date": val_split["date"].values,
        "val_store": val_split["store"].values,
        "val_item": val_split["item"].values,
    }
