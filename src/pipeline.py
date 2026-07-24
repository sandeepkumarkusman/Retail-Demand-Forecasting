"""Orchestration for the retail demand forecasting solution."""

import datetime
import json
import os
from pathlib import Path

import joblib
import pandas as pd

import src.backtesting as backtesting
import src.preprocessing as preprocessing
from src.data_loader import load_forecasting_data
from src.features import prepare_ml_data
from src.guard import validate_ml_predictions, validate_required_files
from src.metrics import calculate_ml_metrics
from src.predict import predict_lightgbm_model
from src.train import fit_lightgbm_model


def run_forecasting_pipeline(
    data_dir: str | Path = "data/raw", config: dict = None
) -> pd.DataFrame:
    """Run the global ML LightGBM multi-series pipeline end-to-end."""
    if config is None:
        config = {}

    validate_required_files(str(data_dir))
    train_raw, test_raw, sample_sub = load_forecasting_data(data_dir)

    processed_dir = config.get("processed_dir", "data/processed")
    os.makedirs(processed_dir, exist_ok=True)

    # Preprocessing padding (wire dead code)
    print("Preprocessing raw data...")
    preprocessing.validate_schema(train_raw, ["date", "store", "item", "sales"])
    train_raw = preprocessing.cast_types(train_raw)
    test_raw = preprocessing.cast_types(test_raw)
    train_raw = preprocessing.clip_sales_outliers(
        train_raw, lower_quantile=0.001, upper_quantile=0.999
    )

    # Combine train and test so that lag features are computed across the boundary
    train_raw = train_raw.copy()
    test_raw = test_raw.copy()
    train_raw["is_train"] = 1
    test_raw["is_train"] = 0

    df = pd.concat([train_raw, test_raw], ignore_index=True)

    cache_path = Path(processed_dir) / "features_cache.parquet"
    use_cache = config.get("cache_features", False)

    if use_cache and cache_path.exists():
        print("Loading ML features from cache...")
        df_features = pd.read_parquet(cache_path)
    else:
        print("Preparing ML features for training and inference...")
        df_features = prepare_ml_data(df, is_train=False)

        # Drop any residual index/reset artifacts before splitting
        for col_to_drop in ["index", "level_0"]:
            if col_to_drop in df_features.columns:
                df_features = df_features.drop(columns=[col_to_drop])

        if use_cache:
            print("Saving ML features to cache...")
            df_features.to_parquet(cache_path, index=False)

    train = df_features[df_features["is_train"] == 1].copy()
    test = df_features[df_features["is_train"] == 0].copy()

    # Run proper 3-fold Walk-Forward Validation
    print("Running 3-fold Walk-Forward Cross Validation...")
    # Fix dates to avoid categorical casting issues in cv
    train["date"] = pd.to_datetime(train["date"])
    cv_metrics_df = backtesting.evaluate_walk_forward(
        df=train,
        config=config,
        model_fit_fn=fit_lightgbm_model,
        model_predict_fn=predict_lightgbm_model,
        n_folds=3,
    )

    mean_cv_metrics = (
        cv_metrics_df[
            [
                "smape",
                "mae",
                "rmse",
                "pinball_q05",
                "pinball_q95",
                "interval_coverage_pct",
            ]
        ]
        .mean()
        .to_dict()
    )
    print(f"CV Average SMAPE: {mean_cv_metrics['smape']:.3f}%")

    # Fit the final model on full available training data
    print("Fitting final models on full training data...")
    model_dict = fit_lightgbm_model(train, config)

    # Generate predictions for the test period
    print("Generating predictions...")
    prediction = predict_lightgbm_model(test, sample_sub, model_dict)

    # Validate output integrity
    validate_ml_predictions(prediction, config)

    # Export Feature Importance
    print("Exporting feature importance...")
    point_model = model_dict["models"]["point_model"]
    features = model_dict["features"]
    importance = point_model.feature_importances_
    fi_df = pd.DataFrame({"feature": features, "importance": importance})
    fi_df = fi_df.sort_values("importance", ascending=False)

    os.makedirs("outputs", exist_ok=True)
    fi_df.to_csv("outputs/feature_importance.csv", index=False)

    # Persist model artifacts
    os.makedirs("models", exist_ok=True)
    joblib.dump(point_model, "models/point_model.joblib")
    joblib.dump(model_dict["models"]["quantile_low"], "models/quantile_low_q05.joblib")
    joblib.dump(
        model_dict["models"]["quantile_high"], "models/quantile_high_q95.joblib"
    )

    metadata = {
        "train_date": datetime.datetime.now().isoformat(),
        "feature_count": len(features),
        "val_smape": mean_cv_metrics.get("smape"),
        "params": {
            "n_estimators": config.get("n_estimators"),
            "learning_rate": config.get("learning_rate"),
            "random_seed": config.get("random_seed"),
        },
    }
    with open("models/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    with open("models/metrics.json", "w") as f:
        json.dump(mean_cv_metrics, f, indent=2)

    prediction.to_parquet("outputs/predictions.parquet", index=False)

    submission = sample_sub.copy()
    id_to_sales = prediction.set_index("id")["sales"]
    submission["sales"] = submission["id"].map(id_to_sales).round().astype(int)
    return submission


def run_active_solution(config_path: str | Path = "config/config.yaml") -> pd.DataFrame:
    """Dispatch the configured primary solution."""
    import yaml

    with Path(config_path).open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    active_solution = config["active_solution"]
    solution_config = config["solutions"][active_solution]

    if active_solution == "default_pipeline":
        submission = run_forecasting_pipeline(
            solution_config["data_dir"], solution_config
        )
        submission.to_csv(solution_config["submission_path"], index=False)
    else:
        raise ValueError(f"Unsupported active_solution: {active_solution}")

    return submission


if __name__ == "__main__":
    generated_predictions = run_active_solution()
    print(f"Wrote {len(generated_predictions)} predictions to the configured path.")
