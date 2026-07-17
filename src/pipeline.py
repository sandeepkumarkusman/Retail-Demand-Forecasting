"""Orchestration for the primary XYZT awesome solution."""

from pathlib import Path

import pandas as pd

from src.data_loader import (
    load_blend32_candidates,
    load_fourth_place_data,
    load_polyfit_showcase_data,
    load_store_prediction_candidates,
    load_xyzt_data,
)
from src.features import (
    add_polyfit_showcase_date_features,
    expand_store_prediction_date_features,
    expand_xyzt_date_features,
    prepare_fourth_place_data,
)
from src.predict import (
    blend32_weighted_average,
    blend_store_prediction_candidates,
    create_blend32_submission,
    create_fourth_place_submission,
    predict_polyfit_showcase,
    predict_fourth_place,
    predict_store_prediction_slightly_better,
    predict_store_prediction_weighted,
    predict_xyzt_awesome,
    round_store_prediction_submission,
    round_xyzt_awesome_predictions,
)
from src.train import (
    fit_fourth_place_factors,
    fit_polyfit_showcase_model,
    fit_store_prediction_unweighted_model,
    fit_store_prediction_weighted_model,
    fit_xyzt_awesome_model,
)


def run_xyzt_awesome_pipeline(data_dir: str | Path = "data/raw") -> pd.DataFrame:
    """Run XYZT cells 3, 9, 23, 30, 34, and 35 through rounded predictions."""
    train, test, sample_sub = load_xyzt_data(data_dir)
    data = expand_xyzt_date_features(train)
    model = fit_xyzt_awesome_model(data)
    pred = predict_xyzt_awesome(test, sample_sub.copy(), model)
    return round_xyzt_awesome_predictions(pred)


def write_xyzt_awesome_submission(
    submission: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Write XYZT cell 35's rounded submission CSV without an index column."""
    submission.to_csv(output_path, index=False)


def run_fourth_place_pipeline(data_dir: str | Path = "data/raw") -> pd.DataFrame:
    """Run 4th_place_sol_n.py lines 23-110 without changing its formula."""
    train, test, sample_submission = load_fourth_place_data(data_dir)
    data = prepare_fourth_place_data(train, test)
    data = fit_fourth_place_factors(data)
    data = predict_fourth_place(data)
    return create_fourth_place_submission(sample_submission, data)


def write_fourth_place_submission(
    submission: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Write 4th_place_sol_n.py's submission CSV without an index column."""
    submission.to_csv(output_path, index=False)


def run_polyfit_showcase_pipeline(data_dir: str | Path = "data/raw") -> pd.DataFrame:
    """Run store-item-polyfit-showcase.ipynb cells 0-2."""
    train, test = load_polyfit_showcase_data(data_dir)
    train = add_polyfit_showcase_date_features(train)
    test = add_polyfit_showcase_date_features(test)
    model = fit_polyfit_showcase_model(train)
    return predict_polyfit_showcase(test, model)


def write_polyfit_showcase_submission(
    submission: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Write store-item-polyfit-showcase.ipynb's submission CSV."""
    submission.to_csv(output_path, index=False)


def run_store_prediction_slightly_better_pipeline(
    data_dir: str | Path = "data/raw",
) -> pd.DataFrame:
    """Run store-prediction.ipynb's unweighted cells 2, 4-21."""
    train, test, sample_submission = load_xyzt_data(data_dir)
    data = expand_store_prediction_date_features(train)
    model = fit_store_prediction_unweighted_model(data)
    prediction = predict_store_prediction_slightly_better(test, sample_submission.copy(), model)
    return round_store_prediction_submission(prediction)


def run_store_prediction_weighted_pipeline(
    data_dir: str | Path = "data/raw",
) -> pd.DataFrame:
    """Run store-prediction.ipynb's final weighted predictor through cell 25."""
    train, test, sample_submission = load_xyzt_data(data_dir)
    data = expand_store_prediction_date_features(train)
    model = fit_store_prediction_weighted_model(data)
    prediction = predict_store_prediction_weighted(test, sample_submission.copy(), model)
    return round_store_prediction_submission(prediction)


def write_store_prediction_submission(
    submission: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Write one store-prediction.ipynb submission CSV without an index."""
    submission.to_csv(output_path, index=False)


def run_store_prediction_blend_pipeline(
    output_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run store-prediction.ipynb cells 27-33 using supplied external files."""
    sub1, sub2, sub3, sub4 = load_store_prediction_candidates(output_dir)
    return blend_store_prediction_candidates(sub1, sub2, sub3, sub4)


def run_blend32_pipeline(candidate_path: str | Path) -> pd.DataFrame:
    """Run blend-boosting-for-best-score-on-demand-forecast.ipynb cells 1 and 4."""
    candidates = load_blend32_candidates(candidate_path)
    candidates = blend32_weighted_average(candidates)
    return create_blend32_submission(candidates)


def write_blend32_submission(
    submission: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Write blend-boosting notebook's final submission CSV without an index."""
    submission.to_csv(output_path, index=False)


def run_active_solution(config_path: str | Path = "config/config.yaml") -> pd.DataFrame:
    """Dispatch the configured primary solution without altering its computation."""
    import yaml

    with Path(config_path).open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if config["active_solution"] != "xyzt_awesome":
        raise ValueError(
            "Only xyzt_awesome is implemented as an active pipeline; "
            f"received {config['active_solution']}"
        )

    solution_config = config["solutions"]["xyzt_awesome"]
    submission = run_xyzt_awesome_pipeline(solution_config["data_dir"])
    write_xyzt_awesome_submission(submission, solution_config["submission_path"])
    return submission
