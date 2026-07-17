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
    predict_unavailable_artifact_fallback_baseline,
    predict_xyzt_awesome,
    round_unavailable_artifact_fallback_submission,
    round_store_prediction_submission,
    round_xyzt_awesome_predictions,
)
from src.train import (
    fit_fourth_place_factors,
    fit_polyfit_showcase_model,
    fit_store_prediction_unweighted_model,
    fit_store_prediction_weighted_model,
    fit_unavailable_artifact_fallback_baseline,
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


def run_unavailable_artifact_fallback_pipeline(
    data_dir: str | Path = "data/raw",
) -> pd.DataFrame:
    """Run the fallback for source components whose original artifacts are absent.

    This is intentionally separate from every notebook implementation. It is a
    runnable project fallback, not a claimed reproduction of Prophet or either
    external-prediction ensemble.
    """
    train, test, sample_submission = load_xyzt_data(data_dir)
    model = fit_unavailable_artifact_fallback_baseline(train)
    prediction = predict_unavailable_artifact_fallback_baseline(
        test, sample_submission, model
    )
    return round_unavailable_artifact_fallback_submission(prediction)


def run_prophet_dumb_reference_fallback_pipeline(
    data_dir: str | Path = "data/raw",
) -> pd.DataFrame:
    """Provide a runnable fallback for missing legacy Prophet source artifacts."""
    return run_unavailable_artifact_fallback_pipeline(data_dir)


def run_store_prediction_blend_fallback_pipeline(
    data_dir: str | Path = "data/raw",
) -> pd.DataFrame:
    """Run the missing-four-candidate blend fallback with identical stand-ins.

    All four stand-ins are the same deterministic fallback prediction. The
    original blend formula is therefore exercised without fabricating distinct
    Kaggle candidate files or implying original ensemble performance.
    """
    fallback = run_unavailable_artifact_fallback_pipeline(data_dir)
    sub1 = fallback.copy()
    sub2 = fallback.copy()
    sub3 = fallback.copy()
    sub4 = fallback.copy()
    _, final_submission = blend_store_prediction_candidates(sub1, sub2, sub3, sub4)
    return final_submission


def run_blend32_fallback_pipeline(
    data_dir: str | Path = "data/raw",
) -> pd.DataFrame:
    """Run the missing-32-candidate blend fallback with identical stand-ins.

    The notebook's blend requires exactly 45,000 competition rows because it
    creates ids from ``range(45000)``. Its original calculation is retained;
    only unavailable candidate values are substituted with the documented
    deterministic fallback.
    """
    fallback = run_unavailable_artifact_fallback_pipeline(data_dir)
    candidates = pd.DataFrame(
        {str(column): fallback["sales"].astype(float).to_numpy() for column in range(32)}
    )
    return create_blend32_submission(blend32_weighted_average(candidates))


def write_unavailable_artifact_fallback_submission(
    submission: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Write a clearly labeled fallback submission without an index column."""
    submission.to_csv(output_path, index=False)


def run_active_solution(config_path: str | Path = "config/config.yaml") -> pd.DataFrame:
    """Dispatch the configured primary solution without altering its computation."""
    import yaml

    with Path(config_path).open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    active_solution = config["active_solution"]
    solution_config = config["solutions"][active_solution]

    if active_solution == "xyzt_awesome":
        submission = run_xyzt_awesome_pipeline(solution_config["data_dir"])
        write_xyzt_awesome_submission(submission, solution_config["submission_path"])
    elif active_solution == "prophet_dumb_reference_fallback":
        submission = run_prophet_dumb_reference_fallback_pipeline(solution_config["data_dir"])
        write_unavailable_artifact_fallback_submission(
            submission, solution_config["submission_path"]
        )
    elif active_solution == "store_prediction_blend_fallback":
        submission = run_store_prediction_blend_fallback_pipeline(solution_config["data_dir"])
        write_unavailable_artifact_fallback_submission(
            submission, solution_config["submission_path"]
        )
    elif active_solution == "blend_32_candidates_fallback":
        submission = run_blend32_fallback_pipeline(solution_config["data_dir"])
        write_unavailable_artifact_fallback_submission(
            submission, solution_config["submission_path"]
        )
    else:
        raise ValueError(f"Unsupported active_solution: {active_solution}")

    return submission


if __name__ == "__main__":
    generated_submission = run_active_solution()
    print(f"Wrote {len(generated_submission)} predictions to the configured submission path.")
