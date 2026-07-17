# Final Report

## Scope and implementation principle

This project preserves the analyzed Store Item Demand Forecasting Challenge notebooks as independent Kaggle solutions. Refactoring will preserve each source's algorithm, feature order, parameters, training order, inference behavior, and output conventions. Implementations must not be merged into hybrid models.

## Analyzed notebook inventory

| Source | Classification | Confirmed role |
|---|---|---|
| `eda-prophet-winning-solution-3-0.ipynb` | Utility/reference | EDA, Prophet diagnostics, reference dumb-model classes, and precomputed CV analysis. |
| `eda-of-total-sales.ipynb` | EDA only | Aggregate total-sales structure and residual analysis. |
| `4th_place_sol_n.py` | Alternative implementation | Standalone global-factor submission pipeline. |
| `blend-boosting-for-best-score-on-demand-forecast.ipynb` | Ensemble-only | Fixed weighted blend of 32 externally supplied prediction vectors. |
| `keeping-it-simple-by-xyzt.ipynb` | Primary executable pipeline | Complete factor-model inference and submission workflow. |
| `store-item-polyfit-showcase.ipynb` | Alternative implementation | Weighted-polyfit factor-model submission variant. |
| `store-prediction.ipynb` | Hyperparameter exploration | XYZT-derived factor variants and external-file weighted blending. |

## Implementation decision

`xyzt_awesome`, from the final executed path in `keeping-it-simple-by-xyzt.ipynb`, is the project default. It is the only analyzed solution that contains all required components to reproduce an end-to-end test submission from raw competition CSVs:

1. Load train, test, and sample submission data.
2. Derive the source notebook's date fields.
3. Fit item-specific weekday, month, store, and annual-growth factors from training data.
4. Apply the source notebook's weighted quadratic annual-growth extrapolation.
5. Generate, round, and write test predictions.

This implementation decision is based on reconstructibility only. It does **not** establish that XYZT was the overall competition winner.

## Missing components preventing Top-1 reproduction

The analyzed Prophet/dumb-model notebook describes a reported Top-1 approach but does not provide the complete executable solution. The following information is unknown or unavailable and must not be inferred:

- Final training and inference code for the reported Top-1 submission.
- Exact rule for selecting or combining the original hardcoded-trend dumb model and the item-specific weekly-seasonality dumb model.
- Executable implementation of the reported hardcoded 2018 trend.
- Final test prediction generation, prediction rounding, and submission-file creation for that approach.
- Time-series cross-validation generation code; the notebook only reads precomputed CV prediction files.
- Required Prophet/dumb CV CSV artifacts and their original generation workflow.
- Tuned Prophet source notebook and final tuned parameters referenced by the analysis notebook.
- Exact package/runtime versions for legacy `fbprophet`, pandas, NumPy, Bokeh, and statsmodels.

## Other unavailable external artifacts

- The 32-column candidate prediction matrix required by `blend-boosting-for-best-score-on-demand-forecast.ipynb`.
- The model/source provenance for those 32 blend candidates, including the referenced LightGBM base notebook.
- `weight_predictor_1.csv`, `weight_predictor_3.csv`, and `weight_predictor_4.csv` required by `store-prediction.ipynb`.
- Candidate validation histories, private scores, and submission provenance for several alternative pipelines.

## Current status

Milestone 1 establishes documentation and the active-solution declaration only. No model code, data handling, feature logic, prediction logic, or leaderboard-affecting behavior has been implemented or changed.
