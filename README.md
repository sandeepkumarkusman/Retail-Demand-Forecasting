# Retail Demand Forecasting

This project records and modularizes independently analyzed Kaggle approaches for the Store Item Demand Forecasting Challenge. The goal is faithful, behavior-preserving reproduction of each notebook's methodology within the established project structure; it is not model redesign, optimization, or a hybrid solution.

## Current default pipeline

The default executable solution is `xyzt_awesome`, derived from `keeping-it-simple-by-xyzt.ipynb`. It is the only analyzed solution with a fully reconstructible end-to-end path from raw competition inputs through test prediction and submission generation.

This choice does **not** mean that `xyzt_awesome` was the overall competition-winning solution. The analyzed Top-1 description references a hardcoded-trend dumb model and an item-specific weekly-seasonality model, but its final executable training, prediction, combination, and submission code were not present in the available notebooks.

Other analyzed Kaggle solutions remain separate, notebook-specific implementations. They are not merged into the default pipeline.

## Data locations

Place the original Kaggle competition files in `data/raw/`:

- `train.csv`
- `test.csv`
- `sample_submission.csv`

Place external Kaggle artifacts that are required by specific notebook-only workflows in `data/external/`. Examples include precomputed cross-validation prediction files, candidate submission matrices for blending, and externally generated submission CSVs. These artifacts are not substituted or recreated by the default pipeline. If they are unavailable, separately named fallback routes are available in `config/config.yaml`; they use a deterministic historical store-item mean and must not be interpreted as the original Prophet or ensemble solutions.

## Project structure

```text
Retail-Demand-Forecasting/
├── README.md
├── requirements.txt
├── Makefile
├── .gitignore
├── data/
│   ├── raw/                 # train.csv, test.csv, sample_submission.csv
│   ├── processed/           # reproducible intermediate artifacts, if required
│   └── external/            # notebook-specific Kaggle artifacts
├── notebooks/
│   ├── 00_Scratch.ipynb
│   ├── 01_EDA.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Baselines.ipynb
│   ├── 04_Modeling.ipynb
│   ├── 05_Backtesting.ipynb
│   ├── 06_Error_Analysis.ipynb
│   └── 07_Presentation_Walkthrough.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── backtesting.py
│   ├── metrics.py
│   ├── explain.py
│   ├── guard.py
│   ├── train.py
│   ├── predict.py
│   ├── pipeline.py
│   └── utils.py
├── config/
│   └── config.yaml
├── models/
├── logs/
├── reports/
│   ├── figures/
│   ├── tables/
│   ├── Final_Report.md
│   └── Final_Report.pdf
├── outputs/
│   ├── predictions.parquet
│   ├── metrics.csv
│   └── feature_importance.csv
├── demo/
│   ├── app.py
│   └── requirements-demo.txt
└── tests/
    ├── test_features.py
    ├── test_metrics.py
    ├── test_backtesting.py
    └── test_guard.py
```

## Solution provenance

The analyzed notebooks include EDA/reference notebooks, several isolated factor-model submission pipelines, and two ensemble-only workflows that depend on unavailable external candidate predictions. The default pipeline is deliberately limited to the reconstructible XYZT implementation. See `reports/Final_Report.md` for the complete notebook inventory, implementation decision, and unreconstructible components.

## Running

Run the default verified XYZT pipeline with `python -m src.pipeline`, or use
`make run` where GNU Make is installed. The minimal local runner is
`python demo/app.py`. All routes write the configured CSV without an index.

The fallback configuration entries are executable only to keep the project
operational when source material is missing. They are not leaderboard-equivalent
implementations and do not alter `xyzt_awesome`.
