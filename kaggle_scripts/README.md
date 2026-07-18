# Kaggle Scripts — README

All scripts output a `submission.csv` ready to upload to the Kaggle competition.
Just copy the script contents into a new Kaggle Notebook and run.

## Scripts

| File | Approach | Public LB | Private LB |
|------|----------|-----------|------------|
| `01_pure_ml_91day.py` | LightGBM, 91-day anchor rolling features | **12.76** | 14.06 |
| `02_pure_ml_364day.py` | LightGBM, 364-day anchor (experimental — should fix Private inversion) | TBD | TBD |
| `03_pure_math_xyzt.py` | Pure XYZT math, zero ML | 12.57 | 13.84 |
| `04_blend_ml_xyzt.py` | 50% ML + 50% XYZT blend | ~12.4 (estimated) | — |
| `05_other_intern_model.py` | Other intern's leaky model (lag_1 on a 90-day horizon) | 25 | 17 |

## How to run on Kaggle

1. Go to [Kaggle](https://www.kaggle.com) → **Create Notebook**
2. Add the competition dataset: `demand-forecasting-kernels-only`
3. Copy the script content into a code cell
4. Run all → download `submission.csv` → Submit

## Data path
All scripts use:
```
/kaggle/input/demand-forecasting-kernels-only/
```
