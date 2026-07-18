"""
Other intern's model — exact replica for benchmarking.
Uses lag_1 and shift(1) rolling features which causes NaN
for days 2-90 of the Kaggle test set. Scored: 25 Public / 17 Private.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import os
import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    DATA_DIR = '/kaggle/input/demand-forecasting-kernels-only/'

    print('Loading data...')
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    train["date"] = pd.to_datetime(train["date"])
    test["date"]  = pd.to_datetime(test["date"])

    test["sales"] = np.nan
    df = pd.concat([train, test], ignore_index=True)
    df.sort_values(["store", "item", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df["year"]      = df.date.dt.year
    df["month"]     = df.date.dt.month
    df["day"]       = df.date.dt.day
    df["dayofweek"] = df.date.dt.dayofweek
    df["weekofyear"] = df.date.dt.isocalendar().week.astype(int)
    df["quarter"]   = df.date.dt.quarter
    df["is_weekend"] = (df.dayofweek >= 5).astype(int)
    df["dayofyear"] = df.date.dt.dayofyear

    for lag in [1, 7, 14, 28, 30, 90]:
        df[f"lag_{lag}"] = df.groupby(["store", "item"])["sales"].transform(lambda x: x.shift(lag))

    for window in [7, 14, 30, 90]:
        df[f"rolling_std_{window}"] = df.groupby(["store", "item"])["sales"].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).std()
        )
        df[f"rolling_mean_{window}"] = df.groupby(["store", "item"])["sales"].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )

    for alpha in [0.95, 0.9, 0.8, 0.7]:
        df[f"ema_{str(alpha).replace('.', '')}"] = df.groupby(["store", "item"])["sales"].transform(
            lambda x: x.shift(1).ewm(alpha=alpha).mean()
        )

    df["expanding_mean"] = df.groupby(["store", "item"])["sales"].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    train_df = df[df["sales"].notna()].sort_values(["store", "item", "date"]).reset_index(drop=True)
    test_df  = df[df["sales"].isna()].sort_values(["store", "item", "date"]).reset_index(drop=True)
    test_ids = test_df["id"].copy()

    DROP = ["sales", "date", "id"]
    FEATURES = [c for c in train_df.columns if c not in DROP]
    X_train = train_df[FEATURES];  y_train = train_df["sales"]
    X_test  = test_df[FEATURES]

    lgb_params = {
        "objective": "regression", "metric": "rmse", "boosting_type": "gbdt",
        "learning_rate": 0.015, "n_estimators": 6000, "num_leaves": 127,
        "max_depth": 12, "min_child_samples": 40, "feature_fraction": 0.90,
        "bagging_fraction": 0.90, "bagging_freq": 5, "lambda_l1": 0.5,
        "lambda_l2": 0.5, "min_gain_to_split": 0.01, "verbosity": -1,
        "random_state": 42, "n_jobs": -1,
    }

    print('Training (this will take a while)...')
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(X_train, y_train)

    preds = np.round(np.clip(model.predict(X_test), 0, None)).astype(int)
    sub = pd.DataFrame({'id': test_ids.astype(int), 'sales': preds})
    sub = sub.sort_values('id').reset_index(drop=True)
    sub.to_csv('submission.csv', index=False)
    print('Done! submission.csv saved.')
