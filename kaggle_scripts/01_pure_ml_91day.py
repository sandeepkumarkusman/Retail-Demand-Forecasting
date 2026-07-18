"""
Pure LightGBM — Version 9 replica (no math blend).
Scored: 12.76 Public / 14.06 Private on notebookb50e8dd130
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from pandas.tseries.holiday import USFederalHolidayCalendar
import os
import warnings

warnings.filterwarnings('ignore')

def smape_lgb(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 200.0
    diff = np.abs(y_true - y_pred) / np.maximum(denominator, 1e-8)
    return 'smape', np.nanmean(diff), False

def expand_ml_date_features(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['year']       = df['date'].dt.year
    df['month']      = df['date'].dt.month
    df['day']        = df['date'].dt.day
    df['dayofweek']  = df['date'].dt.dayofweek
    df['dayofyear']  = df['date'].dt.dayofyear
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
    df['quarter']    = df['date'].dt.quarter
    df['is_weekend']     = (df['dayofweek'] >= 5).astype(int)
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end']   = df['date'].dt.is_month_end.astype(int)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dow_sin']   = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos']   = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['doy_sin']   = np.sin(2 * np.pi * df['dayofyear'] / 365.25)
    df['doy_cos']   = np.cos(2 * np.pi * df['dayofyear'] / 365.25)
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=df['date'].min(), end=df['date'].max())
    df['is_holiday']       = df['date'].isin(holidays).astype(int)
    df['days_since_start'] = (df['date'] - df['date'].min()).dt.days
    return df

def add_lag_and_rolling_features(df):
    df = df.copy()
    df.sort_values(['store', 'item', 'date'], inplace=True)

    for lag in [91, 98, 105, 112, 182, 270, 364, 365, 728]:
        df[f'sales_lag_{lag}'] = df.groupby(['store', 'item'])['sales'].shift(lag)

    store_d = df.groupby(['store', 'date'])['sales'].sum().reset_index()
    store_d['store_total_sales_lag_91'] = store_d.groupby('store')['sales'].shift(91)
    df = df.merge(store_d[['store', 'date', 'store_total_sales_lag_91']], on=['store', 'date'], how='left')

    item_d = df.groupby(['item', 'date'])['sales'].sum().reset_index()
    item_d['item_total_sales_lag_91'] = item_d.groupby('item')['sales'].shift(91)
    df = df.merge(item_d[['item', 'date', 'item_total_sales_lag_91']], on=['item', 'date'], how='left')

    df.sort_values(['store', 'item', 'date'], inplace=True)

    for w in [7, 14, 28, 56, 91]:
        df[f'rolling_mean_{w}'] = df.groupby(['store', 'item'])['sales'].transform(
            lambda x: x.shift(91).rolling(w, min_periods=1).mean()
        )
        df[f'rolling_std_{w}'] = df.groupby(['store', 'item'])['sales'].transform(
            lambda x: x.shift(91).rolling(w, min_periods=1).std()
        )

    df['local_trend']   = df['rolling_mean_28'] - df['rolling_mean_91']
    df['ema_090'] = df.groupby(['store', 'item'])['sales'].transform(
        lambda x: x.shift(91).ewm(alpha=0.90, min_periods=1).mean()
    )
    df['ema_095'] = df.groupby(['store', 'item'])['sales'].transform(
        lambda x: x.shift(91).ewm(alpha=0.95, min_periods=1).mean()
    )
    df['expanding_mean'] = df.groupby(['store', 'item'])['sales'].transform(
        lambda x: x.shift(91).expanding(min_periods=1).mean()
    )
    df['yoy_ratio']                  = df['sales_lag_364'] / df['sales_lag_728'].replace(0, np.nan)
    df['sales_same_month_last_year'] = df['sales_lag_364']
    return df

def add_target_encoding(df):
    df = df.copy()
    mask = df['is_train'] == 1
    for feat, cols in {
        'store_mean':       ['store'],
        'item_mean':        ['item'],
        'store_item_mean':  ['store', 'item'],
        'month_mean':       ['month'],
        'dow_mean':         ['dayofweek'],
        'store_month_mean': ['store', 'month'],
        'item_month_mean':  ['item', 'month'],
        'item_dow_mean':    ['item', 'dayofweek'],
        'store_dow_mean':   ['store', 'dayofweek'],
    }.items():
        means = df.loc[mask].groupby(cols)['sales'].mean()
        if len(cols) == 1:
            df[feat] = df[cols[0]].map(means)
        else:
            df[feat] = df.set_index(cols).index.map(means)
    return df

if __name__ == '__main__':
    DATA_DIR = '/kaggle/input/demand-forecasting-kernels-only/'

    print('Loading data...')
    train_raw = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test_raw  = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

    lo = train_raw['sales'].quantile(0.001)
    hi = train_raw['sales'].quantile(0.999)
    tr = train_raw.copy()
    tr['sales'] = tr['sales'].clip(lo, hi)
    tr['is_train'] = 1
    te = test_raw.copy()
    te['is_train'] = 0

    combined = pd.concat([tr, te], ignore_index=True)
    print('Building date features...')
    combined = expand_ml_date_features(combined)
    print('Building lag/rolling features...')
    combined = add_lag_and_rolling_features(combined)
    print('Building target encodings...')
    combined = add_target_encoding(combined)

    train_df = combined[combined['is_train'] == 1].copy()
    test_df  = combined[combined['is_train'] == 0].copy()

    max_date  = train_df['date'].max()
    val_start = max_date - pd.Timedelta(days=89)
    val_mask  = train_df['date'] >= val_start

    FEATURES = [c for c in train_df.columns if c not in ['date', 'sales', 'is_train', 'id']]

    X_tr  = train_df[~val_mask][FEATURES];  y_tr  = train_df[~val_mask]['sales']
    X_val = train_df[ val_mask][FEATURES];  y_val = train_df[ val_mask]['sales']
    print(f'Train: {len(X_tr):,}  Val: {len(X_val):,}  Features: {len(FEATURES)}')

    params = dict(
        objective='regression_l1', metric='mae', boosting_type='gbdt',
        num_leaves=127, learning_rate=0.05, feature_fraction=0.8,
        bagging_fraction=0.8, bagging_freq=5, min_child_samples=20,
        lambda_l1=0.1, lambda_l2=0.1, n_estimators=1500,
        n_jobs=-1, random_state=42,
    )

    print('Training LightGBM...')
    model = lgb.LGBMRegressor(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              eval_metric=smape_lgb,
              callbacks=[lgb.early_stopping(50, verbose=100)])

    preds = np.clip(model.predict(test_df[FEATURES]), 0, None)

    sub = pd.DataFrame({
        'id':    test_df['id'].fillna(0).astype(int),
        'sales': np.round(preds).astype(int),
    }).sort_values('id').reset_index(drop=True)
    sub.to_csv('submission.csv', index=False)
    print('Done! submission.csv saved.')
