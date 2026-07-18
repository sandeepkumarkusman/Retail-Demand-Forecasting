"""
Pure LightGBM — 364-day anchor fix (experimental).
Rolling windows look at same calendar period last year to eliminate
Q4 holiday contamination bleeding into Q1 Private LB predictions.
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

def expand_date_features(df):
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

def add_features(df):
    df = df.copy()
    df.sort_values(['store', 'item', 'date'], inplace=True)
    g = df.groupby(['store', 'item'])['sales']

    # PRIMARY: 364-day anchored lags (same calendar week last year — zero holiday bleed)
    for lag in [357, 364, 371, 378]:
        df[f'lag_{lag}'] = g.shift(lag)

    # Rolling windows anchored at 364 days back
    for w in [7, 14, 28, 56]:
        df[f'yago_rmean_{w}'] = g.transform(
            lambda x: x.shift(364).rolling(w, min_periods=1).mean()
        )
        df[f'yago_rstd_{w}'] = g.transform(
            lambda x: x.shift(364).rolling(w, min_periods=1).std()
        )

    df['yago_ema_90'] = g.transform(lambda x: x.shift(364).ewm(alpha=0.90, min_periods=1).mean())
    df['yago_ema_95'] = g.transform(lambda x: x.shift(364).ewm(alpha=0.95, min_periods=1).mean())

    for lag in [728, 729]:
        df[f'lag_{lag}'] = g.shift(lag)

    df['yoy_ratio'] = df['lag_364'] / df['lag_728'].replace(0, np.nan)

    # SUPPLEMENTARY: 91-day point lags (leakage-free, not used as rolling anchors)
    for lag in [91, 98, 105, 182]:
        df[f'lag_{lag}'] = g.shift(lag)

    df['lag91_rmean_28'] = g.transform(lambda x: x.shift(91).rolling(28, min_periods=1).mean())
    df['lag91_rmean_91'] = g.transform(lambda x: x.shift(91).rolling(91, min_periods=1).mean())
    df['local_trend']    = df['lag91_rmean_28'] - df['lag91_rmean_91']

    store_d = df.groupby(['store', 'date'])['sales'].sum().reset_index()
    store_d['store_lag_364'] = store_d.groupby('store')['sales'].shift(364)
    df = df.merge(store_d[['store', 'date', 'store_lag_364']], on=['store', 'date'], how='left')

    item_d = df.groupby(['item', 'date'])['sales'].sum().reset_index()
    item_d['item_lag_364'] = item_d.groupby('item')['sales'].shift(364)
    df = df.merge(item_d[['item', 'date', 'item_lag_364']], on=['item', 'date'], how='left')

    df.sort_values(['store', 'item', 'date'], inplace=True)
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
    combined = expand_date_features(combined)
    print('Building 364-day anchored features...')
    combined = add_features(combined)
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
        lambda_l1=0.1, lambda_l2=0.1, n_estimators=2000,
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
