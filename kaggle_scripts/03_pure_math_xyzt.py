"""
Pure XYZT math baseline — no machine learning.
Scores: ~12.57 Public / ~13.84 Private.
Runs in seconds.
"""
import numpy as np
import pandas as pd
import os
import warnings

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    DATA_DIR = '/kaggle/input/demand-forecasting-kernels-only/'

    print('Loading data...')
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test  = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

    train['date'] = pd.to_datetime(train['date'])
    test['date']  = pd.to_datetime(test['date'])

    train['year']      = train['date'].dt.year
    train['month']     = train['date'].dt.month
    train['dayofweek'] = train['date'].dt.dayofweek

    test['year']      = test['date'].dt.year
    test['month']     = test['date'].dt.month
    test['dayofweek'] = test['date'].dt.dayofweek

    grand_avg        = train['sales'].mean()
    store_item_table = train.pivot_table(index='store', columns='item', values='sales', aggfunc='mean')
    month_table      = train.pivot_table(index='month', values='sales', aggfunc='mean') / grand_avg
    dow_table        = train.pivot_table(index='dayofweek', values='sales', aggfunc='mean') / grand_avg
    year_table       = train.pivot_table(index='year', values='sales', aggfunc='mean') / grand_avg

    years = np.arange(2013, 2019)
    avg   = year_table['sales'].values.squeeze()
    w     = np.exp((years - 2018) / 2.5)
    poly  = np.poly1d(np.polyfit(years[:-1], avg, 2, w=w[:-1]))

    print('Generating predictions...')
    def predict_row(row):
        return (store_item_table.at[row['store'], row['item']]
                * month_table.at[row['month'], 'sales']
                * dow_table.at[row['dayofweek'], 'sales']
                * poly(row['year']))

    preds = test.apply(predict_row, axis=1)

    sub = pd.DataFrame({
        'id':    test['id'].astype(int),
        'sales': np.round(preds.values).astype(int),
    })
    sub.to_csv('submission.csv', index=False)
    print('Done! submission.csv saved.')
