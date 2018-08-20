import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import xgboost as xgb


def main():
    print('Loading data')
    x_train, x_test, y_train, id_test = pickle.load(open('feature.bin', 'rb'))

    print('Loaded {} features'.format(x_train.shape[1]))
    print(x_train['delta_lon'])

    # First we take the log1p of the target value
    y_train = np.log1p(y_train)

    # Split off the data set
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=51)

    print('{} training samples,{} validation samples'.format(len(x_train), len(x_valid)))

    print('Constructing XGBoost DMatrices')

    # convert our data into XGBoost's fast C++ format
    d_train = xgb.DMatrix(x_train, label=y_train)
    d_valid = xgb.DMatrix(x_valid, label=y_valid)
    d_test = xgb.DMatrix(x_test)

    # remove these from memeory
    del x_train, x_valid, x_test

    # XGBoost will compute metrics on d_train and d_valid
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    # Set up params for XGBoost
    params = {
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': 1,
        'eta': 0.0482,  # decided by params_opt
        'max_depth': 12,  # decided by params_opt
        'colsample_bylevel': 0.7075,  # decided by params_opt
        'subsample': 0.9962,  # decided by params_opt
        'min_child_weight': 0.6236  # decided by params_opt

    }

    print('Training XGBoost model')

    res = xgb.train(params, d_train, int(1e4), watchlist, early_stopping_rounds=50)

    # Save the model file for later visualization
    res.save_model('model_result_opt_me.mdl')

    # Predict on the test dataset
    p_test = res.predict(d_test)

    # Create a submission file
    sub = pd.DataFrame()
    sub['id'] = id_test
    sub['trip_duration'] = np.expm1(p_test)

    # write the csv
    sub.to_csv('model_opt_sub_me.csv', index=False)


if __name__ == '__main__':
    main()
