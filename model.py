import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.preprocessing import StandardScaler


def main():
    print('loading data')
    x_train, x_test, y_train, id_test = pickle.load(open('feature_engineering.bin', 'rb'))

    print('Loaded {} features'.format(x_train.shape[1]))

    # first we take the log1p of the target value(trip_duration)
    y_train = np.log1p(y_train)

    # split data into training and valid sets
    # x_train and y_train are together as our training sets; x_valid and y_valid are as our valid sets before submitting to Kaggle
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=51)

    print('{} training sample,{}validation samples'.format(len(x_train), len(x_valid)))
    print('standardlize the data')
    scaler = StandardScaler()
    # make normalization on x_train,x_valid and x_test dataset
    x_train = pd.DataFrame(scaler.fit_transform(x_train),
                           index=x_train.index, columns=x_train.columns)
    x_valid = pd.DataFrame(scaler.fit_transform(x_valid), index=x_valid.index, columns=x_valid.columns)
    x_test = pd.DataFrame(scaler.fit_transform(x_test),
                          index=x_test.index, columns=x_train.columns.values)
   # print(list(x_test.columns.values))
    print('{} standardlize training sample, {} standardlize valid sample and {} standardlize test sample'.format(x_train.head(), x_valid.head(), x_test.head()))

    print('Constructing XGBoost DMatrices')

    # convert our data into XGBoost's fast C++ format
    d_train = xgb.DMatrix(x_train, label=y_train)
    d_valid = xgb.DMatrix(x_valid, label=y_valid)
    d_test = xgb.DMatrix(x_test)

    # Remove x_train,x_valid and x_test from memory because they are unnecessary now.
    del x_train, x_valid, x_test

    # When computing metrics, XGBoost will do it on d_train and d_valid these two datasets.
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    # Setup parameters for XGBoost training
    params = {}
    params['objective'] = 'reg:linear'  # this is a regression problem
    params['max_depth'] = 4  # this should be between 3-10.I've started with 4. 4-6 can be good starting points based on experience.
    params['min_child_weight'] = 1
    params['eval_metric'] = 'rmse'  # RMSE for evaluation, and then we use log1p for rmlse evaluation
    params['subsample,colsample_bytree'] = 0.6  # these two params are commonly used for subsample ratio of the training instance and subsample ratio of columns when constructing each tree. The start point normally range from 0.5-0.9.
    params['silent'] = 1  # Don't print debug messages
    params['eta'] = 0.28  # Learning reduced to a more reasonable rate - this does not need to be tuned as lower is always better

    # other parameters are as default at the start stage

    print('Training XGBoost model')
    # early stop after 30 rounds
    num_boost_rounds = int(1e4)
    trained_model = xgb.train(params, d_train, num_boost_rounds, watchlist, early_stopping_rounds=30)

    # Predict results on the test set
    p_test = trained_model.predict(d_test)

    # Create a submission file
    sub = pd.DataFrame()
    sub['id'] = id_test
    sub['trip_duration'] = np.expm1(p_test)

    # Write the predict result as csv to the disk
    sub.to_csv('model_sub.csv', index=False)


if __name__ == '__main__':
    main()
