import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from sklearn.cross_validation import KFold
from bayes_opt import BayesianOptimization


def xgbcv(x_train, y_train, max_depth, min_child_weight, subsample, colsample_bylevel, eta):
    params = {
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': 1,
        'eta': eta,
        'max_depth': int(max_depth),
        'min_child_weight': min_child_weight,
        'subsample': subsample,
        'colsample_bylevel': colsample_bylevel,
        'booster': 'gbtree',
        'nthread': 4
    }

    num_rounds = int(3e3)
    nfolds = 3
    cvscore = KFoldValidation(x_train, y_train, params, num_rounds, nfolds)  # 3-folds cross validation
    return -1.0 * cvscore  # invert the cv score to let bayeseopt maximize


def bayesOpt(x_train, y_train):
    # Search space
    ranges = {
        'max_depth': (6, 13),
        'min_child_weight': (0.6, 3.6),
        'subsample': (0.5, 1.0),
        'colsample_bylevel': (0.5, 1.0),
        'eta': (0.1, 0.3)
    }

    bo = BayesianOptimization(lambda max_depth, min_child_weight, subsample, colsample_bylevel, eta: xgbcv(x_train, y_train, max_depth, min_child_weight, subsample, colsample_bylevel, eta), ranges)
    bo.maximize(init_points=50, n_iter=5, acq='ei')

    best_rmse = round((-1.0 * bo.res['max']['max_val']), 6)
    print('Best RMSE found %f' % best_rmse)
    print('Parameters:%s' % bo.res['max']['max_params'])
    return bo.res['max']['max_params']   # return the best value for parameters


def KFoldValidation(x_train, y_train, params, num_rounds, nfolds, target='loss'):

    kf = KFold(len(x_train), nfolds, shuffle=True, random_state=51)
    fold_scores = []
    for train_index, cv_index in kf:
        print(train_index)
        print(cv_index)
        res_dict = {}
        # split train/validation
        X_train, X_valid = x_train.iloc[train_index, :], x_train.iloc[cv_index, :]
        Y_train, Y_valid = y_train[train_index], y_train[cv_index]

        # converting these data into XGBoost fast C++ format
        d_train = xgb.DMatrix(X_train, label=Y_train)
        d_valid = xgb.DMatrix(X_valid, label=Y_valid)

        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        reg = xgb.train(params, d_train, num_rounds, evals=watchlist, early_stopping_rounds=10, evals_result=res_dict, verbose_eval=50)

        score = np.min(res_dict['valid']['rmse'])
        fold_scores.append(score)
    return np.mean(fold_scores)


def main():
    print('Loading data')
    x_train, x_test, y_train, id_test = pickle.load(open('feature_engineering.bin', 'rb'))
    print('Loaded {} features'.format(x_train.shape[1]))
    # first we take the log1p of the target value(trip_duration)
    y_train = np.log1p(y_train)

    d_test = xgb.DMatrix(x_test)
    # Remove  x_test from our memory
    del x_test

    # apply bayesian opt for max_depth,min_child_weight,subsample,colsampe_bylevel and eta optimization
    params = bayesOpt(x_train, y_train)


if __name__ == '__main__':
    main()
