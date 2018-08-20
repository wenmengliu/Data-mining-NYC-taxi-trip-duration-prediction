# import libraries
import numpy as np
import pandas as pd
import pickle


def main():
    print('Loading data into workspace')
    # load data
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')

    print('Read {} training data and {} testing data, preprocessing...'.format(df_train.shape, df_test.shape))

    # convert boolen type store_and_fwd_flag into a binary numeric feature
    df_train.loc[:,'store_and_fwd_flag'] = df_train['store_and_fwd_flag'].apply(lambda x: int(x == 'Y'))
    df_test.loc[:,'store_and_fwd_flag'] = df_test['store_and_fwd_flag'].apply(lambda x: int(x == 'Y'))

    # parse pickup_datetime feature
    df_train['timestamp'] = pd.to_datetime(df_train['pickup_datetime'])
    df_test['timestamp'] = pd.to_datetime(df_test['pickup_datetime'])

    # create unix epoch time feature usually in nanoseconds, we need to convert seconds to nanoseconds.
    df_train['unix_time'] = [np.int64(t.value) / 1000000000 for t in df_train['timestamp']]
    df_test['unix_time'] = [np.int64(t.value) / 1000000000 for t in df_test['timestamp']]

    # create minute of day feature
    df_train['daily_minute'] = [t.hour * 60 + t.minute for t in df_train['timestamp']]
    df_test['daily_minute'] = [t.hour * 60 + t.minute for t in df_test['timestamp']]

    # create day of week feature
    df_train['day_of_week'] = [t.dayofweek for t in df_train['timestamp']]
    df_test['day_of_week'] = [t.dayofweek for t in df_test['timestamp']]

    # extrct targe variable 'trip_duration' and also do the same with test ID, because they are required for Kaggle submission
    y_train = df_train['trip_duration'].values
    id_test = df_test['id'].values

    # drop non-training features from the train set
    x_train = df_train.drop(['timestamp', 'pickup_datetime', 'id', 'trip_duration', 'dropoff_datetime'], axis=1)
    x_test = df_test.drop(['id', 'timestamp', 'pickup_datetime'], axis=1)

    print('df_train.head()')
    print('preprocessing complete,saving data to disk')

    # use pickle to dump our working data to disk so we can use it again in the future
    pickle.dump([df_train, df_test, x_train, x_test, y_train, id_test], open('preprocessed_data.bin', 'wb'), protocol=2)


if __name__ == '__main__':
    main()
