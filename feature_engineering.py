import numpy as np
import pandas as pd
import pickle
import multiprocessing
import haversine

# the number of CPU threads to use for multithreaded operations
N_Threads = 6

# define Haversin distance for circle_based distance


def haversine_distance(x):
    a_lat, a_lot, b_lat, b_lon = x
    return haversine.haversine((a_lat, a_lot), (b_lat, b_lon))

# multithreaded apply function for a dataframe, which uses multiprocessing to map a function to a series, vastly speeding up feature_generation


def apply_multithreaded(data, func):
    pool = multiprocessing.Pool(N_Threads)  # spawn a pool of processes
    # retrive a numpy array which can be iterated over
    data = data.values
    # map the function ove the data multi-threaded
    result = pool.map(func, data)
    pool.close()  # close the threads
    return result

# This code needs to be removed from global scope, else it will be run by every thread in the mulitprocessing pool when they import this file.


def main():
    print('loading preprocessed data\n')

    df_train, df_test, x_train, x_test, y_train, id_test = pickle.load(open('preprocessed_data.bin', 'rb'))

    print(df_train.columns)

    print('Creating direction of travel features')
    x_train['delta_lat'] = x_train['dropoff_latitude'] - x_train['pickup_latitude']
    x_train['delta_lon'] = x_train['dropoff_longitude'] - x_train['dropoff_longitude']
    x_train['angle'] = (180 / np.pi) * np.arctan2(x_train['delta_lat'], x_train['delta_lon'])

    x_test['delta_lat'] = x_test['dropoff_latitude'] - x_test['dropoff_latitude']
    x_test['delta_lon'] = x_test['dropoff_longitude'] - x_test['dropoff_longitude']
    x_test['angle'] = (180 / np.pi) * np.arctan2(x_test['delta_lat'], x_test['delta_lon'])
    print('x_train direction of travel features \n {}'.format(x_train[['delta_lat', 'delta_lon', 'angle']].head()))
    print('x_test direction of travel features \n {}'.format(x_test[['delta_lat', 'delta_lon', 'angle']].head()))
    print('creating distance features')

    # map the three distance functions over all samples in the training set
    x_train['dist_l1'] = np.abs(x_train['pickup_latitude'] - x_train['dropoff_latitude']) + np.abs(x_train['pickup_longitude'] - x_train['dropoff_longitude'])

    x_train['dist_l2'] = np.sqrt((x_train['pickup_latitude'] - x_train['dropoff_latitude'])**2 + (x_train['pickup_longitude'] - x_train['dropoff_longitude'])**2)

    x_train['dist_haversine'] = apply_multithreaded(
        x_train[['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']], haversine_distance)

    x_test['dist_l1'] = np.abs(x_test['pickup_latitude'] - x_test['dropoff_latitude']) + np.abs(x_test['pickup_longitude'] - x_test['dropoff_longitude'])

    x_test['dist_l2'] = np.sqrt((x_test['pickup_latitude'] - x_test['dropoff_latitude'])**2 + (x_test['pickup_longitude'] - x_test['dropoff_longitude']) ** 2)

    x_test['dist_haversine'] = apply_multithreaded(x_test[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']], haversine_distance)

    print(' three distance features in x_train\n {} '.format(x_train[['dist_l1', 'dist_l2', 'dist_haversine']].head()))
    print(' three distance features in x_test\n {}  '. format(x_test[['dist_l1', 'dist_l2', 'dist_haversine']].head()))

    print('Creating traffic situation features')

    # extract day and hour from each datetime string
    df_train['day'] = df_train['pickup_datetime'].apply(lambda x: x.split(' ')[0])
    df_train['hour'] = df_train['pickup_datetime'].apply(lambda x: x.split(':')[1])
    df_test['day'] = df_test['pickup_datetime'].apply(lambda x: x.split(' ')[0])
    df_test['hour'] = df_test['pickup_datetime'].apply(lambda x: x.split(':')[1])

    # apply a groupby operation over unique dates in order to get the number of trips each day
    trip_all = pd.concat([df_train[['day', 'hour']], df_test[['day', 'hour']]])
    print(trip_all.head())
    # compute the number of trips on each day
    traffic_day = trip_all.groupby('day')['day'].count()
    # compute the number of trips in each hour
    traffic_hour = trip_all.groupby('hour')['hour'].count()

    print(traffic_day.head())
    print(traffic_hour.head())

    # save this feature to training datasets
    x_train['daily_count'] = df_train['day'].apply(lambda day: traffic_day[day])
    x_train['hourly_count'] = df_train['hour'].apply(lambda hour: traffic_hour[hour])
    x_test['daily_count'] = df_test['day'].apply(lambda day: traffic_day[day])
    x_test['hourly_count'] = df_test['hour'].apply(lambda hour: traffic_hour[hour])

    print(x_train[['daily_count', 'hourly_count']].head())
    print(x_test[['daily_count', 'hourly_count']].head())

    print('Creating time estimate features')
    df_train['haversine_speed'] = x_train['dist_haversine'] / df_train['trip_duration']
    hourly_speed = df_train.groupby('hour')['haversine_speed'].mean()  # average speed
    hourly_speed_fill = df_train['haversine_speed'].mean()  # fill this varible to unknown situation

    # create time estimate features
    x_train_hourly_speed = df_train['hour'].apply(lambda hour: hourly_speed[hour])
    x_test_hourly_speed = df_test['hour'].apply(lambda hour: hourly_speed[hour] if hour in hourly_speed else hourly_speed_fill)
    x_train['haversine_time_estim'] = x_train['dist_haversine'] / x_train_hourly_speed
    x_test['haversine_time_estim'] = x_test['dist_haversine'] / x_test_hourly_speed

    print(x_train['haversine_time_estim'].head())
    print(x_test['haversine_time_estim'].head())

    print('Feature engineering process complete, feature list {}'.format(x_train.columns.tolist()))
    print('Feature engineering process complete, feature list {}'.format(x_test.columns.tolist()))
    print('Saving features to disk')

    pickle.dump([x_train, x_test, y_train, id_test], open('feature.bin', 'wb'), protocol=2)


if __name__ == '__main__':
    main()
