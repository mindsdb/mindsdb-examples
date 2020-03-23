import mindsdb
import lightwood
import pandas as pd
import numpy as np

test_prefix = 'test'
run_learn = False
drop_cols = ['days_on_market','sqft','location','initial_price']

backend='lightwood'
lightwood.config.config.CONFIG.HELPER_MIXERS = False
# Instantiate a mindsdb Predictor
mdb = mindsdb.Predictor(name='home_rentals')

# We tell the Predictor what column or key we want to learn and from what data

if run_learn:
    mdb.learn(from_data='dataset/home_rentals_train.csv', to_predict='rental_price')

predictions = mdb.predict(when_data=f'dataset/home_rentals_{test_prefix}.csv')
intervals_all = [x.explanation['rental_price']['explanation']['confidence_interval'] for x in predictions]

droped_data = pd.read_csv('dataset/home_rentals_train.csv')
droped_data = droped_data.drop(columns=drop_cols)

mdb = mindsdb.Predictor(name='home_rentals_dropped')

if run_learn:
    mdb.learn(from_data=droped_data, to_predict='rental_price')

droped_data_test = pd.read_csv(f'dataset/home_rentals_{test_prefix}.csv')
droped_data_test = droped_data_test.drop(columns=drop_cols)

predictions_dropped = mdb.predict(when_data=droped_data_test)

intervals_droped = [x.explanation['rental_price']['explanation']['confidence_interval'] for x in predictions_dropped]
interval_wider = 0
intravel_smaller = 0
intrval_same = 0
for i in range(len(intervals_all)):
    print('------------')
    print(intervals_droped[i])
    print(intervals_all[i])
    print('###############')
    if (intervals_droped[i][1] - intervals_droped[i][0]) > (intervals_all[i][1] - intervals_all[i][0]):
        interval_wider += 1
    elif (intervals_droped[i][1] - intervals_droped[i][0]) == (intervals_all[i][1] - intervals_all[i][0]):
        intrval_same += 1
    else:
        intravel_smaller += 1

print(f'Intravel with dropped columns is wider in {interval_wider} cases and smaller in {intravel_smaller} cases and the same in {intrval_same} cases')
