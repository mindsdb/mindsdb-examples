import mindsdb
import lightwood
import pandas as pd
import numpy as np


backend='lightwood'
lightwood.config.config.CONFIG.HELPER_MIXERS = False
# Instantiate a mindsdb Predictor
mdb = mindsdb.Predictor(name='home_rentals')

# We tell the Predictor what column or key we want to learn and from what data

mdb.learn(
    from_data='dataset/home_rentals_train.csv', # the path to the file where we can learn from, (note: can be url)
    to_predict='rental_price', # the column we want to learn to predict given all the data in the file
)

predictions = mdb.predict(when_data='dataset/home_rentals_train.csv')
intervals_all = [x.explanation['rental_price']['explanation']['confidence_interval'] for x in predictions]


droped_data = pd.read_csv('dataset/home_rentals_train.csv')
droped_data = droped_data.drop(columns=['sqft','location','days_on_market'])

mdb = mindsdb.Predictor(name='home_rentals_dropped')

mdb.learn(
    from_data=droped_data, # the path to the file where we can learn from, (note: can be url)
    to_predict='rental_price', # the column we want to learn to predict given all the data in the file
)

droped_data_test = pd.read_csv('dataset/home_rentals_train.csv')
droped_data_test = droped_data_test.drop(columns=['sqft','location','days_on_market'])

predictions_dropped = mdb.predict(when_data=droped_data)

intervals_droped = [x.explanation['rental_price']['explanation']['confidence_interval'] for x in predictions_dropped]

interval_wider = 0
intravel_smaller = 0
intrval_same = 0
for i in range(len(intervals_all)):
    print(intervals_droped[i][1], intervals_droped[i][0], intervals_droped[i][1] - intervals_droped[i][0])
    print(intervals_all[i][1], intervals_all[i][0], intervals_all[i][1] - intervals_all[i][0])

    if (intervals_droped[i][1] - intervals_droped[i][0]) > (intervals_all[i][1] - intervals_all[i][0]):
        interval_wider += 1
    elif (intervals_droped[i][1] - intervals_droped[i][0]) == (intervals_all[i][1] - intervals_all[i][0]):
        intrval_same += 1
    else:
        intravel_smaller += 1

print(f'Intravel is wider in {interval_wider} cases and smaller in {intravel_smaller} cases and the same in {intrval_same} cases')
