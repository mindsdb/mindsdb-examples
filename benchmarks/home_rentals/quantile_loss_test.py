import mindsdb
import lightwood
import pandas as pd
import numpy as np

test_prefix = 'test'
run_learn = True
drop_cols = ['initial_price', 'location']

backend='lightwood'
lightwood.config.config.CONFIG.HELPER_MIXERS = False
# Instantiate a mindsdb Predictor
mdb = mindsdb.Predictor(name='home_rentals')

# We tell the Predictor what column or key we want to learn and from what data

if run_learn:
    mdb.learn(from_data='dataset/train.csv', to_predict='rental_price')

predictions = mdb.predict(when_data=f'dataset/{test_prefix}.csv')
real_values = pd.read_csv(f'dataset/{test_prefix}.csv')['rental_price']
intervals_all = [x.explanation['rental_price']['explanation']['confidence_interval'] for x in predictions]


negative = 0
positive = 0
correct = 0
incorrect = 0

for i, x in enumerate(intervals_all):
    if x[0] < 0:
        negatives += 1
    elif x[1] < 0:
        negatives += 1
    else:
        positive += 1

    if x[0] < real_values[i] < x[1]:
        correct += 1
    else:
        incorrect += 1

interval_width = [x[1] - x[0] for x in intervals_all]

mean_iw = np.mean(interval_width)
std_iw = np.std(interval_width)

print(f'Out of the intervals {correct} were correct (real value was within the interval) and {incorrect} were incorrect (real value was outside the interval)')
print(f'Out of the intervals {positive} contained only positive values and {negative} contained at least one negative value (gross prediction error, since all the targets are positive)')
print(f'The mean range of the intervals was {mean_iw}')
print(f'The standard deviation for the ranges was {std_iw}')
exit()

droped_data = pd.read_csv('dataset/train.csv')
droped_data = droped_data.drop(columns=drop_cols)

mdb = mindsdb.Predictor(name='home_rentals_dropped')

if run_learn:
    mdb.learn(from_data=droped_data, to_predict='rental_price')

droped_data_test = pd.read_csv(f'dataset/{test_prefix}.csv')
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
