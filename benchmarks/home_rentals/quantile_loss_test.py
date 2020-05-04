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

# ignore_columns=['number_of_rooms','number_of_bathrooms','sqft','location','days_on_market','neighborhood']

predictions = mdb.predict(when_data=f'dataset/{test_prefix}.csv')
real_values = pd.read_csv(f'dataset/{test_prefix}.csv')['rental_price']
intervals_all = [x.explanation['rental_price']['explanation']['confidence_interval'] for x in predictions]


negative = 0
positive = 0
correct = 0
incorrect = 0

for i, x in enumerate(intervals_all):
    if x[0] < 0:
        negative += 1
    elif x[1] < 0:
        negative += 1
    else:
        positive += 1

    if x[0] <= real_values[i] <= x[1]:
        correct += 1
    else:
        incorrect += 1

interval_width = [x[1] - x[0] for x in intervals_all]
for x in intervals_all:
    print(x)
mean_iw = np.mean(interval_width)
std_iw = np.std(interval_width)

print(f'Out of the intervals {correct} were correct (real value was within the interval) and {incorrect} were incorrect (real value was outside the interval)')
print(f'Out of the intervals {positive} contained only positive values and {negative} contained at least one negative value (gross prediction error, since all the targets are positive)')
print(f'The mean range of the intervals was {mean_iw}')
print(f'The standard deviation for the ranges was {std_iw}')
