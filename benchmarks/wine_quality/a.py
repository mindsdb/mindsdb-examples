import requests
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd

root_url = 'http://localhost:47334'
datasource_train = 'wine_train'
datasource_test = 'wine_test'

# Train the predictor
predictor_name = 'wine_price_predictor'

# Get the model to make some predictions
response = requests.request('POST',f'{root_url}/predictors/{predictor_name}/predict_datasource', json={
    'data_source_name': datasource_test
    ,'kwargs': {}
    ,'format_flag': 'new_explain'
})
# Note: `kwargs` here work the same way as with train, they are extra argument for Mindsdb Native's  predict endpoint, you can find all of the potential arguments you can pass here: https://mindsdb.github.io/mindsdb/docs/predictor-interface#predict
predictions = response.json()

# The exact number we are predicting. This is almost always incorrect since pedicting an EXACT number is hard
exact_prediction_arr = [x['price']['predicted_value'] for x in predictions]
# The numerical range for which the confidence value applies, ideally you should always be using this range
predicted_range_arr = [x['price']['confidence_interval'] for x in predictions]
# The confidence we have for the `confidence_interval`, the numerical range in which we believe the value lies
prediction_confidence_arr = [x['price']['confidence'] for x in predictions]


high_percentile_confidence = np.percentile(prediction_confidence_arr, 91)
# The real values
test_data = pd.read_csv(open('processed_data/test.csv', 'r'))
real_val_arr = list(test_data['price'])
ignore_indexes_arr = [index if str(real_val_arr[index]) == 'nan' else -1 for index in range(len(real_val_arr))]
# A good score for comparing the accuracy value between two predictors... but not much meaning unless you are a statistician and know the kind of r2 score you're looking for. Usually better to use r2 score on the log of the values, since otherwise it will be bias to over-penalize miss-prediction on very large numbers (where the error range is usually inherently bigger)

accuracy = mean_absolute_error([x if i not in ignore_indexes_arr else 0  for i, x in enumerate(real_val_arr)], [x if i not in ignore_indexes_arr else 0 for i, x in enumerate(exact_prediction_arr)])
print(f'Mean absolute error: {accuracy}')

# Something better, let's see how many of our predictions fall within the predicted range
correct = 0
incorrect = 0
for i, cr in enumerate(predicted_range_arr):
    if i in ignore_indexes_arr:
        continue
    if cr[0] < real_val_arr[i] < cr[1]:
        correct += 1
    else:
        incorrect += 1

print(f'Out of all predictions {correct} of the intervals contained the actual value and {incorrect} did not !')


# Now, let's do the same thing, but only for predictions that have a very high confidence
correct = 0
incorrect = 0
not_used = 0
for i, cr in enumerate(predicted_range_arr):
    if i in ignore_indexes_arr:
        continue
    if prediction_confidence_arr[i] < high_percentile_confidence:
        not_used += 1
    elif cr[0] < real_val_arr[i] < cr[1]:
        correct += 1
    else:
        incorrect += 1

print(f'Out of all predictions {correct} of the intervals contained the actual value and {incorrect} did not ! We didn\'t use {not_used} predictions because the confidence was not high enough')


# Ok, let's try something useful. Maybe some of our wines are actually under-price and we could be selling them for more. Look for predictions where we have a very high confidence AND the lower bound of the interval is above the predicted value. Whilst not certain, this is a good indicator that a wine is overpriced (or maybe it's indeed cheap, but we are making a branding mistake by describing it as an "expensive" wine but selling it for cheap)

for i, cr in enumerate(predicted_range_arr):
    if i in ignore_indexes_arr:
        continue
    confidence = prediction_confidence_arr[i]
    if confidence >= high_percentile_confidence and real_val_arr[i] < cr[0] < cr[1]:
        print('The wine:\n' + str(test_data.iloc[i]) + f'\nIs potentially being sold at a price lower than it\'s real market value, we have a confidence of {confidence*100}% that this wine could be sold at a price between {cr[0]}$ and {cr[1]}$ but it\'s actual cost was {real_val_arr[i]}$ !')
