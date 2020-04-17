import mindsdb
import pandas as pd
from sklearn.metrics import r2_score
# use the model to make predictions

def run(sample=False):
    backend='lightwood'

    mdb = mindsdb.Predictor(name='wineq')

    #mdb.learn(from_data='processed_data/train.csv', to_predict='price', backend=backend, ignore_columns=['no'], use_gpu=True)

    predictions = mdb.predict(when_data=pd.read_csv('processed_data/test.csv'))

    print(predictions[0].explanation)

    confidence_range_arr = [x.explanation['price']['explanation']['confidence_interval'] for x in predictions]
    predicted_val_arr = [x.explanation['price']['predicted_value'] for x in predictions]

    real_val_arr = list(pd.read_csv(open('processed_data/test.csv', 'r'))['price'])
    real_val_arr = [x if str(x) != 'nan' else 0 for x in real_val_arr]

    correct = 0
    incorrect = 0

    for i, cr in enumerate(confidence_range_arr):
        if cr[0] < real_val_arr[i] < cr[1]:
            correct += 1
        else:
            incorrect += 1

    print(f'Out of all predictions {correct} of the intervals contained the actual value and {incorrect} did not !')

    accuracy = r2_score(real_val_arr, predicted_val_arr)
    print(f'Got an r2 score of: {accuracy}')

    return {
        'accuracy': accuracy
        ,'accuracy_function': 'balanced_accuracy_score'
        ,'backend': backend
    }


# Run as main
if __name__ == '__main__':
    print(run())
