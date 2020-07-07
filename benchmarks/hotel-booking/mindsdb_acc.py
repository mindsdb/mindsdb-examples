import mindsdb
import sys
import pandas as pd
from sklearn.metrics import balanced_accuracy_score


def run():
    backend = 'lightwood'

    mdb = mindsdb.Predictor(name='hotel_booking')

    mdb.learn(from_data='processed_data/train.csv', to_predict='is_canceled', backend=backend)

    test_df = pd.read_csv('processed_data/test.csv')
    predictions = mdb.predict(when_data='processed_data/test.csv',
                              unstable_parameters_dict={'always_use_model_predictions': True})

    results = [str(x['is_canceled']) for x in predictions]
    real = list(map(str, list(test_df['is_canceled'])))

    accuracy = balanced_accuracy_score(real, results)

    #show additional info for each transaction row
    additional_info = [x.explanation for x in predictions]

    return {
        'accuracy': accuracy,
        'accuracy_function': 'balanced_accuracy_score',
        'backend': backend,
        'additional_info': additional_info
    }


if __name__ == '__main__':
    result = run()
    print(result)
