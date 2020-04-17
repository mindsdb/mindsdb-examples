import mindsdb
import sys
import pandas as pd
from sklearn.metrics import balanced_accuracy_score


def run():
    backend = 'lightwood'

    mdb = mindsdb.Predictor(name='hotel_booking')

    mdb.learn(from_data='dataset/train.csv', to_predict='is_canceled', backend=backend,
              disable_optional_analysis=True)

    test_df = pd.read_csv('dataset/test.csv')
    predictions = mdb.predict(when_data='dataset/test.csv',
                              unstable_parameters_dict={'always_use_model_predictions': True})

    results = [str(x['is_canceled']) for x in predictions]
    real = list(map(str, list(test_df['is_canceled'])))

    accuracy = balanced_accuracy_score(real, results)

    return {
        'accuracy': accuracy
        , 'accuracy_function': 'balanced_accuracy_score'
        , 'backend': backend
    }


if __name__ == '__main__':
    result = run()
    print(result)
