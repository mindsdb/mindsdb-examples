import mindsdb
import pandas as pd
from sklearn.metrics import r2_score


def run():
    backend = 'lightwood'

    mdb = mindsdb.Predictor(name='Admission_prediction_model')

    mdb.learn(from_data='dataset/train.csv', to_predict='Chance of Admit ', backend=backend)

    predictions = mdb.predict(when_data='dataset/test.csv',
                              unstable_parameters_dict={'always_use_model_predictions': True})

    pred_val = [x['Chance of Admit '] for x in predictions]
    real_val = list(pd.read_csv('dataset/test.csv', usecols=['Chance of Admit '])['Chance of Admit '])

    accuracy = r2_score(real_val, pred_val)
    print(f'Got an r2 score of: {accuracy}')

    return {
        'accuracy': accuracy
        , 'accuracy_function': 'r2_score'
        , 'backend': backend
    }


if __name__ == '__main__':
    results = run()
    print(results)
