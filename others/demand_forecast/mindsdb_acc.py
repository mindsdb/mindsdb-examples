import mindsdb
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score


def run():
    mdb = mindsdb.Predictor(name='demand_forecast')

    backend = 'lightwood'

    mdb.learn(from_data='dataset/train_data.csv', to_predict='sales', order_by=['date'], group_by=['store', 'item'],
              backend=backend, window_size=7)

    predictions = mdb.predict(when='dataset/test_data.csv')

    pred_val = [x['sales'] for x in predictions]
    real_val = list(pd.read_csv(open('dataset/test_data.csv', 'r'))['sales'])

    accuracy = r2_score(real_val, pred_val)
    print(f'Got an r2 score of: {accuracy}')
    return {
        'accuracy': accuracy
        , 'accuracy_function': 'balanced_accuracy_score'
        , 'backend': backend
    }


# Run as main
if __name__ == '__main__':
    print(run())
