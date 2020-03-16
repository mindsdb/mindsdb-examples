import mindsdb
import pandas as pd
from sklearn.metrics import r2_score


def run():
    mdb = mindsdb.Predictor(name='tweets_forecast')

    backend = 'lightwood'

    mdb.learn(from_data='dataset/train_data.csv', to_predict='value', order_by=['timestamp'], backend=backend, window_size=10)

    predictions = mdb.predict(when='dataset/test_data.csv')

    pred_val = [x['sales'] for x in predictions]
    real_val = list(pd.read_csv(open('dataset/test_data.csv', 'r'))['value'])

    accuracy = r2_score(real_val, pred_val)
    print(f'Got an r2 score of: {accuracy}')
    return {
        'accuracy': accuracy
        ,'accuracy_function': 'balanced_accuracy_score'
        ,'backend': backend
    }

# Run as main
if __name__ == '__main__':
    print(run())
