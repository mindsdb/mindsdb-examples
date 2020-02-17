import mindsdb
import pandas as pd
from sklearn.metrics import r2_score

def run():
    backend='lightwood'

    mdb = mindsdb.Predictor(name='lbs')

    mdb.learn(from_data='processed_data/train.csv', to_predict='cnt', backend=backend, window_size=5)

    predictions = mdb.predict(name='lbs').predict(when_data='processed_data/test.csv')
    test_df = pd.read_csv('processed_data/test.csv')

    print('Predictions result: ', predictions[0])
    pred_val = [x['cnt'] for x in predictions]
    real_val = list(pd.read_csv(open('processed_data/test.csv', 'r'))['cnt'])


    accuracy = r2_score(real_val, pred_val)
    print('Got an r2 score of:'.format(accuracy))

    return {
        'accuracy': accuracy
        ,'accuracy_function': 'balanced_accuracy_score'
        ,'backend': backend
    }


# Run as main
if __name__ == '__main__':
    print(run())