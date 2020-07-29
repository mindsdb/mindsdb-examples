import mindsdb
import sys
import pandas as pd
from sklearn.metrics import balanced_accuracy_score


def run(sample):
    backend='lightwood'

    mdb = mindsdb.Predictor(name='churn_model')

    mdb.learn(from_data=pd.read_csv('dataset/train.csv'), to_predict='Exited', backend=backend, ignore_columns=['RowNumber','CustomerId','Surname'])

    test_df = pd.read_csv('dataset/test.csv')
    predictions = mdb.predict(when_data='dataset/test.csv')

    results = [x['Exited'] for x in predictions]
    real = list(test_df['Exited'])

    accuracy = balanced_accuracy_score(list(map(int,real)), list(map(int,results)))
    print(f'Balacned accuracy score of {accuracy}')

    return {
        'accuracy': accuracy
        ,'accuracy_function': 'balanced_accuracy_score'
        ,'backend': backend
    }

if __name__ == '__main__':
    sample = bool(sys.argv[1]) if len(sys.argv) > 1 else False
    result = run(sample)
    print(result)
