import mindsdb
import sys
import pandas as pd
from sklearn.metrics import acc_fun


def run(sample):
    backend='lightwood'

    mdb = mindsdb.Predictor(name='model_name')

    mdb.learn(from_data='processed_data/train.csv', to_predict='target_class', use_gpu=True, backend=backend, equal_accuracy_for_all_output_categories=True)

    test_df = pd.read_csv('processed_data/test.csv')
    predictions = mdb.predict(when_data='processed_data/test.csv')

    results = [x['target_class'] for x in predictions]
    real = list(test_df['target_class'])

    accuracy = acc_fun(real, result)
    print(f'Balacned accuracy score of {accuracy}')

    return {
        'accuracy': accuracy
        ,'accuracy_function': 'acc_fun'
        ,'backend': backend
    }

if __name__ == '__main__':
    sample = bool(sys.argv[1]) if len(sys.argv) > 1 else False
    result = run(sample)
    print(result)
