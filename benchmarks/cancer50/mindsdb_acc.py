import mindsdb
import sys
import pandas as pd
from sklearn.metrics import balanced_accuracy_score


def run(sample):
    backend='lightwood'

    mdb = mindsdb.Predictor(name='cancer_model')

    mdb.learn(from_data='processed_data/train.csv', to_predict='diagnosis', use_gpu=True, backend=backend, stop_training_in_x_seconds=180, equal_accuracy_for_all_output_categories=True)

    test_df = pd.read_csv('processed_data/test.csv')
    predictions = mdb.predict(when_data='processed_data/test.csv', unstable_parameters_dict={'always_use_model_predictions': True})

    results = [str(x['diagnosis']) for x in predictions]
    real = list(map(str,list(test_df['diagnosis'])))

    accuracy = balanced_accuracy_score(real, results)

    return {
        'accuracy': accuracy
        ,'accuracy_function': 'balanced_accuracy_score'
        ,'backend': backend
    }

if __name__ == '__main__':
    sample = bool(sys.argv[1]) if len(sys.argv) > 1 else False
    result = run(sample)
    print(result)
