from mindsdb import Predictor
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix


def run(sample=False):
    backend = 'lightwood'

    mdb = Predictor(name='german_data')

    mdb.learn(to_predict='class',from_data='processed_data/train.csv', stop_training_in_x_seconds=80,backend=backend, sample_margin_of_error=0.1, equal_accuracy_for_all_output_categories=True, use_gpu=True)

    predictions = mdb.predict(when_data='processed_data/test.csv', use_gpu=True)

    predicted_class = [x['class'] for x in predictions]
    real_class = list(pd.read_csv('processed_data/test.csv')['class'])

    accuracy = balanced_accuracy_score(real_class, predicted_class)
    print(f'Balacned accuracy score of {accuracy}')

    cm = confusion_matrix(real_class, predicted_class)

    print(cm)
    return {
        'accuracy': accuracy
        ,'accuracy_function': 'balanced_accuracy_score'
        ,'backend': backend
    }


# Run as main
if __name__ == '__main__':
    print(run())
