from mindsdb import Predictor
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix


def run(sample=False):
    backend = 'lightwood'

    mdb = Predictor(name='german_data')

    mdb.learn(to_predict='class',from_data='processed_data/train.csv',backend=backend)

    predictions = mdb.predict(when_data='processed_data/test.csv')

    predicted_val = [x.explanation['class']['predicted_value'] for x in predictions]
    real_val = list(pd.read_csv('processed_data/test.csv')['class'])

    accuracy = balanced_accuracy_score(real_val, predicted_val)

    cm = confusion_matrix(real_val, predicted_val)
    print(cm)

    #show additional info for each transaction row
    additional_info = [x.explanation for x in predictions]

    return {
        'accuracy': accuracy,
        'accuracy_function': 'balanced_accuracy_score',
        'backend': backend,
        'single_row_predictions': additional_info
    }


# Run as main
if __name__ == '__main__':
    print(run())