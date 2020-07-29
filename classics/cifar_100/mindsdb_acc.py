import mindsdb
import csv
from sklearn.metrics import accuracy_score
import sys
import pandas as pd

def run(sample):
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    if sample:
        train_df = train_df.loc[train_df['superclass'] == 'fish'].reset_index()
        test_df = test_df.loc[test_df['superclass'] == 'fish'].reset_index()

    backend='lightwood'

    predictor = mindsdb.Predictor(name='CIFRAR_Model')

    predictor.learn(from_data=train_df, to_predict=['class'], ignore_columns='index', advanced_args={'use_selfaware_model': False,'force_disable_cache': False})

    #

    predictions = predictor.predict(when_data=test_df)

    predicted_class = list(map(lambda x: x['class'], predictions))
    real_class = list(test_df['class'])

    acc = accuracy_score(real_class, predicted_class) * 100
    print(f'Log loss accuracy of {acc}% for classes !')

    return {
        'accuracy': acc
        ,'accuracy_function': 'accuracy_score'
        ,'backend': backend
    }


# Run as main
if __name__ == '__main__':
    sample = bool(int(sys.argv[1])) if len(sys.argv) > 1 else False
    run(sample)
