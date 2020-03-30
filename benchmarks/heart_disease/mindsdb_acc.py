import mindsdb
import pandas as pd
import numpy as np
import math
from sklearn.metrics import r2_score, balanced_accuracy_score

def run():
    backend='lightwood'

    mdb = mindsdb.Predictor(name='hd')

    mdb.learn(from_data='processed_data/train.csv', to_predict='target', backend=backend, window_size=5)

    predictions = mdb.predict(when_data='processed_data/test.csv')

    pred_val = [int(x['target']) for x in predictions]
    real_val = [int(x) for x in list(pd.read_csv(open('processed_data/test.csv', 'r'))['target'])]

    accuracy = balanced_accuracy_score(real_val, pred_val)

    #show additional info for each transaction row
    additional_info = [x.explanation for x in predictions]
      
    return {
        'accuracy': accuracy,
        'backend': backend,
        'additional info': additional_info
    }

# Run as main
if __name__ == '__main__':
    print(run())
