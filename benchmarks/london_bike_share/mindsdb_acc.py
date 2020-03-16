import mindsdb
import pandas as pd
import numpy as np
import math
# use the model to make predictions

def pct_error(yt, yp, allowed_err=0):
    y_true = []
    y_pred = []
    for i in range(len(yt)):
        y_pred.append(yp[i])

        if yt[i] == 0:
            y_true.append(0.0001)
            y_pred[i] += 0.0001
        else:
            y_true.append(yt[i])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    pct_errors = np.abs((y_true - y_pred) / y_true)
    pct_errors = [x if x > allowed_err else 0 for x in pct_errors]

    return 1 - np.mean(pct_errors)

def run():
    backend='lightwood'

    mdb = mindsdb.Predictor(name='lbs')

    mdb.learn(from_data='processed_data/train.csv', to_predict='cnt', backend=backend, window_size=5)

    predictions = mdb.predict(when_data='processed_data/test.csv')

    pred_val = [int(x['cnt']) for x in predictions]
    real_val = [int(x) for x in list(pd.read_csv(open('processed_data/test.csv', 'r'))['cnt'])]

    accuracy = r2_score([math.log(x) if x > 0 else 0 for x in real_val], [math.log(x) if x > 0 else 0 for x in pred_val])
    print(f'Got an r2_score for the log predictions of: {accuracy}')

    accuracy = pct_error(real_val, pred_val, 0.05)
    print(f'Got a percentage accuracy score with error-margin 5% of: {accuracy}')

    accuracy = pct_error(real_val, pred_val)
    print(f'Got a percentage accuracy score of: {accuracy}')

    return {
        'accuracy': accuracy
        ,'accuracy_function': 'pct_error_0'
        ,'backend': backend
    }

# Run as main
if __name__ == '__main__':
    print(run())
