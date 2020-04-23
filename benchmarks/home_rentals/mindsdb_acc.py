import mindsdb
import lightwood
import pandas as pd
import numpy as np
import math
from sklearn.metrics import r2_score


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

def run(sample=False):
    backend='lightwood'
    lightwood.config.config.CONFIG.HELPER_MIXERS = False

    # Instantiate a mindsdb Predictor
    mdb = mindsdb.Predictor(name='home_rentals')

    # We tell the Predictor what column or key we want to learn and from what data
    print(111)

    train_df = pd.read_csv('dataset/train.csv')
    #train_df = train_df.drop(columns=['initial_price'])

    #'''
    mdb.learn(
        from_data=train_df, # the path to the file where we can learn from, (note: can be url)
        to_predict='rental_price' # the column we want to learn to predict given all the data in the file
    	,stop_training_in_x_seconds=10
    )
    #'''

    test_df = pd.read_csv('dataset/test.csv')

    for drop_cols in [['number_of_rooms','number_of_bathrooms','location','days_on_market', 'neighborhood'], []]:
        print(f'Predicting without columns: {drop_cols}')
        predictions = mdb.predict(when_data=test_df.drop(columns=drop_cols))

        pred_val = [x.explain()['rental_price'][0]['model_result']['value'] for x in predictions]
        real_val = list(pd.read_csv(open('dataset/test.csv', 'r'))['rental_price'])
        real_val = [x if str(x) != 'nan' else 0 for x in real_val]

        accuracy = r2_score([math.log(x) if x > 0 else 0 for x in real_val], [math.log(x) if x > 0 else 0 for x in pred_val])
        print(f'Got an r2_score for the log predictions of: {accuracy}')

        accuracy = pct_error(real_val, pred_val, 0.05)
        print(f'Got a percentage accuracy score with error-margin 5% of: {accuracy}')

        accuracy = pct_error(real_val, pred_val)
        print(f'Got a percentage accuracy score of: {accuracy}')
        print('\n\n------------------------\n\n')

    print([(x['importance_score'],x['column_name']) for x in mdb.get_model_data()['data_analysis']['input_columns_metadata']])

    return {
        'accuracy': accuracy
        ,'accuracy_function': 'pct_error_0'
        ,'backend': backend
    }


# Run as main
if __name__ == '__main__':
    print(run())
