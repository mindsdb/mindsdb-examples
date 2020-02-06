from mindsdb import Predictor
import pandas as pd
from sklearn.metrics import r2_score
# use the model to make predictions

def run():
    backend='lightwood'

    mdb = mindsdb.Predictor(name='wineq')

    mdb.learn(from_data="dataset/winemag-train.csv", to_predict='price', backend=backend)

    predictions = Predictor(name='wineq').predict(when_data='dataset/winemag-test.csv')

    pred_val = [x['price'] for x in predictions]
    real_val = list(pd.read_csv(open('dataset/winemag-test.csv', 'r'))['price'])
    real_val = [x if str(x) != 'nan' else 0 for x in real_val]

    for v in pred_val:
        print(v,' !')
        float(v)

    for v in real_val:
        print(str(v), ' ?')
        float(v)


    accuracy = r2_score(real_val, pred_val)
    print(f'Got an r2 score of: {accuracy}')

    return {
        'accuracy': accuracy
        ,'accuracy_function': 'balanced_accuracy_score'
        ,'backend': backend
    }


# Run as main
if __name__ == '__main__':
    print(run())
