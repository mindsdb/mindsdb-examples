import sys
sys.path.append('../../helpers')
from confidence_suffle import confidence_suffle
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

def accuracy_func(real, pred):
    return balanced_accuracy_score([str(x) for x in real], [str(x) for x in  pred])

def run():
    df_train = pd.read_csv('dataset/train.csv')
    df_test = pd.read_csv('dataset/test.csv')

    to_predict = 'target_class'
    columns = list(set(df_train.columns) - set([to_predict]))

    confidence_suffle(columns, df_train, df_test, accuracy_func, to_predict)

if __name__ == '__main__':
    result = run()
