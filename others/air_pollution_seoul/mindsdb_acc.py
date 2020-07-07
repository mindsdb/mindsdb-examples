import mindsdb
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

def run():

    mdb = mindsdb.Predictor(name='air_pl')

    mdb.learn(from_data='processed_data/train.csv', to_predict='SO2')

    predictions = mdb.predict(when_data='processed_data/test.csv')

    #show additional info for each transaction row
    additional_info = [x.explanation for x in predictions]
      
    return {
        'additional info': additional_info
    }

# Run as main
if __name__ == '__main__':
    print(run())