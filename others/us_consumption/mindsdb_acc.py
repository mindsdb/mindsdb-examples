import mindsdb
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

def run():

    mdb = mindsdb.Predictor(name='cons')

    timeseries_settings = {'timeseries_settings': {'order_by': ['T'], 
                                        'window': 5,
                                        'use_previous_target': True}}

    mdb.learn(from_data='data.csv', to_predict='Consumption', timeseries_settings=timeseries_settings['timeseries_settings'])

    predictions = mdb.predict(when_data='test.csv')

    #show additional info for each transaction row
    additional_info = [x.explanation for x in predictions]
      
    return {
        'additional info': additional_info
    }

# Run as main
if __name__ == '__main__':
    print(run())