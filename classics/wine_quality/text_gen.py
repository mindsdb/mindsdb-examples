import pandas as pd
from mindsdb_native import Predictor


mdb = Predictor(name='description_predictor')

mdb.learn(from_data=pd.read_csv('processed_data/train.csv'), to_predict='description')

predictions = mdb.predict('processed_data/test.csv')

for pred in predictions:
    print(pred['description'])
