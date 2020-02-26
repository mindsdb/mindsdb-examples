import pandas as pd
from mindsdb import Predictor
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score

class Temperature:

    def __init__(self):
        self.mindsDb = Predictor(name='temperature')

    def temp_train(self):
        self.mindsDb.learn(to_predict='temperature', from_data='train.csv',
                           window_size=20, order_by='index')

    def temp_predict(self):
        y_real = pd.read_csv("test.csv")
        results = self.mindsDb.predict(when_data="test.csv")
        y_pred = []
        for row in results:
            y_pred.append(row['temperature'])
        predictions = pd.DataFrame(y_pred)
        predictions.to_csv(index=False, header=True, path_or_buf="test_pred.csv")
        print(r2_score(y_real['temperature'].tolist(), pd.Series(y_pred).tolist()))


if __name__ == "__main__":
    tTest = Temperature()
    tTest.temp_train()
    tTest.temp_predict()
