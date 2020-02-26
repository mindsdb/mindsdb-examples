import pandas as pd
from mindsdb import Predictor
from sklearn.metrics import r2_score


class Electricity:

    def __init__(self):
        self.mindsDb = Predictor(name='demand_30')

    def train(self):
        self.mindsDb.learn(to_predict='power_consumed', from_data='dataset/mdb_train.csv',
                           window_size=84, order_by=['TimeStamp'], group_by=['customer'],
                           disable_optional_analysis=True)

    def test_predict(self):
        y_real = pd.read_csv("mdb_test.csv")
        y_real = list(y_real["power_consumed"])
        results = self.mindsDb.predict(when_data="dataset/mdb_test.csv")
        y_pred = []
        for row in results:
            y_pred.append(row['power_consumed'])
        print(r2_score(y_real, y_pred))


if __name__ == "__main__":
    tTest = Electricity()
    tTest.train()
    tTest.test_predict()
