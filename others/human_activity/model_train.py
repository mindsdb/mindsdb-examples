import pandas as pd
from mindsdb import Predictor
from sklearn.metrics import accuracy_score


class Robotics:

    def __init__(self):
        self.mindsDb = Predictor(name='human_activity')

    def train(self):
        print("model training started")
        self.mindsDb.learn(from_data="train.csv", to_predict=['target'],
                           order_by=['time'], window_size=128, group_by='id',
                           disable_optional_analysis=True)
        print("model training completed")

    def predict_test(self):
        print("test prediction started")
        y_real = pd.read_csv("test.csv")
        y_real = list(y_real["target"])
        results = self.mindsDb.predict(when_data="test.csv")
        y_pred = []
        for row in results:
            y_pred.append(row['target'])
        predictions = pd.DataFrame(y_pred)
        predictions.to_csv(index=False, header=True, path_or_buf="test_pred.csv")
        acc_score = accuracy_score(y_real, y_pred, normalize=True)
        acc_pct = round(acc_score * 100)
        print(pd.crosstab(pd.Series(y_pred), pd.Series(y_real)))
        test_cm = pd.crosstab(pd.Series(y_pred), pd.Series(y_real))
        test_cm.to_csv('test_final_cm.csv', header=True, index=True)
        print(f'Accuracy of : {acc_pct}%')
        print("test prediction completed")


if __name__ == "__main__":
    tTest = Robotics()
    tTest.train()
    tTest.predict_test()
