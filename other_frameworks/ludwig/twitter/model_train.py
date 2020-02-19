from ludwig.api import LudwigModel
import yaml
from sklearn.metrics import r2_score
import pandas as pd


class Twitter:

    def __init__(self):
        self.x = 'Ludwig'

    def train_with_ludwig(self):
        model_definition = yaml.load(open("model_defination.yaml"))
        print(model_definition)
        ludwig_model = LudwigModel(model_definition)
        train_stats = ludwig_model.train(data_train_csv="train.csv", data_validation_csv="valid.csv",
                                         model_name='twitter')
        print(train_stats)

    def predict_train_with_ludwig(self):
        ludwig_model = LudwigModel.load("results/api_experiment_twitter_0/model")
        predictions = ludwig_model.predict(data_csv="test.csv")
        predictions.to_csv(index=False, header=True, path_or_buf="predicted.csv")
        print(predictions)

    def calculate_score(self):
        y_pred = pd.read_csv('predicted.csv')
        y_true = pd.read_csv('test.csv', usecols=['value'])
        print(r2_score(list(y_true['value']), list(y_pred['value_predictions'])))


if __name__ == "__main__":
    tTest = Twitter()
    tTest.train_with_ludwig()
    tTest.predict_train_with_ludwig()
    tTest.calculate_score()
