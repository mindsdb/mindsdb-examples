from ludwig.api import LudwigModel
import yaml
from sklearn.metrics import r2_score
import pandas as pd
import shutil


class Temperature:

    def __init__(self):
        self.x = 'Ludwig'

    def train_with_ludwig(self):
        path = 'results/'
        shutil.rmtree(path, ignore_errors=True)
        model_definition = yaml.load(open("model_defination.yaml"))
        print(model_definition)
        ludwig_model = LudwigModel(model_definition)
        train_stats = ludwig_model.train(data_train_csv="dataset/train.csv",
                                         model_name='temperature')
        print(train_stats)

    def predict_train_with_ludwig(self):
        ludwig_model = LudwigModel.load("results/api_experiment_temperature/model")
        predictions = ludwig_model.predict(data_csv="dataset/test.csv")
        predictions.to_csv(index=False, header=True, path_or_buf="predicted.csv")
        print(predictions)

    def calculate_score(self):
        y_pred = pd.read_csv('predicted.csv')
        y_true = pd.read_csv('dataset/test.csv', usecols=['temperature'])
        print(r2_score(list(y_true['temperature']), list(y_pred['temperature_predictions'])))


if __name__ == "__main__":
    tTest = Temperature()
    tTest.train_with_ludwig()
    tTest.predict_train_with_ludwig()
    tTest.calculate_score()
