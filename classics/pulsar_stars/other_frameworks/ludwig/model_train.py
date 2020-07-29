import shutil

import pandas as pd
import yaml
from ludwig.api import LudwigModel
from sklearn.metrics import balanced_accuracy_score, accuracy_score


class PulsarStars:

    def __init__(self):
        self.x = 'Ludwig'

    def train_with_ludwig(self):
        path = 'results/'
        shutil.rmtree(path, ignore_errors=True)
        model_definition = yaml.load(open("model_defination.yaml"))
        print(model_definition)
        ludwig_model = LudwigModel(model_definition)
        train_stats = ludwig_model.train(data_train_csv="../../dataset/train.csv",
                                         model_name='pulsar_ludwig')
        print(train_stats)

    def predict_train_with_ludwig(self):
        ludwig_model = LudwigModel.load("results/api_experiment_pulsar_ludwig/model")
        predictions = ludwig_model.predict(data_csv="../../dataset/test.csv")
        predictions.to_csv(index=False, header=True, path_or_buf="predicted.csv")
        print(predictions)

    def calculate_score(self):
        y_pred = pd.read_csv('predicted.csv')
        y_true = pd.read_csv('../../dataset/test.csv')
        results = list(y_pred['target_class_predictions'])
        real = list(y_true['target_class'])

        accuracy = balanced_accuracy_score(list(map(int, real)), list(map(int, results)))
        print(f'Balacned accuracy score of {accuracy}')
        accuracy = accuracy_score(list(map(int, real)), list(map(int, results)))
        print(f'accuracy score of {accuracy}')



if __name__ == "__main__":
    import time
    start_time = time.time()
    tTest = PulsarStars()
    tTest.train_with_ludwig()
    print(time.time() - start_time)
    tTest.predict_train_with_ludwig()
    tTest.calculate_score()
