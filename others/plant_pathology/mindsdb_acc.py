import mindsdb
import csv
from sklearn.metrics import accuracy_score
import sys
import pandas


def run():
    train_file = 'train.csv'
    test_file = 'test.csv'

    backend = 'lightwood'

    predictor = mindsdb.Predictor(name='plant_pathology')

    train_df = pandas.read_csv(train_file)
    predictor.learn(from_data=train_df, to_predict=['class'], disable_optional_analysis=True, use_gpu=False,
                    backend=backend, stop_training_in_x_seconds=round((3600 * 2)))

    test_df = pandas.read_csv(test_file)
    predictions = predictor.predict(when_data=test_df, unstable_parameters_dict={'always_use_model_prediction': True},
                                    use_gpu=False)

    predicted_class = list(map(lambda x: x['class'], predictions))

    real_class = []
    first = True
    with open(test_file) as raw_csv_fp:
        reader = csv.reader(raw_csv_fp)
        for row in reader:
            if first:
                first = False
            else:
                real_class.append(row[2])

    acc = accuracy_score(real_class, predicted_class) * 100
    print(f'Log loss accuracy of {acc}% for classes !')

    return {
        'accuracy': acc
        , 'accuracy_function': 'accuracy_score'
        , 'backend': backend
    }


# Run as main
if __name__ == '__main__':
    run()
