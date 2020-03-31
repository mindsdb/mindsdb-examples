import mindsdb
import sys
import pandas as pd
from sklearn.metrics import balanced_accuracy_score


def run():
    backend = 'lightwood'

    mdb = mindsdb.Predictor(name='employee_retention_model')

    mdb.learn(from_data='dataset/train.csv', to_predict='Attrition', backend=backend,
              output_categories_importance_dictionary={'Yes': 1, 'No': 0.7},
              disable_optional_analysis=True)

    test_df = pd.read_csv('dataset/test.csv')
    predictions = mdb.predict(when_data='dataset/test.csv',
                              unstable_parameters_dict={'always_use_model_predictions': True})

    results = [str(x['Attrition']) for x in predictions]
    real = list(map(str, list(test_df['Attrition'])))

    accuracy = balanced_accuracy_score(real, results)

    return {
        'accuracy': accuracy
        , 'accuracy_function': 'balanced_accuracy_score'
        , 'backend': backend
    }


if __name__ == '__main__':
    result = run()
    print(result)
