import mindsdb
import pandas as pd
from sklearn.metrics import accuracy_score


def run():
    backend = 'lightwood'

    mdb = mindsdb.Predictor(name='employee_retention_model')

    mdb.learn(from_data='dataset/train.csv', to_predict='Churn', backend=backend,
              output_categories_importance_dictionary={'Yes': 1, 'No': 0.5},
              disable_optional_analysis=True)

    test_df = pd.read_csv('dataset/test.csv')
    predictions = mdb.predict(when_data='dataset/test.csv',
                              unstable_parameters_dict={'always_use_model_predictions': True})

    predicted_val = [x.explanation['Churn']['predicted_value'] for x in predictions]

    real_val = list(map(str, list(test_df['Churn'])))

    accuracy = accuracy_score(real_val, predicted_val)

    #show additional info for each transaction row
    additional_info = [x.explanation for x in predictions]

    return {
        'accuracy': accuracy,
        'accuracy_function': 'accuracy_score',
        'backend': backend,
        'single_row_predictions': additional_info
    }


if __name__ == '__main__':
    print(run())