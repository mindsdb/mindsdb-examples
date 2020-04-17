import requests
import json

root_url = 'http://localhost:47334'

# Specify the training data and the value you want predicted
train_data_url = 'https://raw.githubusercontent.com/mindsdb/mindsdb-examples/master/benchmarks/heart_disease/raw_data/heart.csv'
train_data = {
    'to_predict': 'target',
    'from_data': train_data_url
    }

# Run a statistical analysis of the data to gather insights about it
response = requests.request('GET', f'{root_url}/predictors/any/analyse_dataset', params=train_data)
print(response.text)

# Create a Predictor (this will automatically start training the Predictor with the training data specified to predicted the outputs)
predictor_name = 'heart_disease_predictor'
response = requests.request('PUT', f'{root_url}/predictors/{predictor_name}', json=train_data)
print(response.text)

# Get a metadata for the created predictor
response = requests.request('GET', f'{root_url}/predictors/{predictor_name}')
print(response.text)

# Get a prediction based on some incomplete data
test_data = {
    'when': {
        'age': '25',
        'sex': '0',
        'chol': '150',
    },
    'format_flag': 'new_explain' # The format of the returned predictions, new_explain is the best one currently, I recommend only using this, other format serve backwards compatibility purposes
}
response = requests.request('POST',f'{root_url}/predictors/{predictor_name}/predict', json=test_data)
# Note: We only sent one prediction in the form of the `when` parameters, the array being returned will contain multiple predictions if more than one row/object is sent, the order of the predictions is the same as the ordering of the array being sent
first_prediction = response.json()[0]['target'] 

predicted_value = first_prediction[0]['value']
confidence_percentage = round(100* first_prediction[0]['confidence'])
print(f'Predicted a value of {predicted_value} with a confidence of {confidence_percentage}%')

# Note: To increase the prediction confidence we need to add more data to `when` parameters. Usually include those features that have high score in column importance metrics 
new_test_data = {
    'when': {
        'age': '25',
        'sex': '0',
        'chol': '150',
        'thalach': '170',
        'exang': '0',
        'fbs': '0',
        'thal': '3',
    },
    'format_flag': 'new_explain' 
}
response = requests.request('POST',f'{root_url}/predictors/{predictor_name}/predict', json=new_test_data)
new_first_prediction = response.json()[0]['target'] 

predicted_value = new_first_prediction[0]['value']
confidence_percentage = round(100* new_first_prediction[0]['confidence'])
print(f'New predicted value of {predicted_value} with a confidence of {confidence_percentage}%')

# Create a new DataSource. MindsDB Datasources are simmilar to pandas DataFrames with additonal clean and parse functions
datasource_name = 'heart_disease_ds'
datasource_url = f'{root_url}/datasources/{datasource_name}'
ds_data = {
     'name' : datasource_name,
    'source_type' :'url',
    'source' :train_data_url
}
response = requests.request('PUT', datasource_url, json=ds_data)

# Get created datasource
response = requests.request('GET', url=datasource_url)
print(f'{datasource_name} datasource: {response.text}')