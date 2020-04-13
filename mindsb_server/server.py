import requests
import pandas
import json


def create_predictor(url, dataset, to_predict):
    '''
    Learn to predict a given variable from existing data.
    :param url: the create predictor service url
    :param dataset: the data to train the predictor
    :param to_predict: feature to predict
    :returns: status code
    '''
    data = dict(
        to_predict = to_predict,
        from_data = dataset
    )
    response = requests.request('PUT', url, json=data)
    return response


def get_predictor_metadata(url):
    '''
    Get metadatadata for a given predictor.
    :param url: get predictor metadata service url
    :returns: metadata info as json
    '''
    response = requests.request("GET", url)
    return response.text


def query_predictor(url, data):
    '''
    Send prediction queries to a specific predictor.
    :param url: query predictor service url
    :param data: the data contains the query
    :returns: model predictions as json
    '''
    response = requests.request('POST', url + '/predict', json=data)
    return response.text


if __name__ == "__main__":

    # feature that we want to predict
    to_predict = 'target'
    
    # the name of the predictor
    predictor_name = 'test_predictor'
    # endpoint url
    url = 'http://localhost:47334/predictors/' + predictor_name

    # url to dataset
    dataset = 'https://raw.githubusercontent.com/mindsdb/mindsdb-examples/master/benchmarks/heart_disease/processed_data/train.csv'
    

    # create new predictor
    #print(create_predictor(url, dataset, to_predict))

    # get predictor metadata
    #print(get_predictor_metadata(url))

    # get predictions
    data = {
        'when': {
            'age': '25',
            'sex': '0',
            'chol': '150',
            'thalach': '170',
            'exang': '0',
            'fbs': '0',
            'thal': '3',
        }
    }
    print(query_predictor(url, data))