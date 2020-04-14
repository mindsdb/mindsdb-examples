import requests
import pandas as pd
import json
from mindsdb import FileDS

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

def analyse_dataset(url, from_data):
    '''
    Analayse dataset for a specific predictor.
    :param url: analyse_dataset service url
    :param from_data: data that contains the dataset
    :returns: dataset information as json
    '''   
    response = requests.request('GET', url + '/analyse_dataset?from_data='+ from_data)
    return response.text


def create_datasource(url, name, data):
    '''
    Add new datasource
    :param url: create datasource service url
    :param name: the name of the new datasource
    :param data: dict with name, datasources type and datset
    :returns: created datasource as json
    '''
    response = requests.request('PUT', url + name, json=data)
    return response.text


def get_datasource(url, name):
    '''
    Get specific datasource
    :param url: get datasource service url
    :param name: the name of the datasource
    :returns: datasource object as json
    '''
    datasource_api = url + name
    response = requests.request('GET', url= datasource_api)
    return response.text


if __name__ == "__main__":

    # feature that we want to predict
    to_predict = 'target'
    
    # the name of the predictor
    predictor_name = 'test_predictor'
    # endpoint url
    predictor_api = 'http://localhost:47334/predictors/' + predictor_name

    # url to dataset
    dataset = 'https://raw.githubusercontent.com/mindsdb/mindsdb-examples/master/benchmarks/heart_disease/processed_data/train.csv'
    
    # create new predictor   
    #print(create_predictor(predictor_api, dataset, to_predict))

    # get predictor metadata
    #print(get_predictor_metadata(predictor_api))

    # dictionary used for making a single prediction, each key is the name of an input column 
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
    #print(query_predictor(predictor_api, data))

    # analyze dataset    
    #print(analyse_dataset(predictor_api, dataset))

    # create datasource
    ds_api = 'http://127.0.0.1:47334/datasources/'
    ds_name = 'test_ds'
    ds_data = dict(
        name = 'test_ds',
        source_type = 'url',
        source = 'https://raw.githubusercontent.com/mindsdb/mindsdb-examples/master/benchmarks/heart_disease/processed_data/train.csv'
    )
    #create_datasource(ds_api, ds_name, ds_data)

    # get datasource
    #print(get_datasource(ds_api, ds_name))