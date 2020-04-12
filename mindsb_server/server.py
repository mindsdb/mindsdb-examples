import requests
import pandas
import json

class MindsdbServerSDK:

    def __init__(self, data):
        self.url = data['server_url']
        self.dataset = data['dataset']
        self.to_predict = data['to_predict']

    def create_predictor(self):
        '''
        Learn to predict a given variable from existing data.
        '''
        data = dict(
            to_predict = self.to_predict,
            from_data = dataset
        )
        response = requests.request('PUT', self.url, json=data)
        return response

    
    def get_predictor_metadata(self):
        '''
        Get metadatadata for a given predictor.
        '''
        response = requests.request("GET", self.url)
        return response.text


    def query_predictor(self):
        '''
        Send prediction queries to a specific predictor
        '''
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
        response = requests.request('POST', self.url + '/predict', json=data)
        return response.text


if __name__ == "__main__":

    # feature that we want to predict
    to_predict = 'target'
    
    # the name of the predictor
    predictor_name = 'test_predictor'

    # url to dataset
    dataset = 'https://raw.githubusercontent.com/mindsdb/mindsdb-examples/master/benchmarks/heart_disease/processed_data/train.csv'
    
    data = dict(
        server_url = 'http://localhost:47334/predictors/' + predictor_name,
        dataset = dataset,
        to_predict = to_predict
    )
    server = MindsdbServerSDK(data)

    # create new predictor
    #server.create_predictor()

    # get predictor metadata
    print(server.get_predictor_metadata())

    # get predictions
    #print(server.query_predictor())