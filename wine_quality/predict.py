from mindsdb import Predictor

# use the model to make predictions
result = Predictor(name='wineq').predict(when={'Country': 'Spain', 'winery': 'Prim Family', 'region_2' : 'Napa'})
print(result[0])