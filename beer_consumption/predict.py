from mindsdb import Predictor

# use the model to make predictions
result = Predictor(name='beer_consumption').predict(when={'Temperatura_Media': '27.3'})
print(result[0])