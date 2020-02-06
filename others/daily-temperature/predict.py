from mindsdb import Predictor

# use the model to make predictions
result = Predictor(name='temp').predict(when={'Date': '1981-01-01'})
print(result[0])