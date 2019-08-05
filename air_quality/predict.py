from mindsdb import Predictor

# use the model to make predictions
result = Predictor(name='AirQua').predict(when={'RH': 25, 'AH': 1})

print(result[0])