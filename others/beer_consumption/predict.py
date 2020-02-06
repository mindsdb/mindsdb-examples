from mindsdb import Predictor

# use the model to make predictions
result = Predictor(name='beer_consumption').predict(when={'Temperatura Media': '27.3','Final de Semana': 0, 'Precipitacao': '1.2'})
print(result[0])