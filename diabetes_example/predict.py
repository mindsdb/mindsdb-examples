from mindsdb import Predictor

# use the model to make predictions
result = Predictor(name='diabetes-class').predict(when={'Plasma glucose concentration': 93, 'Diastolic blood pressure': 31, 'Number of times pregnant': 1})
print(result[0])