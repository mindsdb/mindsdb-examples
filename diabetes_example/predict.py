from mindsdb import Predictor

# use the model to make predictions
result = Predictor(name='diabetes-class').predict(when={'Plasma glucose concentration': 162, 'Diastolic blood pressure': 84,
 'Triceps skin fold thickness': 0, 'Age': 54, 'Number of times pregnant': 10})
print(result[0])