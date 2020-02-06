from mindsdb import Predictor

# use the model to make predictions
result = Predictor(name='insurance_charges').predict(when={'age': 58, 'sex': 'male', 'smoker' : 'no', 'bmi': '49.06'})

print(result[0])