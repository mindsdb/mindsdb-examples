from mindsdb import Predictor

# use the model to make predictions
result = Predictor(name='restaurant_score').predict(when={'inspection_score': 92, 'business_state': 'CA'})
print(result[0])