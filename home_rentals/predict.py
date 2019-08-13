from mindsdb import Predictor

# use the model to make predictions
result = Predictor(name='home_rentals').predict(when={'number_of_rooms': 3, 'number_of_bathrooms': 1, 'neighborhood' : 'south_side'})

print(result[0])