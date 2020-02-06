from mindsdb import Predictor

# use the model to make predictions
result = Predictor(name='pc').predict(when={'radius': 23, 'texture': 12, 'perimeter' : 151, 'area': 0.143, 'fractal_dimension': 0.079})

print(result[0])