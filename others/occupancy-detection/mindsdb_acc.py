from mindsdb import Predictor

# train
Predictor(name='occu').learn(
    from_data="dataset/datatraining.csv",
    to_predict="Occupancy"
)

# predict
result = Predictor(name='occu').predict(when={'Temperature': '23.18', 'Humidity': '27.272', 'Light': '426', 'CO2': '721.25'})
print(result[0])