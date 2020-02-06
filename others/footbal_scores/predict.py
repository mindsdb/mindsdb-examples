from mindsdb import Predictor

# use the model to make predictions
result = Predictor(name='score').predict(when={'home_team': 'Brazil', 'away_team': 'China', 'tournament' : 'Friendly',
        'country': 'Brazil'})

print(result[0])