from mindsdb import Predictor

# use the model to make predictions
result = Predictor(name='player-stats').predict(when={'home_team': 'Scotland', 'away_team': 'England', 'tournament' : 'Friendly',
        'country': 'Scotland'})

print(result[0])