from mindsdb import Predictor

# use the model to make predictions
result = Predictor(name='btc-price').predict(when={'txVolume(USD)': 6739584540.73, 'adjustedTxVolume(USD)': 3868097401.91,
    'txCount': 204913, 'exchangeVolume(USD)': 7394019840, 'generatedCoins': 1875, 'fees': 35.900, 'blockCount':  150})
print(result[0])