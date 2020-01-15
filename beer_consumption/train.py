
import mindsdb
from mindsdb import Predictor

# We tell mindsDB what we want to learn and from what data
Predictor(name='beer_consumption').learn(
    to_predict='Consumo de cerveja', # the column we want to learn to predict given all the data in the file
    from_data="dataset/Consumo_cerveja_train.csv",# the path to the file where we can learn from, (note: can be url)
)

