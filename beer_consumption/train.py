
import mindsdb
from mindsdb import Predictor
from mindsdb.config import CONFIG

# We tell mindsDB what we want to learn and from what data
Predictor(name='beer_consumption').learn(
    to_predict='Consumo_de_cerveja', # the column we want to learn to predict given all the data in the file
    from_data="Consumo_cerveja.csv" # the path to the file where we can learn from, (note: can be url)
)

