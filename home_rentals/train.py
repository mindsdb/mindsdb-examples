import mindsdb

# Instantiate a mindsdb Predictor
mdb = mindsdb.Predictor(name='real_estate_model')

# We tell the Predictor what column or key we want to learn and from what data
mdb.learn(
    from_data="https://s3.eu-west-2.amazonaws.com/mindsdb-example-data/home_rentals.csv", # the path to the file where we can learn from, (note: can be url)
    to_predict='rental_price', # the column we want to learn to predict given all the data in the file
)