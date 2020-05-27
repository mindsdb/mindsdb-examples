

from mindsdb import Predictor, ClickhouseDS

# Get data
pg_ds = ClickhouseDS(query="SELECT number_of_rooms,number_of_bathrooms,sqft,location,days_on_market,initial_price,neighborhood,rental_price FROM default.home_rentalss",
                     password="pass", port=8123)

# Train model
mdb = Predictor(name="home-rentals")
mdb.learn(from_data=pg_ds, to_predict="rental_price")

# Get prediction
prediction = mdb.predict(when={"number_of_rooms": 3, 'initial_price': 2000})
print(prediction[0].explanation)