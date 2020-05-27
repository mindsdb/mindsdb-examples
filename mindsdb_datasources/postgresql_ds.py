
from mindsdb import Predictor, PostgresDS

# Get data
pg_ds = PostgresDS(query="SELECT age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target FROM public.heart", 
                   user="postgres", password="pass", database="heartdiseases")

# Train model
mdb = Predictor(name="heart-disease")
mdb.learn(from_data=pg_ds, to_predict="target")

# Get prediction
prediction = mdb.predict(when={"age": "40", "sex": 0, "chol": 180, "fbs": 0, "thal": 3, "exang": 0})
print(prediction[0].explanation)