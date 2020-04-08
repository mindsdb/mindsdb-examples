import mindsdb

# Instantiate a mindsdb Predictor
mdb = mindsdb.Predictor(name='diabetes-class')

# We tell the Predictor what column or key we want to learn and from what data
mdb.learn(
    from_data="dataset/diabetes-train.csv",
    to_predict='Class',
    equal_accuracy_for_all_output_categories=True
)
