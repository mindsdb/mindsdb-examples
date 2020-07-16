import mindsdb
import csv
from sklearn.metrics import accuracy_score
import pandas as pd

# Path to our train and test files
train_file = 'train_sample.csv'
test_file = 'test_sample.csv'

# Create predicotr
predictor = mindsdb.Predictor(name='cifrar_100_predictor')

# Train it
predictor.learn(from_data=pd.read_csv(train_file).iloc[0:50], to_predict=['class'], ignore_columns=['superclass'])

# Predict the class for each image on the testing set
predictions = predictor.predict(when_data=pd.read_csv(test_file).iloc[0:5])
predicted_class = list(map(lambda x: x['class'], predictions))

# Get the real class for each image
real_class = []
reader = csv.reader(open(test_file, 'r'))
next(reader, None)
for row in reader:
    real_class.append(row[2])

# Evaluate the accuracy
acc = accuracy_score(real_class, predicted_class) * 100
print(f'Log loss accuracy of {acc}% for classes !')
