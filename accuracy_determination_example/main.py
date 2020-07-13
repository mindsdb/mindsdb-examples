from sklearn.metrics import balanced_accuracy_score
from mindsdb import Predictor


predictor = Predictor(name='heart_disease_predictor')
validation_data = predictor.learn('target', 'heart.csv')
accuracy = predictor.test(when_data=validation_data, accuracy_score_functions=balanced_accuracy_score)
print(accuracy)