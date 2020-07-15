from sklearn.metrics import balanced_accuracy_score
from mindsdb import Predictor
import mindsdb


# Method 1
predictor = Predictor(name='heart_disease_predictor')
predictor.learn('target', 'heart.csv')
accuracy = predictor.test(when_data=predictor.transaction.input_data.validation_df, accuracy_score_functions=balanced_accuracy_score)
print(accuracy)

# Method 2
accuracy =  mindsdb.F.validate('target', 'heart.csv',balanced_accuracy_score)
print(accuracy)

# Method 3 (work in progress)
