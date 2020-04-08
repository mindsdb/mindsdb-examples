from mindsdb import Predictor
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import lightwood


lightwood.config.config.CONFIG.FORCE_HELPER_MIXERS = True
# use the model to make predictions
result = Predictor(name='diabetes-class').predict(when_data='dataset/diabetes-test.csv')
real = pd.read_csv('dataset/diabetes-test.csv')['Class']

p = [str(x['Class']) for x in result]
print(p)
r = [str(x) for x in real]
print(r)
print(accuracy_score(r,p))
