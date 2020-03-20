import mindsdb
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

predictor = mindsdb.Predictor(name='default_on_credit_preditor')
predictions = predictor.predict(when_data='dataset/test.csv')

explainations = [x.explanation for x in predictions]
predict_values = [str(x['default.payment.next.month']['predicted_value']) for x in explainations]

real_values = [str(x) for x in list(pd.read_csv('dataset/test.csv')['default.payment.next.month'])]

acc = balanced_accuracy_score(predict_values, real_values)
print(f'Balanced accuracy of: {acc} !')
