from mindsdb import Predictor

# use the model to make predictions
p1 = {'Time': '23', 'V1': '1.32270726911234', 'V2': '-0.17404083293642',
     'V3': '0.434555031250987', 'V28': '0.0288223002307744', 'V27': '0.042335257639718', 'Amount': '16'}
p2 = {'Time': '24', 'V1': '1.23742903021294', 'V2': '0.0610425841868962',
     'V3': '0.380525879794222', 'V28': '0.011836231430416', 'Amount': '17.28'}
result = Predictor(name='cc-fraud').predict(when=p2)
print(result[0])