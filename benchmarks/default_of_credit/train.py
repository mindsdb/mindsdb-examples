import mindsdb


predictor = mindsdb.Predictor(name='default_on_credit_preditor')
predictor.learn(from_data='dataset/train.csv', to_predict='default.payment.next.month',stop_training_in_x_seconds=60)
