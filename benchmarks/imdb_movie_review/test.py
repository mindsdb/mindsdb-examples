import mindsdb
from sklearn.metrics import accuracy_score


predictor = mindsdb.Predictor(name='movie_sentiment_predictor')
predictor.learn(from_data='train_sample.tsv', to_predict=['sentiment'])

accuracy_data = predictor.test('test_sample.tsv', accuracy_score)
accuracy_pct = accuracy_data['sentiment_accuracy'] * 100
print(f'Accuracy of {accuracy_pct}% !')
