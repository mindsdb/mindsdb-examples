import mindsdb

mdb = mindsdb.Predictor(name='lbs')
md = mdb.get_model_data()
print(md['model_analysis'][0]['confusion_matrix'])
