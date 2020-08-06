from mindsdb_native import Predictor

mdb = Predictor(name='coffee_predictor')
mdb.learn(from_data='data.tsv', to_predict=['Coffe_Malt', 'Chocolat', 'Gold', 'Medium_Barley', 'Dark_Barley', 'Dandelion', 'Beets', 'Chicory_Roots', 'Figs'])
