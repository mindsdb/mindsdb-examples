
import mindsdb
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, accuracy_score, r2_score

def run():

    mdb = mindsdb.Predictor(name='corona-data')

    target = ['Confirmed','Recovered','Deaths']
    # We tell the Predictor what column or key we want to learn and from what data
    mdb.learn(
        from_data="processed_data/train.csv",
        to_predict=target
    )

    test_df = pd.read_csv('processed_data/test.csv')
    predictions = mdb.predict(when_data='processed_data/test.csv')

    results = [str(x[target[0]]) for x in predictions]
    real = list(map(str,list(test_df[target[0]])))
    accuracy = r2_score(real, results)
    print(accuracy)
    
    return {
        'accuracy': accuracy
    }


if __name__ == '__main__':
    result = run()
    print(result)
