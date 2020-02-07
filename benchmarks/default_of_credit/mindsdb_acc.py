from mindsdb import Predictor
import lightwood
import sys
import csv
from sklearn.metrics import balanced_accuracy_score, confusion_matrix


def run(sample):
    train_file = 'train.csv'
    test_file = 'test.csv'

    backend = 'lightwood'

    def get_real_test_data():
        test_reader = csv.reader(open(test_file, 'r'))
        next(test_reader, None)
        test_rows = [x for x in test_reader]
        return list(map(lambda x: int(x[-1]), test_rows))

    target_val_real = get_real_test_data()

    #lightwood.config.config.CONFIG.HELPER_MIXERS = False
    mdb = Predictor(name='default_on_credit_dp4')

    mdb.learn(to_predict='default.payment.next.month',from_data=train_file, backend=backend)

    predictions = mdb.predict(when_data=test_file)

    cfz = 0
    cfo = 0
    lcfz = 0.00001
    lcfo = 0.00001
    for p in predictions:
        tv = str(p['default.payment.next.month'])
        if tv == '0':
            cfz += p['default.payment.next.month_confidence']
            lcfz += 1
        else:
            cfo += p['default.payment.next.month_confidence']
            lcfo += 1

    print('Confidence for 0: ')
    print(cfz/lcfz)

    print('Confidence for 1: ')
    print(cfo/lcfo)

    target_val_predictions = list(map(lambda x: x['default.payment.next.month'], predictions))

    for i in range(len(target_val_predictions)):
        try:
            target_val_predictions[i] = int(str(target_val_predictions[i]))
        except:
            target_val_predictions[i] = 2

    accuracy = balanced_accuracy_score(target_val_real, target_val_predictions)
    print(f'Balacned accuracy score of {accuracy}')

    cm = confusion_matrix(target_val_real, target_val_predictions)

    return {
        'accuracy': accuracy
        ,'accuracy_function': 'balanced_accuracy_score'
        ,'backend': backend
    }


# Run as main
if __name__ == '__main__':
    sample = bool(sys.argv[1]) if len(sys.argv) > 1 else False
    print(run(sample))
