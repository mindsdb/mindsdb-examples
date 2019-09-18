from sklearn.model_selection import train_test_split
import pandas


DATASET = 'creditcard-test.csv'

def read_file(dataset):
    lines = pandas.read_csv(dataset)
    return lines


def create_train_dataset(train_data):   
    df = pandas.DataFrame(data=train_data)
    df.to_csv("./train-{0}.csv".format(DATASET), sep=',',index=False)


def create_test_dataset(test_data):
    df = pandas.DataFrame(data=test_data)
    df.to_csv("./test-{0}.csv".format(DATASET), sep=',',index=False)


def train_test_split_dataset():
    dataset = read_file()
    X_train, X_test, = train_test_split(
        dataset, test_size=0.33, random_state=42)

    create_test_dataset(X_train)
    create_train_dataset(X_test)


if __name__ == "__main__":
    train_test_split_dataset(DATASET)