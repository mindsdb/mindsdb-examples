from sklearn.model_selection import train_test_split
import pandas


def read_file(dataset=None):
    '''Return DataFrame from csv file
    '''
    lines = pandas.read_csv(dataset)
    return lines


def create_train_dataset(train_data=None, name='dataset'):    
    '''Creates new file with train data
    '''   
    df = pandas.DataFrame(data=train_data)
    df.to_csv("./train-{0}".format(name), sep=',',index=False)


def create_test_dataset(test_data=None, name='dataset'):
    '''Creates new file with test data
    '''
    df = pandas.DataFrame(data=test_data)
    df.to_csv("./test-{0}".format(name), sep=',',index=False)


def train_test_split_dataset(dataset=None, name=None):
    '''Split arrays or matrices into random train and test subsets
    '''
    dataset = read_file(dataset)
    X_train, X_test, = train_test_split(
        dataset, test_size=0.20, random_state=42)

    create_test_dataset(X_test, name)
    create_train_dataset(X_train, name)

def get_parser():
    '''
    Example use: python helpers.py --dataset daily-temperature/dataset/daily-min-temperatures.csv --name temperatures
    '''
    from argparse import ArgumentParser

    parser = ArgumentParser(description='')

    # Dataset Location
    parser.add_argument('--dataset',
                        dest='data_loc',
                        required=True,
                        help='File system location of the dataset.')

    # Dataset Location
    parser.add_argument('--name',
                        dest='data_name',
                        required=False,
                        help='The new name of train and test datasets.')
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    train_test_split_dataset(args.data_loc, args.data_name)