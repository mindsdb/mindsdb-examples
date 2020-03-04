import csv
from random import randint


all_reader = csv.reader(open('raw_data/all_data.csv', 'r'))

train_writer = csv.writer(open('dataset/train.csv', 'w'))
test_writer = csv.writer(open('dataset/test.csv', 'w'))

header = None
for row in all_reader:
    if header is None:
        header = row
        train_writer.writerow(header)
        test_writer.writerow(header)
        continue
    if randint(1,5) == 5:
        test_writer.writerow(row)
    else:
        train_writer.writerow(row)
