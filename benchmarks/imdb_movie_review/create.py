import csv
import os
import random


for name in ['train','test']:
    reviews = []
    pos_dir = f'raw_dataset/{name}/pos/'
    neg_dir = f'raw_dataset/{name}/neg/'

    for f in os.listdir(pos_dir):
        with open(pos_dir + f, 'r') as fp:
            text = fp.read()
            reviews.append(text.replace('\t', '  ') + '\t' + 'pos')

    for f in os.listdir(neg_dir):
        with open(neg_dir + f, 'r') as fp:
            text = fp.read()
            reviews.append(text.replace('\t', '  ') + '\t' + 'neg')

    random.shuffle(reviews)

    with open(f'{name}.tsv', 'w') as fp:
        fp.write('review\tsentiment' + '\n' + '\n'.join(reviews))
