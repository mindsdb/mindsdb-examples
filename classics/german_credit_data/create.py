from scipy.io.arff import loadarff
import pandas as pd
import numpy as np


# Load arff into pandas dataframe
dataset = loadarff(open('raw_data/credit_fruad.arff','r'))
data = []
for row in dataset[0]:
    cleaned_row = []
    for val in row:
        if type(val) == np.bytes_:
            val = val.decode('utf-8').lstrip("'").rstrip("'")
        cleaned_row.append(val)
    data.append(cleaned_row)
    
df = pd.DataFrame(data, columns=dataset[1].names())

# Shuffle the rows, using a seed so it always gives the same "random" arrangement.
df = df.sample(frac=1, random_state=7)

# Split into train and test
df_train = df.iloc[:round(len(df)*4/5)]
df_test = df.iloc[round(len(df)*4/5):]

df_train.to_csv('processed_data/train.csv', index=False)
df_test.to_csv('processed_data/test.csv', index=False)
