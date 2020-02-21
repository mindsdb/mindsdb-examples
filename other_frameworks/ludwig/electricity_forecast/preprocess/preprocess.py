import numpy as np
import pandas as pd
from ludwig.utils.data_utils import add_sequence_feature_column

data_kw = pd.read_csv("../dataset/energy_consumption.csv", index_col='TimeStamp', parse_dates=True)

freq = '2H'

# Length of the test data- Predicting energy consumption for 7 days
prediction_length = 7 * 12

start_dataset = pd.Timestamp("2014-01-01 00:00:00", freq=freq)
end_training = pd.Timestamp("2014-09-01 00:00:00", freq=freq)

# As customer type are in column names we have to convert to rows so that
# in mindsDB we can use as groupby column

df_final = data_kw[start_dataset:end_training + prediction_length - 1]

dfObj_Final = pd.DataFrame(columns=[])

for column in df_final.columns:
    print(column)
    if dfObj_Final.empty:
        df_1 = pd.DataFrame({'customer': np.repeat(column, df_final.shape[0]),
                             'power_consumed': df_final[column]},
                            index=df_final.index)
        dfObj_Final = df_1.copy()
    else:
        df_2 = pd.DataFrame({'customer': np.repeat(column, df_final.shape[0]),
                             'power_consumed': df_final[column]},
                            index=df_final.index)
        dfObj_Final = pd.concat([dfObj_Final, df_2])

# Converting to Ludwig timeseries format #

customer_list = list(dfObj_Final['customer'].unique())

dfObj_ludwig = pd.DataFrame(columns=[])

for customer in customer_list:
    print(customer)
    if dfObj_ludwig.empty:
        df_1 = dfObj_Final.loc[(dfObj_Final['customer'] == customer), :]
        add_sequence_feature_column(df_1, 'power_consumed', 84)
        dfObj_ludwig = df_1.copy()
    else:
        df_2 = dfObj_Final.loc[(dfObj_Final['customer'] == customer), :]
        df_2.reset_index(drop=True, inplace=True)
        add_sequence_feature_column(df_2, 'power_consumed', 84)
        dfObj_ludwig = pd.concat([dfObj_ludwig, df_2])

# Dividing the data into train and test data #

train = pd.DataFrame(columns=[])
test = pd.DataFrame(columns=[])

for customer in customer_list:
    print(customer)
    if train.empty and test.empty:
        df_1 = dfObj_ludwig.loc[(dfObj_ludwig['customer'] == customer), :]
        df_train = df_1.iloc[:-84]
        df_test = df_1.iloc[-84:]
        train = df_train.copy()
        test = df_test.copy()
    else:
        df_2 = dfObj_ludwig.loc[(dfObj_ludwig['customer'] == customer), :]
        df_2.reset_index(drop=True, inplace=True)
        df_train = df_2.iloc[:-84]
        df_test = df_2.iloc[-84:]
        train = pd.concat([train, df_train])
        test = pd.concat([test, df_test])

train.to_csv('../dataset/train.csv', header=True, index=False)
test.to_csv('../dataset/test.csv', header=True, index=False)
