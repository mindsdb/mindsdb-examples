import pandas as pd

df = pd.read_csv("../../../other_frameworks/ludwig/demand_sales_forecast/dataset/demand_forecast.csv", parse_dates=True,
                 date_parser=True)

df['date'] = pd.to_datetime(pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d %H:%M:%S'), format='%Y-%m-%d %H:%M:%S')
df.set_index('date', inplace=True)

freq = 'D'

start_dataset = pd.Timestamp("2013-01-01", freq=freq)
end_training = pd.Timestamp("2016-12-31", freq=freq)

store_list = []
for store in list(df['store'].unique()):
    for item in list(df['item'].unique()):
        store_list.append((store, item))

dfObj_Train = pd.DataFrame(columns=[])
dfObj_Test = pd.DataFrame(columns=[])

for i in store_list:
    print(i)
    if (dfObj_Train.empty & dfObj_Test.empty):
        df_cust_empty = df.loc[(df['store'] == i[0]) & (df['item'] == i[1]), :]
        dfObj_Train = df_cust_empty[start_dataset:end_training - 1]
        dfObj_Test = df_cust_empty[end_training:]
    else:
        df_cust = df.loc[(df['store'] == i[0]) & (df['item'] == i[1]), :]
        df_train = df_cust[start_dataset:end_training - 1]
        df_test = df_cust[end_training:]
        dfObj_Train = pd.concat([dfObj_Train, df_train])
        dfObj_Test = pd.concat([dfObj_Test, df_test])

dfObj_Train.to_csv("../dataset/train_data.csv", header=True, index=True)
dfObj_Test.to_csv("../dataset/test_data.csv", header=True, index=True)
