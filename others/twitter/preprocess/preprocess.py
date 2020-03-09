import pandas as pd

url = "https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv"
df_twitter = pd.read_csv(url, header=0, index_col=0)
df_twitter_index = pd.DatetimeIndex(df_twitter.index, freq='5T')
df_twitter.index = df_twitter_index
df_twitter = df_twitter.resample('5T').sum()

freq = '5T'

start_dataset = pd.Timestamp("2015-02-26 21:40:00", freq=freq)
end_training = pd.Timestamp("2015-04-11 21:00:00", freq=freq)

train = df_twitter[start_dataset:end_training - 1]
test = df_twitter[end_training:]
train.to_csv('../dataset/train_data.csv', header=True, index=True)
test.to_csv('../dataset/test_data.csv', header=True, index=True)
