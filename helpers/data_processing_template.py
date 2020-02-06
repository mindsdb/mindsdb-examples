import numpy as np
import pandas as pd


def main():
    df = pd.read_csv('raw_data/rawdataset.csv')

    np.random.seed(555)
    # 80/20 is split is our standard AVOID DOING ANYTHING ELESE !
    split_maks = np.random.rand(len(df)) < 0.8
    train_df = df[split_maks]
    test_df = df[~split_maks]

    train_df.to_csv('processed_data/train.csv', index=False)
    test_df.to_csv('processed_data/test.csv', index=False)

if __name__ == '__main__':
    main()
