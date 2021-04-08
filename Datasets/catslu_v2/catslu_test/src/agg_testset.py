import pandas as pd
import os

def join_tables():
    folders = ['map', 'music', 'video', 'weather']
    fnames = ['labelled_train.csv', 'labelled_test.csv']
    trains = []
    tests = []

    # collect data from each folders
    for folder in folders:
        fpath = f'../data/{folder}/'
        if fnames[0] in os.listdir(f'../data/{folder}/'):
            trains.append(pd.read_csv(f'{fpath}/{fnames[0]}'))
        tests.append(pd.read_csv(f'{fpath}/{fnames[1]}'))

    # join tables
    train_df = pd.concat(trains)
    test_df = pd.concat(tests)

    # output
    train_df.to_csv('../data/processed/train.csv', index=False)
    test_df.to_csv('../data/processed/test.csv', index=False)

if __name__ == '__main__':
    join_tables()