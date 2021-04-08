import pandas as pd
import os

def join_tables():
    folders = ['map', 'music', 'video', 'weather']
    fnames = ['labelled_train.csv', 'labelled_development.csv']
    trains = []
    devs = []

    # collect data from each folders
    for folder in folders:
        fpath = f'../data/{folder}/'
        trains.append(pd.read_csv(f'{fpath}/{fnames[0]}'))
        devs.append(pd.read_csv(f'{fpath}/{fnames[1]}'))

    # join tables
    train_df = pd.concat(trains)
    dev_df = pd.concat(devs)

    # output
    train_df.to_csv('../data/processed/train.csv', index=False)
    dev_df.to_csv('../data/processed/dev.csv', index=False)

if __name__ == '__main__':
    join_tables()