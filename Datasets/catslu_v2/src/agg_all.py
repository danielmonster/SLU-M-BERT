import pandas as pd
import os

def join_all():
    folders = ['catslu_traindev', 'catslu_test']
    fnames = ['train.csv', 'dev.csv', 'test.csv']
    trains, devs, tests = [], [], []
    for folder in folders:
        cur_path = f'../{folder}/data/processed/'
        flist = os.listdir(cur_path)
        if fnames[0] in flist:
            trains.append(pd.read_csv(f'{cur_path}/{fnames[0]}'))
        if fnames[1] in flist:
            devs.append(pd.read_csv(f'{cur_path}/{fnames[1]}'))
        if fnames[2] in flist:
            tests.append(pd.read_csv(f'{cur_path}/{fnames[2]}'))
    
    # join tables
    train_df = pd.concat(trains)
    dev_df = pd.concat(devs)
    test_df = pd.concat(tests)

    # output
    train_df.to_csv('../preprocessed/train.csv', index=False)
    dev_df.to_csv('../preprocessed/dev.csv', index=False)
    test_df.to_csv('../preprocessed/test.csv', index=False)

if __name__ == '__main__':
    join_all()