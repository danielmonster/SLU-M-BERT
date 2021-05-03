import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from ast import literal_eval
import json
import argparse


# mkdir -p memory/roberta/en
# python3 roberta/preprocess_en.py --data_dir=Datasets/smart-devices-en-fr/ \
# --memory_dir=memory/roberta/en

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_ratio', type=float, default=0.25)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--memory_dir', type=str, required=True)
    return parser.parse_args()



# Ignore all the "*-far.csv" files because of low audio quality




def load_df(data_dir):
    df_light_en = pd.read_csv(os.path.join(data_dir, "smart-lights-en-close.csv"), header=0)
    df_speaker_en = pd.read_csv(os.path.join(data_dir, "smart-speaker-en-close.csv"), header=0)
    return df_light_en, df_speaker_en


def get_xy(df):
    df_phones = df['phones'].apply(lambda x: literal_eval(x))
    phones_samples = df_phones.to_numpy()
    labels = df['label'].to_numpy()
    X_all = []
    for sample in phones_samples:
        x = " ".join(sample)
        X_all.append(x)
    X_all = np.array(X_all, dtype=object)
    return X_all, labels


def save_numpy_data(data, filename):
    with open(filename, 'wb') as f:
        np.save(f, data)
    print("Saved in {}".format(filename))



def main(args):
    df_light_en, df_speaker_en = load_df(args.data_dir)
    df_en_all = pd.concat([df_light_en, df_speaker_en], ignore_index=True)
    
    # labels in english dataset are [0, 1, 2, 3, 4, 5, 8], change 8 to 6
    mask = (df_en_all['label'] == df_en_all['label'].max())
    newval = len(df_en_all['label'].unique()) - 1
    df_en_all['label'].mask(mask, newval, inplace=True)

    # Train-test split with ratio = 0.15
    ratio = args.split_ratio
    train_en, valid_en = train_test_split(df_en_all, random_state=1, test_size=ratio, shuffle=True)
    dev_en, test_en = train_test_split(valid_en, random_state=1, test_size=0.5)
    # Reset indices of dataframe
    train_en = train_en.reset_index(drop=True)
    dev_en = dev_en.reset_index(drop=True)
    test_en = test_en.reset_index(drop=True)


    # Save x and y
    train_x, train_labels = get_xy(train_en)
    save_numpy_data(train_x, os.path.join(args.memory_dir, "train_x.npy"))
    save_numpy_data(train_labels, os.path.join(args.memory_dir, "train_y.npy"))

    dev_x, dev_labels = get_xy(dev_en)
    save_numpy_data(dev_x, os.path.join(args.memory_dir, "dev_x.npy"))
    save_numpy_data(dev_labels, os.path.join(args.memory_dir, "dev_y.npy"))

    test_x, test_labels = get_xy(test_en)
    save_numpy_data(test_x, os.path.join(args.memory_dir, "test_x.npy"))
    save_numpy_data(test_labels, os.path.join(args.memory_dir, "test_y.npy"))




if __name__ == '__main__':
    args = get_args()
    main(args)
