import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from ast import literal_eval
import json
import argparse


# python3 preprocess_enfr.py --lang=en
# python3 preprocess_enfr.py --lang=fr


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_ratio', type=float, default=0.25)
    parser.add_argument('--lang', type=str, required=True, help="Either 'en' (english) or 'fr' (french)")
    parser.add_argument('--memory_dir', type=str, default="../memory/enfr")
    return parser.parse_args()

args = get_args()


# Ignore all the "*-far.csv" files because of low audio quality

train_light_en = "../Datasets/smart-devices-en-fr/smart-lights-en-close.csv"
train_speaker_en = "../Datasets/smart-devices-en-fr/smart-speaker-en-close.csv"
train_speaker_fr = "../Datasets/smart-devices-en-fr/smart-speaker-fr-close.csv"
# train_light_fr missing



def build_phone_vocab(df_all):
    phone2idx = {}
    # Convert string of list to string, i.e., "['a', 'b']" to ['a', 'b']
    df_phones = df_all['phones'].apply(lambda x: literal_eval(x))
    phones_samples = df_phones.to_numpy()
    for sample in phones_samples:
        for phone in sample:
            if phone not in phone2idx:
                phone2idx[phone] = len(phone2idx)
    return phone2idx


def save_train_test(df_train, df_dev, df_test, save_dir):
    df_train.to_csv(os.path.join(save_dir, "train.csv"), encoding='utf-8', index=False)
    df_dev.to_csv(os.path.join(save_dir, "dev.csv"), encoding='utf-8', index=False)
    df_test.to_csv(os.path.join(save_dir, "test.csv"), encoding='utf-8', index=False)
    print("train.csv, dev.csv, and valid.csv saved in {}".format(os.path.abspath(save_dir)))


def save_phone_idx(phone2idx, save_dir):
    with open(os.path.join(save_dir, "phone_idx.json"), 'w', encoding='utf-8') as f:
        json.dump(phone2idx, f)
    print("phone_idx.json saved in {}".format(os.path.abspath(save_dir)))



def load_df():
    df_light_en = pd.read_csv(train_light_en, header=0)
    df_speaker_en = pd.read_csv(train_speaker_en, header=0)
    df_speaker_fr = pd.read_csv(train_speaker_fr, header=0) 
    return df_light_en, df_speaker_en, df_speaker_fr


def get_xy(df, phone2idx):
    df_phones = df['phones'].apply(lambda x: literal_eval(x))
    phones_samples = df_phones.to_numpy()
    labels = df['label'].to_numpy()
    X_all = []
    for sample in phones_samples:
        x = [phone2idx[phone] for phone in sample]
        X_all.append(x)
    X_all = np.array(X_all, dtype=object)
    return X_all, labels


def save_numpy_data(data, filename):
    with open(filename, 'wb') as f:
        np.save(f, data)
    print("Saved in {}".format(filename))



def main():
    df_light_en, df_speaker_en, df_speaker_fr = load_df()
    df_en_all = pd.concat([df_light_en, df_speaker_en], ignore_index=True)
    # Build mapping of phone->index
    phone2idx = build_phone_vocab(df_en_all)
    memory_path = os.path.join(args.memory_dir, args.lang)
    save_phone_idx(phone2idx, memory_path)

    # Train-test split with ratio = 0.15
    ratio = args.split_ratio
    train_en, valid_en = train_test_split(df_en_all, random_state=1, test_size=ratio, shuffle=True)
    dev_en, test_en = train_test_split(valid_en, random_state=1, test_size=0.5)
    # Reset indices of dataframe
    train_en = train_en.reset_index(drop=True)
    dev_en = dev_en.reset_index(drop=True)
    test_en = test_en.reset_index(drop=True)
    # Save train and test files in "memory" directory
    save_train_test(train_en, dev_en, test_en, memory_path)

    # Save x and y
    train_x, train_labels = get_xy(train_en, phone2idx)
    save_numpy_data(train_x, os.path.join(memory_path, "train_x.npy"))
    save_numpy_data(train_labels, os.path.join(memory_path, "train_y.npy"))

    dev_x, dev_labels = get_xy(dev_en, phone2idx)
    save_numpy_data(dev_x, os.path.join(memory_path, "dev_x.npy"))
    save_numpy_data(dev_labels, os.path.join(memory_path, "dev_y.npy"))

    test_x, test_labels = get_xy(test_en, phone2idx)
    save_numpy_data(test_x, os.path.join(memory_path, "test_x.npy"))
    save_numpy_data(test_labels, os.path.join(memory_path, "test_y.npy"))




if __name__ == '__main__':
    main()
