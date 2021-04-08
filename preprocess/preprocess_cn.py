import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import json
import argparse


CN_dir = "../Datasets/catslu_v2/preprocessed"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--memory_dir', type=str, default="../memory/cn")
    return parser.parse_args()





def build_phone_vocab(df_all, temp, cnt_):
    length = len(df_all['transcript'])
    cnt = cnt_
    for i in range(length):
        if type(df_all['transcript'][i]) != str:
            continue
        for phone in df_all['transcript'][i].split(" "):
            if phone not in temp:
                temp[phone] = cnt
                cnt += 1
    return temp, cnt




def save_phone_idx(phone2idx, save_dir="../memory/cn"):
    with open(os.path.join(save_dir, "phone_idx.json"), 'w', encoding='utf-8') as f:
        json.dump(phone2idx, f)
    print("phone_idx.json saved in {}".format(os.path.abspath(save_dir)))



def load_phone_idx(file_path="../memory/cn/phone_idx.json"):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_xy(df, phone2idx):
    length = len(df)
    labels = df['label']
    labels_return = list()
    df = df['transcript']
    X_all = list()
    for i in range(length):
        if type(df[i]) != str:
            continue
        sample = df[i]
        x = [phone2idx[phone] for phone in sample.split(" ")]
        X_all.append(x)
        labels_return.append(labels[i])
    X_all = np.array(X_all, dtype=object)
    return X_all, labels_return




def save_numpy_data(filename, data):
    with open(filename, 'wb') as f:
        np.save(f, data)



def main(args):
    train = pd.read_csv(os.path.join(CN_dir, 'train.csv'), header=0)
    vad = pd.read_csv(os.path.join(CN_dir, 'dev.csv'), header=0)
    test = pd.read_csv(os.path.join(CN_dir, 'test.csv'), header=0)

    phone2idx, cnt = build_phone_vocab(train, dict(), 0)
    phone2idx, cnt = build_phone_vocab(vad, phone2idx, cnt)
    phone2idx, cnt = build_phone_vocab(test, phone2idx,cnt)

    save_phone_idx(phone2idx, args.memory_dir)

    train_x, train_labels = get_xy(train, phone2idx)
    vad_x, vad_labels = get_xy(vad, phone2idx)
    test_x, test_labels = get_xy(test, phone2idx)

    save_numpy_data("../memory/cn/train_x.npy", train_x)
    save_numpy_data("../memory/cn/train_y.npy", train_labels)
    save_numpy_data("../memory/cn/dev_x.npy", vad_x)
    save_numpy_data("../memory/cn/dev_y.npy", vad_labels)
    save_numpy_data("../memory/cn/test_x.npy", test_x)
    save_numpy_data("../memory/cn/test_y.npy", test_labels)



if __name__ == '__main__':
    args = get_args()
    main(args)







