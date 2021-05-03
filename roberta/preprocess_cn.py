import pandas as pd
import numpy as np
import os
import argparse


# mkdir -p memory/roberta/cn
# python3 roberta/preprocess_cn.py --data_dir=Datasets/catslu_v2/preprocessed/audio/  \
# --memory_dir=memory/roberta/cn/

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--memory_dir', type=str, required=True)
    return parser.parse_args()




def save_numpy_data(filename, data):
    with open(filename, 'wb') as f:
        np.save(f, data)



def main(args):
    train = pd.read_csv(os.path.join(args.data_dir, 'train.csv'), header=0)
    dev = pd.read_csv(os.path.join(args.data_dir, 'dev.csv'), header=0)
    test = pd.read_csv(os.path.join(args.data_dir, 'test.csv'), header=0)

    train = train.dropna()
    dev = dev.dropna()
    test = test.dropna()
    
    train_x, train_labels = train['transcript'].to_numpy(), train['label'].to_numpy()
    dev_x, dev_labels = dev['transcript'].to_numpy(), dev['label'].to_numpy()
    test_x, test_labels = test['transcript'].to_numpy(), test['label'].to_numpy()

    save_numpy_data(os.path.join(args.memory_dir, "train_x.npy"), train_x)
    save_numpy_data(os.path.join(args.memory_dir, "train_y.npy"), train_labels)
    save_numpy_data(os.path.join(args.memory_dir, "dev_x.npy"), dev_x)
    save_numpy_data(os.path.join(args.memory_dir, "dev_y.npy"), dev_labels)
    save_numpy_data(os.path.join(args.memory_dir, "test_x.npy"), test_x)
    save_numpy_data(os.path.join(args.memory_dir, "test_y.npy"), test_labels)


if __name__ == '__main__':
    args = get_args()
    main(args)







