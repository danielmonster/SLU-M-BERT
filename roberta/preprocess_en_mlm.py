import numpy as np
import pandas as pd
import os
from ast import literal_eval
import argparse

# python3 roberta/preprocess_en_mlm.py --dir=Datasets/en_LM_data --outpath=memory/en_lm.npy

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--outpath', type=str, required=True)
    return parser.parse_args()



def main(args):
    df1 = pd.read_csv(os.path.join(args.dir, "A_0_en.csv"))
    df2 = pd.read_csv(os.path.join(args.dir, "B_1_en.csv"))
    df3 = pd.read_csv(os.path.join(args.dir, "C_2_en.csv"))
    df4 = pd.read_csv(os.path.join(args.dir, "D_3_en.csv"))
    data = pd.concat([df1, df2, df3, df4], ignore_index=True)
    df_phones = data['phones'].apply(lambda x: literal_eval(x))
    phones_samples = df_phones.to_numpy()
    X_all = []
    for sample in phones_samples:
        x = ' '.join(sample)
        X_all.append(x)
    X_all = np.array(X_all, dtype=object)
    with open(args.outpath, 'wb') as f:
        np.save(f, X_all)

if __name__ == '__main__':
    args = get_args()
    main(args)
