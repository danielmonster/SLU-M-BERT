import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from lscn import LSCNsClassifier
from phoneset import PhoneDataset, load_phone_idx
import os
from sklearn.metrics import f1_score, accuracy_score, classification_report

import argparse

# python3 baseline/eval.py --test_dir=memory/enfr/en --model=best_model/en_best.pt

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, required=True) # ex: memory/enfr/en
    parser.add_argument('--model', type=str, default="best_model/en_best.pt")
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--num_filters', type=int, default=128)
    parser.add_argument('--lstm_hidden', type=int, default=128)
    parser.add_argument('--num_lstm_layers', type=int, default=1)
    parser.add_argument('--weight_tying', type=int, default=1)
    return parser.parse_args()



def report_acc_f1(y_pred, y_true):
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    return f1_macro, f1_micro, accuracy, report




def evaluate(model, device, test_loader):
    model.to(device)
    model.eval()
    preds_all = []
    labels_all = []
    with torch.no_grad():
    
        for batch_idx, (x_padded, x_lengths, target) in enumerate(test_loader):
            x_padded = x_padded.to(device)
            target = target.to(device)
            outputs = model(x_padded, x_lengths)
            _, predicted = torch.max(outputs.data, 1)
            preds_all += predicted.cpu().numpy().tolist()
            labels_all += target.cpu().numpy().tolist()
    
    f1_macro, f1_micro, acc, report = report_acc_f1(preds_all, labels_all)
    print("F1 Macro: {}, F1 Micro: {}".format(f1_macro, f1_micro))
    print("Accuracy: {}".format(acc))
    print(report)
    

def main(args):
    test_x = np.load(os.path.join(args.test_dir, "test_x.npy"), allow_pickle=True)
    test_y = np.load(os.path.join(args.test_dir, "test_y.npy"), allow_pickle=True)
    phone2idx = load_phone_idx(os.path.join(args.test_dir, "phone_idx.json"))

    test_dataset = PhoneDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, \
                            num_workers=8, collate_fn=PhoneDataset.collate_fn)

    vocab_size = len(phone2idx) + 1
    num_classes = len(np.unique(test_y))
    embed_dim = args.embed_dim
    num_filters = args.num_filters
    lstm_hidden = args.lstm_hidden
    num_lstm_layers = args.num_lstm_layers
    weight_tying = args.weight_tying

    model = LSCNsClassifier(vocab_size, num_classes, embed_dim, num_filters, \
                            lstm_hidden, num_lstm_layers, weight_tying)
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(args.model))
    evaluate(model, device, test_loader)

if __name__ == '__main__':
    args = get_args()
    main(args)
