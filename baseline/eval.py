import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from lscn import LSCNsClassifier
from phoneset import PhoneDataset, load_phone_idx
import os


import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, required=True) # ex: memory/enfr/en
    parser.add_argument('--model', type=str, default="best_model/en_best.pt")
    return parser.parse_args()

args = get_args()



def evaluate(model, device, test_loader):
    model.to(device)
    model.eval()
    results = []
    with torch.no_grad():
        total_predictions = 0
        correct_predictions = 0
    
        for batch_idx, (x_padded, x_lengths, target) in enumerate(test_loader):
            x_padded = x_padded.to(device)
            target = target.to(device)
            outputs = model(x_padded, x_lengths)
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += target.size(0)
            correct_predictions += (predicted == target).sum().item()
    
    accuracy = (correct_predictions/total_predictions) * 100.0
    print("Accuracy: ", accuracy, "%")
    return accuracy
    

def main():
    test_x = np.load(os.path.join(args.test_dir, "test_x.npy"), allow_pickle=True)
    test_y = np.load(os.path.join(args.test_dir, "test_y.npy"), allow_pickle=True)
    phone2idx = load_phone_idx(os.path.join(args.test_dir, "phone_idx.json"))

    max_label = np.max(test_y)
    test_y[test_y == max_label] = len(np.unique(test_y)) - 1

    test_dataset = PhoneDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, \
                            num_workers=8, collate_fn=PhoneDataset.collate_fn)

    vocab_size = len(phone2idx)
    num_classes = len(np.unique(test_y))

    model = LSCNsClassifier(vocab_size, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(args.model))
    evaluate(model, device, test_loader)

if __name__ == '__main__':
    main()
