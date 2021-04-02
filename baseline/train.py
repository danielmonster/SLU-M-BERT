import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import time
from lscn import LSCNsClassifier
from phoneset import PhoneDataset

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=120)
    return parser.parse_args()

args = get_args()



TRAIN_EN_X = "../memory/enfr/en/train_x.npy"
TRAIN_EN_Y = "../memory/enfr/en/train_y.npy"
VALID_EN_X = "../memory/enfr/en/valid_x.npy"
VALID_EN_Y = "../memory/enfr/en/valid_y.npy"



def load_phone_idx(file_path="../memory/enfr/en/phone_idx.json"):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total_predictions = 0
    correct_predictions = 0
    
    start_time = time.time()
    for batch_idx, (x_padded, x_lengths, target) in enumerate(train_loader):
        optimizer.zero_grad()
        x_padded = x_padded.to(device)
        target = target.to(device)
        outputs = model(x_padded, x_lengths)
        
        loss = criterion(outputs, target)
        _, predicted = torch.max(outputs.data, 1)
        total_predictions += target.size(0)
        correct_predictions += (predicted == target).sum().item()
        running_loss += loss.item()
        
        loss.backward()
        optimizer.step()
    
    end_time = time.time()
    running_loss /= len(train_loader)
    accuracy = (correct_predictions/total_predictions) * 100.0
    print("Training loss: ", running_loss, "Time: ", end_time - start_time, 's')
    print("Training Accuracy", accuracy, "%")
    return running_loss, accuracy


def valid_epoch(model, valid_loader, criterion, device):
    start_time = time.time()
    with torch.no_grad():
        model.eval()
        running_loss = 0.0
        total_predictions = 0
        correct_predictions = 0
    
        for batch_idx, (x_padded, x_lengths, target) in enumerate(train_loader):
            x_padded = x_padded.to(device)
            target = target.to(device)
            outputs = model(x_padded, x_lengths)
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += target.size(0)
            correct_predictions += (predicted == target).sum().item()
            loss = criterion(outputs, target).detach()
            running_loss += loss.item()
    
    end_time = time.time()
    running_loss /= len(train_loader)
    accuracy = (correct_predictions/total_predictions) * 100.0
    print("Validation loss: ", running_loss, "Time: ", end_time - start_time, 's')
    print("Validation Accuracy", accuracy, "%")
    return running_loss, accuracy



def main():
    train_x = np.load(TRAIN_EN_X, allow_pickle=True)
    train_y = np.load(TRAIN_EN_Y, allow_pickle=True)
    valid_x = np.load(VALID_EN_X, allow_pickle=True)
    valid_y = np.load(VALID_EN_Y, allow_pickle=True)
    phone2idx = load_phone_idx()

    train_dataset = PhoneDataset(train_x, train_y)
    valid_dataset = PhoneDataset(valid_x, valid_y)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=32, num_workers=8, collate_fn=PhoneDataset.collate_fn)
    valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=32, collate_fn=PhoneDataset.collate_fn)

    vocab_size =  len(phone2idx)
    num_classes = len(np.unique(train_y))

    model = LSCNsClassifier(vocab_size, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(model)

    num_epochs = 120

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss, valid_acc = valid_epoch(model, valid_loader, criterion, device)
        print("Epoch {} finished.".format(epoch))
        print('='*20)

if __name__ == "__main__":
    main()