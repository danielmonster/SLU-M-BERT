import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
import time
from lscn import LSCNsClassifier
from phoneset import LMDataset, PhoneDataset, load_phone_idx
import os

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lm_epochs', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--num_filters', type=int, default=128)
    parser.add_argument('--lstm_hidden', type=int, default=128)
    parser.add_argument('--num_lstm_layers', type=int, default=1)
    parser.add_argument('--weight_tying', type=int, default=0)
    parser.add_argument('--dir', type=str, required=True) # ex: "memory/enfr/en"
    parser.add_argument('--save_model_path', type=str, default="best_model/en_best.pt")
    parser.add_argument('--verbose', type=int, default=1)
    return parser.parse_args()


# Reproducibility
torch.manual_seed(123)
np.random.seed(123)


def train_epoch_lm(model, train_loader, criterion, optimizer, device, verbose):
    model.train()
    running_loss = 0.0
    start_time = time.time()

    for batch_idx, (x_padded, y_padded, lengths) in enumerate(train_loader):
        optimizer.zero_grad()
        x_padded = x_padded.to(device)
        y_padded = y_padded.to(device)
        logits = model(x_padded, lengths, lm=True)
        loss = criterion(logits, y_padded)
        running_loss += loss.cpu().item()

        loss.backward()
        optimizer.step()

    end_time = time.time()
    running_loss /= len(train_loader)
    if verbose:
        print("Training loss for LM: ", running_loss, "Time: ", end_time - start_time, 's')
    return running_loss


def train_epoch(model, train_loader, criterion, optimizer, device, verbose):
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
    if verbose:
        print("Training loss: ", running_loss, "Time: ", end_time - start_time, 's')
        print("Training Accuracy", accuracy, "%")
    return running_loss, accuracy


def valid_epoch(model, valid_loader, criterion, device, verbose):
    start_time = time.time()
    with torch.no_grad():
        model.eval()
        running_loss = 0.0
        total_predictions = 0
        correct_predictions = 0
    
        for batch_idx, (x_padded, x_lengths, target) in enumerate(valid_loader):
            x_padded = x_padded.to(device)
            target = target.to(device)
            outputs = model(x_padded, x_lengths)
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += target.size(0)
            correct_predictions += (predicted == target).sum().item()
            loss = criterion(outputs, target).detach()
            running_loss += loss.item()
    
    end_time = time.time()
    running_loss /= len(valid_loader)
    accuracy = (correct_predictions/total_predictions) * 100.0
    if verbose:
        print("Validation loss: ", running_loss, "Time: ", end_time - start_time, 's')
        print("Validation Accuracy", accuracy, "%")
    return running_loss, accuracy



def main(args):
    train_x = np.load(os.path.join(args.dir, "train_x.npy"), allow_pickle=True)
    train_y = np.load(os.path.join(args.dir, "train_y.npy"), allow_pickle=True)
    valid_x = np.load(os.path.join(args.dir, "dev_x.npy"), allow_pickle=True)
    valid_y = np.load(os.path.join(args.dir, "dev_y.npy"), allow_pickle=True)
    phone2idx = load_phone_idx(os.path.join(args.dir, "phone_idx.json"))

    eos_index = len(phone2idx)
    vocab_size = len(phone2idx) + 1
    train_dataset_LM = LMDataset(train_x, eos_index)
    train_loader_LM = DataLoader(train_dataset_LM, shuffle=True, \
            batch_size=128, num_workers=8, collate_fn=LMDataset.collate_fn)

    batch_size = args.batch_size
    train_dataset = PhoneDataset(train_x, train_y)
    valid_dataset = PhoneDataset(valid_x, valid_y)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, \
                            num_workers=8, collate_fn=PhoneDataset.collate_fn)
    valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=batch_size, \
                            collate_fn=PhoneDataset.collate_fn)

    num_classes = len(np.unique(train_y))
    
    embed_dim = args.embed_dim
    num_filters = args.num_filters
    lstm_hidden = args.lstm_hidden
    num_lstm_layers = args.num_lstm_layers
    weight_tying = args.weight_tying
    model = LSCNsClassifier(vocab_size, num_classes, embed_dim, num_filters, \
                            lstm_hidden, num_lstm_layers, weight_tying)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(model)
    verbose = args.verbose

    # Train LM model
    for epoch in range(args.lm_epochs):
        train_loss = train_epoch_lm(model, train_loader_LM, criterion, optimizer, device, verbose)
        if verbose:
            print("Epoch {} finished.".format(epoch))
            print('='*20)

    # Train the validate the actual model
    scheduler = ReduceLROnPlateau(optimizer, factor=0.3, patience=2, 
                                    verbose=verbose)
    num_epochs = args.epochs
    
    best_model_dict = None
    best_acc = 0

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, verbose)
        valid_loss, valid_acc = valid_epoch(model, valid_loader, criterion, device, verbose)
        scheduler.step(valid_loss)
        if verbose:
            print("Epoch {} finished.".format(epoch))
            print('='*20)
        
        if valid_acc > best_acc:
            best_model_dict = model.state_dict()
            best_acc = valid_acc
    if best_model_dict is not None:
        torch.save(best_model_dict, args.save_model_path)
    print("Best validation accuracy: ", best_acc, "%")
        

if __name__ == "__main__":
    args = get_args()
    main(args)
