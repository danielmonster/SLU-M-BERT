from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import json


def load_phone_idx(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


class PhoneDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.length = len(Y)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, i):
        return self.X[i], self.Y[i]
    
    def collate_fn(batch):
        batch_x = [torch.as_tensor(x) for x, y in batch]
        batch_x_length = [len(x) for x, y in batch]
        batch_x_padded = pad_sequence(batch_x, batch_first=True)
        batch_y = torch.as_tensor([y for x, y in batch])
        return batch_x_padded, batch_x_length, batch_y


class LMDataset(Dataset):
    def __init__(self, X, eos_index):
        for x in X:
            x.append(eos_index)
        self.X = X
        self.length = len(X)
    
    def __len__(self):
        return self.length

    def __getitem__(self, i):
        x = self.X[i][:-1]
        y = self.X[i][1:]
        return x, y

    def collate_fn(batch):
        batch_x = [torch.as_tensor(x) for x, y in batch]
        batch_y = [torch.as_tensor(y) for x, y in batch]
        # Each sample of x has the same length as y
        batch_length = [len(x) for x, y in batch]
        batch_x_padded = pad_sequence(batch_x, batch_first=True)
        batch_y_padded = pad_sequence(batch_y, batch_first=True)
        return batch_x_padded, batch_y_padded, batch_length
