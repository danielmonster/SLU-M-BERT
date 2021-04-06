from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import json


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

    
def load_phone_idx(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
