import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence



class LSCNsClassifier(nn.Module):
    '''
    Input: X = [x1, ..., xt], i.e., a sequence of phones
    x_i is an one-hot index (a unique integer representing the phone, not one-hot vector)
    
    Output: Intent label class
    ----------------------------------------------------------------------------
    Network architecture:

    X  -> embedding layer -> 1d CNN of 128 filters of size 3 -> A
                          -----------------------------------------> concat(A, B)
                          -> 1d CNN of 128 filters of size 5 -> B
    continuing...
    concat(A, B) -> LSTM with hidden size of 128 -> linear -> Softmax
    '''


    def __init__(self, vocab_size, num_classes, embedding_dim=128, 
                    num_filters=128, lstm_hidden=128):
        super(LSCNsClassifier, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)

        self.convA = nn.Conv1d(embedding_dim, num_filters, 3, padding=1)
        self.convB = nn.Conv1d(embedding_dim, num_filters, 5, padding=2)
        self.bn = nn.BatchNorm1d(num_filters * 2)
        self.lstm = nn.LSTM(256, lstm_hidden, batch_first=True)
        self.linear = nn.Linear(lstm_hidden, num_classes)
    
    def forward(self, x, x_lengths):
        # x: (batch_size, seq_len)
        embeds = self.embed(x) # (bs, seq_len, embed_dim)
        out = embeds.transpose(1, 2) # (bs, embed_dim, seq_len)
        A = self.convA(out) # (bs, num_filters, seq_len)
        B = self.convB(out) # (bs, num_filters, seq_len)
        out = torch.cat((A, B), 1) # (bs, 2 * num_filters, seq_len)
        out = self.bn(out)
        out = out.transpose(1, 2) # (bs, seq_len, 2 * num_filters)
        out = pack_padded_sequence(out, x_lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(out) # (bs, seq_len, lstm_hidden)
        # Unpack output
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        # Get the output of the last time step since this is many-to-one
        out = out[:, -1, :] # (bs, lstm_hidden)
        out = self.linear(out) # (bs, num_classes)
        return out