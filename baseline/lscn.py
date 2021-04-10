import torch
import torch.nn as nn
import torch.nn.functional as F
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


    def __init__(self, vocab_size, num_classes, embedding_dim, 
                    num_filters, lstm_hidden, num_lstm_layers, weight_tying=False):
        super(LSCNsClassifier, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.convA = nn.Conv1d(embedding_dim, num_filters, 3, padding=1)
        self.convB = nn.Conv1d(embedding_dim, num_filters, 5, padding=2)
        self.bn = nn.BatchNorm1d(num_filters * 2)
        self.relu = nn.ReLU()
        
        self.lstm_LM = nn.LSTM(embedding_dim, lstm_hidden, num_layers=1)
        self.lstm = nn.LSTM(num_filters * 2, lstm_hidden, \
                        num_layers=num_lstm_layers, batch_first=True)

        self.linear_LM = nn.Linear(lstm_hidden, embedding_dim)
        self.output_LM = nn.Linear(embedding_dim, vocab_size)
        # nn.init.normal_(self.output_LM.weight, 0, 1)
        if weight_tying:
                self.embed.weight = self.output_LM.weight

        self.linear = nn.Linear(lstm_hidden, num_classes)
    
    def forward(self, x, x_lengths, lm=False):
        # x: (batch_size, seq_len)
        embeds = self.embed(x) # (bs, seq_len, embed_dim)
        if lm:
                packed_tensor = pack_padded_sequence(embeds, x_lengths, batch_first=True, enforce_sorted=False)
                out, _ = self.lstm_LM(packed_tensor)
                out, _ = pad_packed_sequence(out, batch_first=True)
                out = F.relu(self.linear_LM(out))
                logits = self.output_LM(out)
                logits = logits.permute(0, 2, 1)
        else:
                out = embeds.transpose(1, 2) # (bs, embed_dim, seq_len)
                A = self.convA(out) # (bs, num_filters, seq_len)
                B = self.convB(out) # (bs, num_filters, seq_len)
                out = torch.cat((A, B), 1) # (bs, 2 * num_filters, seq_len)
                out = self.bn(out)
                out = self.relu(out)
                out = out.transpose(1, 2) # (bs, seq_len, 2 * num_filters)
                out = pack_padded_sequence(out, x_lengths, batch_first=True, enforce_sorted=False)
                packed_out, (ht, ct) = self.lstm(out) # (bs, seq_len, lstm_hidden)
                logits = self.linear(ht[-1]) # (bs, num_classes)
        return logits