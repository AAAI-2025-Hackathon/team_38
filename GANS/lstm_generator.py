import torch
import torch.nn as nn

class LSTMGenerator(nn.Module):


    def __init__(self, in_dim, out_dim, n_layers=1, hidden_dim=256):
        super().__init__()
        self.in_dim=in_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layers, batch_first=True)
        self.linear = nn.Sequential(nn.Linear(hidden_dim, out_dim), nn.Tanh())

    def forward(self, input):
        batch_size, seq_len = input.size(0), input.size(1)
        # h_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        # c_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        h_0 = torch.zeros(self.n_layers,self.hidden_dim)
        c_0 = torch.zeros(self.n_layers, self.hidden_dim)
        input=input.view(seq_len,self.in_dim)

        recurrent_features, _ = self.lstm(input, (h_0, c_0))
        outputs = self.linear(recurrent_features.contiguous().view(batch_size*seq_len, self.hidden_dim))
        outputs = outputs.view(batch_size, seq_len, self.out_dim)
        return outputs