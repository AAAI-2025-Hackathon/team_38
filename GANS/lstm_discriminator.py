import torch
import torch.nn as nn


class LSTMDiscriminator(nn.Module):

    def __init__(self, in_dim, n_layers=1, hidden_dim=256):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layers, batch_first=True)
        self.linear = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, input):
        batch_size, seq_len = input.size(0), input.size(1)
        # h_0 = torch.zeros(self.n_layers, batch_size,self.hidden_dim)
        # c_0 = torch.zeros(self.n_layers, batch_size,self.hidden_dim)
        input=input.view(input.size(1),1)
        h_0 = torch.zeros(self.n_layers,self.hidden_dim)
        c_0 = torch.zeros(self.n_layers,self.hidden_dim)

        recurrent_features, _ = self.lstm(input, (h_0, c_0))
        outputs = self.linear(recurrent_features.contiguous().view(batch_size*seq_len, self.hidden_dim))
        outputs = outputs.view(batch_size, seq_len, 1)
        return outputs


# if __name__ == "__main__":
#     batch_size = 16
#     seq_len = 32
#     noise_dim = 100
#     seq_dim = 4

#     gen = LSTMGenerator(noise_dim, seq_dim)
#     dis = LSTMDiscriminator(seq_dim)
#     noise = torch.randn(8, 16, noise_dim)
#     gen_out = gen(noise)
#     dis_out = dis(gen_out)
    
#     print("Noise: ", noise.size())
#     print("Generator output: ", gen_out.size())
#     print("Discriminator output: ", dis_out.size())