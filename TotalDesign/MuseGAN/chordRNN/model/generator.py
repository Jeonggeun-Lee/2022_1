import torch
import torch.nn as nn
import math

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.input_size = 12
        self.hidden_size = 12
        self.num_layers = 2
        self.seq_length = 128//8
        self.in_features = 128//8 * 12
        self.out_features = 128//8 * 12
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.in_features, self.out_features)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        output, _ = self.lstm(x)
        output = torch.reshape(output, (-1, 16*12))
        output = self.relu(output)
        output = self.fc(output)
        output = self.sigmoid(output)

        output = torch.reshape(output, (-1, 16, 12))

        return output


class EarlyStopping:
    def __init__(self, patience=5):
        self.loss = math.inf
        self.patience = 0
        self.patience_limit = patience

    def step(self, loss):
        if self.loss > loss:
            self.patience = 0
        else:
            self.patience += 1
        self.loss = loss
    def is_stop(self):
        return self.patience >= self.patience_limit




