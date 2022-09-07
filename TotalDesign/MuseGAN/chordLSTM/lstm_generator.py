import random as rd
import numpy as np
import torch
import torch.nn as nn


class LSTMGenerator(nn.Module):
    def __init__(self):
        super(LSTMGenerator, self).__init__()
        self.num_classes = 12
        self.input_size = 12
        self.hidden_size = 12
        self.num_layers = 1
        self.seq_length = 1
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.layer_out = nn.Linear(12, self.num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        output, _ = self.lstm(x)
        output = self.layer_out(output)
        output = self.relu(output)

        return output


class EarlyStopping:
    def __init__(self, patience=5):
        self.loss = np.inf
        self.patience = 0
        self.patience_limit = patience

    def step(self, loss):
        if self.loss > loss:
            self.loss = loss
            self.patience = 0
        else:
            self.patience += 1

    def is_stop(self):
        return self.patience >= self.patience_limit


def train(chord):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_x = chord[0:96000, :, :]
    train_y = chord[1:96001, :, :]
    #train_x = chord[0:10000, :, :]
    #train_y = chord[1:10001, :, :]

    train_x_tensor = torch.Tensor(train_x)
    train_y_tensor = torch.Tensor(train_y)
    train_x_tensor_final = torch.reshape(
        train_x_tensor,
        (train_x_tensor.shape[0], train_x_tensor.shape[1], train_x_tensor.shape[2])
    )

    lstm_generator = LSTMGenerator().to(device)
    loss_function = torch.nn.MSELoss()
    early_stop = EarlyStopping(patience=5)
    optimizer = torch.optim.Adam(lstm_generator.parameters(), lr=0.01)

    for epoch in range(1000):
        outputs = lstm_generator.forward(train_x_tensor_final.to(device))
        optimizer.zero_grad()
        loss = loss_function(outputs, train_y_tensor.to(device))
        loss.backward()
        optimizer.step() # improve from loss = back propagation
        early_stop.step(loss.item())
        if epoch % 10 == 0:
            print("Epoch : %d, loss : %1.5f" % (epoch, loss.item()))
        if early_stop.is_stop():
            break
    torch.save(lstm_generator.state_dict(), 'parameter/lstm_1000e.prm')


def generate_chord_sample(chord, parameter_path='parameter/lstm_1000e.prm', result_path='result/generated_chord.npy'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    select = rd.randrange(0,len(chord))
    x = chord[select]
    x = np.reshape(x, (1,128,12))
    x = torch.from_numpy(x).float()

    lstm_generator = LSTMGenerator().to(device)
    lstm_generator.load_state_dict(torch.load(parameter_path))
    lstm_generator.eval()

    test_predict = lstm_generator(x.to(device))
    result = test_predict.data.detach().cpu().numpy()

    for j in range(128):
        result_del = np.array(result[0][j])
        max_1 = np.argmax(result_del)
        result_del = np.delete(result_del, max_1)
        max_2 = np.argmax(result_del)
        result_del = np.delete(result_del, max_2)
        max_3 = np.argmax(result_del)
        for k in range(12):
            if result[0][j][k] >= result_del[max_3]:
                result[0][j][k] = 1
            else:
                result[0][j][k] = 0
    
    np.save(result_path, result)



