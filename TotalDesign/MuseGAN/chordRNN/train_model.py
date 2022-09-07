import numpy as np
import torch
import pickle
import random
from MuseGAN.chordRNN.model.generator import Generator, EarlyStopping


def train(train_data, test_data, num_epochs=1000, batch_size=1000):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_function = torch.nn.MSELoss()
    generator = Generator().to(device)
    early_stop = EarlyStopping(patience=5)
    optimizer = torch.optim.Adam(generator.parameters(), lr=0.01)
    num_batches = len(train_data) // batch_size
    train_data = np.reshape(train_data, (-1, 8, 16, 12))
    test_data = np.reshape(test_data, (-1, 8, 16, 12))

    for epoch in range(num_epochs):
        generator.train()
        sum_loss = 0
        for batch in range(num_batches):
            sub_data = train_data[batch * batch_size:(batch + 1) * batch_size]
            sub_data = torch.tensor(sub_data).to(device=device, dtype=torch.float)

            train_x = sub_data[:, :7, :, :]
            train_y = sub_data[:, 1:, :, :]

            output_list = []
            for i in range(7):
                x = train_x[:, i]
                x = torch.squeeze(x, dim=1)
                output = generator(x)
                output_list.append(output)

            output = torch.stack(output_list, dim=1)
            loss = loss_function(output, train_y)
            sum_loss += loss.item()
            generator.zero_grad()
            loss.backward()
            optimizer.step()
        mean_loss = sum_loss/num_batches

        generator.eval()
        test_start = random.randrange(len(test_data)-10000)
        sub_data = torch.tensor(test_data[test_start:test_start+10000]).to(device=device, dtype=torch.float)

        test_x = sub_data[:, :7, :, :]
        test_y = sub_data[:, 1:, :, :]
        output_list = []
        for i in range(7):
            x = test_x[:, i]
            x = torch.squeeze(x, dim=1)
            output = generator(x)
            output_list.append(output)

        output = torch.stack(output_list, dim=1)
        test_loss = loss_function(output, test_y)

        print("Epoch : %d/%d, train_loss : %1.5f, test_loss : %1.5f" % (epoch + 1, num_epochs, mean_loss, test_loss))
        # if epoch % 10 == 0:
        #     print("Epoch : %d/%d, loss : %1.5f" % (epoch+1, num_epochs, mean_loss))

        early_stop.step(test_loss)
        if early_stop.is_stop():
            break

    torch.save(generator.state_dict(), 'parameter/lstm.prm')

if __name__ == '__main__':
    with open('../chordRNN/data/data_train.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open('../chordRNN/data/data_test.pkl', 'rb') as f:
        test_data = pickle.load(f)
    train(train_data, test_data, 500)